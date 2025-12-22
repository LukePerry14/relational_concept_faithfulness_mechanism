from __future__ import annotations
from typing import Dict, List, Literal, Tuple, Optional
import numpy as np
from custom_dataclasses import *
NodeId = int



class Subgraph:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.nodes: Dict[NodeId, Node] = {}
        self.edges: List[Edge] = []
        self.adj: Dict[NodeId, List[Tuple[str, NodeId]]] = {}
        self._next_id: int = 0
        self.root: Optional[NodeId] = None
        self.generating_concepts: Dict[str, int] = {}

    def _new_node(self, node_type: str, time: float, feat: np.ndarray) -> NodeId:
        """Create a new node"""
        next_id = self._next_id
        self._next_id += 1
        self.nodes[next_id] = Node(id=next_id, node_type=node_type, time=float(time), feature_embedding=np.asarray(feat, dtype=float))
        return next_id

    def _add_edge(self, source: NodeId, destination: NodeId, relation: str):
        """Add an edge between given nodes"""
        self.edges.append(Edge(source=source, destination=destination, relation=relation))
        self.adj.setdefault(source, []).append((relation, destination))

    def create_root(self, time: float, feat: np.ndarray) -> NodeId:
        """Create root node for subgraph"""
        rid = self._new_node(self.schema.root_type, time=time, feat=feat)
        self.root = rid
        return rid

    @staticmethod
    def _pick_relation(relation_types: List[str], probs_row: np.ndarray, rng: np.random.Generator, strategy: Literal["mode", "sample"]) -> str:
        # Pick relation with highest probability
        if strategy == "mode":
            return relation_types[int(np.argmax(probs_row))]
        
        # Sample relation using relation probablities as weights
        idx = int(rng.choice(len(relation_types), p=probs_row))
        return relation_types[idx]

    def add_evidence(
        self,
        generating_concept: Concept,
        rng: Optional[np.random.Generator] = None,
        time_noise_std: float = 0.0,
        feat_noise_std: float = 0.0,
        instances: int = 1,
        node_types_for_hops: Optional[List[str]] = None,
        relation_pick: Literal["mode", "sample"] = "mode",
    ):

        rng = rng or np.random.default_rng()
        L = generating_concept.L()

        # Keep record of the generating concept used
        self.generating_concepts[generating_concept.name] = self.generating_concepts.get(generating_concept.name, 0) + int(instances)

        # Generate correct number of instances
        for _ in range(int(instances)):
            
            # Same root node
            current = self.root
            root_time = self.nodes[self.root].time
            current_type = self.nodes[current].node_type

            for k in range(L):
                # Extract relation from concept
                relation = self._pick_relation(generating_concept.relation_types, generating_concept.relation_probs[k], rng, relation_pick)

                # Extract node type (not meaningful in current implementation)
                if node_types_for_hops is not None:
                    destination_type = node_types_for_hops[k]
                else:
                    destination_type = self.schema.transitions[(current_type, relation)]

                # Ensure correctness of proposition
                self.schema.check(current_type, relation, destination_type)

                # time delta 
                time_delta = float(generating_concept.time_deltas[k])
                time = float(root_time + time_delta + rng.normal(0.0, time_noise_std))

                # features
                centroid = generating_concept.feature_centroid[k + 1]
                features = centroid + rng.normal(0.0, feat_noise_std, size=centroid.shape)

                # add node and edge
                nxt = self._new_node(destination_type, time, features)
                self._add_edge(current, nxt, relation)

                # continue to next node
                current = nxt
                current_type = destination_type

    def sample_paths(
        self,
        L: int,
        n_samples: int = 128,
        rng: np.random.Generator | None = None,
    ) -> List[PathSample]:
        """
        Randomly sample path instances of hop-length L from the subgraph.
        """
        if rng is None:
            rng = np.random.default_rng()

        root_id = self.root
        root_time = self.nodes[root_id].time  # seed reference time (t=0)

        samples: List[PathSample] = []

        # Sample required number of paths
        for _ in range(n_samples):
            current = root_id
            relations: List[str] = []
            times: List[float] = []
            feats: List[np.ndarray] = []

            for _depth in range(L):
                out_edges = self.adj.get(current, [])
                if not out_edges:
                    # Dead end before reaching length L; discard this attempt.
                    relations = []
                    times = []
                    feats = []
                    break

                # Choose a random outgoing edge uniformly.
                idx = int(rng.integers(0, len(out_edges)))
                relation, v = out_edges[idx]

                relations.append(relation)
                times.append(self.nodes[v].time - root_time)          # time offset from seed
                feats.append(self.nodes[v].feature_embedding)         # node feature at this hop

                current = v

            # Add sample
            if len(relations) == L:
                samples.append(
                    PathSample(
                        relation_sequence=relations,
                        time_vector=np.array(times, dtype=float),
                        feature_vector=np.stack(feats, axis=0),
                    )
                )

        return samples


    @staticmethod
    def _relational_similarity(path_relations: List[str], hypothesis_concept: Concept) -> float:
        """
        Similarity of relational sequence
        """
        
        # extract relation -> idx mapping
        relation_lookup = hypothesis_concept.relation_index()
        
        # Extract hypothesized relation probabilities
        relation_probabilities = hypothesis_concept.relation_probs
        if len(path_relations) != relation_probabilities.shape[0]:
            return 0.0

        relational_similarity = 1.0
        
        # Iterate the hypothesis concept
        for idx, relation in enumerate(path_relations):
            
            # Extract the amount of probability mass assigned to the correct relation
            correct_probability_assignment = relation_lookup.get(relation)
            if correct_probability_assignment is None:
                return 0.0
            
            # scale overall similarity by the proportion of correctly assigned mass
            relational_similarity *= float(relation_probabilities[idx, correct_probability_assignment])
            if relational_similarity == 0.0:
                return 0.0
            
        return relational_similarity

    @staticmethod
    def _time_and_feature_similarity(
        root_features: np.ndarray,
        path_times: np.ndarray,
        path_feature_centroids: np.ndarray,
        hypothesis_concept: Concept,
    ) -> float:
        """
        Uses the Option-1 reparameterisation:
            S(d) = exp(-gamma d^2),  gamma = -ln(k) / rho^2  so that S(rho)=k.
        Supports per-hop rho vectors (time/features). Falls back to existing gamma if rho not provided.
        """

        # ---- hyperparameter for interpretability: similarity threshold ----
        # Prefer concept-provided threshold; otherwise default to "half-similarity radius".
        k = float(getattr(hypothesis_concept, "rbf_k", 0.5))
        if not (0.0 < k < 1.0):
            raise ValueError(f"rbf_k must be in (0,1), got {k}.")
        neg_log_k = -float(np.log(k))
        eps = 1e-12

        # ---- Concept prototypes ----
        concept_times = np.array(hypothesis_concept.time_deltas, dtype=float)                  # (L,)
        concept_root_features = np.array(hypothesis_concept.feature_centroid[0], dtype=float)  # (D,)
        concept_features = np.stack(hypothesis_concept.feature_centroid[1:], axis=0).astype(float)  # (L,D)

        # ---- Differences ----
        root_diff = root_features - concept_root_features                      # (D,)
        time_diff = path_times - concept_times                                 # (L,)
        feat_diff = path_feature_centroids - concept_features                  # (L,D)
        feat_sq = np.sum(feat_diff * feat_diff, axis=1)                        # (L,)

        L = int(time_diff.shape[0])

        # ---- Retrieve rho (preferred) or gamma (legacy), and convert to gamma ----
        def _gamma_from_rho_or_gamma(rho_attr: str, gamma_vec: np.ndarray) -> np.ndarray:
            """
            If concept has a callable <rho_attr>() returning length-L radii, use:
                gamma = -ln(k) / rho^2
            Else use provided legacy gamma vector.
            """
            rho_fn = getattr(hypothesis_concept, rho_attr, None)
            if callable(rho_fn):
                rho = np.asarray(rho_fn(), dtype=float).reshape(-1)
                if rho.size == 1:
                    rho = np.full(L, float(rho.item()), dtype=float)
                if rho.size != L:
                    raise ValueError(f"{rho_attr} must be length {L}, got {rho.size}.")
                return neg_log_k / np.maximum(rho * rho, eps)
            return gamma_vec

        # Legacy per-hop gamma vectors (still supported)
        gamma_t_legacy = np.asarray(hypothesis_concept.gamma_time_vec(), dtype=float)  # (L,)
        gamma_x_legacy = np.asarray(hypothesis_concept.gamma_feat_vec(), dtype=float)  # (L,)

        # New: per-hop radii -> gamma
        gamma_t = _gamma_from_rho_or_gamma("rho_time_vec", gamma_t_legacy)
        gamma_x = _gamma_from_rho_or_gamma("rho_feat_vec", gamma_x_legacy)

        # Root: either rho_root (preferred) or legacy gamma_root
        rho_root_fn = getattr(hypothesis_concept, "rho_root_scalar", None)
        if callable(rho_root_fn):
            rho_root = float(rho_root_fn())
            gamma_root = neg_log_k / max(rho_root * rho_root, eps)
        else:
            gamma_root = float(hypothesis_concept.gamma_root_scalar())

        # ---- Weighted exponent ----
        root_term = gamma_root * float(np.sum(root_diff * root_diff))
        time_term = float(np.sum(gamma_t * (time_diff * time_diff)))
        feat_term = float(np.sum(gamma_x * feat_sq))

        return float(np.exp(-(root_term + time_term + feat_term)))

    def evidence_score(self, hypothesis_concept: Concept, L: Optional[int] = None) -> float:

        # Determine concept Length
        L = int(L if L is not None else hypothesis_concept.L())
        
        # Sample paths from subgraph
        paths = self.sample_paths(L=L)

        root_features = self.nodes[self.root].feature_embedding
        similarity_sum = 0.0
        
        for path in paths:
            # Calculate relational similarity
            relational_similarity = self._relational_similarity(path.relation_sequence, hypothesis_concept)
            if relational_similarity == 0.0:
                continue
            
            # Calculate time and feature similarity
            time_and_feature_similarity = self._time_and_feature_similarity(root_features, path.time_vector, path.feature_vector, hypothesis_concept)
            similarity_sum += relational_similarity * time_and_feature_similarity

        # Create denominator using hill function style to enforce evidence boundary
        denominator = similarity_sum + float(hypothesis_concept.tau)
        return 0.0 if denominator == 0.0 else float(similarity_sum / denominator)