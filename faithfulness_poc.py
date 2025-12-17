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
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(id=nid, node_type=node_type, time=float(time), feat=np.asarray(feat, dtype=float))
        return nid

    def _add_edge(self, source: NodeId, destination: NodeId, relation: str) -> None:
        self.edges.append(Edge(source=source, destination=destination, relation=relation))
        self.adj.setdefault(source, []).append((relation, destination))

    def create_root(self, time: float, feat: np.ndarray) -> NodeId:
        rid = self._new_node(self.schema.root_type, time, feat)
        self.root = rid
        return rid

    @staticmethod
    def _pick_relation(relation_types: List[str], probs_row: np.ndarray, rng: np.random.Generator, strategy: Literal["mode", "sample"]) -> str:
        if strategy == "mode":
            return relation_types[int(np.argmax(probs_row))]
        # sample
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

        self.generating_concepts[generating_concept.name] = self.generating_concepts.get(generating_concept.name, 0) + int(instances)

        for _ in range(int(instances)):
            cur = self.root
            root_time = self.nodes[self.root].time

            cur_type = self.nodes[cur].node_type

            for k in range(L):
                relation = self._pick_relation(generating_concept.relation_types, generating_concept.relation_probs[k], rng, relation_pick)

                if node_types_for_hops is not None:
                    destination_type = node_types_for_hops[k]
                else:
                    destination_type = self.schema.transitions[(cur_type, relation)]

                self.schema.check(cur_type, relation, destination_type)

                abs_dt = float(generating_concept.time_deltas[k])
                t = float(root_time + abs_dt + rng.normal(0.0, time_noise_std))

                mu = generating_concept.feature_centroid[k + 1]
                feat = mu + rng.normal(0.0, feat_noise_std, size=mu.shape)

                nxt = self._new_node(destination_type, t, feat)
                self._add_edge(cur, nxt, relation)

                cur = nxt
                cur_type = destination_type

    def sample_paths(self, L: int) -> List[PathSample]:

        root_time = self.nodes[self.root].time
        samples: List[PathSample] = []

        def dfs(u: NodeId, depth: int, relations: List[str], times: List[float], feats: List[np.ndarray]) -> None:
            if depth == L:
                samples.append(
                    PathSample(
                        relation_sequence=relations.copy(),
                        time_vector=np.array(times, dtype=float),
                        feature_vector=np.stack(feats, axis=0),
                    )
                )
                return
            for relation, v in self.adj.get(u, []):
                relations.append(relation)
                times.append(self.nodes[v].time - root_time)
                feats.append(self.nodes[v].feature_embedding)
                dfs(v, depth + 1, relations, times, feats)
                feats.pop()
                times.pop()
                relations.pop()

        dfs(self.root, 0, [], [], [])
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
    def _time_and_feature_similarity(root_features: np.ndarray, path_times: np.ndarray, path_feature_centroids: np.ndarray, hypothesis_concept: Concept) -> float:
        
        # Extract the features from the hypothesized concept
        concept_times = np.array(hypothesis_concept.time_deltas, dtype=float)
        concept_root_features = np.array(hypothesis_concept.feature_centroid[0])
        concept_features = np.stack(hypothesis_concept.feature_centroid[1:], axis=0)

        # Determine raw distance of paths from hypothesized concepts
        root_difference = root_features - concept_root_features
        time_difference = path_times - concept_times
        feature_difference = path_feature_centroids - concept_features

        squared_norm = float(np.sum(root_difference * root_difference) + np.sum(time_difference * time_difference) + np.sum(feature_difference * feature_difference))
        return float(np.exp(-hypothesis_concept.gamma * squared_norm))

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