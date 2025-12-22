from __future__ import annotations
from typing import Dict, List, Literal, Tuple, Optional
import numpy as np
from collections import deque

from custom_dataclasses import *
NodeId = int

DEFAULT_TAU = 0.5

class Subgraph:
    """
    Local subgraph with path sampling and PDF-style faithfulness scoring.
    """

    def __init__(self, schema: Schema):
        self.schema = schema
        self.nodes: Dict[NodeId, Node] = {}
        self.edges: List[Edge] = []
        self.adj: Dict[NodeId, List[Tuple[str, NodeId]]] = {}
        self._next_id: int = 0
        self.root: Optional[NodeId] = None

    # ---- graph construction ----

    def _new_node(self, node_type: str, time: float, feat: np.ndarray) -> NodeId:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = Node(
            id=nid,
            node_type=node_type,
            time=float(time),
            feature=np.asarray(feat, dtype=float),
        )
        return nid

    def _add_edge(self, src: NodeId, dst: NodeId, relation: str) -> None:
        self.edges.append(Edge(src=src, dst=dst, relation=relation))
        self.adj.setdefault(src, []).append((relation, dst))

    def create_root(self, time: float, feat: np.ndarray) -> NodeId:
        """
        Create the seed node with an absolute timestamp.
        """
        root_id = self._new_node(self.schema.root_type, time=time, feat=feat)
        self.root = root_id
        return root_id

    def add_evidence(self, metapaths):

        if self.root == None:
            raise Exception("Create the root first")
        
        for metapath in metapaths:
            try:
                node_types = metapath.node_types
                node_times = metapath.node_times
                node_features = metapath.node_features
                
                if not (len(node_types) == len(node_times) == len(node_features)):
                    raise Exception("Size mismatch")
                
                prev = self.nodes[self.root]
                
                for idx in range(len(node_types)):
                    
                    if node_types[idx] not in self.schema.transitions[prev.node_type]:
                        raise Exception(f"Invalid node type at hop {idx}")
                    
                    # add node to adjacency
                    node = self.nodes[self._new_node(
                        node_type=node_types[idx],
                        time=node_times[idx],
                        feat=node_features[idx]
                    )]
                    
                    self._add_edge(prev.id, node.id, "NULL")
                    
                    prev = node
            except Exception as e:
                print(f"Failed on metapath {metapath} with {e}")
            
            
        
    
    # ---- path sampling ----

    def sample_paths(self, max_hop, n_samples=128, rng=None):
        """
        We want to sample meta-paths from the immediate surroundings. The number of possible meta-paths allowing for truncations is combinatorial,
        instead, we sample up to the maximum and allow gamma to ignore irrelevant structural nodes
        """
        if self.root == None:
            raise Exception("Create the root first")
        
        root_node = self.nodes[self.root]
        
        frontier = deque()
        frontier.append(self.root)
        
        back_dict = {}
        
        samples = 0
        
        
        while samples < n_samples:
            
            
    #     # rng is unused now; kept only for API compatibility.

    #     root_id = self.root
    #     root_node = self.nodes[root_id]
    #     root_time = float(root_node.time)
    #     root_feat = root_node.feature
    #     D = int(root_feat.shape[0])

    #     # Each element in the frontier is a list of node IDs [root, v1, ..., vk]
    #     from collections import deque
    #     frontier = deque()
    #     frontier.append([root_id])

    #     path_node_lists: List[List[int]] = []

    #     # BFS over paths (not over nodes) to build distinct simple paths
    #     while frontier and len(path_node_lists) < int(n_samples):
    #         path = frontier.popleft()
    #         depth = len(path) - 1  # number of hops from root

    #         # Record this path if it has at least one hop
    #         if depth >= 1:
    #             path_node_lists.append(path)

    #         # If we've already hit the maximum length, do not expand further
    #         if depth >= int(L):
    #             continue

    #         current = path[-1]
    #         out_edges = self.adj.get(current, [])
    #         for _rel, dst in out_edges:
    #             # Avoid cycles: do not revisit nodes already in this path
    #             if dst in path:
    #                 continue
    #             new_path = path + [dst]
    #             frontier.append(new_path)

    #     samples: List[MetaPath] = []

    #     for path in path_node_lists:
    #         # path = [root, v1, v2, ..., vk]
    #         node_ids = path[1:]  # exclude root for meta-path node-type sequence
    #         types: List[str] = []
    #         offsets: List[float] = []
    #         feats: List[np.ndarray] = []

    #         for nid in node_ids:
    #             v_node = self.nodes[nid]
    #             types.append(v_node.node_type)
    #             offsets.append(float(v_node.time - root_time))  # offset from seed
    #             feats.append(v_node.feature)

    #         # Time signature: absolute root time + offsets, padded to length L+1
    #         t_sig = np.full((L + 1,), MISSING_TIME, dtype=float)
    #         t_sig[0] = root_time
    #         if offsets:
    #             t_sig[1 : 1 + len(offsets)] = np.asarray(offsets, dtype=float)

    #         # Feature signature: root feature + per-hop features, padded to length L+1
    #         mu_sig = np.full((L + 1, D), MISSING_FEAT, dtype=float)
    #         mu_sig[0] = root_feat
    #         if feats:
    #             mu_sig[1 : 1 + len(feats)] = np.stack(feats, axis=0).astype(float)

    #         samples.append(MetaPath(path_name=None, node_types=types, node_times=t_sig, node_features=mu_sig))

    #     return samples


    # ---- similarity primitives ----

    @staticmethod
    def _relational_similarity(P: np.ndarray, M: np.ndarray, X: np.ndarray) -> float:
        """
        S_rel = 1 - (1 / ||X||₁) * Σ_{i,j} X_{i,j} (P_{i,j} - M_{i,j})²
        """
        denom = float(np.sum(X))
        if denom <= 0.0:
            return 0.0

        err = (P - M) ** 2
        masked = err * X
        s = 1.0 - float(np.sum(masked) / denom)
        return float(np.clip(s, 0.0, 1.0))

    @staticmethod
    def _time_similarity(
        proto_t: np.ndarray,
        sample_t: np.ndarray,
        gamma_t: np.ndarray,
        k: float,
    ) -> float:
        """
        RBF-like time kernel:

        S_time = exp( ln(k) * Σ_i ( (Δt_i)² / γ_t_i² ) ),
        where k ∈ (0,1) is similarity at distance γ_t.
        Hops with γ_t_i = ∞ or missing sample values are ignored.
        """
        proto_t = np.asarray(proto_t, dtype=float)
        sample_t = np.asarray(sample_t, dtype=float)
        gamma_t = np.asarray(gamma_t, dtype=float)

        if proto_t.shape != sample_t.shape or proto_t.shape != gamma_t.shape:
            raise ValueError("proto_t, sample_t, gamma_t must have same shape")

        if not (0.0 < k < 1.0):
            raise ValueError(f"k must be in (0,1), got {k}")

        # Mask: finite radius and non-missing sample
        mask = np.isfinite(gamma_t) & (np.abs(sample_t - MISSING_TIME) > 1.0)
        if not np.any(mask):
            return 1.0

        diff = proto_t[mask] - sample_t[mask]
        g2 = np.maximum(gamma_t[mask] ** 2, EPS)
        total = float(np.sum((diff ** 2) / g2))
        return float(np.exp(np.log(k) * total))

    @staticmethod
    def _feature_similarity(
        proto_mu: np.ndarray,
        sample_mu: np.ndarray,
        gamma_mu: np.ndarray,
        k: float,
    ) -> float:
        """
        RBF-like feature kernel:

        For each hop i:
          d_i² = ||μ_i - μ̂_i||²
          contribution = d_i² / γ_μ_i²

        S_feat = exp( ln(k) * Σ_i contribution )

        Hops with γ_μ_i = ∞ or missing sample features are ignored.
        """
        proto_mu = np.asarray(proto_mu, dtype=float)
        sample_mu = np.asarray(sample_mu, dtype=float)
        gamma_mu = np.asarray(gamma_mu, dtype=float)

        if proto_mu.shape != sample_mu.shape:
            raise ValueError("proto_mu and sample_mu must have same shape")
        if proto_mu.shape[0] != gamma_mu.shape[0]:
            raise ValueError("gamma_mu must have length L+1 matching hops")

        if not (0.0 < k < 1.0):
            raise ValueError(f"k must be in (0,1), got {k}")

        Lp1, D = proto_mu.shape
        contribs: List[float] = []

        for i in range(Lp1):
            g_i = gamma_mu[i]
            if not np.isfinite(g_i):
                continue  # ignored hop

            if np.all(np.abs(sample_mu[i] - MISSING_FEAT) <= 1.0):
                continue  # missing sample at this hop

            diff = proto_mu[i] - sample_mu[i]
            d2 = float(np.dot(diff, diff))
            g2 = max(g_i ** 2, EPS)
            contribs.append(d2 / g2)

        if not contribs:
            return 1.0

        total = float(np.sum(contribs))
        return float(np.exp(np.log(k) * total))

    # ---- faithfulness / evidence score ----

    def evidence_score(
        self,
        concept: Concept,
        n_samples: int = 128,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """
        Compute the faithfulness/evidence score for a subgraph with respect
        to a concept:

          1. Sample paths p from the subgraph.
          2. For each p, construct:
               - M: one-hot meta-path matrix over node types + NULL.
               - time_signature and feature_signature as defined above.
          3. Compute:
               S_rel(p)  = relational similarity
               S_time(p) = time kernel similarity
               S_feat(p) = feature kernel similarity
               S_tot(p)  = S_rel(p) * S_time(p) * S_feat(p)
          4. Evidence mass:
               M_e = Σ_p S_tot(p)
          5. Saturated evidence:
               E = M_e / (M_e + τ)

        τ is DEFAULT_TAU unless overridden by concept.tau.
        """
        if self.root is None:
            raise ValueError("Root not set. Call create_root(...) first.")

        L = concept.L()
        K = len(concept.ordered_node_types) + 1  # + NULL_TOKEN column

        # Shape checks
        if concept.P.shape != (L, K):
            raise ValueError(f"P must be shape {(L, K)}, got {concept.P.shape}")
        if concept.t.shape != (L + 1,) or concept.gamma_t.shape != (L + 1,):
            raise ValueError("t and gamma_t must be shape (L+1,)")
        if concept.mu.shape[0] != (L + 1,):
            raise ValueError("mu must have shape (L+1, D)")
        if concept.gamma_mu.shape != (L + 1,):
            raise ValueError("gamma_mu must be shape (L+1,)")

        type_idx = concept.type_index()
        X = self.schema.reachability_mask(L, concept.ordered_node_types)
        paths = self.sample_paths(L=L, n_samples=n_samples, rng=rng)

        mass = 0.0
        for p in paths:
            # Build M: L × K one-hot over node types + NULL_TOKEN
            M = np.zeros((L, K), dtype=float)
            valid = True
            for hop in range(L):
                if hop < len(p.node_type_sequence):
                    t = p.node_type_sequence[hop]
                else:
                    t = NULL_TOKEN
                j = type_idx.get(t)
                if j is None:
                    # Type not in prototype vocabulary => ignore this path
                    valid = False
                    break
                M[hop, j] = 1.0
            if not valid:
                continue

            s_rel = self._relational_similarity(concept.P, M, X)
            if s_rel <= 0.0:
                continue

            s_time = self._time_similarity(
                proto_t=concept.t,
                sample_t=p.time_signature,
                gamma_t=concept.gamma_t,
                k=concept.k_time,
            )
            if s_time <= 0.0:
                continue

            s_feat = self._feature_similarity(
                proto_mu=concept.mu,
                sample_mu=p.feature_signature,
                gamma_mu=concept.gamma_mu,
                k=concept.k_feat,
            )
            if s_feat <= 0.0:
                continue

            mass += float(s_rel * s_time * s_feat)

        tau = concept.tau if concept.tau is not None else DEFAULT_TAU
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")

        return float(mass / (mass + tau))

        # # Determine concept Length
        # L = int(L if L is not None else hypothesis_concept.L())
        
        # # Sample paths from subgraph
        # paths = self.sample_paths(L=L)

        # root_features = self.nodes[self.root].feature_embedding
        # similarity_sum = 0.0
        
        # for path in paths:
        #     # Calculate relational similarity
        #     relational_similarity = self._relational_similarity(path.relation_sequence, hypothesis_concept)
        #     if relational_similarity == 0.0:
        #         continue
            
        #     # Calculate time and feature similarity
        #     time_and_feature_similarity = self._time_and_feature_similarity(root_features, path.time_vector, path.feature_vector, hypothesis_concept)
        #     similarity_sum += relational_similarity * time_and_feature_similarity

        # # Create denominator using hill function style to enforce evidence boundary
        # denominator = similarity_sum + float(hypothesis_concept.tau)
        # return 0.0 if denominator == 0.0 else float(similarity_sum / denominator)