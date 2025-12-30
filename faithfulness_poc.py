from __future__ import annotations
from typing import Dict, List, Literal, Tuple, Optional
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from collections import deque
from custom_dataclasses import *
NodeId = int

DEFAULT_TAU = 0.5
MISSING_TIME = float('inf')
MISSING_FEAT = float('inf')


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
        self.MISSING_TIME = MISSING_TIME

    # --- visualise ----

    def visualize_subgraph_plotly(self,):
        """
        Interactive Plotly visualisation of a Subgraph.

        - Node colours represent node types.
        - Hover text shows:
            * node id
            * node type
            * time
            * feature vector

        Requires:
            pip install plotly networkx
        """
        title = "Subgraph"

        if self.root is None:
            raise ValueError("Subgraph has no root; call create_root(...) first.")

        # Build NX graph
        G = nx.DiGraph()
        for nid, node in self.nodes.items():
            G.add_node(
                nid,
                node_type=node.node_type,
                time=node.time,
                feature=node.feature,
            )
        for edge in self.edges:
            G.add_edge(edge.src, edge.dst, relation=edge.relation)

        # Layout
        pos = nx.spring_layout(G, seed=42)  # consistent layout

        # Edge coordinates
        edge_x, edge_y = [], []
        for src, dst in G.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1),
            hoverinfo="none"
        )

        # Build node hover labels + colors
        node_x, node_y, hovertext, labels, types = [], [], [], [], []

        for nid, data in G.nodes(data=True):
            x, y = pos[nid]
            node_x.append(x)
            node_y.append(y)
            labels.append(str(data["node_type"]))
            types.append(data["node_type"])

            feat_str = np.array2string(
                np.asarray(data["feature"], dtype=float),
                precision=3,
                separator=", ",
                suppress_small=True,
            )

            hovertext.append(
                f"<b>id:</b> {nid}<br>"
                f"<b>type:</b> {data['node_type']}<br>"
                f"<b>time:</b> {data['time']:.3f}<br>"
                f"<b>feature:</b> {feat_str}"
            )

        # Assign colours for node types
        unique = sorted(set(types))
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]
        color_map = {t: palette[i % len(palette)] for i, t in enumerate(unique)}
        node_colors = [color_map[t] for t in types]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=labels,
            mode="markers+text",
            textposition="top center",
            hovertext=hovertext,
            hoverinfo="text",
            marker=dict(
                size=16,
                color=node_colors,
                line=dict(width=1, color="#333333"),
            ),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=title,
                title_x=0.5,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
        )

        fig.show()
        return fig


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
        self.D = feat.shape[0]
        self.MISSING_FEAT = np.asarray([float('inf')] * self.D)
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

    def sample_paths(self, max_hops, n_samples=128, rng=None):
        """
        Updated to align with Concept signatures:
        - All signatures are length max_hops + 1.
        - Index 0 is always the root node.
        """
        if self.root is None:
            raise ValueError("Root not set. Call create_root(...) first.")

        rng = rng or np.random.default_rng()
        root_id = self.root
        D = self.D

        # ---- Step 1: Enumerate all simple maximal paths up to max_hops ----
        all_paths: List[List[int]] = []
        seen_paths: set[tuple[int, ...]] = set()

        def dfs(path: List[int]) -> None:
            depth = len(path) - 1
            current = path[-1]
            out_edges = self.adj.get(current, [])

            is_max_length = depth >= max_hops
            is_dead_end = len(out_edges) == 0

            if depth >= 1 and (is_max_length or is_dead_end):
                key = tuple(path)
                if key not in seen_paths:
                    seen_paths.add(key)
                    all_paths.append(path.copy())

            if is_max_length:
                return

            for _rel, dst in out_edges:
                if dst in path:
                    continue
                path.append(dst)
                dfs(path)
                path.pop()

        dfs([root_id])

        if not all_paths:
            return []

        # ---- Step 2: Subsample paths ----
        num_paths = len(all_paths)
        if num_paths > n_samples:
            idx = rng.choice(num_paths, size=n_samples, replace=False)
            chosen_paths = [all_paths[i] for i in idx]
        else:
            rng.shuffle(all_paths)
            chosen_paths = all_paths

        # ---- Step 3: Convert to MetaPath with consistent lengths (L+1) ----
        samples: List[MetaPath] = []

        for path in chosen_paths:
            # path is [root, v1, ..., vk]
            actual_len = len(path)
            
            # node_types: Include root at index 0, pad to max_hops + 1
            node_types: List[str] = []
            for h in range(max_hops + 1):
                if h < actual_len:
                    node_types.append(self.nodes[path[h]].node_type)
                else:
                    node_types.append(NULL_TOKEN)

            # node_times: Include root at index 0, pad to max_hops + 1
            t_sig = np.full((max_hops + 1,), self.MISSING_TIME, dtype=float)
            for h in range(actual_len):
                t_sig[h] = float(self.nodes[path[h]].time)

            # node_features: Include root at index 0, pad to max_hops + 1
            mu_sig = np.full((max_hops + 1, D), self.MISSING_FEAT, dtype=float)
            for h in range(actual_len):
                mu_sig[h] = self.nodes[path[h]].feature

            samples.append(
                MetaPath(
                    path_name=None,
                    node_types=node_types,
                    node_times=t_sig,
                    node_features=mu_sig,
                )
            )

        return samples
            
            
   

    @staticmethod
    def _relational_similarity(P: np.ndarray, M: np.ndarray, X: np.ndarray) -> float:
        """
        S_rel = 1 - (1 / ||X||₁) * Σ_{i,j} X_{i,j} (P_{i,j} - M_{i,j})²
        """
        denom = float(np.sum(M))
        if denom <= 0.0:
            return 0.0

        masked = (M * P)**2
        s = float(np.sum(masked) / denom)
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

        # Mask: finite radius and non-missing (finite) sample times.
        # Avoid subtracting infinities (which produces NaN warnings) by
        # testing for finite sample_t first.
        mask = np.isfinite(gamma_t) & np.isfinite(sample_t)
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
        use_cosine: bool = True,
    ) -> float:
        """
        RBF-like feature kernel with selectable distance metric:

        If use_cosine=False (Euclidean distance, default):
          For each hop i:
            d_i² = ||μ_i - μ̂_i||²
            contribution = d_i² / γ_μ_i²

        If use_cosine=True (cosine distance):
          For each hop i:
            cos_sim_i = (μ_i · μ̂_i) / (||μ_i|| * ||μ̂_i||)
            cos_dist_i = 1 - cos_sim_i
            contribution = cos_dist_i / γ_μ_i²

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

        metapath_length, D = proto_mu.shape
        contribs: List[float] = []

        for i in range(metapath_length):
            g_i = gamma_mu[i]
            if not np.isfinite(g_i):
                continue  # ignored hop

            # If the sample features at this hop are all missing (non-finite),
            # skip this hop. Avoid subtracting infinities to prevent warnings.
            if not np.any(np.isfinite(sample_mu[i])):
                continue  # missing sample at this hop

            g2 = max(g_i ** 2, EPS)
            
            if use_cosine:
                # Cosine distance metric
                proto_vec = proto_mu[i]
                sample_vec = sample_mu[i]
                
                # Compute norms
                proto_norm = float(np.linalg.norm(proto_vec))
                sample_norm = float(np.linalg.norm(sample_vec))
                
                # Avoid division by zero
                if proto_norm > EPS and sample_norm > EPS:
                    cos_sim = float(np.dot(proto_vec, sample_vec)) / (proto_norm * sample_norm)
                    # Clamp to [-1, 1] to handle numerical errors
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    # Convert similarity to distance: larger distance for lower similarity
                    cos_dist = 1.0 - cos_sim
                    contribs.append(cos_dist / g2)
            else:
                # Euclidean distance metric (original)
                diff = proto_mu[i] - sample_mu[i]
                d2 = float(np.dot(diff, diff))
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
        D = self.D

        # Shape checks
        if concept.P.shape != (L, K):
            raise ValueError(f"P must be shape {(L, K)}, got {concept.P.shape}")
        if concept.t.shape != (L + 1,) or concept.gamma_t.shape != (L + 1,):
            raise ValueError("t and gamma_t must be shape (L+1,)")
        if concept.mu.shape != (L + 1, D):
            raise ValueError("mu must have shape (L+1, D)")
        if concept.gamma_mu.shape != (L + 1, ):
            raise ValueError("gamma_mu must be shape (L+1, 1)")


        type_idx = concept.type_index()
        X = self.schema.reachability_mask(L, concept.ordered_node_types)
        paths = self.sample_paths(max_hops=L, n_samples=n_samples, rng=rng)

        mass = 0.0
        for p in paths:
            # Build M: L × K one-hot over node types + NULL_TOKEN
            M = np.zeros((L, K), dtype=float)
            valid = True
            for hop in range(L):
                t = p.node_types[hop + 1] 
                j = type_idx.get(t)
                if j is None:
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
                sample_t=p.node_times,
                gamma_t=concept.gamma_t,
                k=concept.k_time,
            )
            if s_time <= 0.0:
                continue

            s_feat = self._feature_similarity(
                proto_mu=concept.mu,
                sample_mu=p.node_features,
                gamma_mu=concept.gamma_mu,
                k=concept.k_feat,
            )
            if s_feat <= 0.0:
                continue
            
            # print(f"Path: {p.path_name}, S_rel={s_rel:.4f}, S_time={s_time:.4f}, S_feat={s_feat:.4f}, total={s_rel * s_time * s_feat:.4f}")
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