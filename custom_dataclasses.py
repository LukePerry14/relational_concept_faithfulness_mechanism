from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np

NodeId = int

NULL_TOKEN = "∅"
EPS = 1e-12
DEFAULT_TAU = 0.5

@dataclass(frozen=True)
class Node:
    id: NodeId
    node_type: str
    time: float
    feature: np.ndarray  # shape (D,)


@dataclass(frozen=True)
class Edge:
    src: NodeId
    dst: NodeId
    relation: str


@dataclass
class Schema:
    root_type: str
    transitions: Dict[str, List[str]] # src_type -> [dst_types].


    def reachability_mask(self, L: int, ordered_node_types: List[str]) -> np.ndarray:
        """
        Build the reachability mask. the output is formatted according to the order of nodes given in `ordered_node_types`
        
        L is the size of the HOP NEIGHBOURHOOD (number of nodes in the metapath - 1)
        """
        cols = ordered_node_types + [NULL_TOKEN]
        X = np.zeros((L, len(cols)), dtype=float)

        
        current = set()
        current.add(self.root_type)
        
        for hop in range(L):
            
            reachable_next = set()
            
            for node in current:
                reachable_next = reachable_next | set(self.transitions.get(node, {}))

            reachable_next.add(NULL_TOKEN)

            for j, t in enumerate(cols):
                X[hop, j] = 1.0 if t in reachable_next else 0.0
            
            reachable_next.discard(NULL_TOKEN)
            
            current = reachable_next.copy()


        return X


@dataclass
class MetaPath:
    path_name: Optional[str]
    node_types: List[str]
    node_times: np.ndarray
    node_features: np.ndarray

    def __repr__(self):
        return (
            f"MetaPath(\n"
            f"  path_name={self.path_name!r},\n"
            f"  node_types={self.node_types},\n"
            f"  node_times={self.node_times},\n"
            f"  node_features={self.node_features}\n"
            f")"
        )

@dataclass
class Concept:
    """
    Prototype concept as described in the walkthrough.

    ordered_node_types:
        List of node types T = [t1, ..., tK]. An extra column for NULL_TOKEN
        is appended internally when constructing P, X, and M.

    P:
        Relational prototype distribution over node types and NULL_TOKEN.
        Shape (L, |T| + 1), rows = hops 1..L, columns = ordered_node_types + [NULL_TOKEN].
        Each row is a probability distribution.

    t:
        Time prototype, shape (L+1,).
        t[0] is the absolute prototype time of the root.
        t[1:] are prototype offsets (relative to root) for hops 1..L.

    gamma_t:
        Time window radii, shape (L+1,). Use np.inf to ignore a given hop.

    mu:
        Feature prototype, shape (L+1, D).
        mu[0] is the prototype feature of the root node.
        mu[1:] are prototype features for hops 1..L.

    gamma_mu:
        Feature window radii, shape (L+1,). Use np.inf to ignore a given hop.

    tau:
        Saturation parameter in E = M / (M + tau). If None, DEFAULT_TAU is used.
        Keeping it fixed implements the basic mechanism; making it
        per-concept enables later learning or tying to gamma_·.

    k_time, k_feat:
        Similarity at distance == gamma (0 < k < 1). Controls steepness
        of the RBF-like kernels.
    """
    name: str
    ordered_node_types: List[str]
    P: np.ndarray
    t: np.ndarray
    gamma_t: np.ndarray
    mu: np.ndarray
    gamma_mu: np.ndarray
    tau: Optional[float] = None
    k_time: float = 0.1
    k_feat: float = 0.1

    def L(self) -> int:
        return int(self.P.shape[0])

    def type_index(self) -> Dict[str, int]:
        cols = self.ordered_node_types + [NULL_TOKEN]
        return {t: i for i, t in enumerate(cols)}