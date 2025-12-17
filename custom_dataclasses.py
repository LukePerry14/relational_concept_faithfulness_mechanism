from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

NodeId = int


@dataclass(frozen=True)
class Node:
    id: NodeId
    node_type: str
    time: float
    feature_embedding: np.ndarray


@dataclass(frozen=True)
class Edge:
    source: NodeId
    destination: NodeId
    relation: str


@dataclass(frozen=True)
class Schema:
    root_type: str
    transitions: Dict[Tuple[str, str], str]

    def check(self, source_type: str, relation: str, destination_type: str) -> None:
        exp = self.transitions.get((source_type, relation))
        if exp is None:
            raise ValueError(f"Relation not allowed by schema: ({source_type}) -[{relation}]-> (?)")
        if exp != destination_type:
            raise ValueError(f"Schema mismatch: ({source_type}) -[{relation}]-> ({destination_type}), expected ({exp})")


@dataclass(frozen=True)
class Concept:
    """
    rel_probs: shape (L, R), rows are distributions over rel_types.
    feature_centroid: length L+1 (includes root at index 0).
    time_deltas: length L.
    """
    name: str
    relation_types: List[str]
    relation_probs: np.ndarray
    time_deltas: List[float]
    feature_centroid: List[np.ndarray]
    gamma: float = 1.0
    tau: float = 0.0

    def __post_init__(self) -> None:
        rp = np.asarray(self.relation_probs, dtype=float)
        if rp.ndim != 2:
            raise ValueError("relation_probs must be a 2D array of shape (L, R).")
        L, R = rp.shape
        if R != len(self.relation_types):
            raise ValueError("rel_probs second dimension must equal len(relation_types).")
        if len(self.time_deltas) != L:
            raise ValueError("time_deltas must have length L (same as relation_probs.shape[0]).")
        if len(self.feature_centroid) != L + 1:
            raise ValueError("feature_centroid must have length L+1 (including root).")
        if np.any(rp < 0):
            raise ValueError("relation_probs must be non-negative.")
        row_sums = rp.sum(axis=1, keepdims=True)
        if np.any(row_sums == 0):
            raise ValueError("Each row of relation_probs must sum to > 0.")
        # Normalize to be safe (keeps it continuous and well-defined)
        rp = rp / row_sums
        object.__setattr__(self, "relation_probs", rp)

    def L(self) -> int:
        return int(self.relation_probs.shape[0])

    def relation_index(self) -> Dict[str, int]:
        return {r: i for i, r in enumerate(self.relation_types)}


@dataclass(frozen=True)
class PathSample:
    relation_sequence: List[str]
    time_vector: np.ndarray
    feature_vector: np.ndarray