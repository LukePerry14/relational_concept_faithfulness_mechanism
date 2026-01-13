from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np

NodeId = int

NULL_TOKEN = "∅"
EPS = 1e-12
DEFAULT_TAU = 0.5
MISSING_TIME = float('inf')
MISSING_FEAT = float('inf')

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
    transitions: Dict[str, List[str]]
    node_types: List[str] = field(default_factory=list)
    
    def __post_init__(self): # populate sorted nodetype list after initialisation using tranistion datastructure
        if not self.node_types:
            all_types = set([self.root_type])
            for src, dsts in self.transitions.items():
                all_types.add(src)
                all_types.update(dsts)
            self.node_types = sorted(list(all_types))

    def reachability_mask(self, hop_count, ordered_node_types, zeroed=False):
        """
        Build the reachability mask. the output is formatted according to the order of nodes given in `ordered_node_types`
        
        hop_count is the size of the HOP NEIGHBOURHOOD (number of nodes in the metapath - 1)
        """
        cols = ordered_node_types + [NULL_TOKEN]

        X = np.zeros((hop_count+1, len(cols)), dtype=float)
        X[0][0] = 1
        X[0][-1] = 1
        if zeroed:
            return X
        
        current = set()
        current.add(self.root_type)
        
        for hop in range(1,hop_count+1):
            
            reachable_next = set()
            
            for node in current:
                reachable_next = reachable_next | set(self.transitions.get(node, {}))

            reachable_next.add(NULL_TOKEN)

            for j, t in enumerate(cols):
                X[hop, j] = 1.0 if t in reachable_next else 0.0
            
            reachable_next.discard(NULL_TOKEN)
            
            current = reachable_next.copy()


        return X
    
    def get_adjacency_matrix(self, ordered_node_types):
        """
        Creates an (R+1) x (R+1) square matrix where A[i, j] = 1 
        if node i can transition to node j.
        """
        nodes = ordered_node_types + [NULL_TOKEN]
        R_plus_1 = len(nodes)
        adj = np.zeros((R_plus_1, R_plus_1), dtype=float)

        # Map types to indices for faster lookup
        type_to_idx = {t: i for i, t in enumerate(nodes)}

        for src, targets in self.transitions.items():
            if src in type_to_idx:
                src_idx = type_to_idx[src]
                for dst in targets:
                    if dst in type_to_idx:
                        adj[src_idx, type_to_idx[dst]] = 1.0
        
        # STOP Logic: The null token can only transition to the null token
        # This prevents the path from 'restarting' after a stop.
        null_idx = type_to_idx[NULL_TOKEN]
        adj[null_idx, null_idx] = 1.0
        
        # Every node should also be able to transition to STOP.
        for i in range(R_plus_1):
            adj[i, null_idx] = 1.0

        return adj

    def to_dict(self):
        return {
            "root_type": self.root_type,
            "transitions": self.transitions,
            "node_types": self.node_types
        }

    def from_dict(cls, d):
        return cls(
            root_type=d["root_type"],
            transitions=d["transitions"],
            node_types=d.get("node_types", [])
        )

@dataclass
class MetaPath:
    path_name: Optional[str]
    node_types: List[str]
    node_times: np.ndarray
    node_features: np.ndarray
    node_ids: Optional[List[int]]

    def __repr__(self):
        return (
            f"MetaPath(\n"
            f"  path_name={self.path_name!r},\n"
            f"  node_types={self.node_types},\n"
            f"  node_times={self.node_times},\n"
            f"  node_features={self.node_features}\n"
            f"  node_ids={self.node_ids}\n"
            f")"
        )

@dataclass
class MetaPathSchema:
    """
    Represents a meta-path type (sequence of node types).
    
    Example: driver -> results -> races would be:
        MetaPathSchema(type_sequence=["driver", "results", "races"])
    """
    type_sequence: List[str]
    
    @property
    def length(self) -> int:
        return len(self.type_sequence)
    
    def __hash__(self):
        return hash(tuple(self.type_sequence))
    
    def __eq__(self, other):
        if not isinstance(other, MetaPathSchema):
            return False
        return self.type_sequence == other.type_sequence
    
    def __repr__(self):
        return " → ".join(self.type_sequence)
    
@dataclass
class Concept:
    name: str
    ordered_node_types: List[str]
    relational_prototype: np.ndarray  # Shape (L, |T| + 1)
    time_prototype: np.ndarray       # Shape (L + 1,)
    time_gamma: np.ndarray           # Shape (L + 1,)
    feature_prototype: np.ndarray    # Shape (L + 1, D)
    feature_gamma: np.ndarray        # Shape (L + 1,)
    tau: Optional[float] = None
    similarity_at_time_gamma: float = 0.1
    similarity_at_feature_gamma: float = 0.1

    def __len__(self):
        # Use .shape[0] with square brackets to get the first dimension (L)
        return self.relational_prototype.shape[0]

    def type_index(self) -> Dict[str, int]:
        cols = self.ordered_node_types + [NULL_TOKEN]
        return {t: i for i, t in enumerate(cols)}
    

