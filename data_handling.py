"""
Data Pipeline for Concept Bottleneck Model on RelBench

This module handles:
1. Schema extraction from RelBench HeteroData
2. Meta-path enumeration within k-hop neighborhoods
3. Biased path sampling with temporal constraints
4. Data storage and loading for training
"""
import h5py
import os
import json
import torch
from torch_geometric.data import HeteroData
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from tqdm import tqdm
from utils.custom_dataclasses import (
    Schema,
    MetaPath,
    MetaPathSchema,
    NULL_TOKEN,
    EPS,
    MISSING_TIME,
    MISSING_FEAT
)


# =============================================================================
# Schema Extraction
# =============================================================================

def extract_schema_from_heterodata(
    data: HeteroData,
    root_type: str,
    exclude_self_loops: bool = True
) -> Schema:
    """
    Extract a Schema object from a PyG HeteroData graph.
    """
    transitions = defaultdict(set)
    
    # Extract transitions from edge types
    # Edge types in HeteroData are tuples: (src_type, relation_name, dst_type)
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        
        if exclude_self_loops and src_type == dst_type:
            continue
            
        # Add bidirectional transitions (undirected graph assumption)
        transitions[src_type].add(dst_type)
        transitions[dst_type].add(src_type)
    
    # Convert sets to sorted lists for deterministic ordering
    transitions_dict = {k: sorted(list(v)) for k, v in transitions.items()}
    
    # Collect all node types
    all_types = sorted(list(data.node_types))
    
    return Schema(
        root_type=root_type,
        transitions=transitions_dict,
        node_types=all_types
    )


# =============================================================================
# Meta-path Enumeration
# =============================================================================

def enumerate_metapath_schemas(
    schema: Schema,
    max_hops: int,
    include_shorter: bool = True
) -> List[MetaPathSchema]:
    """
    Enumerate all valid meta-path schemas up to max_hops from the root type.
    
    Args:
        schema: The relational schema
        max_hops: Maximum path length (number of edges)
        include_shorter: If True, include paths shorter than max_hops
        
    Returns:
        List of MetaPathSchema objects representing valid path types
    """
    all_schemas = []
    
    def enumerate_recursive(current_sequence: List[str], current_type: str, remaining_hops: int):
        """Recursively enumerate all valid continuations."""
        if remaining_hops == 0:
            if len(current_sequence) > 1:  # At least root + one hop
                all_schemas.append(MetaPathSchema(type_sequence=current_sequence.copy()))
            return
        
        # Option 1: Stop here (if include_shorter and we have at least one hop)
        if include_shorter and len(current_sequence) > 1:
            all_schemas.append(MetaPathSchema(type_sequence=current_sequence.copy()))
        
        # Option 2: Continue to next hop
        next_types = schema.transitions.get(current_type, [])
        for next_type in next_types:
            # Avoid immediate backtracking (going back to the previous node type)
            # This is a heuristic to reduce redundant paths
            if len(current_sequence) >= 2 and next_type == current_sequence[-2]:
                continue
            current_sequence.append(next_type)
            enumerate_recursive(current_sequence, next_type, remaining_hops - 1)
            current_sequence.pop()
    
    # Start enumeration from root
    enumerate_recursive([schema.root_type], schema.root_type, max_hops)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_schemas = []
    for s in all_schemas:
        if s not in seen:
            seen.add(s)
            unique_schemas.append(s)
    
    return unique_schemas

def print_metapath_schemas(schemas: List[MetaPathSchema], title: str = "Meta-path Schemas"):
    """Pretty-print enumerated meta-path schemas."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Group by length
    by_length = defaultdict(list)
    for s in schemas:
        by_length[s.length].append(s)
    
    for length in sorted(by_length.keys()):
        print(f"\nLength {length} ({length-1} hops):")
        for s in by_length[length]:
            print(f"  {s}")
    
    print(f"\nTotal: {len(schemas)} unique meta-path schemas")
    print(f"{'='*60}\n")

class MetaPathSampler:
    """
    Samples concrete meta-path instances from a HeteroData graph.
    
    Supports:
    - Temporal constraints (only include nodes with time <= seed_time)
    - Schema-guided sampling (only follow valid transitions)
    - Biased sampling to ensure rare meta-path types are represented
    """
    
    def __init__(
        self,
        data: HeteroData,
        schema: Schema,
        max_hops: int,
        metapath_schemas: Optional[List[MetaPathSchema]] = None,
        adjacency: Optional[Dict] = None
    ):
        """
        Initialize the sampler.
        
        Args:
            data: PyG HeteroData graph
            schema: Relational schema
            max_hops: Maximum path length
            metapath_schemas: Optional pre-computed meta-path schemas
        """
        self.data = data
        self.schema = schema
        self.max_hops = max_hops
        
        # Enumerate valid meta-path schemas if not provided
        if metapath_schemas is None:
            self.metapath_schemas = enumerate_metapath_schemas(schema, max_hops)
        else:
            self.metapath_schemas = metapath_schemas

        if adjacency is not None:
            self.adjacency = adjacency
        else:
            self._build_adjacency()
        
        # Get feature dimension from data
        self.feature_dim = self._get_feature_dim()
    
    def _build_adjacency(self):
        """Build adjacency lists indexed by (node_type, node_idx) -> [(neighbor_type, neighbor_idx), ...]"""
        self.adjacency = defaultdict(lambda: defaultdict(set))
        
        for edge_type in self.data.edge_types:
            src_type, _, dst_type = edge_type
            
            if 'edge_index' not in self.data[edge_type]:
                continue
                
            edge_index = self.data[edge_type].edge_index
            src_list = edge_index[0].tolist()
            dst_list = edge_index[1].tolist()
            
            for s, d in zip(src_list, dst_list):
                # Bidirectional (undirected)
                self.adjacency[src_type][s].add((dst_type, d))
                self.adjacency[dst_type][d].add((src_type, s))
    
    def _get_feature_dim(self) -> int:
        """Determine feature dimension from the data."""
        for node_type in self.data.node_types:
            if hasattr(self.data[node_type], 'tf') and self.data[node_type].tf is not None:
                # TorchFrame features - get dimension from first entry
                tf = self.data[node_type].tf
                # This is a TensorFrame object, we need to check its structure
                # For now, return a placeholder - will be updated when we encode features
                return 128  # Default placeholder
            elif hasattr(self.data[node_type], 'x') and self.data[node_type].x is not None:
                return self.data[node_type].x.shape[1]
        return 128  # Default
    
    def _get_node_time(self, node_type: str, node_idx: int) -> float:
        """Get timestamp for a node, or -inf if no timestamp exists."""
        if hasattr(self.data[node_type], 'time'):
            time_tensor = self.data[node_type].time
            if time_tensor is not None and node_idx < len(time_tensor):
                return float(time_tensor[node_idx].item())
        return float('-inf')  # No temporal constraint
    
    def _get_node_features(self, node_type: str, node_idx: int) -> np.ndarray:
        """
        Get feature vector for a node.
        
        Note: In the full pipeline, features will come from the encoded TorchFrame.
        This is a placeholder that returns zeros.
        """
        # Placeholder - will be replaced with actual feature extraction
        return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _get_neighbors_by_type(
        self,
        node_type: str,
        node_idx: int,
        target_type: str,
        seed_time: float
    ) -> List[int]:
        """
        Get neighbors of a specific type that satisfy temporal constraints.
        
        Args:
            node_type: Current node's type
            node_idx: Current node's index
            target_type: Desired neighbor type
            seed_time: Maximum allowed timestamp
            
        Returns:
            List of neighbor indices of the target type
        """
        neighbors = []
        for (nbr_type, nbr_idx) in self.adjacency[node_type][node_idx]:
            if nbr_type != target_type:
                continue
            
            # Check temporal constraint
            nbr_time = self._get_node_time(nbr_type, nbr_idx)
            if nbr_time <= seed_time:
                neighbors.append(nbr_idx)
        
        return neighbors
    
    def sample_paths_for_seed(
        self,
        seed_type: str,
        seed_idx: int,
        seed_time: float,
        n_samples_per_metapath_schema: int = 4,
        max_total_samples: int = 64,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[List[MetaPath], Dict[MetaPathSchema, int]]:
        """
        Sample meta-path instances for a single seed node.
        
        Uses stratified sampling to ensure representation of different meta-path types.
        
        Args:
            seed_type: Node type of the seed
            seed_idx: Index of the seed node
            seed_time: Timestamp of the seed (for temporal filtering)
            n_samples_per_schema: Target samples per meta-path schema
            max_total_samples: Maximum total paths to return
            rng: Random number generator
            
        Returns:
            paths: List of MetaPath instances
            schema_counts: Dict mapping schema -> count of sampled instances
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if seed_type != self.schema.root_type:
            raise ValueError(f"Seed type {seed_type} doesn't match schema root {self.schema.root_type}")
        
        # Get seed node info
        seed_time_actual = self._get_node_time(seed_type, seed_idx)
        seed_features = self._get_node_features(seed_type, seed_idx)
        
        all_paths = []
        schema_counts = defaultdict(int)
        
        # Sample paths for each meta-path schema
        for mp_schema in self.metapath_schemas:
            schema_paths = self._sample_paths_for_schema(
                mp_schema,
                seed_idx,
                seed_time,
                n_samples=n_samples_per_metapath_schema,
                rng=rng
            )
            
            for path in schema_paths:
                all_paths.append(path)
                schema_counts[mp_schema] += 1
        
        # If we have too many paths, subsample while maintaining diversity
        if len(all_paths) > max_total_samples:
            all_paths = self._diverse_subsample(all_paths, max_total_samples, rng)
        
        # Pad all paths to max_hops + 1 length
        padded_paths = [self._pad_path(p) for p in all_paths]
        
        return padded_paths, dict(schema_counts)
    
    def _sample_paths_for_schema(
        self,
        mp_schema: MetaPathSchema,
        seed_idx: int,
        seed_time: float,
        n_samples: int,
        rng: np.random.Generator
    ) -> List[MetaPath]:
        """Sample concrete paths matching a specific meta-path schema."""
        paths = []
        type_sequence = mp_schema.type_sequence
        
        # Early exit if schema doesn't start with root type
        if type_sequence[0] != self.schema.root_type:
            return []
        
        # Try to sample n_samples paths
        attempts = 0
        max_attempts = n_samples * 10  # Avoid infinite loops
        
        while len(paths) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Try to construct a path following the schema
            path_ids = [seed_idx]
            path_types = [type_sequence[0]]
            path_times = [self._get_node_time(type_sequence[0], seed_idx)]
            path_features = [self._get_node_features(type_sequence[0], seed_idx)]
            
            valid = True
            current_idx = seed_idx
            current_type = type_sequence[0]
            
            for hop in range(1, len(type_sequence)):
                target_type = type_sequence[hop]
                
                # Get valid neighbors
                neighbors = self._get_neighbors_by_type(
                    current_type, current_idx, target_type, seed_time
                )
                
                # Exclude already visited nodes to avoid cycles
                neighbors = [n for n in neighbors if n not in path_ids or target_type != current_type]
                
                if not neighbors:
                    valid = False
                    break
                
                # Randomly select a neighbor
                next_idx = rng.choice(neighbors)
                
                path_ids.append(next_idx)
                path_types.append(target_type)
                path_times.append(self._get_node_time(target_type, next_idx))
                path_features.append(self._get_node_features(target_type, next_idx))
                
                current_idx = next_idx
                current_type = target_type
            
            if valid:
                paths.append(MetaPath(
                    path_name=str(mp_schema),
                    node_types=path_types,
                    node_times=np.array(path_times, dtype=np.float32),
                    node_features=np.array(path_features, dtype=np.float32),
                    node_ids=path_ids
                ))
        
        return paths
    
    def _pad_path(self, path: MetaPath) -> MetaPath:
        """Pad a path to max_hops + 1 length with NULL tokens."""
        target_len = self.max_hops + 1
        current_len = len(path.node_types)
        
        if current_len >= target_len:
            return path
        
        # Pad node types with NULL
        padded_types = path.node_types + [NULL_TOKEN] * (target_len - current_len)
        
        # Pad times with inf
        padded_times = np.full(target_len, MISSING_TIME, dtype=np.float32)
        padded_times[:current_len] = path.node_times
        
        # Pad features with inf
        padded_features = np.full((target_len, path.node_features.shape[1]), MISSING_FEAT, dtype=np.float32)
        padded_features[:current_len] = path.node_features
        
        # Pad node_ids with -1
        padded_ids = None
        if path.node_ids is not None:
            padded_ids = path.node_ids + [-1] * (target_len - current_len)
        
        return MetaPath(
            path_name=path.path_name,
            node_types=padded_types,
            node_times=padded_times,
            node_features=padded_features,
            node_ids=padded_ids
        )
    
    def _diverse_subsample(
        self,
        paths: List[MetaPath],
        max_samples: int,
        rng: np.random.Generator
    ) -> List[MetaPath]:
        """Subsample paths while maintaining diversity across meta-path schemas."""
        # Group by schema (approximated by type sequence)
        by_schema = defaultdict(list)
        for p in paths:
            key = tuple(p.node_types)
            by_schema[key].append(p)
        
        # Calculate samples per schema to achieve diversity
        n_schemas = len(by_schema)
        base_per_schema = max_samples // n_schemas
        remainder = max_samples % n_schemas
        
        selected = []
        schema_keys = list(by_schema.keys())
        rng.shuffle(schema_keys)
        
        for i, key in enumerate(schema_keys):
            n_to_sample = base_per_schema + (1 if i < remainder else 0)
            schema_paths = by_schema[key]
            
            if len(schema_paths) <= n_to_sample:
                selected.extend(schema_paths)
            else:
                idxs = rng.choice(len(schema_paths), size=n_to_sample, replace=False)
                selected.extend([schema_paths[i] for i in idxs])
        
        return selected
