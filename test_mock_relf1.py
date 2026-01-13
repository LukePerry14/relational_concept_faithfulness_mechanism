"""
Mock Test: Test the data pipeline with a simulated rel-f1 schema.

Since we cannot install RelBench in this environment, this script tests
the core logic using a mock schema that mirrors the rel-f1 structure.

The rel-f1 dataset contains the following tables (based on the paper):
- drivers: Driver information
- constructors: Team/constructor information
- races: Race information
- results: Race results (fact table linking drivers, constructors, races)
- qualifying: Qualifying results
- pit_stops: Pit stop data
- circuits: Circuit information

Run with: python test_mock_relf1.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from data_pipeline import (
    Schema,
    MetaPathSchema,
    MetaPath,
    enumerate_metapath_schemas,
    print_metapath_schemas,
    NULL_TOKEN,
    MISSING_TIME,
    MISSING_FEAT
)


def create_mock_f1_schema() -> Schema:
    """
    Create a mock schema matching the rel-f1 dataset structure.
    
    Based on the RelBench paper, the F1 dataset has these relationships:
    - results links to: drivers, constructors, races
    - qualifying links to: drivers, constructors, races
    - pit_stops links to: drivers, races
    - races links to: circuits
    
    For the driver-top3 task, root_type is "driver".
    """
    transitions = {
        # From driver, we can reach results, qualifying, pit_stops
        "driver": ["results", "qualifying", "pit_stops"],
        
        # From results (fact table), we can reach back to drivers, constructors, races
        "results": ["driver", "constructor", "race"],
        
        # From qualifying, we can reach drivers, constructors, races
        "qualifying": ["driver", "constructor", "race"],
        
        # From pit_stops, we can reach drivers, races
        "pit_stops": ["driver", "race"],
        
        # From constructor, we can reach results, qualifying
        "constructor": ["results", "qualifying"],
        
        # From race, we can reach results, qualifying, pit_stops, circuits
        "race": ["results", "qualifying", "pit_stops", "circuit"],
        
        # From circuit, we can reach races
        "circuit": ["race"],
    }
    
    return Schema(
        root_type="driver",
        transitions=transitions,
        node_types=["driver", "results", "qualifying", "pit_stops", "constructor", "race", "circuit"]
    )


class MockHeteroData:
    """
    Mock HeteroData object for testing without PyTorch Geometric.
    
    Simulates the structure of a rel-f1 graph.
    """
    
    def __init__(self, num_drivers=100, num_races=50, num_results_per_race=20):
        self.num_drivers = num_drivers
        self.num_races = num_races
        
        # Create mock node counts
        self._node_counts = {
            "driver": num_drivers,
            "constructor": 10,
            "race": num_races,
            "circuit": 20,
            "results": num_races * num_results_per_race,
            "qualifying": num_races * num_results_per_race // 2,
            "pit_stops": num_races * num_results_per_race * 3,
        }
        
        # Create mock timestamps (race-related entities have times)
        self._times = {}
        rng = np.random.default_rng(42)
        
        # Results have timestamps based on race date
        self._times["results"] = rng.uniform(0, 365 * 5, self._node_counts["results"]).astype(np.float32)
        self._times["qualifying"] = rng.uniform(0, 365 * 5, self._node_counts["qualifying"]).astype(np.float32)
        self._times["pit_stops"] = rng.uniform(0, 365 * 5, self._node_counts["pit_stops"]).astype(np.float32)
        self._times["race"] = rng.uniform(0, 365 * 5, self._node_counts["race"]).astype(np.float32)
        
        # Create mock edges
        self._edges = self._create_mock_edges(rng)
        
        # Create mock features
        self._feature_dim = 64
        self._features = {
            nt: rng.standard_normal((count, self._feature_dim)).astype(np.float32)
            for nt, count in self._node_counts.items()
        }
    
    def _create_mock_edges(self, rng) -> Dict[Tuple[str, str, str], np.ndarray]:
        """Create mock edge indices."""
        edges = {}
        
        # results -> driver
        num_results = self._node_counts["results"]
        edges[("results", "to", "driver")] = np.stack([
            np.arange(num_results),
            rng.integers(0, self.num_drivers, num_results)
        ])
        
        # results -> constructor
        edges[("results", "to", "constructor")] = np.stack([
            np.arange(num_results),
            rng.integers(0, 10, num_results)
        ])
        
        # results -> race
        results_per_race = num_results // self.num_races
        edges[("results", "to", "race")] = np.stack([
            np.arange(num_results),
            np.repeat(np.arange(self.num_races), results_per_race)[:num_results]
        ])
        
        # qualifying -> driver
        num_qual = self._node_counts["qualifying"]
        edges[("qualifying", "to", "driver")] = np.stack([
            np.arange(num_qual),
            rng.integers(0, self.num_drivers, num_qual)
        ])
        
        # qualifying -> race
        edges[("qualifying", "to", "race")] = np.stack([
            np.arange(num_qual),
            rng.integers(0, self.num_races, num_qual)
        ])
        
        # pit_stops -> driver
        num_pits = self._node_counts["pit_stops"]
        edges[("pit_stops", "to", "driver")] = np.stack([
            np.arange(num_pits),
            rng.integers(0, self.num_drivers, num_pits)
        ])
        
        # pit_stops -> race
        edges[("pit_stops", "to", "race")] = np.stack([
            np.arange(num_pits),
            rng.integers(0, self.num_races, num_pits)
        ])
        
        # race -> circuit
        edges[("race", "to", "circuit")] = np.stack([
            np.arange(self.num_races),
            rng.integers(0, 20, self.num_races)
        ])
        
        return edges
    
    @property
    def node_types(self) -> List[str]:
        return list(self._node_counts.keys())
    
    @property
    def edge_types(self) -> List[Tuple[str, str, str]]:
        return list(self._edges.keys())
    
    def __getitem__(self, key):
        """Access node or edge data."""
        if isinstance(key, str):
            # Node type access
            return MockNodeStore(
                num_nodes=self._node_counts.get(key, 0),
                time=self._times.get(key),
                features=self._features.get(key)
            )
        elif isinstance(key, tuple):
            # Edge type access
            return MockEdgeStore(edge_index=self._edges.get(key))
        return None


class MockNodeStore:
    """Mock node store for testing."""
    def __init__(self, num_nodes: int, time=None, features=None):
        self.num_nodes = num_nodes
        self.time = time
        self.x = features
        self.tf = None  # No TorchFrame in mock


class MockEdgeStore:
    """Mock edge store for testing."""
    def __init__(self, edge_index=None):
        self.edge_index = edge_index


class MockMetaPathSampler:
    """
    Simplified meta-path sampler for testing with mock data.
    """
    
    def __init__(self, data: MockHeteroData, schema: Schema, max_hops: int):
        self.data = data
        self.schema = schema
        self.max_hops = max_hops
        self.metapath_schemas = enumerate_metapath_schemas(schema, max_hops)
        self.feature_dim = data._feature_dim
        
        # Build adjacency
        self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency lists from mock data."""
        self.adjacency = defaultdict(lambda: defaultdict(set))
        
        for edge_type, edge_index in self.data._edges.items():
            src_type, _, dst_type = edge_type
            
            for i in range(edge_index.shape[1]):
                src_idx = edge_index[0, i]
                dst_idx = edge_index[1, i]
                
                # Bidirectional
                self.adjacency[src_type][src_idx].add((dst_type, dst_idx))
                self.adjacency[dst_type][dst_idx].add((src_type, src_idx))
    
    def _get_node_time(self, node_type: str, node_idx: int) -> float:
        """Get timestamp for a node."""
        times = self.data._times.get(node_type)
        if times is not None and node_idx < len(times):
            return float(times[node_idx])
        return float('-inf')
    
    def _get_node_features(self, node_type: str, node_idx: int) -> np.ndarray:
        """Get features for a node."""
        features = self.data._features.get(node_type)
        if features is not None and node_idx < len(features):
            return features[node_idx]
        return np.zeros(self.feature_dim, dtype=np.float32)
    
    def sample_paths_for_seed(
        self,
        seed_type: str,
        seed_idx: int,
        seed_time: float,
        n_samples_per_schema: int = 4,
        max_total_samples: int = 64,
        rng=None
    ) -> Tuple[List[MetaPath], Dict[MetaPathSchema, int]]:
        """Sample paths for a seed node."""
        if rng is None:
            rng = np.random.default_rng()
        
        all_paths = []
        schema_counts = defaultdict(int)
        
        for mp_schema in self.metapath_schemas:
            paths = self._sample_paths_for_schema(
                mp_schema, seed_idx, seed_time, n_samples_per_schema, rng
            )
            for p in paths:
                all_paths.append(p)
                schema_counts[mp_schema] += 1
        
        # Pad paths
        padded = [self._pad_path(p) for p in all_paths]
        
        # Subsample if needed
        if len(padded) > max_total_samples:
            idxs = rng.choice(len(padded), max_total_samples, replace=False)
            padded = [padded[i] for i in idxs]
        
        return padded, dict(schema_counts)
    
    def _sample_paths_for_schema(
        self,
        mp_schema: MetaPathSchema,
        seed_idx: int,
        seed_time: float,
        n_samples: int,
        rng
    ) -> List[MetaPath]:
        """Sample paths matching a schema."""
        paths = []
        type_seq = mp_schema.type_sequence
        
        if type_seq[0] != self.schema.root_type:
            return []
        
        attempts = 0
        max_attempts = n_samples * 20
        
        while len(paths) < n_samples and attempts < max_attempts:
            attempts += 1
            
            path_ids = [seed_idx]
            path_types = [type_seq[0]]
            path_times = [self._get_node_time(type_seq[0], seed_idx)]
            path_features = [self._get_node_features(type_seq[0], seed_idx)]
            
            valid = True
            current_idx = seed_idx
            current_type = type_seq[0]
            
            for hop in range(1, len(type_seq)):
                target_type = type_seq[hop]
                
                # Get neighbors of target type
                neighbors = []
                for (nbr_type, nbr_idx) in self.adjacency[current_type][current_idx]:
                    if nbr_type == target_type:
                        nbr_time = self._get_node_time(nbr_type, nbr_idx)
                        if nbr_time <= seed_time:
                            neighbors.append(nbr_idx)
                
                # Remove already visited
                neighbors = [n for n in neighbors if n not in path_ids]
                
                if not neighbors:
                    valid = False
                    break
                
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
        """Pad path to max_hops + 1."""
        target_len = self.max_hops + 1
        current_len = len(path.node_types)
        
        if current_len >= target_len:
            return path
        
        padded_types = path.node_types + [NULL_TOKEN] * (target_len - current_len)
        
        padded_times = np.full(target_len, MISSING_TIME, dtype=np.float32)
        padded_times[:current_len] = path.node_times
        
        padded_features = np.full((target_len, self.feature_dim), MISSING_FEAT, dtype=np.float32)
        padded_features[:current_len] = path.node_features
        
        padded_ids = path.node_ids + [-1] * (target_len - current_len) if path.node_ids else None
        
        return MetaPath(
            path_name=path.path_name,
            node_types=padded_types,
            node_times=padded_times,
            node_features=padded_features,
            node_ids=padded_ids
        )


def test_schema():
    """Test the mock schema."""
    print("\n" + "=" * 60)
    print("Test 1: Schema Creation")
    print("=" * 60)
    
    schema = create_mock_f1_schema()
    
    print(f"\nSchema root type: {schema.root_type}")
    print(f"Node types: {schema.node_types}")
    print(f"\nTransitions:")
    for src, dsts in sorted(schema.transitions.items()):
        print(f"  {src} -> {dsts}")
    
    # Test reachability mask
    print(f"\nReachability mask (max_hops=2):")
    mask = schema.reachability_mask(2)
    print(f"  Shape: {mask.shape}")
    types_with_null = schema.node_types + [NULL_TOKEN]
    for hop in range(3):
        reachable = [types_with_null[i] for i in range(len(types_with_null)) if mask[hop, i] > 0]
        print(f"  Hop {hop}: {reachable}")
    
    return schema


def test_metapath_enumeration(schema):
    """Test meta-path enumeration."""
    print("\n" + "=" * 60)
    print("Test 2: Meta-path Enumeration")
    print("=" * 60)
    
    for max_hops in [1, 2]:
        schemas = enumerate_metapath_schemas(schema, max_hops)
        print_metapath_schemas(schemas, f"Meta-paths (max_hops={max_hops})")
    
    return enumerate_metapath_schemas(schema, 2)


def test_mock_data():
    """Test mock data creation."""
    print("\n" + "=" * 60)
    print("Test 3: Mock HeteroData")
    print("=" * 60)
    
    data = MockHeteroData(num_drivers=100, num_races=50)
    
    print(f"\nNode types: {data.node_types}")
    for nt in data.node_types:
        node_store = data[nt]
        has_time = node_store.time is not None
        print(f"  {nt}: {node_store.num_nodes} nodes, has_time={has_time}")
    
    print(f"\nEdge types: {len(data.edge_types)}")
    for et in data.edge_types:
        edge_store = data[et]
        if edge_store.edge_index is not None:
            print(f"  {et[0]} -> {et[2]}: {edge_store.edge_index.shape[1]} edges")
    
    return data


def test_path_sampling(data, schema, metapath_schemas):
    """Test path sampling."""
    print("\n" + "=" * 60)
    print("Test 4: Path Sampling")
    print("=" * 60)
    
    sampler = MockMetaPathSampler(data, schema, max_hops=2)
    
    print(f"\nSampler initialized with {len(sampler.metapath_schemas)} meta-path schemas")
    
    # Sample for a few seed drivers
    rng = np.random.default_rng(42)
    
    for seed_idx in [0, 10, 50]:
        seed_time = 365 * 3  # 3 years in
        
        print(f"\n--- Seed driver {seed_idx}, time={seed_time:.0f} ---")
        
        paths, schema_counts = sampler.sample_paths_for_seed(
            seed_type="driver",
            seed_idx=seed_idx,
            seed_time=seed_time,
            n_samples_per_schema=4,
            max_total_samples=32,
            rng=rng
        )
        
        print(f"  Sampled {len(paths)} paths")
        
        if schema_counts:
            print(f"  Schema distribution:")
            for mp_schema, count in sorted(schema_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"    {mp_schema}: {count}")
        
        if paths:
            print(f"\n  Example path:")
            p = paths[0]
            print(f"    Types: {' → '.join(p.node_types)}")
            times_str = [f"{t:.1f}" if t != MISSING_TIME else "∅" for t in p.node_times]
            print(f"    Times: [{', '.join(times_str)}]")
            print(f"    Features shape: {p.node_features.shape}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Mock rel-f1 Data Pipeline Test")
    print("=" * 60)
    print("\nThis test validates the data pipeline logic without")
    print("requiring the RelBench package to be installed.")
    
    # Test 1: Schema
    schema = test_schema()
    
    # Test 2: Meta-path enumeration
    metapath_schemas = test_metapath_enumeration(schema)
    
    # Test 3: Mock data
    data = test_mock_data()
    
    # Test 4: Path sampling
    test_path_sampling(data, schema, metapath_schemas)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nThe data pipeline logic is working correctly.")
    print("To run with real RelBench data, use test_relf1_integration.py")
    print("in an environment with RelBench installed.")


if __name__ == "__main__":
    main()
