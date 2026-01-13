"""
CBM Dataset for Concept Bottleneck Model Training

This module provides a PyTorch Dataset class that:
1. Loads precomputed meta-path samples for each seed node
2. Provides batching with proper collation
3. Integrates with the PredictionHead from models.py

Compatible with both mock data (for testing) and real RelBench data.
"""

import os
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np

# Optional imports
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    TorchDataset = object

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import torch.nn.functional as F
except ImportError:
    F = None

from data_pipeline import (
    Schema,
    MetaPath,
    MetaPathSchema,
    MetaPathSampler,
    enumerate_metapath_schemas,
    NULL_TOKEN,
    MISSING_TIME,
    MISSING_FEAT
)


@dataclass
class CBMConfig:
    """Configuration for CBM Dataset."""
    max_hops: int = 2
    n_samples_per_schema: int = 4
    max_paths_per_seed: int = 64
    feature_dim: int = 128
    cache_dir: str = ".cache/cbm"
    precompute: bool = True


class CBMDataset(TorchDataset):
    """
    Dataset for Concept Bottleneck Model training on relational data.
    
    This dataset:
    - Samples meta-paths from a relational entity graph
    - Stores precomputed samples in HDF5 for efficiency
    - Provides batched tensors compatible with EvidenceScorer
    
    The output format matches what the PredictionHead expects:
    - relations: [batch, n_paths, max_hops+1, num_types+1] one-hot
    - times: [batch, n_paths, max_hops+1] relative timestamps
    - features: [batch, n_paths, max_hops+1, feature_dim] node features
    """
    
    def __init__(
        self,
        schema: Schema,
        node_types: List[str],
        seed_indices: List[int],
        seed_times: Optional[List[float]] = None,
        labels: Optional[List[Any]] = None,
        config: Optional[CBMConfig] = None,
        sampler: Optional[MetaPathSampler] = None,
        split: str = "train",
        precomputed_paths: Optional[Dict[int, List[MetaPath]]] = None
    ):
        """
        Initialize the CBM Dataset.
        
        Args:
            schema: Relational schema
            node_types: Ordered list of node types (for one-hot encoding)
            seed_indices: List of seed node indices
            seed_times: Optional timestamps for each seed
            labels: Optional labels for supervised training
            config: Dataset configuration
            sampler: MetaPathSampler for generating paths (required if not precomputed)
            split: Data split name (train/val/test)
            precomputed_paths: Optional precomputed paths dict
        """
        super().__init__()
        
        self.schema = schema
        self.node_types = node_types
        self.seed_indices = seed_indices
        self.seed_times = seed_times if seed_times is not None else [float('inf')] * len(seed_indices)
        self.labels = labels
        self.config = config or CBMConfig()
        self.sampler = sampler
        self.split = split
        
        # Build type-to-index mapping (including NULL token)
        self.type_to_idx = {t: i for i, t in enumerate(node_types)}
        self.type_to_idx[NULL_TOKEN] = len(node_types)
        self.num_types = len(node_types) + 1  # +1 for NULL
        
        # Path storage
        self.paths_by_seed: Dict[int, List[MetaPath]] = precomputed_paths or {}
        
        # Precompute or load paths
        if not self.paths_by_seed:
            self._initialize_paths()
    
    def _initialize_paths(self):
        """Initialize path storage, either by loading or computing."""
        cache_path = self._get_cache_path()
        
        if self.config.precompute and os.path.exists(cache_path):
            print(f"[{self.split}] Loading precomputed paths from {cache_path}")
            self._load_paths(cache_path)
        elif self.sampler is not None:
            print(f"[{self.split}] Computing paths for {len(self.seed_indices)} seeds...")
            self._compute_and_save_paths(cache_path)
        else:
            raise ValueError("Either provide precomputed_paths or a sampler")
    
    def _get_cache_path(self) -> str:
        """Get the cache file path."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / f"{self.split}_paths.json")
    
    def _compute_and_save_paths(self, cache_path: str):
        """Compute paths for all seeds and save to cache."""
        rng = np.random.default_rng(42)
        
        for i, (seed_idx, seed_time) in enumerate(zip(self.seed_indices, self.seed_times)):
            paths, _ = self.sampler.sample_paths_for_seed(
                seed_type=self.schema.root_type,
                seed_idx=seed_idx,
                seed_time=seed_time,
                n_samples_per_schema=self.config.n_samples_per_schema,
                max_total_samples=self.config.max_paths_per_seed,
                rng=rng
            )
            self.paths_by_seed[seed_idx] = paths
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(self.seed_indices)} seeds")
        
        # Save to cache
        if self.config.precompute:
            self._save_paths(cache_path)
    
    def _save_paths(self, cache_path: str):
        """Save paths to JSON cache."""
        data = {}
        for seed_idx, paths in self.paths_by_seed.items():
            data[str(seed_idx)] = [
                {
                    "node_types": p.node_types,
                    "node_times": p.node_times.tolist(),
                    "node_features": p.node_features.tolist()
                }
                for p in paths
            ]
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved paths to {cache_path}")
    
    def _load_paths(self, cache_path: str):
        """Load paths from JSON cache."""
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        for seed_idx_str, path_dicts in data.items():
            seed_idx = int(seed_idx_str)
            paths = []
            for pd in path_dicts:
                paths.append(MetaPath(
                    path_name=None,
                    node_types=pd["node_types"],
                    node_times=np.array(pd["node_times"], dtype=np.float32),
                    node_features=np.array(pd["node_features"], dtype=np.float32)
                ))
            self.paths_by_seed[seed_idx] = paths
    
    def __len__(self) -> int:
        return len(self.seed_indices)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, Any], Optional[Any]]:
        """
        Get a single sample.
        
        Returns:
            sample: Dict with 'relations', 'times', 'features' tensors
            label: Optional label for this sample
        """
        seed_idx = self.seed_indices[idx]
        seed_time = self.seed_times[idx]
        paths = self.paths_by_seed.get(seed_idx, [])
        
        # Convert paths to tensors
        sample = self._paths_to_tensors(paths, seed_time)
        
        # Get label if available
        label = self.labels[idx] if self.labels is not None else None
        
        return sample, label
    
    def _paths_to_tensors(self, paths: List[MetaPath], seed_time: float) -> Dict[str, Any]:
        """
        Convert a list of MetaPaths to tensor format.
        
        Args:
            paths: List of MetaPath objects
            seed_time: Seed node timestamp for relative time computation
            
        Returns:
            Dict with:
                - relations: [n_paths, max_hops+1, num_types] one-hot
                - times: [n_paths, max_hops+1] relative times
                - features: [n_paths, max_hops+1, feature_dim] features
        """
        n_paths = len(paths)
        path_len = self.config.max_hops + 1
        
        if n_paths == 0:
            # Return empty tensors
            if HAS_TORCH:
                return {
                    'relations': torch.zeros(1, path_len, self.num_types),
                    'times': torch.full((1, path_len), MISSING_TIME),
                    'features': torch.full((1, path_len, self.config.feature_dim), MISSING_FEAT)
                }
            else:
                return {
                    'relations': np.zeros((1, path_len, self.num_types), dtype=np.float32),
                    'times': np.full((1, path_len), MISSING_TIME, dtype=np.float32),
                    'features': np.full((1, path_len, self.config.feature_dim), MISSING_FEAT, dtype=np.float32)
                }
        
        # Get feature dimension from first path
        feat_dim = paths[0].node_features.shape[1] if len(paths[0].node_features.shape) > 1 else self.config.feature_dim
        
        # Initialize arrays
        relations = np.zeros((n_paths, path_len, self.num_types), dtype=np.float32)
        times = np.full((n_paths, path_len), MISSING_TIME, dtype=np.float32)
        features = np.full((n_paths, path_len, feat_dim), MISSING_FEAT, dtype=np.float32)
        
        for i, path in enumerate(paths):
            # One-hot encode node types
            for j, node_type in enumerate(path.node_types[:path_len]):
                type_idx = self.type_to_idx.get(node_type, self.num_types - 1)
                relations[i, j, type_idx] = 1.0
            
            # Relative times (relative to seed = root node)
            root_time = path.node_times[0]
            for j in range(min(len(path.node_times), path_len)):
                if path.node_times[j] != MISSING_TIME and root_time != MISSING_TIME:
                    if np.isfinite(path.node_times[j]) and np.isfinite(root_time):
                        times[i, j] = path.node_times[j] - root_time
                    else:
                        times[i, j] = 0.0  # Same time if root has no time
            
            # Features
            for j in range(min(len(path.node_features), path_len)):
                if not np.any(np.isinf(path.node_features[j])):
                    features[i, j] = path.node_features[j]
        
        if HAS_TORCH:
            return {
                'relations': torch.from_numpy(relations),
                'times': torch.from_numpy(times),
                'features': torch.from_numpy(features)
            }
        else:
            return {
                'relations': relations,
                'times': times,
                'features': features
            }
    
    @staticmethod
    def collate_fn(batch: List[Tuple[Dict[str, Any], Any]]) -> Tuple[Dict[str, Any], Any]:
        """
        Collate function for DataLoader.
        
        Handles variable numbers of paths per sample by padding to max.
        
        Args:
            batch: List of (sample_dict, label) tuples
            
        Returns:
            batched_sample: Dict with batched tensors
            batched_labels: Stacked labels tensor
        """
        samples, labels = zip(*batch)
        
        # Find max paths in this batch
        max_paths = max(s['relations'].shape[0] for s in samples)
        
        batched = {}
        
        for key in ['relations', 'times', 'features']:
            tensors = []
            for s in samples:
                t = s[key]
                n_paths = t.shape[0]
                
                if n_paths < max_paths:
                    # Pad with zeros/inf
                    if HAS_TORCH:
                        if key == 'relations':
                            pad = torch.zeros(max_paths - n_paths, *t.shape[1:])
                        else:
                            pad = torch.full((max_paths - n_paths, *t.shape[1:]), 
                                           MISSING_TIME if key == 'times' else MISSING_FEAT)
                        t = torch.cat([t, pad], dim=0)
                    else:
                        if key == 'relations':
                            pad = np.zeros((max_paths - n_paths, *t.shape[1:]), dtype=np.float32)
                        else:
                            pad = np.full((max_paths - n_paths, *t.shape[1:]), 
                                        MISSING_TIME if key == 'times' else MISSING_FEAT, dtype=np.float32)
                        t = np.concatenate([t, pad], axis=0)
                
                tensors.append(t)
            
            if HAS_TORCH:
                batched[key] = torch.stack(tensors, dim=0)
            else:
                batched[key] = np.stack(tensors, axis=0)
        
        # Handle labels
        if labels[0] is not None:
            if HAS_TORCH:
                batched_labels = torch.tensor(labels)
            else:
                batched_labels = np.array(labels)
        else:
            batched_labels = None
        
        return batched, batched_labels


def create_cbm_datasets_from_mock(
    mock_data,
    schema: Schema,
    config: CBMConfig,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[CBMDataset, CBMDataset, CBMDataset]:
    """
    Create train/val/test CBM datasets from mock data.
    
    Args:
        mock_data: MockHeteroData instance
        schema: Schema object
        config: CBMConfig
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from test_mock_relf1 import MockMetaPathSampler
    
    # Get all driver indices
    num_drivers = mock_data._node_counts["driver"]
    all_indices = list(range(num_drivers))
    
    # Create mock labels (random binary for testing)
    rng = np.random.default_rng(42)
    all_labels = rng.integers(0, 2, num_drivers).tolist()
    
    # Create mock times
    all_times = [float('inf')] * num_drivers  # No temporal constraint for drivers
    
    # Split indices
    rng.shuffle(all_indices)
    n_train = int(len(all_indices) * train_ratio)
    n_val = int(len(all_indices) * val_ratio)
    
    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    test_idx = all_indices[n_train + n_val:]
    
    # Create sampler
    sampler = MockMetaPathSampler(mock_data, schema, max_hops=config.max_hops)
    
    # Create datasets
    datasets = []
    for split, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ds = CBMDataset(
            schema=schema,
            node_types=schema.node_types,
            seed_indices=indices,
            seed_times=[all_times[i] for i in indices],
            labels=[all_labels[i] for i in indices],
            config=config,
            sampler=sampler,
            split=split
        )
        datasets.append(ds)
    
    return tuple(datasets)


# =============================================================================
# Testing
# =============================================================================

def test_cbm_dataset():
    """Test CBMDataset with mock data."""
    print("\n" + "=" * 60)
    print("CBMDataset Test")
    print("=" * 60)
    
    # Import mock utilities
    from test_mock_relf1 import create_mock_f1_schema, MockHeteroData
    
    # Create mock data
    schema = create_mock_f1_schema()
    mock_data = MockHeteroData(num_drivers=50, num_races=20)
    
    # Configuration
    config = CBMConfig(
        max_hops=2,
        n_samples_per_schema=2,
        max_paths_per_seed=16,
        feature_dim=64,
        cache_dir="/tmp/cbm_test",
        precompute=False  # Don't cache for test
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_cbm_datasets_from_mock(
        mock_data, schema, config,
        train_ratio=0.6, val_ratio=0.2
    )
    
    print(f"\nDataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    # Test single item access
    print("\nTesting single item access...")
    sample, label = train_ds[0]
    print(f"  Relations shape: {sample['relations'].shape}")
    print(f"  Times shape: {sample['times'].shape}")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Label: {label}")
    
    # Test batching
    print("\nTesting batching...")
    batch_samples = [train_ds[i] for i in range(4)]
    batched, labels = CBMDataset.collate_fn(batch_samples)
    
    print(f"  Batched relations shape: {batched['relations'].shape}")
    print(f"  Batched times shape: {batched['times'].shape}")
    print(f"  Batched features shape: {batched['features'].shape}")
    print(f"  Labels shape: {labels.shape if HAS_TORCH else labels.shape}")
    
    print("\n" + "=" * 60)
    print("CBMDataset test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cbm_dataset()
