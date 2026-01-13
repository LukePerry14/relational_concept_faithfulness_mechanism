"""
CBM Dataset for Concept Bottleneck Model Training on RelBench

This module provides a dataset class similar to RelGTTokens but for meta-path sampling
needed by the Concept Bottleneck Model. Reuses existing infrastructure where possible.
"""
import os
import gc
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from relbench.modeling.graph import get_node_train_table_input
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch import Tensor

from torch_geometric.data import HeteroData

from data_handling import extract_schema_from_heterodata, enumerate_metapath_schemas, MetaPathSampler
from utils.custom_dataclasses import NULL_TOKEN, MISSING_TIME, MISSING_FEAT


class GloveTextEmbedding:
    def __init__(self, device: torch.device):
        self.model = SentenceTransformer("sentence-transformers/average_word_embeddings_glove.6B.300d", device=device)
    
    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))

def build_adjacency_hetero(hetero_data: HeteroData, undirected: bool = True):
    """Taken from Rel-GT github repo"""
    adjacency = {
        node_type: [set() for _ in range(hetero_data[node_type].num_nodes)]
        for node_type in hetero_data.node_types
    }
    for edge_type in hetero_data.edge_types:
        src_type, _, dst_type = edge_type
        if 'edge_index' not in hetero_data[edge_type]:
            continue
        edge_index = hetero_data[edge_type].edge_index
        src_list = edge_index[0].tolist()
        dst_list = edge_index[1].tolist()
        for s, d in zip(src_list, dst_list):
            adjacency[src_type][s].add((dst_type, d))
            if undirected:
                adjacency[dst_type][d].add((src_type, s))
    return adjacency

GLOBAL_PATH_SAMPLER = None
GLOBAL_NODE_TYPE = None
GLOBAL_SCHEMA = None

def init_worker(path_sampler, node_type, schema):
    """Initialize worker process with shared objects"""
    global GLOBAL_PATH_SAMPLER, GLOBAL_NODE_TYPE, GLOBAL_SCHEMA
    GLOBAL_PATH_SAMPLER = path_sampler
    GLOBAL_NODE_TYPE = node_type
    GLOBAL_SCHEMA = schema

def process_seed(args):
    """
    Worker function to process one seed node.
    Returns paths sampled for this seed.
    """
    global GLOBAL_PATH_SAMPLER, GLOBAL_NODE_TYPE, GLOBAL_SCHEMA
    
    node_id, seed_time, seed_val, n_samples_per_metapath_schema, max_total_samples = args
    
    # Set seed for reproducibility
    rng = np.random.default_rng(seed=seed_val)
    
    # Sample paths
    paths, schema_counts = GLOBAL_PATH_SAMPLER.sample_paths_for_seed(
        seed_type=GLOBAL_NODE_TYPE,
        seed_idx=int(node_id),
        seed_time=seed_time,
        n_samples_per_metapath_schema=n_samples_per_metapath_schema,
        max_total_samples=max_total_samples,
        rng=rng
    )
    
    return paths

class CBMTokens(Dataset):
    """
    Pytorch-geometric  Dataset wrapper for sampling meta-paths for Concept Bottleneck Model training.
    
    Reuses code from rel-GT github
    """
    
    def __init__(
        self,
        data: HeteroData,
        task,
        split: str = "train",
        max_hops: int = 3,
        n_paths_per_seed: int = 64,
        n_samples_per_metapath_schema: int = 4,
        precompute: bool = True,
        precomputed_dir: str = None,
        num_workers: int = None,
        undirected: bool = True,
    ):
        super().__init__()
        self.data = data
        self.task = task
        self.split = split
        self.max_hops = max_hops
        self.n_paths_per_seed = n_paths_per_seed
        self.n_samples_per_metapath_schema = n_samples_per_metapath_schema
        self.undirected = undirected
        self.num_workers = num_workers
        self.precompute = precompute
        self.precomputed_dir = precomputed_dir

        # Get training table
        self.table = self.task.get_table(split=self.split)
        self.table_input = get_node_train_table_input(self.table, self.task)
        self.node_type, self.node_idxs = self.table_input.nodes
        self.target = self.table_input.target if self.table_input.target is not None else None
        self.time = getattr(self.table_input, "time", None)
        
        # Extract schema
        print(f"Extracting schema for root type: {self.node_type}")
        self.schema = extract_schema_from_heterodata(
            self.data,
            root_type=self.node_type,
            exclude_self_loops=True
        )
        
        # Enumerate metapath schemas
        print(f"Enumerating metapath schemas up to {self.max_hops} hops...")
        self.metapath_schemas = enumerate_metapath_schemas(
            self.schema,
            max_hops=self.max_hops
        )
        print(f"Found {len(self.metapath_schemas)} metapath schemas")
        
        # Build adjacency
        print("Building adjacency list...")
        self.adjacency = build_adjacency_hetero(self.data, undirected=self.undirected)
        
        # Create path sampler
        self.path_sampler = MetaPathSampler(
            data=self.data,
            schema=self.schema,
            metapath_schemas=self.metapath_schemas,
            max_hops=self.max_hops,
            adjacency=self.adjacency,
        )
        
        # Type mappings
        self.node_types = self.data.node_types
        self.type_to_idx = {nt: idx for idx, nt in enumerate(self.node_types)}
        self.type_to_idx[NULL_TOKEN] = len(self.node_types)
        self.num_types = len(self.node_types) + 1
        
        # HDF5 path
        self.precomputed_path = self._construct_precomputed_path()
        
        if self.precompute:
            if os.path.exists(self.precomputed_path):
                print(f"[{self.split}] Found existing HDF5 at {self.precomputed_path}")
            else:
                print(f"[{self.split}] Precomputing path sampling...")
                self._precompute_paths()

    
    def _construct_precomputed_path(self) -> str:
        if not self.precomputed_dir:
            raise ValueError("must provide a 'precomputed_dir' to store paths.")
        path = os.path.join(
            self.precomputed_dir,
            f"hops{self.max_hops}_paths{self.n_paths_per_seed}",
            f"{self.split}.h5"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    def __len__(self):
        return len(self.node_idxs)
    
    def _get_smart_chunk_size(self, total_samples: int) -> int:
        """
        Calculate chunk size that balances:
        1. Progress update frequency
        2. HDF5 write efficiency
        3. Memory usage
        
        Strategy from RelGT:
        - Use fixed chunk size for HDF5 (good for I/O)
        - But update progress per chunk (good for visibility)
        """
        # For HDF5 chunking: use min(total, 10000) like RelGT
        hdf5_chunk = min(total_samples, 10000)

        # For processing chunks: smaller if dataset is small
        if total_samples <= 500:
            processing_chunk = 50  # Update every 50 samples
        elif total_samples <= 2000:
            processing_chunk = 100  # Update every 100 samples
        elif total_samples <= 10000:
            processing_chunk = 500  # Update every 500 samples
        else:
            processing_chunk = 1000  # Update every 1000 samples
            
        return hdf5_chunk, processing_chunk
    
    def _precompute_paths(self):
        """
        Precompute paths for all seeds and store in HDF5.
        
        Storage format:
        - For each seed, store multiple paths
        - Each path has: types, times, features (all padded to max_hops+1)
        - Store as variable-length dataset per seed
        """
        total = len(self.node_idxs)
        hdf5_chunk_size, processing_chunk_size = self._get_smart_chunk_size(total)
        
        print(f"Processing {total} samples:")
        print(f"  - HDF5 chunk size: {hdf5_chunk_size}")
        print(f"  - Processing chunk size: {processing_chunk_size}")
        print(f"  - Progress updates every {processing_chunk_size} samples")
        
        max_paths = self.n_paths_per_seed
        path_len = self.max_hops + 1

        with h5py.File(self.precomputed_path, 'w') as hf:
            # Create datasets with HDF5 chunking
            path_types_dset = hf.create_dataset(
                "path_types",
                shape=(total, max_paths, path_len),
                dtype='int16',
                chunks=(hdf5_chunk_size, max_paths, path_len),
                compression='gzip',
                compression_opts=4
            )
            path_indices_dset = hf.create_dataset(
                "path_node_indices",
                shape=(total, max_paths, path_len),
                dtype='int32',
                chunks=(hdf5_chunk_size, max_paths, path_len),
                compression='gzip',
                compression_opts=4
            )
            path_times_dset = hf.create_dataset(
                "path_times",
                shape=(total, max_paths, path_len),
                dtype='float32',
                chunks=(hdf5_chunk_size, max_paths, path_len),
                compression='gzip',
                compression_opts=4
            )
            path_n_paths_dset = hf.create_dataset(
                "path_n_paths",
                shape=(total,),
                dtype='int16',
                chunks=(hdf5_chunk_size,)
            )
            
            # Store metadata
            hf.attrs['max_hops'] = self.max_hops
            hf.attrs['n_paths_per_seed'] = self.n_paths_per_seed
            hf.attrs['split'] = self.split
            hf.attrs['node_type'] = self.node_type
            
            # Process in chunks with proper progress tracking
            # KEY FIX: Wrap outer loop with tqdm like RelGT does
            with tqdm(total=total, desc=f"Precomputing '{self.split}'") as pbar:
                for start_idx in range(0, total, processing_chunk_size):
                    end_idx = min(start_idx + processing_chunk_size, total)
                    size_chunk = end_idx - start_idx
                    
                    # Prepare tasks for this chunk
                    tasks = []
                    for i in range(start_idx, end_idx):
                        node_idx_t = self.node_idxs[i]
                        node_idx = node_idx_t.item() if hasattr(node_idx_t, 'item') else int(node_idx_t)
                        
                        seed_time = self.time[i].item() if self.time is not None else float('inf')
                        seed_val = hash((self.node_type, node_idx, seed_time, self.n_paths_per_seed)) & 0xffffffff
                        
                        tasks.append((
                            node_idx,
                            seed_time,
                            seed_val,
                            self.n_samples_per_metapath_schema,
                            max_paths
                        ))
                    
                    # Process chunk with multiprocessing or sequential
                    if self.num_workers is not None and self.num_workers > 1:
                        num_workers_actual = min(self.num_workers, size_chunk, cpu_count() - 2)
                        num_workers_actual = max(1, num_workers_actual)
                        
                        with Pool(
                            processes=num_workers_actual,
                            initializer=init_worker,
                            initargs=(self.path_sampler, self.node_type, self.schema)
                        ) as pool:
                            chunk_results = pool.map(process_seed, tasks)
                    else:
                        # Sequential processing - initialize globals first!
                        init_worker(self.path_sampler, self.node_type, self.schema)
                        chunk_results = [process_seed(t) for t in tasks]
                    
                    # Convert results to padded arrays
                    c_types = np.full((size_chunk, max_paths, path_len), -1, dtype=np.int16)
                    c_indices = np.full((size_chunk, max_paths, path_len), -1, dtype=np.int32)
                    c_times = np.full((size_chunk, max_paths, path_len), MISSING_TIME, dtype=np.float32)
                    c_n_paths = np.zeros(size_chunk, dtype=np.int16)
                    
                    for i, paths in enumerate(chunk_results):
                        n_paths = min(len(paths), max_paths)
                        c_n_paths[i] = n_paths
                        
                        for p in range(n_paths):
                            path = paths[p]
                            
                            for l in range(min(len(path.node_types), path_len)):
                                node_type = path.node_types[l]
                                node_id = path.node_ids[l]
                                node_time = path.node_times[l]
                                
                                type_idx = self.type_to_idx.get(node_type, self.num_types - 1)
                                c_types[i, p, l] = type_idx
                                c_indices[i, p, l] = node_id
                                c_times[i, p, l] = node_time
                    
                    # Write chunk to HDF5
                    path_types_dset[start_idx:end_idx] = c_types
                    path_indices_dset[start_idx:end_idx] = c_indices
                    path_times_dset[start_idx:end_idx] = c_times
                    path_n_paths_dset[start_idx:end_idx] = c_n_paths
                    
                    # Update progress by chunk size
                    pbar.update(size_chunk)
                    
                    # Cleanup
                    del chunk_results
                    gc.collect()
        
        print(f"\n✓ Precomputation complete: {self.precomputed_path}")
    
    def __getitem__(self, idx: int):
        """Retrieve samples from HDF5 and the label from self.target"""
        with h5py.File(self.precomputed_path, 'r') as hf:
            sample = {
                "path_types": torch.from_numpy(hf["path_types"][idx]).long(),
                "path_indices": torch.from_numpy(hf["path_node_indices"][idx]).long(),
                "path_times": torch.from_numpy(hf["path_times"][idx]).float(),
                "global_idx": idx,
            }
        
        label = self.target[idx] if self.target is not None else None
        return sample, label
    
    def collate(self, batch: List[Tuple[dict, Optional[torch.Tensor]]]):
        """Collate function for DataLoader"""
        samples, labels = zip(*batch)
        
        path_types = torch.stack([s["path_types"] for s in samples], dim=0)
        path_indices = torch.stack([s["path_indices"] for s in samples], dim=0)
        path_times = torch.stack([s["path_times"] for s in samples], dim=0)
        global_idxs = torch.tensor([s["global_idx"] for s in samples], dtype=torch.long)
        
        out = {
            "path_types": path_types,
            "path_indices": path_indices,
            "path_times": path_times,
            "global_idx": global_idxs,
        }
        
        if self.target is not None:
            out["labels"] = torch.stack(labels, dim=0)
        else:
            out["labels"] = None
        
        return out


# =============================================================================
# Testing Code
# =============================================================================

if __name__ == "__main__":
    """
    Quick test to make sure the class can be instantiated.
    For full testing, use test_cbm_dataset.py
    """
    print("CBMTokens dataset class loaded successfully")
    print("\nTo test with real data, use:")
    print("  from cbm_dataset import CBMTokens")
    print("  dataset = CBMTokens(data, task, split='train', ...)")