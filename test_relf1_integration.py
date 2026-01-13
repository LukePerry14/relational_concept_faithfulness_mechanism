"""
Integration Test: Load rel-f1 dataset and test the data pipeline.

This script:
1. Loads the rel-f1 dataset from RelBench
2. Extracts the schema for the driver-top3 task
3. Enumerates meta-path schemas
4. Tests path sampling for a few seed nodes

Run with: python test_relf1_integration.py
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from tqdm import tqdm

# RelBench imports
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.base import Dataset, EntityTask, TaskType
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.utils import get_stype_proposal

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig

# Local imports
from data_pipeline import (
    Schema,
    MetaPathSchema,
    MetaPath,
    extract_schema_from_heterodata,
    enumerate_metapath_schemas,
    print_metapath_schemas,
    MetaPathSampler,
    NULL_TOKEN
)


def get_simple_text_embedder():
    """Create a simple text embedder for testing (avoids heavy model loading)."""
    try:
        from sentence_transformers import SentenceTransformer
        
        class SimpleTextEmbedding:
            def __init__(self, device: str = "cpu"):
                # Use a lightweight model for testing
                self.model = SentenceTransformer(
                    "sentence-transformers/average_word_embeddings_glove.6B.300d",
                    device=device
                )
            
            def __call__(self, sentences):
                return torch.from_numpy(self.model.encode(sentences))
        
        return SimpleTextEmbedding()
    except Exception as e:
        print(f"Warning: Could not load text embedder: {e}")
        print("Using placeholder embeddings.")
        return None


def load_rel_f1_data(cache_dir: str = None, device: str = "cpu"):
    """
    Load the rel-f1 dataset and construct the heterogeneous graph.
    
    Args:
        cache_dir: Directory for caching processed data
        device: Device for text embedding computation
        
    Returns:
        data: HeteroData object
        task: EntityTask object for driver-top3
        col_stats_dict: Column statistics
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/cbm_relbench")
    
    print("=" * 60)
    print("Loading rel-f1 dataset...")
    print("=" * 60)
    
    # Load dataset and task
    dataset: Dataset = get_dataset("rel-f1", download=True)
    task: EntityTask = get_task("rel-f1", "driver-top3", download=True)
    
    print(f"\nDataset: {dataset}")
    print(f"Task: {task}")
    print(f"Task type: {task.task_type}")
    
    # Get column type proposals
    stypes_cache_path = Path(f"{cache_dir}/rel-f1/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
        print("\nLoaded cached column types.")
    except FileNotFoundError:
        print("\nComputing column type proposals...")
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
    
    # Build the heterogeneous graph
    print("\nBuilding heterogeneous graph...")
    text_embedder = get_simple_text_embedder()
    
    if text_embedder is not None:
        text_cfg = TextEmbedderConfig(text_embedder=text_embedder, batch_size=256)
    else:
        text_cfg = None
    
    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_cfg,
        cache_dir=f"{cache_dir}/rel-f1/materialized",
    )
    
    return data, task, col_stats_dict


def analyze_heterodata(data):
    """Print statistics about the HeteroData graph."""
    print("\n" + "=" * 60)
    print("HeteroData Graph Analysis")
    print("=" * 60)
    
    print(f"\nNode types ({len(data.node_types)}):")
    for nt in data.node_types:
        num_nodes = data[nt].num_nodes
        has_time = hasattr(data[nt], 'time') and data[nt].time is not None
        has_tf = hasattr(data[nt], 'tf') and data[nt].tf is not None
        has_x = hasattr(data[nt], 'x') and data[nt].x is not None
        
        time_info = f", time: {has_time}" if has_time else ""
        feat_info = ", features: TorchFrame" if has_tf else (", features: x" if has_x else "")
        
        print(f"  {nt}: {num_nodes:,} nodes{time_info}{feat_info}")
    
    print(f"\nEdge types ({len(data.edge_types)}):")
    for et in data.edge_types:
        src_type, rel, dst_type = et
        if 'edge_index' in data[et]:
            num_edges = data[et].edge_index.shape[1]
            print(f"  {src_type} --[{rel}]--> {dst_type}: {num_edges:,} edges")
        else:
            print(f"  {src_type} --[{rel}]--> {dst_type}: (no edges)")


def test_schema_extraction(data, task):
    """Test schema extraction for the driver-top3 task."""
    print("\n" + "=" * 60)
    print("Schema Extraction Test")
    print("=" * 60)
    
    # The target entity type for driver-top3 is "driver"
    # We need to find this from the task definition
    table = task.get_table("train")
    table_input = get_node_train_table_input(table, task)
    root_type, _ = table_input.nodes
    
    print(f"\nRoot type (from task): {root_type}")
    
    # Extract schema
    schema = extract_schema_from_heterodata(data, root_type)
    
    print(f"\nExtracted Schema:")
    print(f"  Root type: {schema.root_type}")
    print(f"  Node types: {schema.node_types}")
    print(f"\n  Transitions:")
    for src, dsts in sorted(schema.transitions.items()):
        print(f"    {src} -> {dsts}")
    
    return schema


def test_metapath_enumeration(schema, max_hops=2):
    """Test meta-path schema enumeration."""
    print("\n" + "=" * 60)
    print(f"Meta-path Enumeration Test (max_hops={max_hops})")
    print("=" * 60)
    
    schemas = enumerate_metapath_schemas(schema, max_hops)
    print_metapath_schemas(schemas, f"Meta-paths from {schema.root_type}")
    
    return schemas


def test_path_sampling(data, schema, task, metapath_schemas, n_seeds=5, max_hops=2):
    """Test path sampling for a few seed nodes."""
    print("\n" + "=" * 60)
    print(f"Path Sampling Test (n_seeds={n_seeds})")
    print("=" * 60)
    
    # Get training table to find seed nodes
    table = task.get_table("train")
    table_input = get_node_train_table_input(table, task)
    seed_type, seed_idxs = table_input.nodes
    seed_times = table_input.time
    
    print(f"\nTotal training seeds: {len(seed_idxs)}")
    print(f"Seed type: {seed_type}")
    
    # Create sampler
    sampler = MetaPathSampler(
        data=data,
        schema=schema,
        max_hops=max_hops,
        metapath_schemas=metapath_schemas
    )
    
    print(f"\nSampler initialized with {len(sampler.metapath_schemas)} meta-path schemas")
    
    # Sample paths for first n_seeds
    rng = np.random.default_rng(42)
    
    for i in range(min(n_seeds, len(seed_idxs))):
        seed_idx = seed_idxs[i].item()
        seed_time = seed_times[i].item() if seed_times is not None else float('inf')
        
        print(f"\n--- Seed {i}: idx={seed_idx}, time={seed_time:.2f} ---")
        
        paths, schema_counts = sampler.sample_paths_for_seed(
            seed_type=seed_type,
            seed_idx=seed_idx,
            seed_time=seed_time,
            n_samples_per_schema=4,
            max_total_samples=32,
            rng=rng
        )
        
        print(f"  Sampled {len(paths)} paths")
        print(f"  Schema distribution:")
        for mp_schema, count in sorted(schema_counts.items(), key=lambda x: -x[1]):
            print(f"    {mp_schema}: {count}")
        
        # Print first few paths
        print(f"\n  First 3 paths:")
        for p in paths[:3]:
            types_str = " → ".join(p.node_types)
            times_str = ", ".join([f"{t:.1f}" if t != float('inf') else "∅" for t in p.node_times])
            print(f"    Types: {types_str}")
            print(f"    Times: [{times_str}]")


def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("CBM-RelBench Integration Test: rel-f1 / driver-top3")
    print("=" * 60)
    
    # Configuration
    cache_dir = os.path.expanduser("~/.cache/cbm_relbench")
    max_hops = 2
    
    # Step 1: Load data
    data, task, col_stats = load_rel_f1_data(cache_dir=cache_dir)
    
    # Step 2: Analyze the graph
    analyze_heterodata(data)
    
    # Step 3: Extract schema
    schema = test_schema_extraction(data, task)
    
    # Step 4: Enumerate meta-path schemas
    metapath_schemas = test_metapath_enumeration(schema, max_hops=max_hops)
    
    # Step 5: Test path sampling
    test_path_sampling(data, schema, task, metapath_schemas, n_seeds=3, max_hops=max_hops)
    
    # Save schema for later use
    schema_path = f"{cache_dir}/rel-f1/schema_driver-top3.json"
    Path(schema_path).parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump(schema.to_dict(), f, indent=2)
    print(f"\nSchema saved to: {schema_path}")
    
    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
