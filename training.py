import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Dict
import wandb

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task

from models import RelCBM
from cbm_dataset import GloveTextEmbedding, CBMTokens
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import h5py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-top3")
    parser.add_argument("--precompute", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--gt_conv_type", type=str, default="full")

    parser.add_argument("--ablate", type=str, default="none")
    parser.add_argument("--gnn_pe_dim", type=int, default=0)
    parser.add_argument("--num_neighbors", type=int, default=300)
    parser.add_argument("--num_centroids", type=int, default=4096)
    parser.add_argument("--ff_dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--temporal_strategy", type=str, default="uniform")
    parser.add_argument("--pos_enc", type=str, default="none")
    parser.add_argument("--max_degree", type=int, default=10000)
    parser.add_argument("--pos_enc_dim", type=int, default=128)
    parser.add_argument("--max_steps_per_epoch", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/debug")
    parser.add_argument("--run_name", type=str, default="debug")
    parser.add_argument('--model_parameters', type=int, default=0, help='Number of model parameters')
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    parser.add_argument("--train_stage", type=str, default="finetune", choices=["finetune"])

    # My arguments
    parser.add_argument("--max_hops", type=int, default=2) 
    parser.add_argument("--n_paths_per_seed", type=int, default=4)
    parser.add_argument("--n_samples_per_metapath_schema", type=int, default=2)

    return parser.parse_args()

def inspect_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in H5: {list(f.keys())}") # path_types, path_node_indices, path_times 
        
        # Look at the first 5 entries of path_types
        n_paths = f['path_n_paths'][:5]
        indices = f['path_node_indices'][:5]
        times = f['path_times'][:5]
        types = f['path_types'][:5]

        for i in range(5):
            print(f"Inspecting subgraph {i}")
            print(f"n_paths: {n_paths[i]}")
            print(f"indices: {indices[i]}")
            print(f"times: {times[i]}")
            print(f"types: {types[i]}")

        # Check if any non-padding data exists
        if (types != -1).any():
            print("Status: Success! Non-empty paths found.")
        else:
            print("Status: Warning! All paths appear to be empty padding.")

def prepare_data(args):
    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: EntityTask = get_task(args.dataset, args.task, download=True)

    stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(dataset.get_db())
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)

    data, col_stats_dict = make_pkey_fkey_graph(
        dataset.get_db(),
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=TextEmbedderConfig(
            text_embedder=GloveTextEmbedding(device=device), batch_size=256
        ),
        cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
    )

    data = {
        split: CBMTokens(
            data=data,
            task=task,
            split=split,
            max_hops=args.max_hops,
            n_paths_per_seed=args.n_paths_per_seed,
            n_samples_per_metapath_schema=args.n_samples_per_metapath_schema,
            precompute=args.precompute,
            precomputed_dir=f"{args.cache_dir}/precomputed/{args.dataset}/{args.task}",
            num_workers=args.num_workers,
            undirected=True,
        )
        for split in ["train", "val", "test"]
        }
    
    loader_train = DataLoader(
        data["train"], 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=data["train"].collate,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True)

    loader_val = DataLoader(
        data["val"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data["val"].collate,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True
    )

    loader_test = DataLoader(
        data["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data["test"].collate,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        pin_memory=True
    )


    loader_dict: Dict[str, DataLoader] = {"train": loader_train, "val": loader_val, "test": loader_test}

    return data, loader_dict, task, col_stats_dict

def build_model(args, data, col_stats_dict):

    out_channels = 1
    col_names_dict = {node_type: data["train"].data[node_type].tf.col_names_dict for node_type in data["train"].data.node_types}
    
    model = RelCBM(
        num_nodes=data["train"].data.num_nodes,
        max_neighbor_hop=data["train"].max_hops,
        node_type_map=data["train"].type_to_idx,
        col_names_dict=col_names_dict,
        col_stats_dict=col_stats_dict,
        local_num_layers=args.num_layers,
        channels=args.channels,
        out_channels=out_channels,
        global_dim=args.channels//2,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        conv_type=args.gt_conv_type,
        sample_node_len=args.num_neighbors,
        args=args
    ).to(device)

    return model

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    loss_accum = 0
    count = 0
    
    pbar = tqdm(loader, desc="Train", mininterval=1.0)
    for batch in pbar:
        batch = {
            k: v.to(device) if hasattr(v, 'to') else v 
            for k, v in batch.items()
        }
        optimizer.zero_grad()
        
        pred = model(batch)
        pred = pred.view(-1)
        pred = torch.sigmoid(pred)

        
        labels = batch.labels.float()
        
        loss = loss_fn(pred, labels)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_value = loss.item()
        loss_accum += loss_value * len(labels)
        count += len(labels)
        
        pbar.set_postfix({'loss': loss_value})

    return loss_accum / count if count > 0 else 0

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds = []
    
    for batch in tqdm(loader, desc="Eval"):
        batch = {
            k: v.to(device) if hasattr(v, 'to') else v 
            for k, v in batch.items()
        }
        pred = model(batch).view(-1)
        preds.append(pred.cpu().numpy())
        
    return np.concatenate(preds)

def main():
    args = parse_args()
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {args.dataset}:{args.task} with device: {current_device}")

    data, loaders, task, col_stats_dict = prepare_data(args)

    exit()
    model = build_model(args, data, col_stats_dict)
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"

    print(f"Model built: {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_metric = -math.inf
    wandb.init(project="rel-cbm", config=vars(args))

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # Train
        train_loss = train_epoch(model, loaders['train'], optimizer, current_device, loss_fn)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_preds = evaluate(model, loaders['val'], current_device)
        val_metrics = task.evaluate(val_preds, task.get_table("val"))
        print(f"Val Metrics: {val_metrics}")
        
        # Log to W&B
        wandb.log({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        # Checkpoint logic (Save best model)
        current_metric_value = val_metrics[tune_metric]
        
        if current_metric_value > best_val_metric:
            best_val_metric = current_metric_value
            print(f"New best {tune_metric}! Saving checkpoint...")
            # torch.save(model.state_dict(), f"{args.out_dir}/best_model.pt")

    # 5. Final Test
    print("\n=== Final Test ===")
    test_preds = evaluate(model, loaders['test'], current_device)
    test_metrics = task.evaluate(test_preds)
    print(f"Test Metrics: {test_metrics}")



if __name__ == "__main__":
    main()