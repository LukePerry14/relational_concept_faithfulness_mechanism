"""
CBM Training Script for Relational Data

This script demonstrates the full training pipeline:
1. Load data (mock or RelBench)
2. Extract schema and enumerate meta-paths
3. Create CBM datasets with sampled paths
4. Train the PredictionHead (ConceptDecoder + EvidenceScorer + LogicHead)
5. Evaluate and interpret learned concepts

Usage:
    # With mock data (for testing):
    python train_cbm.py --mock
    
    # With RelBench data (requires relbench package):
    python train_cbm.py --dataset rel-f1 --task driver-top3
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Handle optional imports gracefully
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Running in numpy-only mode.")

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import (
    Schema,
    MetaPathSchema,
    enumerate_metapath_schemas,
    print_metapath_schemas,
    NULL_TOKEN
)

from models import (
    ConceptDecoder,
    EvidenceScorer,
    SEL_LogicHead,
    PredictionHead
)

from cbm_dataset import CBMDataset, CBMConfig, create_cbm_datasets_from_mock


# =============================================================================
# Model Components
# =============================================================================

if HAS_TORCH:
    class ConceptDecoder(nn.Module):
        """
        Decodes concept embeddings into interpretable meta-path prototypes.
        """
        
        def __init__(
            self,
            concept_dim: int,
            feature_dim: int,
            max_hops: int,
            num_types: int,
            schema_adj: torch.Tensor
        ):
            super().__init__()
            
            self.L = max_hops + 1
            self.R = num_types
            self.D = feature_dim
            
            self.register_buffer('adj', schema_adj)
            
            self.trunk = nn.Sequential(
                nn.Linear(concept_dim, concept_dim * 2),
                nn.GELU(),
                nn.Linear(concept_dim * 2, concept_dim)
            )
            
            self.relation_rnn = nn.Linear(concept_dim + self.R, self.R)
            self.time_head = nn.Linear(concept_dim, self.L)
            self.gamma_time_head = nn.Linear(concept_dim, self.L)
            self.gamma_feat_head = nn.Linear(concept_dim, self.L)
            self.feature_head = nn.Linear(concept_dim, self.L * self.D)
            self.tau_head = nn.Linear(concept_dim, 1)
        
        def forward(self, concepts: torch.Tensor):
            num_concepts = concepts.shape[0]
            device = concepts.device
            
            h = self.trunk(concepts)
            
            # Autoregressive relation decoding
            rel_probs = []
            prev = torch.zeros(num_concepts, self.R, device=device)
            prev[:, 0] = 1.0  # Root type
            rel_probs.append(prev)
            
            for i in range(self.L - 1):
                logits = self.relation_rnn(torch.cat([h, rel_probs[-1]], dim=-1))
                mask = self.adj[min(i+1, self.adj.shape[0]-1)]
                masked = logits.masked_fill(mask.unsqueeze(0) == 0, -1e9)
                rel_probs.append(F.softmax(masked, dim=-1))
            
            relations = torch.stack(rel_probs, dim=1)
            times = F.softplus(self.time_head(h))
            gamma_times = F.softplus(self.gamma_time_head(h)) + 0.1
            gamma_feats = F.sigmoid(self.gamma_feat_head(h))
            features = self.feature_head(h).view(num_concepts, self.L, self.D)
            features = F.normalize(features, dim=-1)
            taus = F.softplus(self.tau_head(h)) + 0.1
            
            return relations, times, gamma_times, gamma_feats, features, taus
    
    
    class EvidenceScorer(nn.Module):
        """Computes evidence scores between sampled paths and concept prototypes."""
        
        def __init__(self, sharpness: float = 5.0):
            super().__init__()
            self.sharpness = sharpness
        
        def forward(
            self,
            path_relations: torch.Tensor,
            path_times: torch.Tensor,
            path_features: torch.Tensor,
            proto_relations: torch.Tensor,
            proto_times: torch.Tensor,
            proto_gamma_t: torch.Tensor,
            proto_gamma_f: torch.Tensor,
            proto_features: torch.Tensor,
            proto_taus: torch.Tensor
        ) -> torch.Tensor:
            B, N, L, R = path_relations.shape
            C = proto_relations.shape[0]
            
            # Expand for broadcasting
            path_rel = path_relations.unsqueeze(2)
            path_t = path_times.unsqueeze(2)
            path_f = path_features.unsqueeze(2)
            
            proto_rel = proto_relations.unsqueeze(0).unsqueeze(0)
            proto_t = proto_times.unsqueeze(0).unsqueeze(0)
            proto_gt = proto_gamma_t.unsqueeze(0).unsqueeze(0)
            proto_f = proto_features.unsqueeze(0).unsqueeze(0)
            
            # Similarities
            rel_sim = (path_rel * proto_rel).sum(dim=-1)
            
            # Handle missing times
            valid_time = (path_t < 1e6) & (proto_t < 1e6)
            time_diff = torch.where(valid_time, (path_t - proto_t).abs(), torch.zeros_like(path_t))
            time_sim = torch.exp(-time_diff / (proto_gt + 1e-6))
            time_sim = torch.where(valid_time, time_sim, torch.ones_like(time_sim))
            
            # Feature similarity
            valid_feat = ~torch.isinf(path_f).any(dim=-1, keepdim=True).squeeze(-1)
            path_f_safe = torch.where(torch.isinf(path_f), torch.zeros_like(path_f), path_f)
            path_f_norm = F.normalize(path_f_safe, dim=-1)
            proto_f_norm = F.normalize(proto_f, dim=-1)
            feat_sim = (path_f_norm * proto_f_norm).sum(dim=-1)
            feat_sim = (feat_sim + 1) / 2
            feat_sim = torch.where(valid_feat, feat_sim, torch.ones_like(feat_sim))
            
            # Combine
            total_sim = rel_sim * time_sim * feat_sim
            path_scores = total_sim.prod(dim=-1)
            
            # Aggregate over paths
            concept_evidence = torch.logsumexp(
                self.sharpness * path_scores, dim=1
            ) / self.sharpness
            
            # Hill function
            tau = proto_taus.squeeze(-1).unsqueeze(0)
            activations = concept_evidence / (concept_evidence + tau + 1e-8)
            
            return activations
    
    
    class LogicHead(nn.Module):
        """Neuro-symbolic logic layer for combining concept activations."""
        
        def __init__(self, num_concepts: int, num_classes: int, num_conjuncts: int = 4):
            super().__init__()
            self.conjunction_weights = nn.Parameter(torch.randn(num_conjuncts, num_concepts) * 0.1)
            self.disjunction_weights = nn.Parameter(torch.randn(num_classes, num_conjuncts) * 0.1)
        
        def forward(self, concept_activations: torch.Tensor, temperature: float = 1.0):
            conj_attn = F.softmax(self.conjunction_weights / temperature, dim=-1)
            log_acts = torch.log(concept_activations + 1e-8)
            log_conj = torch.matmul(log_acts, conj_attn.T)
            conjunctions = torch.exp(log_conj)
            
            disj_attn = F.softmax(self.disjunction_weights / temperature, dim=-1)
            logits = torch.matmul(conjunctions, disj_attn.T)
            
            return logits
    
    
    class PredictionHead(nn.Module):
        """Full prediction head: ConceptDecoder + EvidenceScorer + LogicHead"""
        
        def __init__(
            self,
            num_concepts: int,
            concept_dim: int,
            feature_dim: int,
            max_hops: int,
            num_types: int,
            num_classes: int,
            schema_adj: torch.Tensor
        ):
            super().__init__()
            
            self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.1)
            
            self.decoder = ConceptDecoder(
                concept_dim=concept_dim,
                feature_dim=feature_dim,
                max_hops=max_hops,
                num_types=num_types,
                schema_adj=schema_adj
            )
            
            self.scorer = EvidenceScorer(sharpness=5.0)
            self.logic = LogicHead(num_concepts, num_classes, max(4, num_concepts // 2))
        
        def forward(self, relations, times, features, temperature=1.0):
            proto_rel, proto_t, proto_gt, proto_gf, proto_f, proto_tau = \
                self.decoder(self.concept_embeddings)
            
            activations = self.scorer(
                relations, times, features,
                proto_rel, proto_t, proto_gt, proto_gf, proto_f, proto_tau
            )
            
            logits = self.logic(activations, temperature)
            
            return logits, activations, {
                'relations': proto_rel, 'times': proto_t,
                'gamma_times': proto_gt, 'gamma_feats': proto_gf,
                'features': proto_f, 'taus': proto_tau
            }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, optimizer, device, temperature=1.0):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for batch, labels in dataloader:
        relations = batch['relations'].to(device)
        times = batch['times'].to(device)
        features = batch['features'].to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _, _ = model(relations, times, features, temperature)
        loss = F.cross_entropy(logits, labels.long())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.shape[0]
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += labels.shape[0]
    
    return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}


def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    with torch.no_grad():
        for batch, labels in dataloader:
            relations = batch['relations'].to(device)
            times = batch['times'].to(device)
            features = batch['features'].to(device)
            labels = labels.to(device)
            
            logits, _, _ = model(relations, times, features)
            loss = F.cross_entropy(logits, labels.long())
            
            total_loss += loss.item() * labels.shape[0]
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += labels.shape[0]
    
    return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}


def interpret_concepts(model, schema):
    model.eval()
    with torch.no_grad():
        proto_rel, proto_t, _, _, _, proto_tau = model.decoder(model.concept_embeddings)
    
    node_types = schema.node_types + [NULL_TOKEN]
    
    print("\n" + "=" * 60)
    print("Learned Concept Prototypes")
    print("=" * 60)
    
    for c in range(proto_rel.shape[0]):
        rel_probs = proto_rel[c].cpu().numpy()
        path_types = []
        for pos in range(rel_probs.shape[0]):
            top_idx = rel_probs[pos].argmax()
            type_name = node_types[top_idx] if top_idx < len(node_types) else "?"
            path_types.append(f"{type_name}({rel_probs[pos, top_idx]:.2f})")
        
        print(f"\nConcept {c}: {' → '.join(path_types)}")
        print(f"  Tau: {proto_tau[c].item():.3f}")


# =============================================================================
# Main
# =============================================================================

def train_with_mock_data(args):
    from test_mock_relf1 import create_mock_f1_schema, MockHeteroData
    
    print("\n" + "=" * 60)
    print("Training CBM with Mock rel-f1 Data")
    print("=" * 60)
    
    config = CBMConfig(
        max_hops=args.max_hops,
        n_samples_per_schema=args.samples_per_schema,
        max_paths_per_seed=args.max_paths,
        feature_dim=args.feature_dim,
        cache_dir=args.cache_dir,
        precompute=False
    )
    
    schema = create_mock_f1_schema()
    mock_data = MockHeteroData(num_drivers=args.num_drivers, num_races=args.num_races)
    
    print(f"\nSchema root: {schema.root_type}")
    metapath_schemas = enumerate_metapath_schemas(schema, args.max_hops)
    print(f"Meta-path schemas: {len(metapath_schemas)}")
    
    train_ds, val_ds, test_ds = create_cbm_datasets_from_mock(mock_data, schema, config)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    if not HAS_TORCH:
        print("\nPyTorch not available - skipping model training.")
        return
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=CBMDataset.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=CBMDataset.collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=CBMDataset.collate_fn)
    
    schema_adj = torch.from_numpy(schema.reachability_mask(args.max_hops)).float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PredictionHead(
        num_concepts=args.num_concepts,
        concept_dim=args.concept_dim,
        feature_dim=args.feature_dim,
        max_hops=args.max_hops,
        num_types=len(schema.node_types) + 1,
        num_classes=2,
        schema_adj=schema_adj
    ).to(device)
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("\nTraining...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        temp = max(0.1, 1.0 - epoch / args.epochs * 0.9)
        train_metrics = train_epoch(model, train_loader, optimizer, device, temp)
        val_metrics = evaluate(model, val_loader, device)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), f"{args.cache_dir}/best_model.pt")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | temp={temp:.2f} | "
                  f"train_acc={train_metrics['accuracy']:.4f} | val_acc={val_metrics['accuracy']:.4f}")
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {test_metrics['accuracy']:.4f}")
    
    interpret_concepts(model, schema)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train CBM on relational data")
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--cache_dir', type=str, default='/tmp/cbm_train')
    parser.add_argument('--num_drivers', type=int, default=100)
    parser.add_argument('--num_races', type=int, default=50)
    parser.add_argument('--max_hops', type=int, default=2)
    parser.add_argument('--samples_per_schema', type=int, default=4)
    parser.add_argument('--max_paths', type=int, default=64)
    parser.add_argument('--feature_dim', type=int, default=64)
    parser.add_argument('--num_concepts', type=int, default=8)
    parser.add_argument('--concept_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    
    args = parser.parse_args()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mock:
        train_with_mock_data(args)
    else:
        print("Use --mock flag to test with synthetic data.")


if __name__ == "__main__":
    main()
