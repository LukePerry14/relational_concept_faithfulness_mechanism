from relbench.base import Database, Dataset, Table
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional
from utils.custom_dataclasses import Schema, Concept, MetaPath, NULL_TOKEN
from utils.faithfulness_poc import Subgraph
from evidence_scoring_head import PredictionHead
import random
import torch.optim as optim
from concept_dashboard import run_dashboard

class SchemaDrivenGenerator:
    def __init__(self, schema, embed_dim):
        self.schema = schema
        self.D = embed_dim
        self.subgraphs: List[Subgraph] = []
        self.labels: List[int] = []

    def _generate_motif_path(self, concept, start_time):
        """
        Creates a path strictly aligned with the Concept prototype.
        """
        concept_length = len(concept)
        node_types, node_times, node_features = [], [], []
        
        # ordered_node_types defines the task-specific 'alphabet' for this concept
        type_options = concept.ordered_node_types + [NULL_TOKEN]
        
        for i in range(1, concept_length):
            # Select type based on the prototype's probability distribution P
            p_dist = concept.relational_prototype[i]
            t_name = np.random.choice(type_options, p=p_dist)
            node_types.append(t_name)

            if t_name == NULL_TOKEN:
                node_times.append(float('inf'))
                node_features.append(np.full(self.D, float('inf')))
            else:
                # Add temporal and feature jitter based on concept parameters
                node_times.append(start_time + concept.time_prototype[i] + np.random.uniform(-0.02, 0.02))
                node_features.append(concept.feature_prototype[i] + np.random.normal(0, 0.01, self.D))
        
        return MetaPath("motif_path", node_types, np.array(node_times), np.array(node_features))

    def _fill_with_noise(self, subGraph, max_nodes, max_length):
        """
        Adds random paths until the node limit is reached. 
        Follows schema transitions to ensure structural plausibility.
        """
        
        while len(subGraph.nodes) < max_nodes:
            curr_id = subGraph.root
            # Random walk length
            path_len = np.random.randint(1, max_length+1)
            
            for _ in range(path_len):
                
                curr_node_type = subGraph.nodes[curr_id].node_type
                
                next_node_options = self.schema.transitions.get(curr_node_type, [])
                
                if not next_node_options or len(subGraph.nodes) >= max_nodes:
                    break
                
                ntype = np.random.choice(next_node_options)
                ntime = subGraph.nodes[curr_id].time + np.random.uniform(0.1, 5.0)
                nfeat = np.random.randn(self.D)
                
                new_id = subGraph._new_node(ntype, ntime, nfeat)
                subGraph._add_edge(curr_id, new_id, "background_noise")
                curr_id = new_id

    def create_toy_dataset(self, positive_concept: Concept, K: int = 20, max_nodes_per_sub: int = 15):
        """
        Builds K independent subgraphs.
        Positive instances (Label 1) receive the motif.
        All instances receive background noise until max_nodes is reached.
        """
        self.subgraphs = []
        self.labels = []

        # Build K paths
        for i in range(K):
            
            # Half positive half negative
            label = 1 if i < K // 2 else 0
            
            # Insantiate Subgraphs
            sub = Subgraph(self.schema)
            
            # Root node initialized at random baseline
            
            root_time = 0# np.random.uniform(0, 100)
            
            if label == 1:
                sub.create_root(time=root_time, feat=np.asarray([1.0, 1.0]))
            else:
                sub.create_root(time=root_time, feat=np.random.randn(self.D))

            if label == 1:
                motif = self._generate_motif_path(positive_concept, root_time)
                sub.add_evidence([motif])
            
            # Fill the remainder of the subgraph with noise
            self._fill_with_noise(sub, max_nodes_per_sub, max_length=2)
            
            self.subgraphs.append(sub)
            self.labels.append(label)

        print(f"Generated {K} subgraphs (Balanced: {K//2} pos, {K//2} neg).")


def toy_amazon(params):
    toy_amazon_schema = Schema(
        root_type="customer",
        transitions={
            "customer": ["review"],
            "review": ["product"],
            "product": [],
        }
    )
    
    
    max_hops = params["max_hops"]
    node_type_count = len(params["node_types"])
    feature_embedding_dimension = params["feature_embed_dim"]


    relation_prototype = np.zeros((max_hops+1, node_type_count+1))
    relation_prototype[0, 0] = 1.0
    relation_prototype[1, 1] = 1.0
    relation_prototype[2, 2] = 1.0


    time_prototype = np.array([0.0, 10.0, 20.0])
    
    feature_prototype = np.array([
        [1.0, 1.0], # Customer vector
        [1.0, 0.0], # Review vector
        [0.0, 1.0] , # Product vector
    ])
    
    
    time_prototype = np.array([0.0, 10.0, 20.0])

    target_concept = Concept(
        name="FraudMotif",
        ordered_node_types=params["node_types"],
        relational_prototype=relation_prototype,
        time_prototype=time_prototype,
        time_gamma=np.array([float('inf'), float('inf'), float('inf')]),
        feature_prototype=feature_prototype,
        feature_gamma=np.array([0.3, 0.3, 0.3]),
        tau=0.5
    )

    dataset = SchemaDrivenGenerator(toy_amazon_schema, embed_dim=params["feature_embed_dim"])
    dataset.create_toy_dataset(target_concept, K=params["K"], max_nodes_per_sub=10)

    return toy_amazon_schema, target_concept, dataset


def collate_metapaths(subgraphs, max_hops, embed_dim, relation_count, n_samples=32):
    batch_relations, batch_times, batch_features = [], [], []

    # Ensure we are handling a list even if a single subgraph is passed
    if not isinstance(subgraphs, list):
        subgraphs = [subgraphs]

    for sub in subgraphs:
        # Sample paths from subgraph
        paths = sub.sample_paths(max_hops=max_hops, n_samples=n_samples)
        
        # Duplicate paths up until desired sample_count
        if len(paths) < n_samples:
            paths = (paths * (n_samples // len(paths) + 1))[:n_samples]

        sub_rels, sub_times, sub_feats = [], [], []
        
        for p in paths:
            # Root is index 0, Review is index 1, etc.
            rel_idx = [params["node_types"].index(t) if t != "∅" else relation_count for t in p.node_types]
            
            # This ensures the root is always 0.0, matching the prototype
            root_time = p.node_times[0]
            norm_times = torch.tensor(p.node_times) - root_time
            
            sub_rels.append(F.one_hot(torch.tensor(rel_idx), num_classes=relation_count + 1).float())
            sub_times.append(norm_times)
            sub_feats.append(torch.tensor(p.node_features))

        batch_relations.append(torch.stack(sub_rels))
        batch_times.append(torch.stack(sub_times))
        batch_features.append(torch.stack(sub_feats))

    return {
        'relations': torch.stack(batch_relations), 
        'times': torch.stack(batch_times).float(), 
        'features': torch.stack(batch_features).float() 
    }

def print_ground_truth(concept, node_types):
    """
    Summarizes the Ground Truth concept by pairing Node Types, 
    Timestamps, and Features into a single readable path.
    """
    vocab = node_types + ["∅ (STOP)"]
    
    # Extract raw data
    rel_p = concept.relational_prototype  # [L, R+1]
    time_p = concept.time_prototype       # [L+1]
    g_time_p = concept.time_gamma         # [L+1]
    feat_p = concept.feature_prototype    # [L+1, D]
    g_feat_p = concept.feature_gamma      # [L+1]
    
    # Get node types via argmax
    path_indices = np.argmax(rel_p, axis=-1)
    full_type_path = [vocab[idx] for idx in path_indices]

    print(f"\n{'='*20} GROUND TRUTH: {concept.name} {'='*20}")
    print(f"{'Step':<10} | {'Node Type':<15} | {'Time (±γ)':<15} | {'Features (±γ)':<20}")
    print(f"{'-'*85}")

    for h in range(len(full_type_path)):
        step_label = "ROOT" if h == 0 else f"HOP {h}"
        node_type  = full_type_path[h]
        time_str   = f"{time_p[h]:.1f} (±{g_time_p[h]:.1f})"
        
        # Format features: [val, val] (±gamma)
        feat_vals  = ", ".join([f"{v:+.2f}" for v in feat_p[h]])
        feat_str   = f"[{feat_vals}] (±{g_feat_p[h]:.2f})"
        
        print(f"{step_label:<10} | {node_type:<15} | {time_str:<15} | {feat_str:<20}")
    
    print(f"{'-'*85}")
    print(f"Saturation Threshold (Tau): {concept.tau:.3f}")
    print(f"{'='*85}\n")


def print_contribution_report(components_list, labels):
    """
    components_list: List of dicts from pHead forward pass
    labels: Ground truth labels for the batch
    """
    # Focus only on True Positives to see if the motif is being caught
    pos_mask = (labels == 1)
    
    print(f"\n{'-'*20} SIMILARITY BREAKDOWN (Positive Samples) {'-'*20}")
    print(f"{'Concept':<10} | {'Relational':<12} | {'Temporal':<12} | {'Feature':<12}")
    
    for c_idx, comp in enumerate(components_list["components"]):
        # comp['rel'] is [Batch, Path_Samples]
        # We want the average log-similarity across all paths in positive subgraphs
        rel_avg = np.exp(comp['rel'][pos_mask].mean().item())
        time_avg = np.exp(comp['time'][pos_mask].mean().item())
        feat_avg = np.exp(comp['feat'][pos_mask].mean().item())
        
        print(f"ID {c_idx:<7} | {rel_avg:>12.2f} | {time_avg:>12.2f} | {feat_avg:>12.2f}")
    print(f"{'-'*75}")



if __name__ == "__main__":
    params = {
        "concept_dim": 6,
        "feature_embed_dim":2,
        "K_train": 100,
        "K_test": 40,
        "node_types": ["customer", "review", "product"],
        "max_hops": 2,
        "num_concepts": 1,
        "num_epochs": 50,
        "batch_size": 10,
        "training_steps": 500,
        "relational_sharpness": 10,
        "lr": 0.005,
        "gamma_floor": 0.1,
        "min_tau": 0.1,

    }
    


    toy_amazon_schema, target_concept, train_dataset = toy_amazon({**params, "K": params["K_train"]})
    _, _, test_dataset = toy_amazon({**params, "K": params["K_test"]})  

    
    params["schema"] = toy_amazon_schema

    pHead = PredictionHead(params)


    # initialise optimiser
    optimizer = optim.Adam(pHead.parameters(), lr=0.01, weight_decay=1e-4)
    
    train_idxs = list(range(params["K_train"]))
    test_idxs = list(range(params["K_test"]))
    train_labels = torch.tensor(train_dataset.labels)
    test_labels = torch.tensor(test_dataset.labels)
    
    # train
    print("Beginning training...")
    for i in range(params["training_steps"]):

        batch_indices = random.sample(train_idxs, params["batch_size"])
        batch_subgraphs = [train_dataset.subgraphs[idx] for idx in batch_indices]
        batch_labels = train_labels[batch_indices]

        sampled_train = collate_metapaths(
            batch_subgraphs, 
            params["max_hops"], 
            params["feature_embed_dim"], 
            len(params["node_types"])
        )

        optimizer.zero_grad()

        prediction_logit, _ = pHead(sampled_train)

        # ortho_loss = pHead.concept_orthogonality_regularisation()
        loss = F.binary_cross_entropy_with_logits(prediction_logit, batch_labels.float()) #+ (1 * ortho_loss)
    
        loss.backward()
        optimizer.step()


        if i % 50 == 0:
            pHead.eval()
            with torch.no_grad():
                sampled_test = collate_metapaths(test_dataset.subgraphs, params["max_hops"], params["feature_embed_dim"], len(params["node_types"]))
                test_logit, components = pHead(sampled_test)
                
                # Metrics
                test_preds = (test_logit > 0).long()
                test_acc = (test_preds == test_labels).float().mean()
                
                train_preds = (prediction_logit > 0).long()
                train_acc = (train_preds == batch_labels).float().mean()

                print_contribution_report(components, test_labels)
                
            pHead.train()
            print(f"Step {i:03d} | Loss: {loss.item():.4f} | train Acc: {train_acc:.2f} | test Acc: {test_acc:.2f}")

    print("\nTraining Complete.")
    print_ground_truth(target_concept, params["node_types"])
    
    pHead.inspect_concepts(params["node_types"])

    # run_dashboard(pHead, params["node_types"])
    
"""
TODO:
- make visualisation better
- push relational mass towards only possible relation matrices
- implement successful sparsity
- Ensure path generation complies with tau etc.
- combat subpath matching
"""