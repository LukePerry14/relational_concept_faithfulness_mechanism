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

class SchemaDrivenGenerator:
    def __init__(self, schema, embed_dim):
        self.schema = schema
        self.D = embed_dim
        self.subgraphs: List[Subgraph] = []
        self.labels: List[int] = []

    def _generate_motif_path(self, concept: Concept, start_time: float) -> MetaPath:
        """
        Creates a path strictly aligned with the Concept prototype.
        """
        L = len(concept)
        node_types, node_times, node_features = [], [], []
        
        # ordered_node_types defines the task-specific 'alphabet' for this concept
        type_options = concept.ordered_node_types + [NULL_TOKEN]
        
        for i in range(L):
            # Select type based on the prototype's probability distribution P
            p_dist = concept.relational_prototype[i]
            t_name = np.random.choice(type_options, p=p_dist)
            node_types.append(t_name)

            if t_name == NULL_TOKEN:
                node_times.append(float('inf'))
                node_features.append(np.full(self.D, float('inf')))
            else:
                # Add temporal and feature jitter based on concept parameters
                node_times.append(start_time + concept.time_prototype[i+1] + np.random.uniform(-0.02, 0.02))
                node_features.append(concept.feature_prototype[i+1] + np.random.normal(0, 0.01, self.D))
        
        return MetaPath("motif_path", node_types, np.array(node_times), np.array(node_features))

    def _fill_with_noise(self, sub: Subgraph, max_nodes: int):
        """
        Adds random paths until the node limit is reached. 
        Follows schema transitions to ensure structural plausibility.
        """
        while len(sub.nodes) < max_nodes:
            curr_id = sub.root
            # Random walk length
            path_len = np.random.randint(1, 4)
            
            for _ in range(path_len):
                curr_type = sub.nodes[curr_id].node_type
                options = self.schema.transitions.get(curr_type, [])
                if not options or len(sub.nodes) >= max_nodes:
                    break
                
                ntype = np.random.choice(options)
                ntime = sub.nodes[curr_id].time + np.random.uniform(0.1, 5.0)
                nfeat = np.random.randn(self.D)
                
                new_id = sub._new_node(ntype, ntime, nfeat)
                sub._add_edge(curr_id, new_id, "background_noise")
                curr_id = new_id

    def create_toy_dataset(self, 
                          positive_concept: Concept, 
                          K: int = 20, 
                          max_nodes_per_sub: int = 15):
        """
        Builds K independent subgraphs.
        Positive instances (Label 1) receive the motif.
        All instances receive background noise until max_nodes is reached.
        """
        self.subgraphs = []
        self.labels = []

        for i in range(K):
            label = 1 if i < K // 2 else 0
            sub = Subgraph(self.schema)
            
            # Root node initialized at random baseline
            root_time = np.random.uniform(0, 100)
            sub.create_root(time=root_time, feat=np.random.randn(self.D))

            if label == 1:
                # Inject ground truth motif
                motif = self._generate_motif_path(positive_concept, root_time)
                sub.add_evidence([motif])
            
            # Fill the remainder of the subgraph with schema-compliant noise
            self._fill_with_noise(sub, max_nodes_per_sub)
            
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

    relation_prototype = toy_amazon_schema.reachability_mask(params["max_hops"], params["node_types"], zeroed=True)
    relation_prototype[0, 1] = 1.0
    relation_prototype[1, 2] = 1.0


    feature_prototype=np.random.randn(params["max_hops"]+1, params["feature_embed_dim"])

    target_concept = Concept(
        name="FraudMotif",
        ordered_node_types=params["node_types"],
        relational_prototype=relation_prototype,
        time_prototype=np.array([0, 10, 20, 30]),
        time_gamma=np.array([1, 1, 1, 1]),
        feature_prototype=feature_prototype,
        feature_gamma=np.array([1, 1, 1, 1]),
        tau=0.5
    )

    dataset = SchemaDrivenGenerator(toy_amazon_schema, embed_dim=params["feature_embed_dim"])
    dataset.create_toy_dataset(target_concept, K=params["K"], max_nodes_per_sub=100)

    return toy_amazon_schema, target_concept, dataset


def collate_metapaths(subgraphs, max_hops, embed_dim, relation_count, n_samples=32):
    """
    Groups paths by subgraph to maintain the [Batch, Path_Count] structure.
    """
    batch_relations = []
    batch_times = []
    batch_features = []

    for sub in subgraphs:
        # Sample n_samples for EACH subgraph
        paths = sub.sample_paths(max_hops=max_hops, n_samples=n_samples)
        
        # If a subgraph has fewer paths than n_samples, we need to pad or wrap
        # For simplicity in this toy, we'll just ensure we have exactly n_samples
        if len(paths) < n_samples:
            paths = (paths * (n_samples // len(paths) + 1))[:n_samples]

        sub_rels, sub_times, sub_feats = [], [], []
        
        for p in paths:
            # Type to Index mapping
            rel_idx = [params["node_types"].index(t) if t != "∅" else len(params["node_types"]) 
                       for t in p.node_types[1:]]
            
            sub_rels.append(F.one_hot(torch.tensor(rel_idx), num_classes=relation_count + 1).float())
            sub_times.append(torch.tensor(p.node_times[1:]))
            sub_feats.append(torch.tensor(p.node_features[1:]))

        batch_relations.append(torch.stack(sub_rels))
        batch_times.append(torch.stack(sub_times))
        batch_features.append(torch.stack(sub_feats))

    # Return tensors shaped [Batch, N_samples, L, ...]
    return {
        'relations': torch.stack(batch_relations), 
        'times': torch.stack(batch_times).float(), 
        'features': torch.stack(batch_features).float() 
    }


if __name__ == "__main__":
    params = {
        "concept_dim": 8,
        "feature_embed_dim":3,
        "K": 100,
        "node_types": ["customer", "review", "product"],
        "max_hops": 2,
        "num_concepts": 3,
        "num_epochs": 20, # single pass over data
        "batch_size": 5,
        "training_steps": 500,
        "lr": 0.01
    }

    toy_amazon_schema, target_concept, dataset = toy_amazon(params)

    pHead = PredictionHead(params)

    
    idxs = list(range(params["K"]))
    true_labels = torch.tensor(dataset.labels)

    # initialise optimiser
    optimizer = optim.Adam(pHead.parameters(), lr=0.01)

    # train
    for i in range(params["training_steps"]):

        batch_indices = random.sample(idxs, params["batch_size"])
        batch_subgraphs = [dataset.subgraphs[idx] for idx in batch_indices]
        batch_labels = true_labels[batch_indices]

        sampled_data = collate_metapaths(
            batch_subgraphs, 
            params["max_hops"], 
            params["feature_embed_dim"], 
            len(params["node_types"])
        )

        optimizer.zero_grad()

        prediction_logits = pHead(sampled_data)

        loss = F.cross_entropy(prediction_logits, batch_labels)
        loss.backward()
        optimizer.step()


        if i % 50 == 0:
            # Track progress
            with torch.no_grad():
                preds = torch.argmax(prediction_logits, dim=1)
                accuracy = (preds == batch_labels).float().mean()
            print(f"Step {i:03d} | Loss: {loss.item():.4f} | Batch Acc: {accuracy:.2f}")

    print("\nTraining Complete.")