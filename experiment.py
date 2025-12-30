# experiment with taus

# add in gradient descent

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# Existing imports from your file
from custom_dataclasses import Schema
from faithfulness_poc import Subgraph
from path_creation import *
# New import for the differentiable components
from torch_evidence_scoring import DifferentiableEvidenceScorer, prepare_batch

import torch.nn.functional as F

def _test_learned_gamma_mu(amazon_schema, ordered_node_types):
    # Build Subgraph (4 metapath types in pdf)
    amazon_has_baby = Subgraph(
        schema=amazon_schema
    )
    amazon_has_baby.create_root(time=100, feat=np.asanyarray(get_bert_embeddings(["adult"])[0]))
    relevant_paths = build_relevant_paths()
    amazon_has_baby.add_evidence(relevant_paths)

    amazon_no_baby = Subgraph(
        schema=amazon_schema
    )
    amazon_no_baby.create_root(time=300, feat=np.asanyarray(get_bert_embeddings(["child"])[0]))
    distant_paths = build_distant_paths()
    amazon_no_baby.add_evidence(distant_paths)


    buys_nappies_concept = build_graph_concept_with_BERT(ordered_node_types, learnable_gamma_mu=False)
    
    L_plus_1 = len(buys_nappies_concept.mu) # Length is L+1
    
    # 2. Initialize a VECTOR of gammas (one per hop)
    # Initializing with 1.0 for all hops
    gamma_mu_vector = nn.Parameter(torch.ones(L_plus_1, dtype=torch.float32))
    
    # Optimizer tracks the entire vector
    optimizer = optim.Adam([gamma_mu_vector], lr=0.05)
    criterion = nn.BCELoss()

    print(f"Optimizing {L_plus_1} separate gamma_mu parameters...")

    for epoch in range(100):
        optimizer.zero_grad()
        
        # Inject current vector into concept for the NumPy-based scoring
        buys_nappies_concept.gamma_mu = gamma_mu_vector.detach().numpy()
        
        # Calculate Evidence Scores
        score_has_baby = amazon_has_baby.evidence_score(buys_nappies_concept)
        score_no_baby = amazon_no_baby.evidence_score(buys_nappies_concept)
        
        # Create tensors for loss calculation
        pred = torch.stack([torch.tensor(score_has_baby, requires_grad=True), 
                            torch.tensor(score_no_baby, requires_grad=True)])
        target = torch.tensor([1.0, 0.0])
        
        loss = criterion(pred, target)
        loss.backward()

        # --- Gradient Proxy Logic ---
        # Since evidence_score is NumPy, we manually apply the gradient direction:
        # If no_baby score is > 0, we want to SHRINK gammas (reduce radius)
        # If has_baby score is < 1, we want to GROW gammas (increase radius)
        with torch.no_grad():
            error_signal = (score_has_baby - 1.0) + (score_no_baby - 0.0)
            
            # We apply the update. You can even apply different weights 
            # to specific hops here if you had hop-specific labels.
            gamma_mu_vector -= 0.1 * error_signal * torch.ones_like(gamma_mu_vector)
            
            # Ensure radii stay positive and physically meaningful
            gamma_mu_vector.clamp_(min=0.01, max=50.0)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"Scores: [Has:{score_has_baby:.3f}, No:{score_no_baby:.3f}]")

    print("\nFinal Optimized Gamma Vector (one per hop):")
    for i, g in enumerate(gamma_mu_vector):
        print(f"  Hop {i} ({buys_nappies_concept.ordered_node_types[i-1] if i>0 else 'Root'}): {g.item():.4f}")


def _test_differentiable_gamma_mu(amazon_schema, ordered_node_types):
    print("\n--- Running Differentiable Evidence Scoring Test ---")
    
    # 1. Build Subgraphs (Data generation)
    amazon_has_baby = Subgraph(schema=amazon_schema)
    amazon_has_baby.create_root(time=100, feat=np.asanyarray(get_bert_embeddings(["adult"])[0]))
    amazon_has_baby.add_evidence(build_relevant_paths())

    amazon_no_baby = Subgraph(schema=amazon_schema)
    amazon_no_baby.create_root(time=300, feat=np.asanyarray(get_bert_embeddings(["child"])[0]))
    amazon_no_baby.add_evidence(build_distant_paths())

    # 1. Prepare data
    concept = build_graph_concept_with_BERT(ordered_node_types)
    has_baby_data = prepare_batch(amazon_has_baby, concept)
    no_baby_data = prepare_batch(amazon_no_baby, concept)
    
    # 2. Initialize Scorer
    # Use a safe tau if the concept doesn't have one
    tau_val = concept.tau if (concept.tau and concept.tau > 0) else 0.5

    freeze_config = {
        "gamma_mu": False,
        "gamma_t": True,
        "tau": True 
    }
    
    scorer = DifferentiableEvidenceScorer(
        concept=concept,
        k_feat=concept.k_feat,
        k_time=concept.k_time,
        tau=tau_val,
        D=768,
        freeze_config=freeze_config
    )

    optimizer = optim.Adam(scorer.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    for epoch in range(101):
        optimizer.zero_grad()
        
        score_has = scorer(*has_baby_data)
        score_no = scorer(*no_baby_data)
        
        preds = torch.stack([score_has, score_no])
        targets = torch.tensor([1.0, 0.0])
        
        loss = criterion(preds, targets)
        
        if torch.isnan(loss):
            print(f"Bailing at epoch {epoch}: Loss became NaN!")
            break
            
        loss.backward()

        # 1. Clip gradients ONCE
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)

        optimizer.step()

        scorer.apply_constraints()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Has: {score_has.item():.3f} | No: {score_no.item():.3f}")

    print("\nFinal Optimized Gamma Vector (one per hop):")
    for i, g in enumerate(scorer.gamma_mu):
        node_name = "Root" if i == 0 else ordered_node_types[i-1]
        print(f"  Hop {i} ({node_name}): {g.item():.4f}")


def _test_differentiable_gamma_mu_tau(amazon_schema, ordered_node_types):
    print("\n--- Running Differentiable Evidence Scoring Test ---")
    
    # 1. Build Subgraphs (Data generation)
    amazon_has_baby = Subgraph(schema=amazon_schema)
    amazon_has_baby.create_root(time=100, feat=np.asanyarray(get_bert_embeddings(["adult"])[0]))
    amazon_has_baby.add_evidence(build_relevant_paths())

    amazon_no_baby = Subgraph(schema=amazon_schema)
    amazon_no_baby.create_root(time=300, feat=np.asanyarray(get_bert_embeddings(["child"])[0]))
    amazon_no_baby.add_evidence(build_distant_paths())

    # 1. Prepare data
    concept = build_graph_concept_with_BERT(ordered_node_types)
    has_baby_data = prepare_batch(amazon_has_baby, concept)
    no_baby_data = prepare_batch(amazon_no_baby, concept)
    
    # 2. Initialize Scorer
    # Use a safe tau if the concept doesn't have one
    tau_val = concept.tau if (concept.tau and concept.tau > 0) else 0.5
    concept.gamma_mu = np.asarray([1, 1, 1, 1])
    concept.gamma_t = np.asarray([1, 1, 1, 1])
    concept.k_feat = 0.1
    concept.k_time = 0.9

    freeze_config = {
        "gamma_mu": False,
        "gamma_t": False,
        "tau": False 
    }
    
    scorer = DifferentiableEvidenceScorer(
        concept=concept,
        k_feat=concept.k_feat,
        k_time=concept.k_time,
        tau=tau_val,
        D=768,
        freeze_config=freeze_config
    )

    optimizer = optim.Adam(scorer.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    for epoch in range(101):
        optimizer.zero_grad()
        
        score_has = scorer(*has_baby_data)
        score_no = scorer(*no_baby_data)
        
        # Force the targets to float32 AND ensure the stacked scores are float32
        preds = torch.stack([score_has, score_no]).float()
        targets = torch.tensor([1.0, 0.0], dtype=torch.float32)

        loss = criterion(preds, targets)            
        loss.backward()

        # update
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Has: {score_has.item():.3f} | No: {score_no.item():.3f}")

    with torch.no_grad():
        gmu = (F.softplus(scorer.gamma_mu_raw) + 0.01).cpu()
        gt  = (F.softplus(scorer.gamma_t_raw)  + 0.1).cpu()
        tau = (F.softplus(scorer.tau_raw)      + 0.01).cpu()

    print("gamma_mu (positive):", gmu)
    print("gamma_t  (positive):", gt)
    print("tau      (positive):", tau)


if __name__ == "__main__":
    # Build Schema (Customer, Review, Product)
    amazon_schema = Schema(
        root_type="users",
        transitions= {
            "users": ["orders", "subscriptions"],
            "orders": ["products"],
            "subscriptions": ["subscriptionProducts"],
            "subscriptionProducts": ["products"]
        }
        )
    
    ordered_node_types = ["orders", "subscriptions", "subscriptionProducts", "products"]
    
    # _test_differentiable_gamma_mu(amazon_schema, ordered_node_types)
    _test_differentiable_gamma_mu_tau(amazon_schema, ordered_node_types)
    # learned_feat_gamma_test(amazon_schema, ordered_node_types)

    


# decoded concepts

# sparse concepts

# full toy dataset

# lambda + concepts

# sparse concepts

# real data performance

# competitive performance


    """
    
    keep track of:
    
    - absolute time and relative time both important - resolved?

    """