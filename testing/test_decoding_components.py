import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evidence_scoring_head import ConceptDecoder, EvidenceScorer
from utils.custom_dataclasses import Schema, NULL_TOKEN

def run_relational_unit_test():
    print(f"{'='*20} ATOMIC TEST: RELATIONAL CONVERGENCE {'='*20}")
    
    # 1. SETUP: A Tiny Schema
    # Root -> A -> B (Target)
    # Root -> C -> D (Distractor)
    node_types = ["root", "A", "B", "C", "D"]
    schema_config = {
        "root": ["A", "C"],
        "A": ["B"],
        "C": ["B", "D"],
        "B": [], "D": []
    }
    
    # Create the Adjacency Matrix (needed for your new decoder)
    schema = Schema("root", schema_config)
    
    # HYPERPARAMETERS
    params = {
        "concept_dim": 16, "feature_embed_dim": 2, "max_hops": 2,
        "node_types": node_types, "schema": schema, "num_concepts": 1
    }
    
    # Initialize Components
    decoder = ConceptDecoder(params)
    scorer = EvidenceScorer(relational_sharpness=20) # Lower sharpness aids convergence
    
    # The Latent Concept we want to learn
    concept_z = torch.nn.Parameter(torch.randn(1, params["concept_dim"]))
    optimizer = optim.Adam([concept_z], lr=0.05) # Aggressive LR for unit test
    
    # 3. DATA GENERATION (Fixed Tensors)
    # We construct "perfect" matches for Time and Features so they contribute 0 loss.
    # The only variable is the Relation Sequence.
    
    # Target Path: Root -> A -> B
    # Indices: Root=0, A=1, B=2, C=3, D=4, STOP=5
    target_seq = torch.tensor([0, 1, 2]) 
    distractor_seq = torch.tensor([0, 3, 4]) 
    
    # Expand to One-Hot: [Batch=1, P=1, L=3, R=6]
    pos_rels = F.one_hot(target_seq, num_classes=6).float().view(1, 1, 3, 6)
    neg_rels = F.one_hot(distractor_seq, num_classes=6).float().view(1, 1, 3, 6)
    
    # Dummy Time/Feats (Perfect match placeholders)
    # We will feed the DECODED prototype back as input to ensure perfect time/feat match
    # This isolates the Relational gradient completely.

    print("Target Sequence: Root -> A -> B")
    print("Distractor:      Root -> C -> D")
    print("Training Start...\n")

    for step in range(200):
        optimizer.zero_grad()
        
        # A. Decode the Concept
        rel_p, t_p, gt_p, gf_p, mu_p, tau = decoder(concept_z)
        
        # B. Construct inputs where Time/Feat are PERFECT matches
        # This ensures S_time and S_feat are approx 1.0 (Log Sim approx 0)
        # So the only error comes from S_rel.
        
        # Positive Sample Input
        pos_sample = {
            'relations': pos_rels,
            'times': t_p.unsqueeze(0),       # [1, 1, L] - Matches prototype exactly
            'features': mu_p.unsqueeze(0)    # [1, 1, L, D] - Matches prototype exactly
        }
        
        # Negative Sample Input
        neg_sample = {
            'relations': neg_rels,
            'times': t_p.unsqueeze(0),       # Matches prototype exactly
            'features': mu_p.unsqueeze(0)    # Matches prototype exactly
        }

        # C. Construct Prototype Object
        proto_obj = type('Obj', (object,), {
            'relations': rel_p[0], 'times': t_p[0], 'gamma_times': gt_p[0],
            'features': mu_p[0], 'gamma_features': gf_p[0], 'tau': tau[0]
        })()

        # D. Calculate Evidence
        # We want Positive Evidence -> High, Negative Evidence -> Low
        pos_logit, pos_comps = scorer(proto_obj, pos_sample)
        neg_logit, neg_comps = scorer(proto_obj, neg_sample)
        
        # E. Loss Function: Maximize Gap
        # We want pos_logit > 0 and neg_logit < 0
        loss = F.binary_cross_entropy_with_logits(
            torch.stack([pos_logit, neg_logit]).squeeze(), 
            torch.tensor([1.0, 0.0])
        )
        
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            # Check Probability Mass on the Target Nodes
            # Hop 1 Target is Index 1 (A), Hop 2 Target is Index 2 (B)
            p_A = rel_p[0, 1, 1].item()
            p_B = rel_p[0, 2, 2].item()
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | P(A): {p_A:.2f} | P(B): {p_B:.2f}")
            
            if p_A > 0.9 and p_B > 0.9:
                print("\nSUCCESS: Model converged on the hidden motif!")
                break
                
    # F. Final Inspection
    print("\nFinal Learned Distribution:")
    vocab = node_types + ["STOP"]
    for h in range(3):
        probs = rel_p[0, h]
        best_idx = torch.argmax(probs).item()
        print(f"Hop {h}: Best guess = {vocab[best_idx]} ({probs[best_idx]:.4f})")

def run_feature_unit_test():
    print(f"{'='*20} TRUE ATOMIC TEST: DUAL-FEATURE CONVERGENCE {'='*20}")
    
    # 1. SETUP: Schema (Structure doesn't matter as we will match it perfectly)
    node_types = ["root", "target"]
    schema_config = { "root": ["target"], "target": [] }
    schema = Schema("root", schema_config)
    
    params = {
        "concept_dim": 16, "feature_embed_dim": 2, "max_hops": 1,
        "node_types": node_types, "schema": schema, "num_concepts": 1
    }
    
    decoder = ConceptDecoder(params)
    scorer = EvidenceScorer()
    
    # Learnable Concept (external from prediction head)
    concept_z = torch.nn.Parameter(torch.randn(1, params["concept_dim"]))
    optimizer = optim.Adam([concept_z], lr=0.1) # Aggressive LR for demonstration
    

    # Ground truth features    
    target_root = torch.tensor([1.0, 1.0])
    target_root = F.normalize(target_root, p=2, dim=0)
    
    target_hop1 = torch.tensor([1.0, -1.0])
    target_hop1 = F.normalize(target_hop1, p=2, dim=0)
    
    # Distractors (Opposites)
    dist_root = -target_root
    dist_hop1 = -target_hop1

    print(f"Target Root: {target_root.numpy()}")
    print(f"Target Hop1: {target_hop1.numpy()}")
    print("-" * 50)

    for step in range(10000):
        optimizer.zero_grad()
        
        # A. Decode CURRENT Concept State
        rel_p, t_p, gt_p, gf_p, mu_p, tau = decoder(concept_z)
        
        # B. Construct "Perfect Context" Samples
        # We construct inputs that match the DECODED relation/time exactly.
        # This neutralizes S_rel and S_time (makes them ~1.0), isolating S_feat.
        
        # 1. Relations: Use the Argmax of the current decoded relation
        # This mimics a sampled path that perfectly aligns with the model's current structural preference
        path_indices = torch.argmax(rel_p[0], dim=-1) # [L]
        num_classes = len(params["node_types"]) + 1
        # One-hot encode for the scorer: [1, 1, L, R]
        perfect_rels = F.one_hot(path_indices, num_classes=num_classes).float().view(1, 1, 2, num_classes)
        
        # 2. Times: Use the Decoded Time exactly
        # [1, 1, L]
        perfect_times = t_p.unsqueeze(0) 
        
        # 3. Features: This is the ONLY variable
        # Positive: [Target_Root, Target_Hop1]
        pos_feats_data = torch.stack([target_root, target_hop1]).view(1, 1, 2, 2)
        
        # Negative: [Distractor_Root, Distractor_Hop1]
        neg_feats_data = torch.stack([dist_root, dist_hop1]).view(1, 1, 2, 2)
        
        # Assemble Batches
        pos_sample = { 'relations': perfect_rels, 'times': perfect_times, 'features': pos_feats_data }
        neg_sample = { 'relations': perfect_rels, 'times': perfect_times, 'features': neg_feats_data }

        # C. Score against the Decoded Prototype
        proto_obj = type('Obj', (object,), {
            'relations': rel_p[0], 'times': t_p[0], 'gamma_times': gt_p[0],
            'features': mu_p[0], 'gamma_features': gf_p[0], 'tau': tau[0]
        })()

        pos_logit, _ = scorer(proto_obj, pos_sample)
        neg_logit, _ = scorer(proto_obj, neg_sample)
        
        # task loss
        task_loss = F.binary_cross_entropy_with_logits(
            torch.stack([pos_logit, neg_logit]).squeeze(), 
            torch.tensor([1.0, 0.0])
        )

        gamma_reg = 0.5 * gf_p.mean()

        loss = task_loss + gamma_reg
        
        loss.backward()
        optimizer.step()
        
        # Monitoring
        if step % 100 == 0:
            current_root = mu_p[0, 0].detach()
            current_hop1 = mu_p[0, 1].detach()
            
            dist_r = torch.norm(current_root - target_root).item()
            dist_h1 = torch.norm(current_hop1 - target_hop1).item()

            gamma_r = gf_p[0, 0].item()
            gamma_h1 = gf_p[0, 1].item()
            
            print(f"Step {step:03d} | Loss: {task_loss.item():.4f} (+{gamma_reg.item():.4f}) | "
                  f"Root: d={dist_r:.4f}, g={gamma_r:.4f} | "
                  f"Hop1: d={dist_h1:.4f}, g={gamma_h1:.4f}")
            
            if dist_r < 0.01 and dist_h1 < 0.01:
                print("\nSUCCESS: Model converged on BOTH feature clusters!")
                break

    # Final Report
    print(f"\n{'='*20} RESULTS {'='*20}")
    print(f"Target Root: {target_root.numpy()} | Learned: {mu_p[0, 0].detach().numpy()}")
    print(f"Target Hop1: {target_hop1.numpy()} | Learned: {mu_p[0, 1].detach().numpy()}")


def run_temporal_unit_test():
    print(f"{'='*20} TRUE ATOMIC TEST: TEMPORAL CONVERGENCE {'='*20}")
    
    # 1. SETUP: 3-Node Path
    node_types = ["customer", "review", "product"]
    schema_config = { "customer": ["review"], "review": ["product"], "product": [] }
    schema = Schema("customer", schema_config)
    
    params = {
        "concept_dim": 16, "feature_embed_dim": 2, "max_hops": 2,
        "node_types": node_types, "schema": schema, "num_concepts": 1,
        "gamma_floor": 0.1,
        "min_tau": 0.1
    }
    
    decoder = ConceptDecoder(params)
    scorer = EvidenceScorer()
    
    # Learnable Concept
    concept_z = torch.nn.Parameter(torch.randn(1, params["concept_dim"]))
    optimizer = optim.Adam([concept_z], lr=0.1)
    
    # 2. DEFINE GROUND TRUTH TARGETS
    # We expect these absolute times relative to root
    target_times = torch.tensor([0.0, 10.0, 2000.0])
    
    # Distractors: Significant temporal shifts
    dist_times = torch.tensor([0.0, 50.0, 100.0])

    print(f"Target Times: {target_times.numpy()}")
    print("-" * 50)

    for step in range(10000):
        optimizer.zero_grad()
        
        # A. Decode
        rel_p, t_p, gt_p, gf_p, mu_p, tau = decoder(concept_z)
        
        # B. Construct "Perfect Context" Samples
        # Use decoded relations and features so they match prototype exactly (S=1.0)
        path_indices = torch.argmax(rel_p[0], dim=-1)
        num_classes = len(params["node_types"]) + 1
        perfect_rels = F.one_hot(path_indices, num_classes=num_classes).float().view(1, 1, 3, num_classes)
        perfect_feats = mu_p.unsqueeze(0)
        
        # C. Construct Temporal Samples
        # Positive: Matches target [0, 10, 20]
        pos_times = target_times.view(1, 1, 3)
        neg_times = dist_times.view(1, 1, 3)
        
        pos_sample = { 'relations': perfect_rels, 'times': pos_times, 'features': perfect_feats }
        neg_sample = { 'relations': perfect_rels, 'times': neg_times, 'features': perfect_feats }

        # Scores
        proto_obj = type('Obj', (object,), {
            'relations': rel_p[0], 'times': t_p[0], 'gamma_times': gt_p[0],
            'features': mu_p[0], 'gamma_features': gf_p[0], 'tau': tau[0]
        })()

        pos_logit, _ = scorer(proto_obj, pos_sample)
        neg_logit, _ = scorer(proto_obj, neg_sample)
        
        # Task Loss
        task_loss = F.binary_cross_entropy_with_logits(
            torch.stack([pos_logit, neg_logit]).squeeze(), 
            torch.tensor([1.0, 0.0])
        )

        excess_gamma = F.relu(gt_p - params["gamma_floor"]).mean()

        # gamma_reg = F.softplus(torch.log(excess_gamma / task_loss))
        # gamma_reg = 0
        
        gamma_reg = 0.1 * gt_p.mean()
        # gamma_reg = F.relu(F.softplus(torch.log(gt_p).mean()))
        
        # ceiling = task_loss.detach() + 0.1
        # gamma_reg = raw_gamma_penalty * (ceiling / (ceiling + raw_gamma_penalty))

        # excess_gamma = F.relu(gt_p - params["gamma_floor"]).mean()
        # reg_weight = torch.exp(-task_loss.detach())

        # gamma_reg = (reg_weight * excess_gamma).mean()

        # Temporal Causality loss (ignore for now)
        causal_reg = 0 # 1.0 * causality_regularization(t_p)
        # gamma_reg = F.softplus()
        loss = task_loss + gamma_reg + causal_reg
        loss.backward()

        torch.nn.utils.clip_grad_norm_([concept_z], max_norm=1.0)

        optimizer.step()
        
        if step % 100 == 0:
            learned_t = t_p[0].detach().numpy()
            learned_g = gt_p[0].detach().numpy()
            
            # Root is fixed at 0, so we check Hop 1 and Hop 2
            diff_h0 = abs(learned_t[0] - target_times[0].item())
            diff_h1 = abs(learned_t[1] - target_times[1].item())
            diff_h2 = abs(learned_t[2] - target_times[2].item())
            
            print(f"Step {step:03d} | Task_loss: {task_loss.item():.4f}, Gamma_loss = {gamma_reg:.4f}  | "
                    f"root: t={learned_t[0]:.1f}(diff:{diff_h0:.2f}, g:{learned_g[0]:.2f}) | "
                  f"H1: t={learned_t[1]:.1f}(diff:{diff_h1:.2f}, g:{learned_g[1]:.2f}) | "
                  f"H2: t={learned_t[2]:.1f}(diff:{diff_h2:.2f}, g:{learned_g[2]:.2f})")
            
            if diff_h1 < 0.1 and diff_h2 < 0.1:
                print("\nSUCCESS: Model converged on temporal motif!")
                break

    print(f"\nResults | Target: {target_times.numpy()} | Learned: {t_p[0].detach().numpy()} | Learned Gamma: {gt_p[0].detach().numpy()}")

if __name__ == "__main__":
    # run_relational_unit_test()
    # run_feature_unit_test()
    run_temporal_unit_test()