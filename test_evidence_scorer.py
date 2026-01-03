import unittest
import torch
import math
from dataclasses import dataclass
from evidence_scoring_head import EvidenceScorer, ConceptDecoder
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

# Mock Concept Prototype Class
@dataclass
class MockPrototype:
    relations: torch.Tensor
    times: torch.Tensor
    gamma_times: torch.Tensor
    features: torch.Tensor
    gamma_features: torch.Tensor
    tau_raw: torch.Tensor

class TestEvidenceScorer(unittest.TestCase):
    def setUp(self):
        # Initialize Scorer with specific constants for easy math
        # k = ln(0.1) approx -2.3026
        # relational_sharpness = 10.0
        self.scorer = EvidenceScorer(relational_sharpness=10.0, k=0.1)
        self.k_val = math.log(0.1)

    def test_relational_similarity(self):
        print("\n--- Testing Relational Similarity ---")
        # Setup: 1 Step, 2 Relation Types
        # Prototype: Type 0 ([1, 0])
        proto = torch.tensor([[1.0, 0.0]])
        
        # Batch of 2 paths: 
        # Path A: Perfect Match ([1, 0])
        # Path B: Total Mismatch ([0, 1])
        batch = torch.tensor([
            [[1.0, 0.0]], 
            [[0.0, 1.0]]
        ])
        
        scores = self.scorer._relational_similarity_log(proto, batch)
        
        # Check Path A (Match) -> MSE=0 -> Score=0
        self.assertAlmostEqual(scores[0].item(), 0.0, places=5)
        print(f"Match Score: {scores[0].item()} (Expected: 0.0)")
        
        # Check Path B (Mismatch) -> DiffSq=[1, 1], Mean=1.0 -> Score = -10 * 1 = -10
        self.assertAlmostEqual(scores[1].item(), -10.0, places=5)
        print(f"Mismatch Score: {scores[1].item()} (Expected: -10.0)")

    def test_temporal_similarity(self):
        print("\n--- Testing Temporal Similarity ---")
        # Prototype: Time=10, Gamma=5
        proto_t = torch.tensor([10.0])
        gamma_t = torch.tensor([5.0])
        
        # Batch:
        # Path A: Time=10 (Diff=0)
        # Path B: Time=15 (Diff=5, exactly Gamma)
        batch_t = torch.tensor([
            [10.0],
            [15.0]
        ])
        
        scores = self.scorer._time_similarity_log(proto_t, gamma_t, batch_t)
        
        # Path A: Dist=0 -> Score=0
        self.assertAlmostEqual(scores[0].item(), 0.0, places=5)
        
        # Path B: Dist=Gamma -> (Dist/Gamma)^2 = 1 -> Score = k * 1
        self.assertAlmostEqual(scores[1].item(), self.k_val, places=4)
        print(f"Boundary Match Score: {scores[1].item()} (Expected: {self.k_val:.4f})")

    def test_feature_similarity(self):
        print("\n--- Testing Feature Similarity ---")
        # Prototype: [1, 0]
        # Gamma: 1.0 (Strictness)
        proto_f = torch.tensor([[1.0, 0.0]])
        gamma_f = torch.tensor([1.0])
        
        # Batch:
        # Path A: [1, 0] (Identical, Cos=1)
        # Path B: [0, 1] (Orthogonal, Cos=0)
        batch_f = torch.tensor([
            [[1.0, 0.0]],
            [[0.0, 1.0]]
        ])
        
        scores = self.scorer._feature_similarity_log(proto_f, gamma_f, batch_f)
        
        # Path A: Cos=1 -> Dist=0 -> Score=0
        self.assertAlmostEqual(scores[0].item(), 0.0, places=5)
        
        # Path B: Cos=0 -> Dist = 1 - (1/2) = 0.5
        # NormDist = 0.5 / 1.0 = 0.5
        # Sq = 0.25
        # Score = k * 0.25
        expected_score = self.k_val * 0.25
        self.assertAlmostEqual(scores[1].item(), expected_score, places=4)
        print(f"Orthogonal Score: {scores[1].item()} (Expected: {expected_score:.4f})")

    def test_full_pipeline_forward(self):
        print("\n--- Testing Full Forward Pipeline ---")
        # Scenario:
        # We have a Concept.
        # We have a User with 2 paths.
        # Path 1 is PERFECT (Score should be 0 in log space, mass contribution = 1).
        # Path 2 is TERRIBLE (Score should be very negative, mass contribution approx 0).
        # Total Mass M approx 1.
        # Tau is set to 1.0 (raw=inverse_softplus(1)).
        # Expected Logit: ln(M) - ln(tau) = ln(1) - ln(1) = 0.
        
        # 1. Setup Prototype
        # To get tau=1.0 from softplus(raw), raw approx 0.55 (since softplus(0.55)~=1.0)
        # Actually let's just set raw large so softplus is linear, or use inverse.
        # Let's set raw=100, then tau approx 100. Then expected logit is ln(1) - ln(100) = -4.6
        # Let's try to get tau=1.0. softplus(x)=1 => ln(1+e^x)=1 => 1+e^x=e => e^x=e-1 => x=ln(e-1) approx 0.541
        tau_raw_val = math.log(math.e - 1)
        
        proto = MockPrototype(
            relations=torch.tensor([[1.0, 0.0]]),
            times=torch.tensor([10.0]),
            gamma_times=torch.tensor([5.0]),
            features=torch.tensor([[1.0, 0.0]]),
            gamma_features=torch.tensor([1.0]),
            tau_raw=torch.tensor(tau_raw_val)
        )
        
        # 2. Setup Batch Data (Dict)
        batch_data = {
            'relations': torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), # Path 1 Match, Path 2 Fail
            'times':     torch.tensor([[10.0],       [100.0]]),      # Path 1 Match, Path 2 Fail
            'features':  torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]]) # Path 1 Match, Path 2 Opposite
        }
        
        # 3. Run Forward
        logit = self.scorer(proto, batch_data)
        
        # 4. Verify
        # Path 1 Score: Rel=0 + Time=0 + Feat=0 = 0
        # Path 2 Score: Rel=-10 + Time=Huge + Feat=Huge = Very Negative
        # LogSumExp(0, -Large) approx 0.
        # Mass M approx e^0 = 1.
        # Tau = 1.
        # Logit = ln(1) - ln(1) = 0.
        
        print(f"Final Logit: {logit.item()}")
        print(f"final Score Activation: {torch.sigmoid(torch.tensor(logit.item())).item()}")
        self.assertAlmostEqual(logit.item(), 0.0, delta=0.01)


def print_comparison(name, target, prediction, limit=3):
    """
    Prints target vs prediction side-by-side with error.
    limit: max number of rows to print per tensor to keep output clean.
    """
    print(f"\n>> {name.upper()} COMPARISON:")
    
    # Flatten to list for easy iteration if dimensionality is high
    # We just grab the first batch item for simplicity
    t_flat = target[0].detach().numpy()
    p_flat = prediction[0].detach().numpy()
    
    # Calculate Residual
    error = abs(t_flat - p_flat)
    
    print(f"{'INDEX':<8} | {'TARGET':<25} | {'DECODED':<25} | {'ERROR':<10}")
    print("-" * 75)
    
    # Iterate through first 'limit' items (e.g. first 3 hops)
    for i in range(min(len(t_flat), limit)):
        # Format array as string with 4 decimals
        t_str = np.array2string(t_flat[i], formatter={'float_kind':lambda x: "%.4f" % x}, suppress_small=True)
        p_str = np.array2string(p_flat[i], formatter={'float_kind':lambda x: "%.4f" % x}, suppress_small=True)
        e_str = f"{np.mean(error[i]):.4f}" # Mean error for that row
        
        print(f"Hop {i:<4} | {t_str:<25} | {p_str:<25} | {e_str:<10}")
    print("-" * 75)
    
    
def test_basic_decoder_learning():
    print("\n" + "="*40)
    print(" STARTING DECODER OVERFITTING TEST ")
    print("="*40)
    
    # 1. Hyperparameters
    Z_DIM = 16
    EMBED_DIM = 16  # Reduced for cleaner printing
    MAX_HOPS = 3
    RELATION_COUNT = 4 
    LR = 0.01
    STEPS = 500
    
    # 2. Initialize Model & Optimizer
    decoder = ConceptDecoder(Z_DIM, EMBED_DIM, MAX_HOPS, RELATION_COUNT)
    z_latent = torch.nn.Parameter(torch.randn(1, Z_DIM))
    optimizer = optim.Adam(decoder.parameters(), lr=LR)
    
    # 3. Create "Ground Truth" Targets
    target_rel = F.softmax(torch.randn(1, MAX_HOPS, RELATION_COUNT + 1), dim=-1)
    target_time = torch.randn(1, MAX_HOPS)
    target_gamma_t = torch.rand(1, MAX_HOPS) + 0.5 
    target_gamma_f = torch.rand(1, MAX_HOPS) + 0.5
    target_feat = F.normalize(torch.randn(1, MAX_HOPS, EMBED_DIM), dim=-1)
    
    print(f"Target Feature [0,0] (First val): {target_feat[0,0,0]:.4f}")
    
    # 4. Training Loop
    import numpy as np # Import locally for the helper function
    
    for i in range(STEPS):
        optimizer.zero_grad()
        
        # Forward Pass
        P, t, g_t, g_f, mu = decoder(z_latent)
        
        # Compute Loss
        loss_rel = F.mse_loss(P, target_rel)
        loss_time = F.mse_loss(t, target_time)
        loss_gammas = F.mse_loss(g_t, target_gamma_t) + F.mse_loss(g_f, target_gamma_f)
        loss_feat = F.mse_loss(mu, target_feat)
        
        total_loss = loss_rel + loss_time + loss_gammas + loss_feat
        
        # Backward Pass
        total_loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i:03}: Loss = {total_loss.item():.6f}")

    # 5. Visual Evaluation
    print("\n" + "="*40)
    print(" FINAL RESULTS ")
    print("="*40)
    print(f"Final Total Loss: {total_loss.item():.6f}\n")
    
    # Use helper to print components
    print_comparison("Relations (Probabilities)", target_rel, P)
    
    # Combine Gammas into one print for brevity
    # Stack them: [Hop, 2] -> Col 0 is Time Gamma, Col 1 is Feat Gamma
    # Need to unsqueeze last dim to stack
    target_gammas = torch.stack((target_gamma_t, target_gamma_f), dim=2)
    pred_gammas = torch.stack((g_t, g_f), dim=2)
    print_comparison("Gammas [Time, Feat]", target_gammas, pred_gammas)
    
    # Time (Reshape for printer: [1, L] -> [1, L, 1])
    print_comparison("Time Offsets", target_time.unsqueeze(-1), t.unsqueeze(-1))
    
    # Features (First 4 dims only to fit screen)
    print_comparison("Node Features (First 4 dims)", target_feat[:, :, :4], mu[:, :, :4])

    # 6. Success Check
    if total_loss.item() < 0.05:
        print("\n✅ SUCCESS: Decoder learned the target distribution.")
    else:
        print("\n❌ FAILURE: Decoder did not converge.")
        
        
if __name__ == "__main__":
    # unittest.main(argv=[''], exit=False)

