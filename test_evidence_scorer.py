import unittest
import torch
import math
from dataclasses import dataclass
from evidence_scoring_head import EvidenceScorer
import numpy as np
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

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)