import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def prepare_batch(subgraph, concept, n_samples=128):
    paths = subgraph.sample_paths(max_hops=concept.L(), n_samples=n_samples)
    type_idx = concept.type_index()
    L = concept.L()
    K = len(concept.ordered_node_types) + 1
    
    m_list, t_list, mu_list = [], [], []
    
    for p in paths:
        M = np.zeros((L, K))
        valid = True
        for hop in range(L):
            t_type = p.node_types[hop+1]
            j = type_idx.get(t_type)
            if j is None: 
                valid = False; break
            M[hop, j] = 1.0
        
        if valid:
            m_list.append(M)
            t_list.append(p.node_times)
            mu_list.append(p.node_features)
            
    return (torch.tensor(np.array(m_list), dtype=torch.float32),
            torch.tensor(np.array(t_list), dtype=torch.float32),
            torch.tensor(np.array(mu_list), dtype=torch.float32))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DifferentiableEvidenceScorer(nn.Module):
    def __init__(self, concept, k_feat, k_time, tau, D, freeze_config=None):
        """
        Args:
            concept: The initial concept data.
            k_feat: Threshold constant for feature similarity (k in paper)
            k_time: Threshold constant for temporal similarity (k in paper)
            tau: Half-saturation mass for Hill function
            D: Dimensionality of feature embeddings
            freeze_config: Dict specifying what to freeze. 
                           Example: {"gamma_mu": True, "gamma_t": False, "tau": True}
        """
        super().__init__()
        
        if freeze_config is None:
            freeze_config = {}

        # Use raw parameters with softplus activation for positive constraints
        # This prevents gradient issues with clamps
        self.gamma_mu_raw = nn.Parameter(
            torch.log(torch.exp(torch.tensor(concept.gamma_mu, dtype=torch.float32)) - 1)
        )
        self.gamma_t_raw = nn.Parameter(
            torch.log(torch.exp(torch.tensor(concept.gamma_t, dtype=torch.float32)) - 1)
        )
        self.tau_raw = nn.Parameter(
            torch.log(torch.exp(torch.tensor(tau, dtype=torch.float32)) - 1)
        )

        # Apply freezing
        if freeze_config.get("gamma_mu", False):
            self.gamma_mu_raw.requires_grad = False
            print("--- Parameter 'gamma_mu' frozen ---")
            
        if freeze_config.get("gamma_t", False):
            self.gamma_t_raw.requires_grad = False
            print("--- Parameter 'gamma_t' frozen ---")
            
        if freeze_config.get("tau", False):
            self.tau_raw.requires_grad = False
            print("--- Parameter 'tau' frozen ---")

        # Static Buffers
        self.register_buffer("P", torch.tensor(concept.P, dtype=torch.float32))
        self.register_buffer("mu", torch.tensor(concept.mu, dtype=torch.float32))
        self.register_buffer("t", torch.tensor(concept.t, dtype=torch.float32))
        
        # According to paper: k should be 0.1 for threshold-based similarity
        # Using ln(0.1) for RBF kernel
        self.register_buffer("ln_k_feat", torch.tensor(np.log(k_feat), dtype=torch.float32))
        self.register_buffer("ln_k_time", torch.tensor(np.log(k_time), dtype=torch.float32))
        
        self.EPS = 1e-8

    def get_safe_params(self):
        """Get positive-constrained parameters using softplus"""
        gamma_mu = F.softplus(self.gamma_mu_raw) + 0.01
        gamma_t = F.softplus(self.gamma_t_raw) + 0.1
        tau = F.softplus(self.tau_raw) + 0.01
        return gamma_mu, gamma_t, tau

    def _relational_similarity(self, M_batch):
        """
        Calculate relational similarity between sampled paths and prototype.
        M_batch: [N, L, K] - sampled meta-paths
        
        Simple MSE-based similarity without masking
        """
        # Element-wise squared difference
        diff_sq = (self.P.unsqueeze(0) - M_batch) ** 2  # [N, L, K]
        
        # Mean squared error
        mse = torch.mean(diff_sq, dim=(1, 2))  # [N]
        
        # Convert to similarity score (1 - normalized error)
        similarity = 1.0 - mse
        return torch.clamp(similarity, 0.0, 1.0)

    def _time_similarity(self, t_batch, gamma_t):
        """
        Calculate temporal similarity using RBF kernel.
        
        Formula: exp(ln(k) * Σ[(t_i - t̂_i)² / γ_i²])
        where k=0.1 as threshold value
        """
        # Handle infinite gammas (irrelevant time positions)
        mask = torch.isfinite(gamma_t) & torch.isfinite(self.t)
        
        # Compute squared differences only where relevant
        diff = torch.where(mask, self.t - t_batch, torch.zeros_like(t_batch))
        
        # Normalize by gamma squared (with epsilon for stability)
        normalized_diff_sq = (diff ** 2) / ((gamma_t ** 2) + self.EPS)
        
        # Apply mask and sum across time dimensions
        total_diff = torch.sum(normalized_diff_sq * mask.float(), dim=1)  # [N]
        
        # Apply RBF kernel: exp(ln(k) * total_diff)
        # This equals: k^(total_diff)
        similarity = torch.exp(self.ln_k_time * total_diff)
        
        return torch.clamp(similarity, self.EPS, 1.0)

    def _feature_similarity(self, mu_batch, gamma_mu):
        """
        Calculate feature similarity using cosine similarity with RBF-style scaling.
        
        Uses cosine similarity for stability, then scales by gamma to control relevance window.
        """
        # Handle NaN/Inf in feature embeddings
        valid_mask = torch.isfinite(mu_batch).all(dim=-1) & torch.isfinite(gamma_mu)
        
        # Clean embeddings
        mu_batch_clean = torch.where(
            torch.isfinite(mu_batch), 
            mu_batch, 
            torch.zeros_like(mu_batch)
        )
        
        # Normalize embeddings for cosine similarity
        mu_proto_norm = F.normalize(self.mu, p=2, dim=-1, eps=self.EPS)  # [L, D]
        mu_batch_norm = F.normalize(mu_batch_clean, p=2, dim=-1, eps=self.EPS)  # [N, L, D]
        
        # Compute cosine similarity
        cos_sim = torch.sum(mu_batch_norm * mu_proto_norm.unsqueeze(0), dim=-1)  # [N, L]
        
        # Convert cosine similarity to distance: (1 - cos_sim) / 2 maps [-1, 1] -> [0, 1]
        # This gives 0 for identical, 1 for opposite
        cos_dist = (1.0 - cos_sim) / 2.0
        
        # Scale by gamma (wider gamma = more tolerance)
        normalized_dist = cos_dist / (gamma_mu.unsqueeze(0) + self.EPS)
        
        # Apply mask and sum
        total_dist = torch.sum((normalized_dist ** 2) * valid_mask.float(), dim=1)  # [N]
        
        # Clip to prevent overflow in exp
        total_dist = torch.clamp(total_dist, max=50.0)
        
        # Apply RBF-style kernel
        similarity = torch.exp(self.ln_k_feat * total_dist)
        
        return torch.clamp(similarity, self.EPS, 1.0)

    def forward(self, M_batch, t_batch, mu_batch):
        """
        Compute evidence score for a batch of meta-paths.
        
        Returns: Evidence score E ∈ [0,1] using Hill saturation
        """
        # Get positive-constrained parameters
        gamma_mu, gamma_t, tau = self.get_safe_params()

        # Compute individual similarity components
        s_rel = self._relational_similarity(M_batch)
        s_time = self._time_similarity(t_batch, gamma_t)
        s_feat = self._feature_similarity(mu_batch, gamma_mu)
        
        # Total similarity per path
        path_scores = s_rel * s_time * s_feat
        
        # Evidence mass aggregation (sum over paths)
        mass = torch.sum(path_scores)
        
        # Hill-style saturation: E = M / (M + τ)
        evidence = mass / (mass + tau + self.EPS)
        
        return evidence
