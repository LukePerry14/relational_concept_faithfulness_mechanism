import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from dataclasses import dataclass
import unittest

# class ConceptDecoder(nn.Module):
#     def __init__(self, embedding_dim, max_hops, num_relations, valid_hops_mask):
#         super().__init__()
#         self.embedding_dim = embedding_dim

#         self.decoder = nn.

#     def forward(self, )



class PredictionHead(nn.Module):
    def __init__(self, number_of_concepts):
        super().__init__()

        self.evidence_scorer = EvidenceScorer()
        pass

    def forward(self, z_concepts, sampled_metapaths):
        # 1. Decode global concepts from latent z
        prototypes = self.decoder(z_concepts)
        
        # 2. Calculate Log Evidence Mass (log_M) for each concept
        #    (No weights here, just math)
        log_M = self.functional_scorer(sampled_metapaths, prototypes)
        
        # 3. Compute Evidence Logit: x = ln(M) - ln(tau)
        #    Note: log_tau comes from the decoded prototype
        evidence_logits = log_M - prototypes.log_tau
        
        # 4. Task Prediction
        #    We can now use a simple Linear layer on these logits
        #    task_logit = Linear(evidence_logits)
        task_logit = None
        return task_logit, evidence_logits



class EvidenceScorer(nn.Module):
    def __init__(self, relational_sharpness = 10, k=0.1):
        super().__init__()
        self.relational_sharpness = relational_sharpness # complete guess
        self.k = k
        self.ln_k = math.log(k)
        self.EPS = 1e-10
        pass

    def _relational_similarity_log(self, prototype_relations, batch_relations):
        """
        Computes Log-Similarity for relation sequence.
        Exponential similarity = e^{-relational_sharpness*MSE} (updated from initial description to be logspace compatible)
        relational_sharpness is functionaly identical to the gamma values, however, we want this to be as discrete as possible, we instead treat this as a hyperparameter
        logspace similarity  = ln(-relational_sharpness*MSE) = -MSE

        Parameter sizes:
        prototype_relations = [L x R]
        batch_relations = [L x R x B]

        where:
            - L is max metapath length
            - R is the number of candidate relations
            - B is the batch size
        """
        # Calculate Squared Difference between path encodings and prototype
        diff_sq = (prototype_relations.unsqueeze(0) - batch_relations) ** 2
        
        # Mean Squared Error per path as before
        mse = torch.mean(diff_sq, dim=(1, 2))  # [N]
        
        log_similarity = - 1 * self.relational_sharpness * mse 
        
        return log_similarity  # Range: (-inf, 0]

    def _time_similarity_log(self, prototype_time, gamma_time, batch_times):
        """
        Computes Log-Similarity for time.
        Formula: ln(0.1) * sum(normalized_diff)

        Parameter sizes:
        prototype_time = [L]
        gamma_time = [L]
        batch_relations = [L x B]

        where:
            - L is max metapath length
            - B is the batch size
        """

        # Ensure gamma
        # mask = torch.isfinite(gamma_time) * 
        # diff = torch.where(mask, prototype_time - batch_times, torch.zeros_like(batch_times))
        
        diff = prototype_time.unsqueeze(0) - batch_times
        # Normalized Squared Difference - add epsilon for divide by zero ?
        normalized_diff_sq = (diff ** 2) / ((gamma_time ** 2) + self.EPS)
        
        # Sum over path length - can be of arbitrary size
        total_diff = torch.sum(normalized_diff_sq, dim=1)
        
        # Return Log Similarity
        log_similarity = self.ln_k * total_diff
        
        return log_similarity # Range: (-inf, 0]

    def _feature_similarity_log(self, prototype_features, gamma_features, batch_features):
        """
        Computes Log-Similarity for features using cosine similarity.
        Formula:

        Parameter sizes:
        prototype_features = [L x D]
        gamma_features = [L]
        batch_features = [L x D x B]

        where:
            - L is max metapath length
            - D is the embedding dimension
            - B is the batch size
        """
        
        # Normalize embeddings for cosine similarity
        prototype_features_normalised = F.normalize(prototype_features, p=2, dim=-1, eps=self.EPS)  # [L, D]
        batch_features_normalised = F.normalize(batch_features, p=2, dim=-1, eps=self.EPS)  # [L x D x B]
        
        # Compute cosine similarity using unit vector dot product between prototype and batch
        cosine_similarity = torch.sum(batch_features_normalised * prototype_features_normalised.unsqueeze(0), dim=-1)  # [B, L]
        
        # distance in terms of cosine similarity
        similarity_distance = 1 - ((cosine_similarity + 1) / 2.0) # [B, L]
        
        # Scale by gamma (wider gamma = more tolerance) - epsilon for stabiility
        gamma_modified_distance = similarity_distance / (gamma_features.unsqueeze(0) + self.EPS)
        
        # sum cosine similarity "distances" along paths
        total_dist = torch.sum((gamma_modified_distance ** 2), dim=1)  # [B]
        
        log_similarity = self.ln_k * total_dist
    
        return log_similarity

    def aggregate_evidence_log(self, log_s_rel, log_s_time, log_s_feat):
        # Combine log similarities (Multiplication becomes Addition)
        log_s_tot = log_s_rel + log_s_time + log_s_feat  # [N_paths]
        
        # Aggregate mass 
        log_M = torch.logsumexp(log_s_tot, dim=0)
    
        return log_M 
    
    def forward(self, concept_prototype, sampled_metapaths):
        """
        Calculate evidence scores all in Logspace to maintain smooth gradients

        Evidence is calculated using a hill function to allow multiple small activations to contribute equally to one large activation. Let total evidence be M and evidence score be E:

        E = M / (M + tau)

        Evidence score E is a value between 0 and 1. We can therefore treat it as a probability of concept activation and calculate the logit for this directly with

        ln(E / (1-E))

        we can now substitute my hill function back in for E

        ln ((M / (M + tau)) / (1 - (M / (M + tau))))

        with a little rearranging, this becomes:
        
        ln(M / tau) = ln(M) - ln(tau)
        """
        # Generate Evidence Scores
        log_relational_similarity = self._relational_similarity_log(concept_prototype.relations, sampled_metapaths['relations'])
        log_temporal_similarity = self._time_similarity_log(concept_prototype.times, concept_prototype.gamma_times, sampled_metapaths['times'])
        log_feature_similarity = self._feature_similarity_log(concept_prototype.features, concept_prototype.gamma_features, sampled_metapaths['features'])

        # Aggregate evidence
        total_log_evidence = self.aggregate_evidence_log(log_relational_similarity, log_temporal_similarity, log_feature_similarity)
        
        # Extract tau and convert to logspace (ensure positive with softplus)
        log_tau = torch.log(F.softplus(concept_prototype.tau_raw) + self.EPS)     # tau raw is placeholder before I define decoding process    

        # return evidence scores for each concept
        return total_log_evidence - log_tau


if __name__ == "__main__":
    tester = LogitSanityTests()
    tester.test_relational()
    tester.test_temporal()
    tester.test_aggregation_and_hill()
    tester.run_full_pipeline()