import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np

class PredictionHead(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.evidence_scorer = EvidenceScorer()
        self.concept_decoder = ConceptDecoder(params["concept_dim"], params["feature_embed_dim"], max_hops=params["max_hops"], relation_count=len(params["node_types"]))

        # Store concepts locally and optimise on them directly
        self.concepts = nn.Parameter(torch.randn(params["num_concepts"], params["concept_dim"]))

        # Interpretable combination of activation scores. What is an acceptable level of tradeoff in simplicity versus expressivity? 
        self.prediction_head = nn.Linear(params["num_concepts"], 2)

    

    def forward(self, sampled_metapaths):
        # Decode global concepts from latent z
        rel_proto, t_proto, gt_proto, gf_proto, mu_proto, tau = self.concept_decoder(self.concepts)
 
        # Calculate evidence mass over all concepts
        concept_activations = []
        for i in range(self.concepts.shape[0]):
            # Extract the i-th decoded prototype
            prototype_object = type('Obj', (object,), {
                'relations': rel_proto[i],
                'times': t_proto[i],
                'gamma_times': gt_proto[i],
                'features': mu_proto[i],
                'gamma_features': gf_proto[i],
                'tau': tau[i]
            })()
            log_logit = self.evidence_scorer(prototype_object, sampled_metapaths)
            concept_activations.append(log_logit)
        

        # Task Prediction
        activation_tensor = torch.sigmoid(torch.stack(concept_activations)).t() # [num_concepts]

        return self.prediction_head(activation_tensor)

class ConceptDecoder(nn.Module):
    def __init__(self, concept_dim, feature_embed_dim, max_hops, relation_count):
        super().__init__()

        self.L = max_hops
        self.R = relation_count + 1
        self.D = feature_embed_dim
        
        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 4),
            nn.GELU(),
            nn.Linear(concept_dim * 4, concept_dim * 2)
        )
                
        self.relation_head = nn.Linear(concept_dim * 2, self.L * self.R)
        self.meta_head = nn.Linear(concept_dim * 2, (self.L * 3) + 1) # Handles (time, gamma_time, gamma_feat, tau)
        self.feature_head = nn.Linear(concept_dim * 2, self.L * self.D)

    def forward(self, concept_z):
        batch_size = concept_z.shape[0]
        
        concept_state = self.trunk(concept_z)
        
        # Decode Relations
        rel_flat = self.relation_head(concept_state)
        relation_matrix = F.softmax(rel_flat.view(batch_size, self.L, self.R), dim=-1)
        
        # Decode time and gamma tensors
        meta_flat = self.meta_head(concept_state)
        time, gamma_time, gamma_feat, tau_raw = torch.split(
            meta_flat, 
            [self.L, self.L, self.L, 1], 
            dim=1
        )
        
        # Enforce positivity on gammas
        gamma_time = F.softplus(gamma_time) # Ensure n divide by 0 error
        gamma_feat = F.softplus(gamma_feat)
        tau = F.softplus(tau_raw)
        
        # Decode Features
        feat_flat = self.feature_head(concept_state)
        mu = feat_flat.view(batch_size, self.L, self.D)
        
        # Ensure unit sphere
        mu = F.normalize(mu, dim=-1)
        
        return relation_matrix, time, gamma_time, gamma_feat, mu, tau



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
        batch_relations = [B x P x L x R]

        where:
            - L is max metapath length
            - R is the number of candidate relations
            - B is the batch size
            - P is the number of sampled metapaths per subgraph
        """
        # Calculate Squared Difference between path encodings and prototype
        expanded_prototype = prototype_relations[None, None, :, :]
        diff_sq = (expanded_prototype - batch_relations) ** 2 # TODO: Ensure data is unpacked correctly
        
        # Mean Squared Error per path as before
        mse = torch.mean(diff_sq, dim=(2, 3))  # [N]
        
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
        mask = torch.isfinite(batch_times)
        # diff = torch.where(mask, prototype_time - batch_times, torch.zeros_like(batch_times))
        
        prototype_time_expanded = prototype_time[None, None, :]
        gamma_time_expanded = gamma_time[None, None, :]
        diff = prototype_time_expanded - batch_times

        masked_diffs = torch.where(mask, diff, torch.zeros_like(diff))

        # Normalized Squared Difference - add epsilon for divide by zero ?
        normalized_diff_sq = (masked_diffs ** 2) / ((gamma_time_expanded ** 2) + self.EPS)
        
        # Sum over path length - can be of arbitrary size
        total_diff = torch.sum(normalized_diff_sq, dim=2)
        
        # Return Log Similarity
        log_similarity = self.ln_k * total_diff
        
        return log_similarity # Range: (-inf, 0]

    def _feature_similarity_log(self, prototype_features, gamma_features, batch_features):
        """
        Computes Log-Similarity for features with padding mask.
        batch_features shape: [B, P, L, D]
        """
        # 1. Create the Boolean Mask
        # Check if the feature vector at each hop is finite.
        # Resulting shape: [B, P, L]
        mask = torch.isfinite(batch_features).all(dim=-1)

        # 2. Prevent Normalization NaNs
        # F.normalize will produce NaN if it encounters 'inf'. We replace 'inf' 
        # with 0.0 only for the normalization step.
        safe_batch_features = torch.where(mask.unsqueeze(-1), batch_features, torch.zeros_like(batch_features))

        # 3. Normalize prototype and batch embeddings
        # prototype_features: [L, D] -> [1, 1, L, D]
        # batch_features: [B, P, L, D]
        prototype_norm = F.normalize(prototype_features, p=2, dim=-1, eps=self.EPS)
        batch_norm = F.normalize(safe_batch_features, p=2, dim=-1, eps=self.EPS)

        # 4. Compute Cosine Similarity
        # Sum across the D dimension (dim=-1)
        # Resulting shape: [B, P, L]
        cosine_similarity = torch.sum(batch_norm * prototype_norm[None, None, :, :], dim=-1)

        # 5. Convert Similarity to Distance
        # Similarity range [-1, 1] -> Distance range [0, 1]
        similarity_distance = 1.0 - ((cosine_similarity + 1.0) / 2.0)

        # 6. Apply Gamma Scaling
        # gamma_features shape: [L] -> [1, 1, L]
        gamma_features_expanded = gamma_features[None, None, :]
        gamma_modified_distance = similarity_distance / (gamma_features_expanded + self.EPS)

        # 7. Apply the Mask to Distances
        # We explicitly zero out the distance for padding hops so they 
        # contribute nothing to the sum.
        final_distance_sq = torch.where(mask, gamma_modified_distance ** 2, torch.zeros_like(gamma_modified_distance))

        # 8. Sum over path length (L) and convert to Log Similarity
        # Resulting shape: [B, P]
        total_dist = torch.sum(final_distance_sq, dim=-1)
        log_similarity = self.ln_k * total_dist

        return log_similarity

    def aggregate_evidence_log(self, log_s_rel, log_s_time, log_s_feat):
        # Combine log similarities (Multiplication becomes Addition)
        log_s_tot = log_s_rel + log_s_time + log_s_feat  # [N_paths]
        
        # Aggregate mass 
        log_M = torch.logsumexp(log_s_tot, dim=-1)
    
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
        
        # Convert tau to logspace
        log_tau = torch.log(concept_prototype.tau)

        # return evidence scores for each concept
        return total_log_evidence - log_tau
    


    
# class QueryBasedConceptDecoder(nn.Module):
#     """WIP Query Based concept decoder aiming to allow global embeddings to be generated"""
#     def __init__(self, z_dim, embed_dim, max_hops, relation_count):
#         super().__init__()
#         self.L = max_hops
#         self.D = embed_dim
        
#         # 1. THE HOP QUERIES
#         # Learnable vectors that represent "Step 1", "Step 2", etc.
#         # Shape: [L, z_dim]
#         self.hop_queries = nn.Parameter(torch.randn(max_hops, z_dim))
        
#         # 2. THE SHARED TRUNK (Contextualizer)
#         # Takes (z + query) and expands it
#         # Note: We process L items per concept, so this acts on the 'sequence' dim
#         self.trunk = nn.Sequential(
#             nn.Linear(z_dim, z_dim * 4),
#             nn.LayerNorm(z_dim * 4),
#             nn.GELU(),
#             nn.Linear(z_dim * 4, z_dim * 2) # Compress slightly before heads
#         )
        
#         # 3. THE BRANCHES (Heads)
#         # These now operate on the "Hops" dimension
        
#         # Feature Head: Projects back to embedding dimension
#         self.feat_head = nn.Linear(z_dim * 2, embed_dim)
        
#         # Relation Head: Predicts relation leading TO this node
#         self.rel_head = nn.Linear(z_dim * 2, relation_count + 1)
        
#         # Meta Head: Time and Gammas
#         self.meta_head = nn.Linear(z_dim * 2, 3) # t, gamma_t, gamma_mu

#     def forward(self, z):
#         """
#         z: [Batch_Size, z_dim] (The Concept Tokens)
#         """
#         batch_size = z.shape[0]
        
#         # 1. Expand Z for each Hop
#         # z_expanded: [Batch, L, z_dim]
#         z_expanded = z.unsqueeze(1).repeat(1, self.L, 1)
        
#         # 2. Add Hop Queries (Broadcasting)
#         # This injects the "Step Info" into the concept
#         # queries: [1, L, z_dim]
#         queries = self.hop_queries.unsqueeze(0)
#         query_state = z_expanded + queries 
        
#         # 3. Pass through Trunk (Shared processing)
#         hidden_state = self.trunk(query_state)
                
#         # Features: [Batch, L, D]
#         mu = F.normalize(self.feat_head(hidden_state), p=2, dim=-1)
        
#         # Relations: [Batch, L, R]
#         rel_logits = self.rel_head(hidden_state)
#         P = F.softmax(rel_logits, dim=-1)
        
#         # Meta: [Batch, L, 3]
#         meta = self.meta_head(hidden_state)
#         t = meta[:, :, 0]
#         gamma_t = F.softplus(meta[:, :, 1])
#         gamma_f = F.softplus(meta[:, :, 2])
        
#         return P, t, gamma_t, gamma_f, mu
