import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np

class PredictionHead(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.evidence_scorer = EvidenceScorer(relational_sharpness=params["relational_sharpness"])
        self.concept_decoder = ConceptDecoder(params)#params["concept_dim"], params["feature_embed_dim"], max_hops=params["max_hops"], relation_count=len(params["node_types"]), schema = params["schema"])

        # Store concepts locally and optimise on them directly
        self.concepts = nn.Parameter(torch.randn(params["num_concepts"], params["concept_dim"]))

        # Interpretable combination of activation scores. What is an acceptable level of tradeoff in simplicity versus expressivity? 
        self.prediction_head = nn.Linear(params["num_concepts"], 2)
        self.concept_weights = nn.Parameter(torch.ones(params["num_concepts"]))
        self.bias = nn.Parameter(torch.tensor([-1.0]))

    def inspect_concepts(self, node_types):
        """
        Summarizes concepts by zipping relations, times, and features into 
        a vertical, readable path for each concept archetype.
        """
        vocab = node_types + ["∅ (STOP)"]
        
        with torch.no_grad():
            rel_p, time_p, g_time_p, g_feat_p, feat_p, tau_p, _ = self.concept_decoder(self.concepts)
        
        # Calculate non-negative weights from the additive head [cite: 328]
        positive_weights = torch.nn.functional.softplus(self.concept_weights).detach().cpu()
        
        print(f"\n{'='*100}")
        print(f"{'ID':<4} | {'Tau (Saturation)':<18} | {'Prediction Weight':<18}")
        print(f"{'='*100}")

        for i in range(self.concepts.shape[0]):
            tau_val = tau_p[i].item()
            weight_val = positive_weights[i].item()
            
            print(f"{i:<4} | {tau_val:<18.3f} | {weight_val:<+18.4f}")
            print(f"{'-'*100}")
            print(f"{'Step':<10} | {'Node Type Distribution':<35} | {'Time (±γ)':<15} | {'Features (±γ)':<20}")
            print(f"{'-'*100}")

            # Iterate through the hops (L) to pair data together 
            for h in range(rel_p.shape[1]):
                # 1. Format Node Type Probs [cite: 49, 53]
                probs = rel_p[i, h]
                active_idx = torch.where(probs > 0.01)[0]
                type_str = ", ".join([f"{vocab[idx]}({probs[idx]:.2f})" for idx in active_idx])
                
                # 2. Format Time [cite: 67, 71]
                t_val = time_p[i, h].item()
                t_gam = g_time_p[i, h].item()
                time_str = f"{t_val:>4.1f} (±{t_gam:<4.1f})"
                
                # 3. Format Features [cite: 74, 75]
                f_vec = feat_p[i, h].detach().cpu().numpy()
                f_gam = g_feat_p[i, h].item()
                f_str = f"[{', '.join([f'{v:+.2f}' for v in f_vec])}] (±{f_gam:.2f})"
                
                step_label = "ROOT" if h == 0 else f"HOP {h}"
                print(f"{step_label:<10} | {type_str:<35} | {time_str:<15} | {f_str}")

            print(f"{'='*100}\n")
        
        return rel_p, time_p, g_time_p, g_feat_p, feat_p, tau_p

    # def concept_orthogonality_regularisation(self):
    #     """
    #     Ensures latent concepts are distinct by penalizing cosine similarity.
    #     """
    #     # Normalize latent vectors
    #     z_norm = F.normalize(self.concepts, p=2, dim=1)
    #     # Compute self-similarity matrix [num_concepts, num_concepts]
    #     sim_matrix = torch.matmul(z_norm, z_norm.t())
        
    #     # Identity matrix (we don't penalize a concept matching itself)
    #     identity = torch.eye(self.concepts.shape[0], device=self.concepts.device)
        
    #     # Penalize any off-diagonal similarity
    #     loss = torch.mean((sim_matrix - identity) ** 2)
    #     return loss

    def interpretable_forward_pass(self, sampled_metapaths):
        logits, pred_data = self.forward(sampled_metapaths)

    def forward(self, sampled_metapaths):
        # Decode global concepts from latent z
        rel_proto, t_proto, gt_proto, gf_proto, mu_proto, tau, regularisation_terms = self.concept_decoder(self.concepts)

        # Calculate evidence mass over all concepts
        concept_activations = []
        components_log = []
        
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
            log_logit, components = self.evidence_scorer(prototype_object, sampled_metapaths)

            activation = torch.sigmoid(log_logit)
            concept_activations.append(activation)
            # concept_activations.append(log_logit)

            components_log.append(components)
        

        # Task Prediction
        activation_tensor = torch.stack(concept_activations).t()

        weights = F.softplus(self.concept_weights)

        pred_logit = (activation_tensor * weights).sum(dim=1) + self.bias

        return pred_logit, {"components": components_log, "activations": activation_tensor}, {"reg_terms": regularisation_terms, "activation_weights": weights}

class ConceptDecoder(nn.Module):
    def __init__(self, params):#concept_dim, feature_embed_dim, max_hops, relation_count, schema):
        super().__init__()
        concept_dim = params["concept_dim"]
        feature_embed_dim = params["feature_embed_dim"]
        max_hops = params["max_hops"]
        relation_count = len(params["node_types"])
        schema = params["schema"]
        
        self.gamma_floor = params["gamma_floor"]
        self.min_tau = params["min_tau"]

        self.L = max_hops + 1
        self.R = relation_count + 1
        self.D = feature_embed_dim

        self.reachability_mask = torch.from_numpy(schema.reachability_mask(hop_count = params["max_hops"], ordered_node_types = params["node_types"]))
        self.adj = torch.from_numpy(schema.get_adjacency_matrix(ordered_node_types = params["node_types"])).float()

        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 4),
            nn.GELU(),
            nn.Linear(concept_dim * 4, concept_dim * 2)
        )
                
        self.relation_head = nn.Linear(concept_dim * 2, self.L * self.R)
        self.meta_head = nn.Linear(concept_dim * 2, (self.L * 3) + 1) # Handles (time, gamma_time, gamma_feat, tau)
        self.feature_head = nn.Linear(concept_dim * 2, self.L * self.D)
        
        
        self.relation_head.apply(self._init_weights)
        self.meta_head.apply(self._init_weights)
        self.feature_head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use a higher gain for the heads to ensure they vary with input
            torch.nn.init.orthogonal_(m.weight, gain=2.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def sparsity_regularisation(self, relation_matrix, gamma_time, gamma_feat):
        
        # Relational_sparsity with shannon entropy
        epsilon = 1e-10
        entropy = -torch.sum(relation_matrix * torch.log(relation_matrix + epsilon), dim=-1)
        relational_sparsity = entropy.mean()
        # temporal sparsity

        # feature sparsity

        return relational_sparsity

    def forward(self, concept_z):
        batch_size = concept_z.shape[0]
        
        # Viterbi algorithm

        concept_state = self.trunk(concept_z)
        
        # Decode Relations
        rel_logits = self.relation_head(concept_state).view(batch_size, self.L, self.R)

        adj = self.adj.to(concept_z.device)

        relation_probs = []
        # Step 1: Root is fixed (e.g., Customer)
        curr_prob = torch.zeros(batch_size, self.R).to(concept_z.device)
        curr_prob[:, 0] = 1.0 # Index 0 is Root/Customer
        relation_probs.append(curr_prob)
        
        # Sequential Masking
        for l in range(1, self.L):
            # Calculate reachability based on previous hop: [Batch, R] @ [R, R]
            reachable = torch.matmul(curr_prob, adj) 
            
            # Use reachability mask to ensure logit mass isn't applied to unreachable nodes
            hop_logits = rel_logits[:, l, :]
            masked_logits = hop_logits.masked_fill(reachable == 0, -1e9)
            
            curr_prob = F.softmax(masked_logits, dim=-1)
            relation_probs.append(curr_prob)
            
        relation_matrix = torch.stack(relation_probs, dim=1)

        # Decode time and gamma tensors
        meta_flat = self.meta_head(concept_state)
        time_raw, gamma_time, gamma_feat, tau_raw = torch.split(
            meta_flat, 
            [self.L, self.L, self.L, 1], 
            dim=1
        )
        
        # time = torch.cumsum(F.softplus(time_raw), dim=1)
        # Treat time_raw as a the exponent in an exponentiation, allowing us to handle varying scales better and increase convergence time
        time = torch.exp(time_raw)
        
        # Enforce positivity on gammas
        # make gamma_time exist in the same logspace for stability under different temporal resolutions
        gamma_time = torch.exp(gamma_time) + self.gamma_floor
        gamma_feat = F.softplus(gamma_feat)
        
        
        tau = F.softplus(tau_raw) + self.min_tau # minimum tau requirement
        
        # Decode Features
        feat_flat = self.feature_head(concept_state)
        mu = feat_flat.view(batch_size, self.L, self.D)
        
        # Ensure unit sphere
        mu = F.normalize(mu, dim=-1)

        regularisation_terms = self.sparsity_regularisation(relation_matrix, gamma_time, gamma_feat)

        return relation_matrix, time, gamma_time, gamma_feat, mu, tau, regularisation_terms

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
        diff_sq = (expanded_prototype - batch_relations) ** 2
        
        # Mean Squared Error per path as before
        mse = torch.mean(diff_sq, dim=(2, 3))  # [N]
        
        log_similarity = - 1 * self.relational_sharpness * mse 
        
        return log_similarity  # Range: (-inf, 0]
    
    def _time_similarity_log(self, prototype_time, gamma_time, batch_times):
        """
        Computes Box-Distance in Log-Space with Huber Centering.
        Prevents evidence vanishing for large distances.
        """
        # 1. Masking & Input Sanitization
        mask = torch.isfinite(batch_times)
        safe_batch_times = torch.where(mask, batch_times, torch.zeros_like(batch_times))

        # 2. Log-Space Conversion
        log_proto = torch.log(prototype_time + 1.0)
        log_batch = torch.log(safe_batch_times + 1.0)
        log_width = torch.log(gamma_time + 1.0)
        
        # 3. Box Boundaries
        box_min = log_proto - log_width
        box_max = log_proto + log_width
        
        # 4. Box Distance (L1)
        closest_point = torch.max(box_min, torch.min(log_batch, box_max))
        dist_to_box = torch.abs(log_batch - closest_point)
        
        # 5. Huber Center Alignment [The Fix]
        # Quadratic when error < 1.0 (Log-Scale), Linear when error > 1.0.
        # This caps the penalty growth, keeping the Time Evidence "alive" 
        # (e.g., -15 instead of -168) so the model attends to it.
        dist_to_center = torch.abs(log_batch - log_proto)
        
        # We use a weight of 5.0 to balance with Feature/Relation scales
        center_pull = 5.0 * torch.where(
            dist_to_center < 1.0,
            dist_to_center ** 2,       # Quadratic (Precision)
            (2.0 * dist_to_center) - 1.0 # Linear (Robustness) - matched derivative at 1.0
        )
        
        # Combine
        total_log_dist = dist_to_box + center_pull
        
        # 6. Re-Apply Mask & Aggregate
        final_dist = torch.where(mask, total_log_dist, torch.zeros_like(total_log_dist))
        path_dist = torch.sum(final_dist, dim=-1)
        
        return self.ln_k * path_dist
    
    def _feature_similarity_log(self, prototype_features, gamma_features, batch_features):
            """
            Calculates Log-Similarity for features following the reparameterized RBF kernel.
            Returns: [Batch, Path_Count]
            """
            # Create mask for padded data [subgraphs, paths, path_length]
            mask = torch.isfinite(batch_features).all(dim=-1)

            # Convert embeddings to unit vectors for cosine
            prototype_norm = F.normalize(prototype_features, p=2, dim=-1, eps=self.EPS).unsqueeze(0).unsqueeze(0)
            
            safe_batch_features = torch.where(mask.unsqueeze(-1), batch_features, torch.zeros_like(batch_features))
            batch_norm = F.normalize(safe_batch_features, p=2, dim=-1, eps=self.EPS)

            # Compute Cosine Similarity: [subgraphs, paths, path_length]
            cosine_similarity = torch.sum(batch_norm * prototype_norm, dim=-1)

            # Convert to Distance: [-1, 1] -> [0, 1] (lower is closer)
            similarity_distance = 1.0 - ((cosine_similarity + 1.0) / 2.0)

            # Apply Gamma Scaling using gamma reparameterisation
            gamma_features_expanded = gamma_features[None, None, :] + self.EPS

            # Clamp similarity ratio to prevent huge distances from exploding gradients
            similarity_ratio = similarity_distance / gamma_features_expanded
            # clamped_similarity_ratio = torch.clamp(similarity_ratio, max=50)
            clamped_similarity_ratio = F.softsign(similarity_ratio)
            norm_dist_sq = (clamped_similarity_ratio) ** 2

            # reapply mask to ensure null nodes don't generate signal
            final_penalty = torch.where(mask, norm_dist_sq, torch.zeros_like(norm_dist_sq))

            # Sum penalty over nodes to create penalty for full path [subgraphs, paths, path_length] -> [subgraphs, paths]
            total_path_penalty = torch.sum(final_penalty, dim=-1)

            # Return Log-Similarity for EACH path
            return self.ln_k * total_path_penalty

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
        return (total_log_evidence - log_tau), {
            "rel": log_relational_similarity,
            "time": log_temporal_similarity,
            "feat": log_feature_similarity,
            "tau": concept_prototype.tau
        }
    


    
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
