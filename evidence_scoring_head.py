import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np

class PredictionHead(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Scorer and concept decoder
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
        # vocab set for printing
        vocab = node_types + ["∅ (STOP)"]
        
        # decode concept components
        with torch.no_grad():
            rel_p, time_p, g_time_p, g_feat_p, feat_p, tau_p, _ = self.concept_decoder(self.concepts)
        
        
        # Calculate concept weights for importance in activation
        positive_weights = torch.nn.functional.softplus(self.concept_weights).detach().cpu()
        
        
        print(f"\n{'='*100}")
        print(f"{'ID':<4} | {'Tau (Saturation)':<18} | {'Prediction Weight':<18}")
        print(f"{'='*100}")

        # Print on a per-concept basis
        for i in range(self.concepts.shape[0]):
            
            # Get tau and weight to understand concept
            tau_val = tau_p[i].item()
            weight_val = positive_weights[i].item()
            
            print(f"{i:<4} | {tau_val:<18.3f} | {weight_val:<+18.4f}")
            print(f"{'-'*100}")
            print(f"{'Step':<10} | {'Node Type Distribution':<35} | {'Time (±γ)':<15} | {'Features (±γ)':<20}")
            print(f"{'-'*100}")

            # Iterate through the hops
            for h in range(rel_p.shape[1]):
                # Format Node Type probabilites
                probs = rel_p[i, h]
                active_idx = torch.where(probs > 0.01)[0]
                type_str = ", ".join([f"{vocab[idx]}({probs[idx]:.2f})" for idx in active_idx])
                
                # Format Time
                t_val = time_p[i, h].item()
                t_gam = g_time_p[i, h].item()
                time_str = f"{t_val:>4.1f} (±{t_gam:<4.1f})"
                
                # Format Features
                f_vec = feat_p[i, h].detach().cpu().numpy()
                f_gam = g_feat_p[i, h].item()
                f_str = f"[{', '.join([f'{v:+.2f}' for v in f_vec])}] (±{f_gam:.2f})"
                
                step_label = "ROOT" if h == 0 else f"HOP {h}"
                print(f"{step_label:<10} | {type_str:<35} | {time_str:<15} | {f_str}")

            print(f"{'='*100}\n")
        
        return rel_p, time_p, g_time_p, g_feat_p, feat_p, tau_p

    def concept_diversity_loss(self, rel_proto):
        """Want to enforce concept sparsity, use rel_proto as first method for this"""
        
        # Flatten relational matrix into vector
        flat_protos = rel_proto.view(self.concepts.shape[0], -1)
        
        # Normalize to unit vectors for cosine similarity (dot product)
        flat_protos = F.normalize(flat_protos, p=2, dim=1)
        
        # Compute Similarity Matrix over concept vectors
        similarity_matrix = torch.matmul(flat_protos, flat_protos.t())
        
        # Identity matrix represents perfect concept orthogonality
        identity = torch.eye(self.concepts.shape[0], device=rel_proto.device)
        
        # MSE loss over non diagonal values
        diversity_loss = torch.mean((similarity_matrix - identity) ** 2)
        
        return diversity_loss
    

    def forward(self, sampled_metapaths):
        # Decode global concepts from latent z
        rel_proto, t_proto, gt_proto, gf_proto, mu_proto, tau, reg_terms = self.concept_decoder(self.concepts)

        # Calculate evidence mass over all concepts
        concept_activations = []
        components_log = []
        
        # For each concept
        for i in range(self.concepts.shape[0]):
            
            # Store concept components as object
            prototype_object = type('Obj', (object,), {
                'relations': rel_proto[i],
                'times': t_proto[i],
                'gamma_times': gt_proto[i],
                'features': mu_proto[i],
                'gamma_features': gf_proto[i],
                'tau': tau[i]
            })()
            
            # Calculate per subgraph similarity logits
            log_logit, components = self.evidence_scorer(prototype_object, sampled_metapaths)

            # convert to similarity probability
            activation = torch.sigmoid(log_logit)
            
            # Store Concept Activations
            concept_activations.append(activation)
            
            # Store runtime metadata
            components_log.append(components)
        

        # Task Prediction
        activation_tensor = torch.stack(concept_activations).t()
        
        # How many tasks are activating
        sparsity_loss = activation_tensor.mean()
        
        # How different are the concepts
        diversity_loss = self.concept_diversity_loss(rel_proto)
        
        # Prediction logit using weighted sum of concept activations
        weights = F.softplus(self.concept_weights)
        pred_logit = (activation_tensor * weights).sum(dim=1) + self.bias

        return pred_logit, {
            "components": components_log, 
            "activations": activation_tensor,
            "sparsity_loss": sparsity_loss,
            "diversity_loss": diversity_loss,
            "reg_terms": reg_terms
        }

class ConceptDecoder(nn.Module):
    """torch module to decode concepts from concept latents"""
    
    def __init__(self, params):
        super().__init__()
        
        # Parameter Extraction
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

        # Adjacency matrix to reduce relational search space        
        self.adj = torch.from_numpy(schema.get_adjacency_matrix(ordered_node_types = params["node_types"])).float()

        # Shared Trunk decoder
        self.trunk = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 4),
            nn.GELU(),
            nn.Linear(concept_dim * 4, concept_dim * 2)
        )
        
        # Decoders
        self.relation_head = nn.Linear(concept_dim * 2, self.L * self.R)
        self.meta_head = nn.Linear(concept_dim * 2, (self.L * 3) + 1) # Handles (time, gamma_time, gamma_feat, tau)
        self.feature_head = nn.Linear(concept_dim * 2, self.L * self.D)
        
        # Weight initialisation
        self.trunk.apply(self._init_trunk_weights)
        self.relation_head.apply(self._init_head_weights)
        self.meta_head.apply(self._init_head_weights)
        self.feature_head.apply(self._init_head_weights)

    def _init_trunk_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=math.sqrt(2)) # use root 2 initialisation to prevent killed gradients from GELU
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def _init_head_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=1.0) # use gain of 1 to make training more stable with exp and softplus functions
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def soft_discrete_regularisation(self, relation_matrix, gamma_time, gamma_feat):
        # relational discretisation with shannon entropy
        epsilon = 1e-10
        entropy = -torch.sum(relation_matrix * torch.log(relation_matrix + epsilon), dim=-1)
        relational_sparsity = entropy.mean()
        
        # temporal sparsity

        # feature sparsity

        return relational_sparsity

    def forward(self, concept_z):
        batch_size = concept_z.shape[0]
        
        # 1) Decode concepts through trunk
        concept_state = self.trunk(concept_z)
        
        # 2.1) Decode Relational matrix logits
        rel_logits = self.relation_head(concept_state).view(batch_size, self.L, self.R)

        # 2.2) use schema defined reachability matrix to push logit mass towards possible paths
        adj = self.adj.to(concept_z.device)
        relation_probs = []
        
        # Define new probability matrix
        curr_prob = torch.zeros(batch_size, self.R).to(concept_z.device)
        
        # Always start from root node
        curr_prob[:, 0] = 1.0
        relation_probs.append(curr_prob)
        
        # sequential masking by hop
        for l in range(1, self.L):
            # Calculate reachability based on previous hop: [Batch, R] @ [R, R]
            reachable = torch.matmul(curr_prob, adj)
            
            # Use reachability mask to ensure logit mass isn't applied to unreachable nodes
            hop_logits = rel_logits[:, l, :]
            masked_logits = hop_logits.masked_fill(reachable < 1e-6, -1e9)            
            curr_prob = F.softmax(masked_logits, dim=-1)
            relation_probs.append(curr_prob)
            
        # restack relation matirx
        relation_matrix = torch.stack(relation_probs, dim=1)

        # 3) Decode time and gamma tensors
        meta_flat = self.meta_head(concept_state)
        time_raw, gamma_time, gamma_feat, tau_raw = torch.split(
            meta_flat, 
            [self.L, self.L, self.L, 1], 
            dim=1
        )
        
        # 3.1) treat time and time gamma as existing in logspace
        time = torch.exp(time_raw)
        gamma_time = torch.exp(gamma_time) + self.gamma_floor


        # 3.2) simply enforce feature gammas to be positive
        gamma_feat = F.softplus(gamma_feat)
        
        # 3.3) Positive tau
        tau = F.softplus(tau_raw) + self.min_tau
        
        # 4) Decode prototype feature vectors
        feat_flat = self.feature_head(concept_state)
        mu = feat_flat.view(batch_size, self.L, self.D)
        
        # Keep features as unit vectors for ease of similarity comparison
        mu = F.normalize(mu, dim=-1)

        # Generate sparsity regularisation terms
        discretisation_regularisation_terms = self.soft_discrete_regularisation(relation_matrix, gamma_time, gamma_feat)

        return relation_matrix, time, gamma_time, gamma_feat, mu, tau, discretisation_regularisation_terms

class EvidenceScorer(nn.Module):
    def __init__(self, relational_sharpness = 10, k=0.1):
        super().__init__()
        
        self.relational_sharpness = relational_sharpness
        self.k = k
        self.ln_k = math.log(k)
        self.EPS = 1e-10
        pass

    def _relational_similarity_log(self, prototype_relations, batch_relations):
        """
        Compute relational similarity as MSE from prototype
        Handle in logspace for stability and similarity compatibility
        Exponential similarity = e^{-relational_sharpness*MSE}, keeps values between 0-1
        """
        # Calculate Squared Difference between path encodings and prototype
        expanded_prototype = prototype_relations[None, None, :, :]
        diff_sq = (expanded_prototype - batch_relations) ** 2
        
        # Simple Mean Squared Error
        mse = torch.mean(diff_sq, dim=(2, 3))
        
        # e^{-relational_sharpness*MSE}, keeps values between 0-1
        log_similarity = - 1 * self.relational_sharpness * mse 
        
        return log_similarity
    
    def _time_similarity_log(self, prototype_time, gamma_time, batch_times):
        """
        Computes logspace Box-Distance for stability in unbounded search space in with Huber Centering for precision
        """
        
        # Masking to ignore irreleavant features
        mask = torch.isfinite(batch_times)
        safe_batch_times = torch.where(mask, batch_times, torch.zeros_like(batch_times))

        # Update parameters to logspace
        log_proto = torch.log(prototype_time + 1.0)
        log_batch = torch.log(safe_batch_times + 1.0)
        log_width = torch.log(gamma_time + 1.0)
        
        # Define Box Boundaries
        box_min = log_proto - log_width
        box_max = log_proto + log_width
        
        # Compute distance to nearest face of hyper-box
        closest_point = torch.max(box_min, torch.min(log_batch, box_max))
        dist_to_box = torch.abs(log_batch - closest_point)
        
        # Huber style box Center Alignment
        dist_to_center = torch.abs(log_batch - log_proto)
        
        # Quadratic when error < 1.0 (Log-Scale), Linear when error > 1.0
        center_pull = 5.0 * torch.where( # 5 is chosen arbitrarily
            dist_to_center < 1.0, # Assumes log-scale sub 1 is reasonable (should be true for day scale)
            dist_to_center ** 2, # Quadratic center
            (2.0 * dist_to_center) - 1.0 # Linear outside of box
        )
        
        # Combine center pull (box centralisation) and outside the box (box boundaries) pulls
        total_log_dist = dist_to_box + center_pull
        
        # Reapply mask to prevent gradient from dummy nodes
        final_dist = torch.where(mask, total_log_dist, torch.zeros_like(total_log_dist))
        path_dist = torch.sum(final_dist, dim=-1)
        
        return self.ln_k * path_dist
    
    def _feature_similarity_log(self, prototype_features, gamma_features, batch_features):
            """
            Calculates Log-Similarity for features usine cosine similarity on unit sphere representations
            """
            
            # Create mask for padded data
            mask = torch.isfinite(batch_features).all(dim=-1)

            # Convert embeddings to unit vectors for cosine
            prototype_norm = F.normalize(prototype_features, p=2, dim=-1, eps=self.EPS).unsqueeze(0).unsqueeze(0)
            
            safe_batch_features = torch.where(mask.unsqueeze(-1), batch_features, torch.zeros_like(batch_features))
            batch_norm = F.normalize(safe_batch_features, p=2, dim=-1, eps=self.EPS)

            # Compute Cosine Similarity
            cosine_similarity = torch.sum(batch_norm * prototype_norm, dim=-1)

            # Convert to Distance: [-1, 1] -> [0, 1] (lower is closer)
            similarity_distance = 1.0 - ((cosine_similarity + 1.0) / 2.0)

            # Apply Gamma Scaling using gamma reparameterisation
            gamma_features_expanded = gamma_features[None, None, :] + self.EPS

            # Clamp similarity ratio to prevent huge distances from exploding gradients
            similarity_ratio = similarity_distance / gamma_features_expanded
            clamped_similarity_ratio = F.softsign(similarity_ratio)
            norm_dist_sq = (clamped_similarity_ratio) ** 2

            # reapply mask to ensure null nodes don't generate signal
            final_penalty = torch.where(mask, norm_dist_sq, torch.zeros_like(norm_dist_sq))

            # Sum penalty over nodes to create penalty for full path
            total_path_penalty = torch.sum(final_penalty, dim=-1)

            # Return Log-Similarity for EACH path
            return self.ln_k * total_path_penalty

    def aggregate_evidence_log(self, log_s_rel, log_s_time, log_s_feat):
        # Combine log similarities (Multiplication becomes Addition)
        log_s_tot = log_s_rel + log_s_time + log_s_feat
        
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
