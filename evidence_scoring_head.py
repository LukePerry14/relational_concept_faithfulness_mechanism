import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class SEL_LogicHead(nn.Module):
    """
    Fuzzy Logic DNF prediction head over concepts for expressivity and easier regularisation
    
    https://github.com/pietrobarbiero/pytorch_explain
    """
    def __init__(self, num_concepts, num_clauses=4):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_clauses = num_clauses
        
        # Literal Selection weights - size 2 vector, one for concept activation and one for negation activation
        self.Literal_selection_weights = nn.Parameter(torch.empty(num_clauses, num_concepts, 2))        
        # Clause grouping weights
        self.clause_selection_weights = nn.Parameter(torch.empty(num_clauses))

        # Temperature for annealing
        self.register_buffer("temp", torch.tensor(1.0))

        self._init_random_weights(self)

    def _init_random_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal(m.weight, std=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        for name, param in m.named_parameters(recurse=False):
            if 'weights' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.1)

    def forward(self, activation_tensor):
        # Generate positive and negative literal selection weights
        literal_weights = torch.sigmoid(self.Literal_selection_weights / self.temp) # act as gates for whether a literal is active

        # We now have 2C activations clause requirements over meta-path concepts now treated as literals
        positive_literal_weights = literal_weights[:, :, 0]
        negative_literal_weights = literal_weights[:, :, 1]

        # Use Fuzzy Product T-Norm (https://en.wikipedia.org/wiki/T-norm) AND gate with concept activation 
        # to see if the concept is activated and the clause contains it 
        activation_tensor_expanded = activation_tensor.unsqueeze(1)
        positive_term = 1.0 - (positive_literal_weights.unsqueeze(0) * (1.0 - activation_tensor_expanded))
        negative_term = 1.0 - (negative_literal_weights.unsqueeze(0) * activation_tensor_expanded)
        
        # Use Fuzzy Product T-Norm Again to create the Clauses
        """
        1) Multiplying positive and negative clauses forces positive and negative concepts to be present
        2) taking the log of the pairwise terms then summing over per literal activations is more stable
        3) re-exponentiate to return to linear space
        """
        log_clauses = torch.sum(torch.log(positive_term * negative_term + 1e-10), dim=-1)
        clauses = torch.exp(log_clauses) # [Batch, Clauses]

        # Extract clause selection weights and sigmoid pushes weights to be between 0-1 
        # as in (https://arxiv.org/pdf/2108.05149 section 5.1)
        clause_weights = torch.sigmoid(self.clause_selection_weights / self.temp)
        
        # At least one clause is true is identical to 1 minus everything is false (De Morgan's Law)
        inverse_or_term = 1.0 - (clause_weights.unsqueeze(0) * clauses)
        
        # Calculation: 1 - exp(sum(log(1 - weight * clause)))
        log_failure_prob = torch.sum(torch.log(inverse_or_term + 1e-10), dim=-1)
        prediction_prob = 1.0 - torch.exp(log_failure_prob)
        
        # Convert prob to logit for BCEWithLogitsLoss
        prediction_prob_clamped = torch.clamp(prediction_prob, min=1e-7, max=1.0 - 1e-7)
        prediction_logit = torch.logit(prediction_prob_clamped)

        return prediction_logit, {"clauses": clauses, "rule_weights": literal_weights, "clause_weights": clause_weights}

    def print_logic(self, concept_names=None):
        """
        Extracts human-readable rules from the weights.
        """
        print(f"\n{'='*20} LEARNED SYMBOLIC LOGIC {'='*20}")
        
        with torch.no_grad():
            # Apply temperature sharpening for a cleaner printout
            w = (torch.sigmoid(self.Literal_selection_weights / self.temp) > 0.5).float()
            v = (torch.sigmoid(self.clause_selection_weights / self.temp) > 0.5).float()
        
        active_rules = []
        for j in range(self.num_clauses):
            if v[j] < 0.5: continue # Skip rules the OR gate ignores
                
            literals = []
            for i in range(self.num_concepts):
                c_name = concept_names[i] if concept_names else f"C{i}"
                if w[j, i, 0] > 0.5 and w[j, i, 1] > 0.5:
                    literals.append(f"FALSE({c_name})")
                elif w[j, i, 0] > 0.5:
                    literals.append(f"{c_name}")
                elif w[j, i, 1] > 0.5:
                    literals.append(f"NOT {c_name}")
            
            rule_str = "(" + " AND ".join(literals) + ")" if literals else "(TRUE)"
            active_rules.append(rule_str)
            
        if not active_rules:
            print("Logic: FALSE (No active rules)")
        else:
            print(f"Logic: IF {' OR '.join(active_rules)} THEN POSITIVE")

class PredictionHead(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Scorer and concept decoder
        self.evidence_scorer = EvidenceScorer(relational_sharpness=params["relational_sharpness"])
        self.concept_decoder = ConceptDecoder(params)#params["concept_dim"], params["feature_embed_dim"], max_hops=params["max_hops"], relation_count=len(params["node_types"]), schema = params["schema"])

        # Store concepts locally and optimise on them directly
        self.concepts = nn.Parameter(torch.randn(params["num_concepts"], params["concept_dim"]))

        # Interpretable combination of activation scores. What is an acceptable level of tradeoff in simplicity versus expressivity? 
        self.logic_head = SEL_LogicHead(params["num_concepts"], num_clauses=4)

            
    def inspect_concepts(self, node_types):
        """
        Summarizes concepts and their status within the Symbolic Logic Rules.
        """
        # 1. Print the Logic Rules first (Context)
        self.logic_head.print_logic()
        
        vocab = node_types + ["∅ (STOP)"]
        
        # 2. Decode Prototypes
        with torch.no_grad():
             rel_p, time_p, g_time_p, g_feat_p, feat_p, tau_p, _ = self.concept_decoder(self.concepts)
             
             # Check Logic Status per concept
             # Get binary weights from logic head
             w = (torch.sigmoid(self.logic_head.Literal_selection_weights / self.logic_head.temp) > 0.5).float()
             v = (torch.sigmoid(self.logic_head.clause_selection_weights / self.logic_head.temp) > 0.5).float()
             
        print(f"{'='*100}")
        print(f"{'ID':<4} | {'Tau':<6} | {'Logic Status'}")
        print(f"{'='*100}")
        
        for i in range(self.concepts.shape[0]):
            # Determine how this concept is used
            # It is "Active" if it appears in at least one clause that is used by the final OR (v=1)
            used_pos = ((w[:, i, 0] * v) > 0.5).any().item()
            used_neg = ((w[:, i, 1] * v) > 0.5).any().item()
            
            if used_pos and used_neg:
                status = "Contradiction (Used as POS & NEG)"
            elif used_pos:
                status = "ACTIVE (Positive Literal)"
            elif used_neg:
                status = "ACTIVE (Negative Literal)"
            else:
                status = "Ignored (Pruned)"

            tau_val = tau_p[i].item()
            print(f"{i:<4} | {tau_val:<6.3f} | {status}")
            print(f"{'-'*100}")
            print(f"{'Step':<10} | {'Node Type Distribution':<35} | {'Time (±γ)':<15} | {'Features (±γ)':<20}")
            print(f"{'-'*100}")

            for h in range(rel_p.shape[1]):
                probs = rel_p[i, h]
                active_idx = torch.where(probs > 0.01)[0]
                type_str = ", ".join([f"{vocab[idx]}({probs[idx]:.2f})" for idx in active_idx])
                
                t_val = time_p[i, h].item()
                t_gam = g_time_p[i, h].item()
                time_str = f"{t_val:>4.1f} (±{t_gam:<4.1f})"
                
                f_vec = feat_p[i, h].detach().cpu().numpy()
                f_gam = g_feat_p[i, h].item()
                f_str = f"[{', '.join([f'{v:+.2f}' for v in f_vec])}] (±{f_gam:.2f})"
                
                step_label = "ROOT" if h == 0 else f"HOP {h}"
                print(f"{step_label:<10} | {type_str:<35} | {time_str:<15} | {f_str}")

            print(f"{'='*100}\n")

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
    
    def activation_weighted_diversity_loss(self, relation_prototype, time_prototype, feature_prototype, activations):
        """
        Calculate the orthogonality between decoded concept prototypes.
        Scale this loss by concept co-activation in the logic head to prevent garbage concepts from becoming blocking
        """
        K = relation_prototype.shape[0]
        
        # 1. FLATTEN AND NORMALIZE COMPONENTS SEPARATELY
        # We normalize each modality *before* concatenation so that 
        # Relations (probs), Time (years), and Features (embeddings) contribute equally.
        
        # Relations: [K, L, R] -> [K, L*R]
        flat_rel = F.normalize(relation_prototype.view(K, -1), p=2, dim=1)
        
        # Time: [K, L] -> [K, L]
        # Note: We use softsign/tanh on time first to bound it, as raw years (e.g. 2022) 
        # can dominate the norm.
        norm_time = F.normalize(torch.tanh(time_prototype).view(K, -1), p=2, dim=1)
        
        # Features: [K, L, D] -> [K, L*D]
        flat_feat = F.normalize(feature_prototype.view(K, -1), p=2, dim=1)
        
        # 2. CREATE UNIFIED CONCEPT FINGERPRINT
        # [K, (LR + L + LD)]
        fingerprints = torch.cat([flat_rel, norm_time, flat_feat], dim=1)
        
        # Normalize the combined fingerprint so dot product = cosine similarity
        fingerprints = F.normalize(fingerprints, p=2, dim=1)
        
        # 3. COMPUTE SIMILARITY MATRIX (S)
        # S_ij = Similarity between Concept i and Concept j
        # Range: [-1, 1]
        similarity_matrix = torch.matmul(fingerprints, fingerprints.t())
        
        # 4. COMPUTE ACTIVATION MASK (M)
        # We want to penalize S_ij only if BOTH i and j are active.
        # activations: [Batch, K]
        # mean_act: [K] (Average activation of each concept across the batch)
        mean_act = activations.mean(dim=0)
        
        # mask_ij = mean_act[i] * mean_act[j]
        # Shape: [K, K]
        activation_mask = torch.outer(mean_act, mean_act)
        
        # 5. WEIGHTED LOSS
        # Target: Identity Matrix (I). We only care about Off-Diagonals.
        identity = torch.eye(K, device=relation_prototype.device)
        
        # Squared Error on Off-Diagonals
        off_diagonal_error = (similarity_matrix - identity) ** 2
        
        # Apply Mask: Overlap is only penalized if concepts are active
        # We detach the mask so we don't discourage activation just to minimize diversity loss.
        weighted_error = off_diagonal_error * activation_mask.detach()
        
        # Sum only off-diagonal elements (Identity spots are 0 error anyway)
        # We divide by K*(K-1) to average over pairs, or just sum for stronger gradients
        loss = weighted_error.sum() / (K * (K - 1) + 1e-8)
        
        return loss
    
    def forward(self, sampled_metapaths):
        # Decode global concepts from latent z
        rel_proto, t_proto, gt_proto, gf_proto, mu_proto, tau, discrete_relations_loss = self.concept_decoder(self.concepts)

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

        div_loss = self.activation_weighted_diversity_loss(
            rel_proto, t_proto, mu_proto, activation_tensor
        )

        pred_logit, logic_info = self.logic_head(activation_tensor)

        w_sparsity = logic_info["rule_weights"].mean()
        v_sparsity = logic_info["clause_weights"].mean()
        sparsity_loss = w_sparsity + v_sparsity


        return pred_logit, {
            "components": components_log, 
            "activations": activation_tensor,
            "sparsity_loss": sparsity_loss,
            "diversity_loss": div_loss,
            "discrete_relations": discrete_relations_loss,
            "logic_debug": logic_info
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

    def soft_discrete_regularisation(self, relation_prototype):
        # relational discretisation with shannon entropy
        epsilon = 1e-10
        entropy = -torch.sum(relation_prototype * torch.log(relation_prototype + epsilon), dim=-1)
        discrete_relations_loss = entropy.mean()

        return discrete_relations_loss


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
        features_flat = self.feature_head(concept_state)
        features = features_flat.view(batch_size, self.L, self.D)
        
        # Keep features as unit vectors for ease of similarity comparison
        features = F.normalize(features, dim=-1)

        # Generate sparsity regularisation terms
        discretisation_regularisation_terms = self.soft_discrete_regularisation(relation_matrix)

        return relation_matrix, time, gamma_time, gamma_feat, features, tau, discretisation_regularisation_terms

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
