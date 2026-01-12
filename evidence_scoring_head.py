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
    
    def activation_weighted_diversity_loss(self, relation_prototype, time_prototype, feature_prototype, activations, activation_method="max", path_normalisation=False):
        """
        Calculate the orthogonality between decoded concept prototypes.
        Scale this loss by concept co-activation in the logic head to prevent garbage concepts from becoming blocking
        """
        num_concepts = relation_prototype.shape[0]
        
        # Flatten and normalise meta-path concept components
        flat_rel = F.normalize(relation_prototype.view(num_concepts, -1), p=2, dim=1)
        norm_time = F.normalize(torch.sigmoid(time_prototype).view(num_concepts, -1), p=2, dim=1)
        flat_feat = F.normalize(feature_prototype.view(num_concepts, -1), p=2, dim=1)
        
        # Flattened concept vector
        concat_vectors = F.normalize(torch.cat([flat_rel, norm_time, flat_feat], dim=1), p=2, dim=1)
                
        # Similarity matrix of flattened vectors
        similarity_matrix = torch.matmul(concat_vectors, concat_vectors.t())
        
        # Only penalise concepts which are being used
        if activation_method == "max":
            max_activations = torch.max(activations, 0).values
        elif activation_method == "mean":
            max_activations = torch.mean(max_activations, dim=0)
        
        # Use outer product to produce weights for orthogonality only if both concepts are non-garbage
        activation_mask = torch.outer(max_activations, max_activations)
        
        # Determine vector similarity using MSE style calculation (ignore simlarity to self)
        identity = torch.eye(num_concepts, device=relation_prototype.device)
        off_diagonal_error = torch.abs(similarity_matrix - identity)
        
        # Mask these values by max activations of concepts
        weighted_error = off_diagonal_error * activation_mask.detach()
        
        # Sum off diagonal "errors"
        loss = weighted_error.sum()
        
        if path_normalisation: # Make error invariant to number of concepts by averaging over concept pairs, 
            loss /= (num_concepts * (num_concepts - 1) + 1e-8)
        
        return loss
    
    def forward(self, sampled_metapaths):
        # Decode global concepts from latent z
        rel_proto, t_proto, gt_proto, gf_proto, mu_proto, tau, regularisation_terms = self.concept_decoder(self.concepts)

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
            "logic_debug": logic_info,
            **regularisation_terms
        }

class ConceptDecoder(nn.Module):
    """Autoregressive style decoder to decode human interpretable concepts from latents"""
    
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
        
        self.hidden_dim = concept_dim * 2

        # Decoders
        self.conditional_GRU_cell = nn.GRUCell(self.hidden_dim + self.R, self.hidden_dim) # Decode hop level conditional state
        self.relation_head = nn.Linear(self.hidden_dim, self.R) # decode next relation
        self.time_head = nn.Linear(self.hidden_dim, 2) # Decode Time and time gamma
        self.feature_head = nn.Linear(self.hidden_dim, self.D + 1) # decode feature vector and feature gamma
        self.tau_head = nn.Linear(self.hidden_dim, 1) # decode concept level tau
        
        # Weight initialisation
        self.trunk.apply(self._init_trunk_weights)
        self.conditional_GRU_cell.apply(self._init_head_weights)
        self.relation_head.apply(self._init_head_weights)
        self.time_head.apply(self._init_head_weights)
        self.feature_head.apply(self._init_head_weights)
        self.tau_head.apply(self._init_head_weights)

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
        num_concepts = concept_z.shape[0]
        
        # 1) Decode concepts through trunk
        concept_state = self.trunk(concept_z)
        
        # Use temporary hidden state tracking for proper RNN unraveling
        hidden = concept_state

        # 2) Build valid relations
        adj = self.adj.to(concept_z.device)
        relation_probs = [torch.zeros(num_concepts, self.R).to(concept_z.device)]
        relation_probs[0][:, 0] = 1.0

        # Decode global tau
        tau = F.softplus(self.tau_head(hidden)) + self.min_tau

        features, gamma_feats, times, gamma_times = [], [], [], []

        # Use RNN to unravel concept autoregressively
        for hop in range(self.L):
            conditonal_concept_state = self.conditional_GRU_cell(torch.cat([concept_state, relation_probs[-1]], dim=-1), hidden)

            curr_features, curr_gamma_feats = torch.split(self.feature_head(conditonal_concept_state), [self.D, 1], dim=1)

            curr_features = F.normalize(curr_features.view(num_concepts, self.D), dim=-1)
            curr_gamma_feats = F.sigmoid(curr_gamma_feats)

            features.append(curr_features)
            gamma_feats.append(curr_gamma_feats)

            curr_times, curr_gamma_times = torch.split(self.time_head(conditonal_concept_state), [1, 1], dim=1)
            
            if hop == 0:
                time_delta = torch.zeros_like(curr_times)
            else:
                time_delta = torch.exp(curr_times)
            curr_gamma_times = torch.exp(curr_gamma_times) + self.gamma_floor
            
            
            times.append(time_delta)
            gamma_times.append(curr_gamma_times)


            next_rel_prob_logits = self.relation_head(conditonal_concept_state)
            reachable = torch.matmul(relation_probs[-1], adj)
            masked_logits = next_rel_prob_logits.masked_fill(reachable == 0, -1e9)

            chosen_relation = F.gumbel_softmax(masked_logits, tau=5, hard=False)

            if hop != (self.L-1):
                relation_probs.append(chosen_relation)

                hidden = conditonal_concept_state
            

        # Stack data together
        relation_matrix = torch.stack(relation_probs, dim = 1)
        stacked_time = torch.cumsum(torch.stack(times, dim=1), dim=1).squeeze(-1)
        stacked_gamma_time = torch.stack(gamma_times, dim=1).squeeze(-1)
        stacked_features = torch.stack(features, dim=1)
        stacked_gamma_features = torch.stack(gamma_feats, dim =1).squeeze(-1)

        # Generate regularisation terms
        gamma_time_penalty = torch.log(stacked_gamma_time + 1.0).mean()
        gamma_feature_penalty = torch.log(stacked_gamma_features + 1.0).mean()
        discrete_loss = self.soft_discrete_regularisation(relation_matrix)

        return relation_matrix, stacked_time, stacked_gamma_time, stacked_gamma_features, stacked_features, tau, {"discrete_loss": discrete_loss, "gamma_time_penalty": gamma_time_penalty, "gamma_feature_penalty": gamma_feature_penalty}

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
        Gumbel-Softmax based
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
        # 1. Masking
        mask = torch.isfinite(batch_times)
        safe_batch_times = torch.where(mask, batch_times, torch.zeros_like(batch_times))

        # 2. Absolute Distance (The Gravity Source)
        dist_to_center = torch.abs(safe_batch_times - prototype_time)
        
        # 3. The Hinge (The Constraint)
        excess_dist = dist_to_center - gamma_time
        constraint_penalty = F.softplus(excess_dist, beta=5.0)
        
        # 4. Interior Gravity (The Centering Fix)
        # Adds a tiny pull towards the center even for points inside the box.
        # This prevents the prototype from drifting aimlessly within the valid window.
        gravity_penalty = 0.01 * dist_to_center 
        
        # 5. Combined Penalty
        # Constraint (Hard Boundaries) + Gravity (Soft Centering)
        total_penalty = constraint_penalty + gravity_penalty
        
        # 6. Apply Relational Sharpness
        weighted_penalty = self.relational_sharpness * total_penalty
        
        # 7. Reapply Mask and Sum
        final_penalty = torch.where(mask, weighted_penalty, torch.zeros_like(weighted_penalty))
        path_penalty = torch.sum(final_penalty, dim=-1)
        
        return -1 * path_penalty
    
    def _feature_similarity_log(self, prototype_features, gamma_features, batch_features):
            """
            Calculates Log-Similarity for features usine cosine similarity on unit sphere representations
            similar to Von Mises-Fisher Distribution
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

        final_scores = torch.clamp((total_log_evidence - log_tau), min=-20, max=20)
        # return evidence scores for each concept
        return final_scores, {
            "rel": log_relational_similarity,
            "time": log_temporal_similarity,
            "feat": log_feature_similarity,
            "tau": concept_prototype.tau
        }
