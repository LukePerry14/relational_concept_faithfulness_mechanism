# Initial testing to ensure new version accuracy

# Add node feature embeddings using BERT

# experiment with taus

# add in gradient descent

from custom_dataclasses import (
    Schema,
    Concept,
    MetaPath
    
)
import numpy as np
from faithfulness_poc import Subgraph
from transformers import AutoTokenizer, AutoModel
import torch
from embedding_viz import compare_embeddings

def build_buys_nappies_concept(ordered_node_types):
    buys_nappies_relations = np.asarray([
        [0.98, 0.01, 0.0, 0.0, 0.01],
        [0.0, 0.0, 0.01, 0.98, 0.01],
        [0.0, 0.0, 0.0, 0.01, 0.99]
    ])
    buys_nappies_time = np.asarray([float('inf'), 0, float('inf'), float('inf')])
    buys_nappies_time_gama = np.asarray([float('inf'), 14, float('inf'), float('inf')])
    
    buys_nappies_mu = np.asarray([[-1.0, -1.0], [float('inf'), float('inf')], [0.0, 0.9], [float('inf'), float('inf')]])
    buys_nappies_mu_gamma = np.asarray([2, float('inf'), 0.5, float('inf')])

    buys_nappies_concept = Concept(
        name="buys nappies",
        ordered_node_types=ordered_node_types,
        P = buys_nappies_relations,
        t = buys_nappies_time,
        gamma_t= buys_nappies_time_gama,
        mu = buys_nappies_mu,
        gamma_mu= buys_nappies_mu_gamma,
        tau=0.5,
        k_time = 0.1,
        k_feat=0.1
    )
    
    return buys_nappies_concept

def build_graph_evidence():
    recent_nappy_purchase = MetaPath(
        path_name="recent_nappy_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([[float('inf'), float('inf')], [0.0,1.0]])
    )
    distant_nappy_purchase = MetaPath(
        path_name="distant_nappy_purchase",
        node_types=["orders", "products"],
        node_times=[11, 500],
        node_features=np.asarray([[float('inf'), float('inf')], [0.0,1.0]])
    )
    recent_irrelevant_purchase = MetaPath(
        path_name="recent_irrelevant_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([[float('inf'), float('inf')], [1.0,0.0]])
    )
    subscription_to_baby_newsletter = MetaPath(
        path_name = "subscription_to_baby_newsletter",
        node_types=["subscriptions", "subscriptionProducts", "products"],
        node_times=[10, 500, 500],
        node_features=np.asarray([[float('inf'), float('inf')], [float('inf'), float('inf')], [0.1,0.8]])
    )
    
    return [recent_nappy_purchase, distant_nappy_purchase, recent_irrelevant_purchase, subscription_to_baby_newsletter]
    

def build_graph_evidence_with_BERT(ordered_node_types):
    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    
    # Get embeddings for target words
    def get_bert_embedding(word):
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.hidden_states[-1][0, 1, :].numpy()  # [CLS] token embedding
    
    parent_embedding = get_bert_embedding("parent")
    diapers_embedding = get_bert_embedding("diapers")
    nappies_embedding = get_bert_embedding("nappies")
    baby_newsletter_embedding = get_bert_embedding("baby newsletter")
    irrelevant_embedding = get_bert_embedding("irrelevant")
    uninformative_embedding = get_bert_embedding("uninformative")

    
    compare_embeddings(
        embeddings=[parent_embedding, diapers_embedding, nappies_embedding, baby_newsletter_embedding, uninformative_embedding, irrelevant_embedding],
        labels=["parent", "diapers", "nappies", "baby newsletter", "[PAD]", "irrelevant"],
        show_plots=True
    )
    # print("Parent Embedding:", parent_embedding)
    # print("Diapers Embedding:", diapers_embedding)
    # print("Uninformative Embedding:", uninformative_embedding)
    exit()
    buys_nappies_mu = np.asarray([
        parent_embedding,
        uninformative_embedding,
        diapers_embedding,
        uninformative_embedding
    ])
    buys_nappies_mu_gamma = np.asarray([2, float('inf'), 0.5, float('inf')])

    buys_nappies_concept = Concept(
        name="buys nappies",
        ordered_node_types=ordered_node_types,
        P=np.asarray([[0.98, 0.01, 0.0, 0.0, 0.01], [0.0, 0.0, 0.01, 0.98, 0.01], [0.0, 0.0, 0.0, 0.01, 0.99]]),
        t=np.asarray([float('inf'), 0, float('inf'), float('inf')]),
        gamma_t=np.asarray([float('inf'), 14, float('inf'), float('inf')]),
        mu=buys_nappies_mu,
        gamma_mu=buys_nappies_mu_gamma,
        tau=0.5,
        k_time=0.1,
        k_feat=0.1
    )
    
    return buys_nappies_concept



if __name__ == "__main__":
    # Build Schema (Customer, Review, Product)
    amazon_schema = Schema(
        root_type="users",
        transitions= {
            "users": ["orders", "subscriptions"],
            "orders": ["products"],
            "subscriptions": ["subscriptionProducts"],
            "subscriptionProducts": ["products"]
        }
        )
    
    ordered_node_types = ["orders", "subscriptions", "subscriptionProducts", "products"]
    
    # reachability_mask = amazon_schema.reachability_mask(3, ordered_node_types)

    # Build Subgraph (4 metapath types in pdf)
    amazon_has_baby = Subgraph(
        schema=amazon_schema
    )
    amazon_has_baby.create_root(time=100, feat=np.asanyarray([-0.5,-0.5]))
    
    metapaths = build_graph_evidence()
    
    amazon_has_baby.add_evidence(metapaths)
    
    # amazon_has_baby.visualize_subgraph_plotly()
    input("Press Enter to Continue...")
    # print(amazon_has_baby.sample_paths(3, 4))
    # Build buying nappies concept
    buys_nappies_concept = build_graph_evidence_with_BERT(ordered_node_types)
    
    total_evidence = amazon_has_baby.evidence_score(buys_nappies_concept)
    print(f"Total Evidence for 'buys nappies' concept: {total_evidence:.4f}")
    # Build subscribes to newsletter concept
    
    # Test concepts
    

# absolute time and relative time both important