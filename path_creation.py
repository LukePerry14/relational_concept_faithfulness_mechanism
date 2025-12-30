from custom_dataclasses import (
    Concept,
    MetaPath
    
)
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


# Get embeddings for target words
def get_bert_embeddings(words):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    
    # Process each word individually
    embeddings = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.hidden_states[-1][0, 1:-1, :]
        # Average them along the sequence dimension
        embedding = torch.mean(token_embeddings, dim=0).numpy()
        embeddings.append(embedding)
    
    return np.array(embeddings)
    
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
    
def build_graph_evidence_BERT():
    
    embs = get_bert_embeddings(["adult", "[CLS]", "diapers", "nappies", "baby newsletter"])
    adult_emd = embs[0]
    irrelevant_emd = embs[0]
    diapers_emd = embs[0]
    nappies_emd = embs[0]
    baby_newsletter_emd = embs[0]

    recent_diaper_purchase = MetaPath(
        path_name="recent_nappy_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, diapers_emd])
    )
    distant_nappy_purchase = MetaPath(
        path_name="distant_nappy_purchase",
        node_types=["orders", "products"],
        node_times=[11, 500],
        node_features=np.asarray([irrelevant_emd, diapers_emd])
    )
    recent_nappy_purchase = MetaPath(
        path_name="recent_nappy_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, nappies_emd])
    )
    recent_irrelevant_purchase = MetaPath(
        path_name="recent_irrelevant_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, irrelevant_emd])
    )
    subscription_to_baby_newsletter = MetaPath(
        path_name = "subscription_to_baby_newsletter",
        node_types=["subscriptions", "subscriptionProducts", "products"],
        node_times=[10, 500, 500],
        node_features=np.asarray([irrelevant_emd, irrelevant_emd, baby_newsletter_emd])
    )
    
    return [recent_nappy_purchase, distant_nappy_purchase, recent_diaper_purchase, recent_irrelevant_purchase, subscription_to_baby_newsletter]
        
def build_graph_concept_with_BERT(ordered_node_types, learnable_gamma_mu=False):
    embs = get_bert_embeddings(["parent", "[CLS]", "diapers"])
    parent_embedding = embs[0]
    uninformative_embedding = embs[1]
    diapers_embedding = embs[2]


    buys_nappies_mu = np.asarray([
        parent_embedding,
        uninformative_embedding,
        diapers_embedding,
        uninformative_embedding
    ])

    buys_nappies_mu_gamma = np.asarray([0.5, float('inf'), 0.5, float('inf')])

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



def build_relevant_paths():
    embs = get_bert_embeddings(["[PAD]", "diapers", "nappies", "baby newsletter"])
    irrelevant_emd = embs[0]
    diapers_emd = embs[1]
    nappies_emd = embs[2]
    baby_newsletter_emd = embs[3]

    recent_diaper_purchase = MetaPath(
        path_name="recent_diaper_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, diapers_emd])
    )
    recentish_diaper_purchase = MetaPath(
        path_name="recentish_diaper_purchase",
        node_types=["orders", "products"],
        node_times=[11, 500],
        node_features=np.asarray([irrelevant_emd, diapers_emd])
    )
    recent_nappy_purchase = MetaPath(
        path_name="recent_nappy_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, nappies_emd])
    )
    recent_irrelevant_purchase = MetaPath(
        path_name="recent_irrelevant_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, irrelevant_emd])
    )
    subscription_to_baby_newsletter = MetaPath(
        path_name = "subscription_to_baby_newsletter",
        node_types=["subscriptions", "subscriptionProducts", "products"],
        node_times=[10, 500, 500],
        node_features=np.asarray([irrelevant_emd, irrelevant_emd, baby_newsletter_emd])
    )
    
    return [recent_nappy_purchase, recentish_diaper_purchase, recent_diaper_purchase, recent_irrelevant_purchase, subscription_to_baby_newsletter]


def build_distant_paths():
    # We use tokens that are semantically unrelated to "diapers" or "baby"
    # to test if the similarity threshold (gamma) and tau filter them out.
    embs = get_bert_embeddings(["[PAD]", "coffee", "power tools", "car tires"])
    irrelevant_emd = embs[0]
    coffee_emd = embs[1]
    tools_emd = embs[2]
    tires_emd = embs[3]

    # Structure: Same ["orders", "products"], but the product is "coffee"
    distant_irrelevant_purchase_1 = MetaPath(
        path_name="coffee_purchase",
        node_types=["orders", "products"],
        node_times=[1, 500],
        node_features=np.asarray([irrelevant_emd, coffee_emd])
    )

    # Structure: Same ["orders", "products"], but the product is "power tools"
    distant_irrelevant_purchase_2 = MetaPath(
        path_name="tools_purchase",
        node_types=["orders", "products"],
        node_times=[5, 500],
        node_features=np.asarray([irrelevant_emd, tools_emd])
    )

    # Structure: Same ["subscriptions", "subscriptionProducts", "products"], 
    # but for "car tires" instead of a "baby newsletter"
    irrelevant_subscription = MetaPath(
        path_name = "tire_subscription",
        node_types=["subscriptions", "subscriptionProducts", "products"],
        node_times=[10, 500, 500],
        node_features=np.asarray([irrelevant_emd, irrelevant_emd, tires_emd])
    )
    
    # Return a list containing these semantically distant paths
    return [distant_irrelevant_purchase_1, distant_irrelevant_purchase_2, irrelevant_subscription]
