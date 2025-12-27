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
        # Extract embedding for the word token (position 1, after [CLS])
        embedding = outputs.hidden_states[-1][0, 1, :].numpy()
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
    
    
def build_graph_concept_with_BERT(ordered_node_types):
    

    embs = get_bert_embeddings(["parent", "[CLS]", "diapers"])
    parent_embedding = embs[0]
    uninformative_embedding = embs[1]
    diapers_embedding = embs[2]
    
    
    # compare_embeddings(
    #     embeddings=[parent_embedding, uninformative_embedding, diapers_embedding],
    #     labels=["parent", "irrelevant", "diapers"],
    #     show_plots=True
    # )

    buys_nappies_mu = np.asarray([
        parent_embedding,
        uninformative_embedding,
        diapers_embedding,
        uninformative_embedding
    ])
    # 7.35680521e-01, -5.22486866e-03,  2.59613812e-01, (proto)
    # -1.19801593e+00, -1.59482241e-01, -5.89826703e-01 (sample)
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


def lambda_regression_paths():
    
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
    
    return [[recent_nappy_purchase, distant_nappy_purchase, recent_diaper_purchase, recent_irrelevant_purchase, subscription_to_baby_newsletter],[]]
    


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
    amazon_has_baby.create_root(time=100, feat=np.asanyarray(get_bert_embeddings(["adult"])[0]))
    # BERT_metapaths = build_graph_evidence_BERT()
    # amazon_has_baby.add_evidence(BERT_metapaths)


    amazon_no_baby = Subgraph(
        schema=amazon_schema
    )
    amazon_no_baby.create_root(time=300, feat=np.asanyarray(get_bert_embeddings(["child"])[0]))
    
    # metapaths = build_graph_evidence()
    
    # amazon_has_baby.add_evidence(metapaths)
    
    # amazon_has_baby.visualize_subgraph_plotly()

    # Build buying nappies concept

    
    buys_nappies_concept = build_graph_concept_with_BERT(ordered_node_types)
    total_evidence = amazon_has_baby.evidence_score(buys_nappies_concept)
    print(f"Total Evidence for 'buys nappies' concept: {total_evidence:.4f}")
    # Build subscribes to newsletter concept
    
    # Test concepts
    


# regress lambda based on toy dataset

# multiple concepts

# sparse concepts

# full toy dataset

# lamvda + concepts

# sparse concepts

# real data performance

# competitive performance

    """
    
    keep track of:
    
    - absolute time and relative time both important - resolved?

    """