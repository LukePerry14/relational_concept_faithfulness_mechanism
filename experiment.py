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

def build_buys_nappies_concept(ordered_node_types):
    buys_nappies_relations = np.asarray([
        [0.98, 0.01, 0.0, 0.0, 0.01],
        [0.0, 0.0, 0.01, 0.98, 0.01],
        [0.0, 0.0, 0.0, 0.01, 0.99]
    ])
    buys_nappies_time = np.asarray([float('inf'), float('inf'), float('inf'), float('inf')])
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
    
    ordered_node_types = ["order", "subscriptions", "subscriptionProducts", "products"]
    
    reachability_mask = amazon_schema.reachability_mask(3, ordered_node_types)
    
    # Build Subgraph (4 metapath types in pdf)
    amazon_has_baby = Subgraph(
        schema=amazon_schema
    )
    amazon_has_baby.create_root(time=100, feat=np.asanyarray([-0.5,-0.5]))
    
    Example_paths = build_graph_evidence()
    metapaths = build_graph_evidence()
    
    amazon_has_baby.add_evidence(metapaths)
    
    print(amazon_has_baby.sample_paths(3, 4))
    exit()
    # Build buying nappies concept
    buys_nappies_concept = build_buys_nappies_concept(ordered_node_types)
    
    amazon_has_baby.evidence_score(buys_nappies_concept)
    # Build subscribes to newsletter concept
    
    # Test concepts
    