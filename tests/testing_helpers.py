
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

from faithfulness_poc import Schema, Concept, Subgraph


# ============================================================
# SECTION 1: Report utilities
# ============================================================
# Functions for writing test results to CSV files for inspection.
# ============================================================

REPORT_DIR = Path(__file__).resolve().parent / "evidence_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, rows: List[Dict[str, Any]]) -> Path:
    """
    Write a list of dictionaries to a CSV file in the report directory.
    """
    path = REPORT_DIR / name
    if not rows:
        # Create empty file if no rows
        path.write_text("")
        return path
    
    # Extract fieldnames from first row
    fieldnames = list(rows[0].keys())
    
    # Write CSV with header and all rows
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


# ============================================================
# SECTION 2: Minimal construction helpers
# ============================================================
# Factory functions for creating test schemas, concepts, and subgraphs
# with controlled parameters.
# ============================================================

def make_schema() -> Schema:
    """
    Create a minimal test schema: ROOT -> A -> B with two relation choices at each hop.
    
    Structure:
        ROOT --(r1 or rx)--> A --(r2 or ry)--> B
    
    Returns:
        Schema object with the above transition rules
    """
    return Schema(
        root_type="ROOT",
        transitions={
            ("ROOT", "r1"): "A",
            ("ROOT", "rx"): "A",
            ("A", "r2"): "B",
            ("A", "ry"): "B",
        },
    )


# Global list of all relation types in the test schema
REL_TYPES = ["r1", "r2", "rx", "ry"]


def concept_template(
    *,
    name: str,
    rel_probs: np.ndarray,
    time_deltas: List[float],
    feat_mu: List[np.ndarray],
    gamma: float,
    tau: float,
) -> Concept:
    """
    Create a Concept with specified parameters.
    
    Args:
        name: Concept name/identifier
        rel_probs: (num_hops, num_rel_types) probability matrix over relations per hop
        time_deltas: Time intervals (list of floats, one per hop)
        feat_mu: Feature centroids (list of arrays, one per hop + root)
        gamma: Sharpness parameter for similarity scoring
        tau: Normalization/threshold parameter for evidence
    
    Returns:
        Concept object
    """
    return Concept(
        name=name,
        rel_types=REL_TYPES,
        rel_probs=np.asarray(rel_probs, dtype=float),
        time_deltas=list(map(float, time_deltas)),
        feat_mu=[np.asarray(x, dtype=float) for x in feat_mu],
        gamma=float(gamma),
        tau=float(tau),
    )


def base_concept_r1r2(*, gamma: float = 5.0, tau: float = 1.0) -> Concept:
    """
    Create a prototype concept: prefers r1 then r2 path.
    
    Structure:
        - Hop 0: takes r1 with prob 1.0
        - Hop 1: takes r2 with prob 1.0
        - Feature centroids: root=[1,0], hop1=[0,1], hop2=[1,1]
        - Time deltas: [2.0, 5.0] seconds
    
    Args:
        gamma: Sharpness parameter (default 5.0)
        tau: Threshold parameter (default 1.0)
    
    Returns:
        Concept object with r1->r2 preference
    """
    rel_probs = np.array([
        [1.0, 0.0, 0.0, 0.0],  # hop 0: r1 with 100%
        [0.0, 1.0, 0.0, 0.0],  # hop 1: r2 with 100%
    ])
    feat_mu = [
        np.array([1.0, 0.0]),   # root centroid
        np.array([0.0, 1.0]),   # hop 1 centroid
        np.array([1.0, 1.0]),   # hop 2 centroid
    ]
    return concept_template(
        name="C_r1r2",
        rel_probs=rel_probs,
        time_deltas=[2.0, 5.0],
        feat_mu=feat_mu,
        gamma=gamma,
        tau=tau,
    )


def base_concept_rxry(*, gamma: float = 5.0, tau: float = 1.0) -> Concept:
    """
    Create an alternative concept: prefers rx then ry path (orthogonal to r1r2).
    
    Structure:
        - Hop 0: takes rx with prob 1.0
        - Hop 1: takes ry with prob 1.0
        - Feature centroids: root=[-1,0], hop1=[0,-1], hop2=[-1,-1]
        - Time deltas: [3.0, 4.0] seconds
    
    Args:
        gamma: Sharpness parameter (default 5.0)
        tau: Threshold parameter (default 1.0)
    
    Returns:
        Concept object with rx->ry preference
    """
    rel_probs = np.array([
        [0.0, 0.0, 1.0, 0.0],  # hop 0: rx with 100%
        [0.0, 0.0, 0.0, 1.0],  # hop 1: ry with 100%
    ])
    feat_mu = [
        np.array([-1.0, 0.0]),  # root centroid
        np.array([0.0, -1.0]),  # hop 1 centroid
        np.array([-1.0, -1.0]), # hop 2 centroid
    ]
    return concept_template(
        name="C_rxry",
        rel_probs=rel_probs,
        time_deltas=[3.0, 4.0],
        feat_mu=feat_mu,
        gamma=gamma,
        tau=tau,
    )


def build_graph(
    sch: Schema,
    *,
    root_time: float,
    root_feat: np.ndarray,
    planted: List[tuple],
) -> Subgraph:
    """
    Build a subgraph by creating a root and planting evidence instances.
    
    Args:
        sch: Schema defining the graph structure
        root_time: Timestamp for the root node
        root_feat: Feature vector for the root node
        planted: List of (Concept, kwargs_dict) pairs where kwargs_dict is passed
                 to add_evidence() (typically includes rng, instances, rel_pick, etc.)
    
    Returns:
        Subgraph object with all planted evidence
    """
    # Initialize empty subgraph with schema
    g = Subgraph(schema=sch)
    
    # Create root node with time and features
    g.create_root(time=float(root_time), feat=np.asarray(root_feat, dtype=float))
    
    # Plant evidence instances for each concept
    for c, kw in planted:
        # `kw` is passed directly to `Subgraph.add_evidence` and may include
        # keys such as `rng`, `instances`, `time_noise_std`, `feat_noise_std`,
        # and `rel_pick`. This allows each concept to be planted under
        # different sampling/noise conditions in the same graph.
        g.add_evidence(c, **kw)
    
    return g


def score(g: Subgraph, c: Concept) -> float:
    """
    Compute the evidence score for a concept against a subgraph.
    
    Args:
        g: Subgraph to score
        c: Concept to score against
    
    Returns:
        Evidence score (float in [0, 1])
    """
    return float(g.evidence_score(c))


def nonincreasing(x: np.ndarray, atol: float = 1e-12) -> bool:
    """
    Check if an array is non-increasing (allowing small numerical tolerance).
    
    Args:
        x: 1D array to check
        atol: Absolute tolerance for comparison
    
    Returns:
        True if x[i] >= x[i+1] for all i (within tolerance)
    """
    return bool(np.all(x[1:] <= x[:-1] + atol))

