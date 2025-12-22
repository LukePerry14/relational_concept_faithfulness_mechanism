import csv
from pathlib import Path
from typing import List, Dict, Any, Union, Sequence

import numpy as np
import pytest

from faithfulness_poc import Schema, Concept, Subgraph

# ============================================================
# SECTION 1: Report utilities
# ============================================================

REPORT_DIR = Path(__file__).resolve().parent / "evidence_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, rows: List[Dict[str, Any]]) -> Path:
    """
    Write a list of dictionaries to a CSV file in the report directory.
    """
    path = REPORT_DIR / name
    if not rows:
        path.write_text("")
        return path

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


# ============================================================
# SECTION 2: Minimal construction helpers
# ============================================================

def make_schema() -> Schema:
    """
    Minimal test schema: ROOT -> A -> B with two relation choices at each hop.

        ROOT --(r1 or rx)--> A --(r2 or ry)--> B
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


# In the new mechanism, prototypes are over node types, not relations.
# We keep a simple ordered node-type vocabulary for the non-root hops.
ORDERED_NODE_TYPES = ["A", "B"]


def _build_time_vector(time_deltas: List[float]) -> np.ndarray:
    """
    Convert per-hop offsets (relative to root) into the prototype time vector.

    New convention:
      - t[0] = absolute time of root prototype (we take 0.0 in tests)
      - t[i] = offset-from-root for hop i, i >= 1
    """
    L = len(time_deltas)
    t = np.zeros(L + 1, dtype=float)
    t[1:] = np.asarray(time_deltas, dtype=float)
    return t


def _build_gamma_vector(L: int, gamma: float) -> np.ndarray:
    """
    Use the same scalar gamma for each hop (including root) for simplicity.
    """
    return np.full(L + 1, float(gamma), dtype=float)


def concept_template(
    *,
    name: str,
    time_deltas: List[float],
    feat_mu: List[np.ndarray],
    gamma: float,
    tau: float,
    ordered_node_types: List[str] = None,
) -> Concept:
    """
    Create a Concept with specified parameters in the *new* format.

    Args:
        name:
            Concept name/identifier.

        time_deltas:
            Per-hop offsets relative to the seed node for hops 1..L.

        feat_mu:
            List of feature centroids, one for root + each hop, length L+1.
            feat_mu[0] is the root feature; feat_mu[i] is hop i feature.

        gamma:
            Window size parameter used for both time and features in tests.

        tau:
            Saturation parameter for evidence (E = M / (M + tau)).

        ordered_node_types:
            Node-type vocabulary for hops (excluding root). Defaults to
            ["A", "B"] for the 2-hop test concepts.

    Returns:
        Concept object usable with the new evidence mechanism.
    """
    if ordered_node_types is None:
        ordered_node_types = ORDERED_NODE_TYPES

    L = len(time_deltas)
    if len(feat_mu) != L + 1:
        raise ValueError(
            f"feat_mu must have length L+1={L+1}, got {len(feat_mu)}"
        )
    if L > len(ordered_node_types):
        raise ValueError(
            f"Need at least {L} node types for L hops, got {len(ordered_node_types)}"
        )

    # Relational prototype P over node types (+ NULL) at each hop.
    # For tests, we use a 1-hot pattern: hop i prefers ordered_node_types[i].
    K = len(ordered_node_types) + 1  # + NULL_TOKEN
    P = np.zeros((L, K), dtype=float)
    for hop in range(L):
        P[hop, hop] = 1.0  # hop 0 -> A, hop 1 -> B, etc.

    # Time prototype and windows
    t = _build_time_vector(time_deltas)
    gamma_t = _build_gamma_vector(L, gamma)

    # Feature prototype and windows
    mu = np.stack([np.asarray(x, dtype=float) for x in feat_mu], axis=0)
    gamma_mu = _build_gamma_vector(L, gamma)

    return Concept(
        name=name,
        ordered_node_types=list(ordered_node_types),
        P=P,
        t=t,
        gamma_t=gamma_t,
        mu=mu,
        gamma_mu=gamma_mu,
        tau=float(tau),
        # keep defaults for k_time, k_feat (0.1) from faithfulness_poc
    )


def base_concept_r1r2(*, gamma: float = 5.0, tau: float = 1.0) -> Concept:
    """
    Prototype concept with path structure corresponding to ROOT -> A -> B.

    In the original relation-based version this was "r1 then r2", but in the
    new mechanism prototypes are over node types. Distinction between r1/r2
    and rx/ry is now carried by time/features instead.
    """
    time_deltas = [2.0, 5.0]  # offsets from root time
    feat_mu = [
        np.array([1.0, 0.0]),   # root centroid
        np.array([0.0, 1.0]),   # hop 1 centroid
        np.array([1.0, 1.0]),   # hop 2 centroid
    ]
    return concept_template(
        name="C_r1r2",
        time_deltas=time_deltas,
        feat_mu=feat_mu,
        gamma=gamma,
        tau=tau,
        ordered_node_types=ORDERED_NODE_TYPES,
    )


def base_concept_rxry(*, gamma: float = 5.0, tau: float = 1.0) -> Concept:
    """
    Alternative prototype concept with different time/feature semantics.

    Originally this corresponded to rx->ry. Here it shares the same node-type
    structure (A then B) but uses different time_deltas and feature centroids,
    making it "orthogonal" to base_concept_r1r2 via time/feature components.
    """
    time_deltas = [3.0, 4.0]  # offsets from root time
    feat_mu = [
        np.array([-1.0, 0.0]),  # root centroid
        np.array([0.0, -1.0]),  # hop 1 centroid
        np.array([-1.0, -1.0]), # hop 2 centroid
    ]
    return concept_template(
        name="C_rxry",
        time_deltas=time_deltas,
        feat_mu=feat_mu,
        gamma=gamma,
        tau=tau,
        ordered_node_types=ORDERED_NODE_TYPES,
    )


# ============================================================
# SECTION 3: Graph construction and scoring helpers
# ============================================================

def _as_per_hop(
    value: Union[float, Sequence[float]],
    L: int,
) -> np.ndarray:
    """
    Normalize scalar-or-sequence noise specs to a length-L array.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        if arr.shape[0] != L:
            raise ValueError(f"Expected length-{L} noise sequence, got {arr.shape[0]}")
        return arr
    return np.full(L, float(value), dtype=float)


def _subgraph_add_evidence(
    self: Subgraph,
    concept: Concept,
    *,
    rng: np.random.Generator,
    instances: int,
    relation_pick: str = "mode",
    time_noise_std: Union[float, Sequence[float]] = 0.0,
    feat_noise_std: Union[float, Sequence[float]] = 0.0,
    **_ignored,
) -> None:
    """
    Monkey-patched helper for tests: plant `instances` noisy paths consistent
    with a given concept into the subgraph.

    Semantics are analogous to the old relation-based `add_evidence`, but now
    work via node-type prototypes:

      - Each instance is a walk of length L = concept.L()
      - At hop h, we create a node of type ordered_node_types[h]
      - Time is sampled around concept.t[h+1] (offset-from-root)
      - Features are sampled around concept.mu[h+1]
      - We choose any schema-consistent relation from current node-type to
        the required next node-type (relation labels themselves do not affect
        scoring in the new mechanism, only node types do).

    Parameters beyond those used (e.g. rel_pick) are accepted for backwards
    compatibility with existing tests but are ignored for scoring.
    """
    if self.root is None:
        raise ValueError("Root must be created before adding evidence.")

    L = concept.L()
    if L == 0:
        return

    # Normalize noise specification to per-hop arrays
    time_noise = _as_per_hop(time_noise_std, L)
    feat_noise = _as_per_hop(feat_noise_std, L)

    root_id = self.root
    root_time = self.nodes[root_id].time

    for _ in range(int(instances)):
        current = root_id

        for hop in range(L):
            src_type = self.nodes[current].node_type
            target_type = concept.ordered_node_types[hop]

            # Find schema transitions from src_type to target_type
            candidates = [
                rel
                for (s_type, rel), dst_type in self.schema.transitions.items()
                if s_type == src_type and dst_type == target_type
            ]
            if not candidates:
                # Can't extend further under schema; path terminates early
                break

            # Choose a relation deterministically; relation_pick is ignored for scoring.
            relation = sorted(candidates)[0]

            # Time: root absolute time + (prototype offset + noise)
            t_offset = concept.t[hop + 1] + time_noise[hop] * rng.normal()
            node_time = float(root_time + t_offset)

            # Features: prototype + Gaussian noise
            base_feat = concept.mu[hop + 1]
            feat_vec = base_feat + feat_noise[hop] * rng.normal(size=base_feat.shape)

            # Create node and edge
            node_id = self._new_node(target_type, node_time, feat_vec)
            self._add_edge(current, node_id, relation)
            current = node_id


# Attach the helper to Subgraph so tests can call `g.add_evidence(...)`
if not hasattr(Subgraph, "add_evidence"):
    setattr(Subgraph, "add_evidence", _subgraph_add_evidence)


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
        sch:
            Schema defining the graph structure.

        root_time:
            Timestamp for the root node (absolute time).

        root_feat:
            Feature vector for the root node.

        planted:
            List of (Concept, kwargs_dict) pairs where kwargs_dict is passed
            to Subgraph.add_evidence (typically includes rng, instances,
            time_noise_std, feat_noise_std, relation_pick, etc.).
    """
    g = Subgraph(schema=sch)
    g.create_root(time=float(root_time), feat=np.asarray(root_feat, dtype=float))

    for c, kw in planted:
        g.add_evidence(c, **kw)

    return g


def score(g: Subgraph, c: Concept) -> float:
    """
    Compute the evidence score for a concept against a subgraph.
    """
    return float(g.evidence_score(c))


def nonincreasing(x: np.ndarray, atol: float = 1e-12) -> bool:
    """
    Check if an array is non-increasing (within a small numerical tolerance).
    """
    x = np.asarray(x, dtype=float)
    return bool(np.all(x[1:] <= x[:-1] + atol))
