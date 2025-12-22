# ============================================================
# Test Suite: Faithfulness Evidence Score Sensitivity Analysis
# ============================================================
# This module contains characterization-oriented tests for the evidence-score
# sensitivity metric described in Strategy_discussion.pdf (Eq. 8–13).
#
# NOTE (updated for node-type prototypes + new Concept API):
#   - Relational prototypes are now over node types via Concept.P
#   - Time prototypes are in Concept.t (root absolute time + offsets)
#   - Feature prototypes are in Concept.mu
#   - Kernel windows are Concept.gamma_t and Concept.gamma_mu
#
# Purpose:
#   - Probe the expressiveness/sensitivity of the evidence scoring function
#   - Sweep controlled parameters (time, features, relations, noise)
#   - Record results to CSV for inspection and analysis
#   - Verify monotonicity properties (sanity checks only)
#
# Outputs: CSV files written to <this file's dir>/evidence_reports/
#
# Run with: pytest -q -s
# ============================================================

import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest

from faithfulness_poc import Schema, Concept, Subgraph
from testing_helpers import *


# ============================================================
# SECTION 1: Smoke tests (sanity checks only)
# ============================================================

def test_smoke_concept_normalizes_rel_probs_rows():
    """
    Verify that concept relational prototype rows (over node types) sum to 1.0.
    """
    c = base_concept_r1r2()
    rs = c.P.sum(axis=1)
    assert np.allclose(rs, 1.0)


def test_smoke_evidence_in_range():
    """
    Verify that evidence score is always in valid range [0, 1].
    """
    sch = make_schema()
    c = base_concept_r1r2(gamma=3.0, tau=1.0)

    # Build graph with one perfect instance of the concept
    g = build_graph(
        sch,
        root_time=float(c.t[0]),
        root_feat=c.mu[0],
        planted=[(c, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
    )

    # Score the concept against itself
    E = score(g, c)

    # Check range
    assert 0.0 <= E <= 1.0


# ============================================================
# SECTION 2: Sensitivity tests - Time component
# ============================================================

def test_sensitivity_time_similarity_sweep():
    """
    Create single-path subgraphs with time prototypes progressively further
    from the ground truth and measure evidence score dropoff.

    Hypothesis: As temporal mismatch increases, evidence should decrease.
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=5.0, tau=1.0)

    # Sweep time offsets applied to all non-root hops
    offsets = np.linspace(0.0, 4.0, 9)
    rows: List[Dict[str, Any]] = []

    for off in offsets:
        # Create concept with shifted time prototype (keep everything else fixed)
        t_gen = C.t.copy()
        t_gen[1:] = C.t[1:] + off

        C_gen = Concept(
            name=f"C_gen_time_off_{off:.2f}",
            ordered_node_types=C.ordered_node_types,
            P=C.P.copy(),
            t=t_gen,
            gamma_t=C.gamma_t.copy(),
            mu=C.mu.copy(),
            gamma_mu=C.gamma_mu.copy(),
            tau=C.tau,
            k_time=C.k_time,
            k_feat=C.k_feat,
        )

        # Build graph with shifted concept, score with original
        g = build_graph(
            sch,
            root_time=float(C.t[0]),
            root_feat=C.mu[0],
            planted=[(C_gen, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
        )
        E = score(g, C)

        rows.append(
            dict(
                kind="time_offset",
                offset=float(off),
                evidence=float(E),
                gamma=float(C.gamma_t[1]),
                tau=float(C.tau),
            )
        )

    out = write_csv("time_similarity_offset_sweep.csv", rows)

    # Verify monotonicity property
    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert nonincreasing(ev, atol=1e-8)

    print(f"\n[time similarity] wrote {out}")
    print("offset -> evidence:", [(float(r["offset"]), float(r["evidence"])) for r in rows])


def test_sensitivity_path_noise_accumulation_same_concept():
    """
    Continue adding increasingly noisy base_concept paths and inspect change in evidence score.

    Hypothesis: the evidence score will initially dampen, then quickly level out
    due to overly noisy paths.
    """
    schema = make_schema()
    concept = base_concept_r1r2(gamma=5.0, tau=1.0)

    # One shared graph for the entire sweep (keeps scores comparable)
    graph = Subgraph(schema)
    graph.create_root(time=float(concept.t[0]), feat=concept.mu[0])

    # Start with a single clean instance as a baseline signal
    graph.add_evidence(
        concept,
        rng=np.random.default_rng(0),
        instances=1,
        relation_pick="mode",
        time_noise_std=0.0,
        feat_noise_std=0.0,
    )

    noise_std_sweep = np.linspace(0.0, 1.0, 9)
    instances_per_noise_level = 10

    rows: List[Dict[str, Any]] = []
    total_instances = 1  # already added the clean baseline instance

    for i, noise_std in enumerate(noise_std_sweep):
        # Add a batch of instances at this noise level to the SAME graph
        graph.add_evidence(
            concept,
            rng=np.random.default_rng(100 + i),
            instances=instances_per_noise_level,
            relation_pick="mode",
            time_noise_std=float(noise_std),
            feat_noise_std=0.0,
        )
        total_instances += instances_per_noise_level

        evidence = float(graph.evidence_score(concept))

        rows.append(
            dict(
                kind="accumulated_noise_single_graph",
                noise_std=float(noise_std),
                instances_added=int(instances_per_noise_level),
                total_instances=int(total_instances),
                evidence=evidence,
                gamma=float(concept.gamma_t[1]),
                tau=float(concept.tau),
            )
        )

    out = write_csv("path_noise_accumulation_irrelevant_concept_sweep.csv", rows)
    print(f"\n[path noise accumulation - irrelevant concept] wrote {out}")
    print("noise_std -> evidence:", [(float(r["noise_std"]), float(r["evidence"])) for r in rows])


def test_sensitivity_path_noise_accumulation_second_concept():
    """
    Continue adding increasingly noisy irrelevant_concept paths and inspect change in evidence score.

    Hypothesis: the evidence score will initially dampen, then quickly level out due to overly noisy paths.
    """
    schema = make_schema()
    base_concept = base_concept_r1r2(gamma=5.0, tau=1.0)

    # One shared graph for the whole sweep (keeps scores directly comparable)
    graph = Subgraph(schema=schema)
    graph.create_root(time=float(base_concept.t[0]), feat=base_concept.mu[0])

    # Start with a small clean baseline so the curve has an anchor point
    baseline_instances = 1
    graph.add_evidence(
        base_concept,
        rng=np.random.default_rng(0),
        instances=baseline_instances,
        relation_pick="mode",
        time_noise_std=0.0,
        feat_noise_std=0.0,
    )

    irrelevant_concept = base_concept_rxry(gamma=5.0, tau=1.0)

    noise_levels = np.linspace(0.0, 1.0, 9)
    instances_per_noise_level = 10

    rows: List[Dict[str, Any]] = []
    total_instances = baseline_instances

    for i, noise_std in enumerate(noise_levels):
        graph.add_evidence(
            irrelevant_concept,
            rng=np.random.default_rng(1 + i),
            instances=instances_per_noise_level,
            relation_pick="mode",
            time_noise_std=float(noise_std),
            feat_noise_std=0.0,
        )
        total_instances += instances_per_noise_level

        E = float(graph.evidence_score(base_concept))

        rows.append(
            dict(
                kind="accumulated_noise_single_graph",
                noise_std=float(noise_std),
                instances_added=int(instances_per_noise_level),
                total_instances=int(total_instances),
                evidence=E,
                gamma=float(base_concept.gamma_t[1]),
                tau=float(base_concept.tau),
            )
        )

    out = write_csv("path_noise_accumulation_sweep_second.csv", rows)
    print(f"\n[path noise accumulation] wrote {out}")
    print("noise_std -> evidence:", [(float(r["noise_std"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 3: Sensitivity tests - Feature component
# ============================================================

def test_sensitivity_feature_centroid_shift_sweep_writes_curve():
    """
    Sweep feature centroid shift and measure evidence degradation.

    Hypothesis: As feature mismatch increases, evidence should decrease.
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=5.0, tau=1.0)

    # Direction vector for consistent shift (normalized diagonal in feature space)
    D = C.mu.shape[1]
    direction = np.ones(D, dtype=float) / np.sqrt(float(D))

    # Sweep shift distances
    shifts = np.linspace(0.0, 2.0, 9)
    rows: List[Dict[str, Any]] = []

    for s in shifts:
        # Shift hop-1 and hop-2 feature centroids in the same direction
        mu_shifted = C.mu.copy()
        if mu_shifted.shape[0] > 1:
            mu_shifted[1] = C.mu[1] + s * direction
        if mu_shifted.shape[0] > 2:
            mu_shifted[2] = C.mu[2] + s * direction

        C_gen = Concept(
            name=f"C_gen_feat_shift_{s:.2f}",
            ordered_node_types=C.ordered_node_types,
            P=C.P.copy(),
            t=C.t.copy(),
            gamma_t=C.gamma_t.copy(),
            mu=mu_shifted,
            gamma_mu=C.gamma_mu.copy(),
            tau=C.tau,
            k_time=C.k_time,
            k_feat=C.k_feat,
        )

        g = build_graph(
            sch,
            root_time=float(C.t[0]),
            root_feat=C.mu[0],
            planted=[(C_gen, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
        )
        E = score(g, C)

        rows.append(
            dict(
                kind="feat_shift",
                shift=float(s),
                evidence=float(E),
                gamma=float(C.gamma_mu[1]),
                tau=float(C.tau),
            )
        )

    out = write_csv("feature_centroid_shift_sweep.csv", rows)

    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert nonincreasing(ev, atol=1e-8)

    print(f"\n[feature centroid] wrote {out}")
    print("shift -> evidence:", [(float(r["shift"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 4: Sensitivity tests - Relational component
# ============================================================

def test_sensitivity_relation_similarity_sweep_writes_curve():
    """
    Sweep relational probability mass and measure evidence response.

    Hypothesis: As probability mass on the node-types actually used in the
    planted paths increases, evidence increases.
    """
    sch = make_schema()

    # Build graph: one perfect base_concept instance
    C_planted = base_concept_r1r2(gamma=5.0, tau=1.0)
    g = build_graph(
        sch,
        root_time=float(C_planted.t[0]),
        root_feat=C_planted.mu[0],
        planted=[(C_planted, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
    )

    # Identify, per hop, the dominant node-type column for the planted concept
    L, K = C_planted.P.shape
    dominant_cols = np.argmax(C_planted.P, axis=1)

    # Sweep probability mass from 0.05 to 0.95
    ps = np.linspace(0.05, 0.95, 10)
    rows: List[Dict[str, Any]] = []

    for p in ps:
        P_score = np.zeros_like(C_planted.P)
        for i in range(L):
            j_star = dominant_cols[i]
            if K == 1:
                P_score[i, 0] = 1.0
            else:
                P_score[i, :] = (1.0 - p) / float(K - 1)
                P_score[i, j_star] = p

        C_score = Concept(
            name=f"C_score_rel_p_{p:.2f}",
            ordered_node_types=C_planted.ordered_node_types,
            P=P_score,
            t=C_planted.t.copy(),
            gamma_t=C_planted.gamma_t.copy(),
            mu=C_planted.mu.copy(),
            gamma_mu=C_planted.gamma_mu.copy(),
            tau=C_planted.tau,
            k_time=C_planted.k_time,
            k_feat=C_planted.k_feat,
        )

        E = score(g, C_score)

        rows.append(
            dict(
                kind="rel_mass",
                p=float(p),
                p_product=float(p ** L),
                evidence=float(E),
                gamma=float(C_planted.gamma_t[1]),
                tau=float(C_planted.tau),
            )
        )

    out = write_csv("relation_similarity_mass_sweep.csv", rows)

    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert bool(np.all(ev[1:] >= ev[:-1] - 1e-10))

    print(f"\n[relation similarity] wrote {out}")
    print("p -> evidence:", [(float(r["p"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 5: Sensitivity tests - Time + Feature component
# ============================================================

def test_sensitivity_stepwise_noise_independent_time_and_feature():
    """
    Add noise independently per relation-step (time + feature), then compare evidence.

    Requirement: noise is independent per relation along the path (stepwise stds).
    Hypothesis: No-noise > single-step noise > both-steps noise (in expectation).
    """
    schema = make_schema()
    concept = base_concept_r1r2(gamma=5.0, tau=1.0)

    instances = 40
    replicates = 5

    # Each entry is (label, time_noise_by_step, feat_noise_by_step)
    profiles = [
        ("clean",            [0.0, 0.0], [0.0, 0.0]),
        ("time_step0",       [0.6, 0.0], [0.0, 0.0]),
        ("time_step1",       [0.0, 0.6], [0.0, 0.0]),
        ("time_both",        [0.6, 0.6], [0.0, 0.0]),
        ("feat_step0",       [0.0, 0.0], [0.6, 0.0]),
        ("feat_step1",       [0.0, 0.0], [0.0, 0.6]),
        ("feat_both",        [0.0, 0.0], [0.6, 0.6]),
        ("time_and_feat",    [0.6, 0.6], [0.6, 0.6]),
    ]

    rows: List[Dict[str, Any]] = []
    avg_evidence: Dict[str, float] = {}

    for label, time_std_steps, feat_std_steps in profiles:
        evidences: List[float] = []

        for r in range(replicates):
            g = Subgraph(schema=schema)
            g.create_root(time=float(concept.t[0]), feat=concept.mu[0])

            g.add_evidence(
                concept,
                rng=np.random.default_rng(10_000 + 100 * r),
                instances=instances,
                relation_pick="mode",
                time_noise_std=list(map(float, time_std_steps)),
                feat_noise_std=list(map(float, feat_std_steps)),
            )

            E = float(g.evidence_score(concept))
            assert 0.0 <= E <= 1.0
            evidences.append(E)

        avg = float(np.mean(evidences))
        avg_evidence[label] = avg

        rows.append(
            dict(
                kind="stepwise_noise_profile",
                profile=label,
                time_noise_step0=float(time_std_steps[0]),
                time_noise_step1=float(time_std_steps[1]),
                feat_noise_step0=float(feat_std_steps[0]),
                feat_noise_step1=float(feat_std_steps[1]),
                instances=int(instances),
                replicates=int(replicates),
                evidence_mean=avg,
                evidence_std=float(np.std(evidences)),
                gamma=float(concept.gamma_t[1]),
                tau=float(concept.tau),
            )
        )

    assert avg_evidence["clean"] >= avg_evidence["time_step0"]
    assert avg_evidence["clean"] >= avg_evidence["time_step1"]
    assert avg_evidence["clean"] >= avg_evidence["feat_step0"]
    assert avg_evidence["clean"] >= avg_evidence["feat_step1"]
    assert avg_evidence["time_both"] <= max(avg_evidence["time_step0"], avg_evidence["time_step1"])
    assert avg_evidence["feat_both"] <= max(avg_evidence["feat_step0"], avg_evidence["feat_step1"])
    assert avg_evidence["time_and_feat"] <= min(avg_evidence["time_both"], avg_evidence["feat_both"])

    out = write_csv("stepwise_noise_profiles.csv", rows)
    print(f"\n[stepwise noise profiles] wrote {out}")
    print("profile -> evidence_mean:", [(r["profile"], float(r["evidence_mean"])) for r in rows])


# ============================================================
# SECTION 6: Sensitivity tests - Noise component
# ============================================================

@pytest.mark.parametrize("gamma", [1.0, 5.0, 15.0])
def test_sensitivity_noise_sweep_time_and_feature_writes_curve(gamma: float):
    """
    Sweep generation noise (time and feature) across different gamma values.

    Hypothesis: As noise increases, evidence should decrease. Different gamma
    values should show different sensitivity curves.
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=gamma, tau=1.0)

    noise = np.linspace(0.0, 1.0, 9)
    rows: List[Dict[str, Any]] = []

    for n in noise:
        g = build_graph(
            sch,
            root_time=float(C.t[0]),
            root_feat=C.mu[0],
            planted=[(C, dict(
                rng=np.random.default_rng(0),
                instances=1,
                rel_pick="mode",
                time_noise_std=float(n),
                feat_noise_std=float(n),
            ))],
        )

        E = score(g, C)

        rows.append(
            dict(
                kind="noise",
                noise_std=float(n),
                evidence=float(E),
                gamma=float(gamma),
                tau=float(C.tau),
            )
        )

    out = write_csv(f"noise_sweep_gamma_{gamma:.1f}.csv", rows)

    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert nonincreasing(ev, atol=1e-8)

    print(f"\n[noise sweep] gamma={gamma} wrote {out}")
    print("noise -> evidence:", [(float(r["noise_std"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 7: Concept sizes / coverage
# ============================================================

def test_sensitivity_concept_sizes_and_coverage_cases():
    """
    Check concept evidence under size/coverage mismatches.

    Required cases:
      1) Concept longer than any present in graph
      2) Concept includes relations not in graph
      3) Concept where only part exists in graph (graph has r1->r2, concept is only r1)
    """
    schema = make_schema()
    full = base_concept_r1r2(gamma=5.0, tau=1.0)
    irrelevant = base_concept_rxry(gamma=5.0, tau=1.0)

    # Build "r1-only" concept by slicing the first step from the full concept.
    L_full = full.P.shape[0]
    assert L_full >= 1

    P_r1 = full.P[:1, :]
    t_r1 = full.t[:2]            # root + first hop
    gamma_t_r1 = full.gamma_t[:2]
    mu_r1 = full.mu[:2, :]
    gamma_mu_r1 = full.gamma_mu[:2]

    r1_only = Concept(
        name="C_r1_only",
        ordered_node_types=full.ordered_node_types,
        P=P_r1,
        t=t_r1,
        gamma_t=gamma_t_r1,
        mu=mu_r1,
        gamma_mu=gamma_mu_r1,
        tau=full.tau,
        k_time=full.k_time,
        k_feat=full.k_feat,
    )

    # Case A: Graph contains full concept evidence
    g_full = build_graph(
        schema,
        root_time=float(full.t[0]),
        root_feat=full.mu[0],
        planted=[(full, dict(rng=np.random.default_rng(0), instances=25, rel_pick="mode"))],
    )

    E_full = float(g_full.evidence_score(full))
    E_partial = float(g_full.evidence_score(r1_only))
    E_irrelevant = float(g_full.evidence_score(irrelevant))

    assert 0.0 <= E_full <= 1.0
    assert 0.0 <= E_partial <= 1.0
    assert 0.0 <= E_irrelevant <= 1.0

    # Relation-not-in-graph should score worst
    assert E_full > E_irrelevant
    assert E_partial > E_irrelevant

    # Case B: Graph contains only r1-only evidence; score longer concept
    g_r1 = build_graph(
        schema,
        root_time=float(r1_only.t[0]),
        root_feat=r1_only.mu[0],
        planted=[(r1_only, dict(rng=np.random.default_rng(1), instances=25, rel_pick="mode"))],
    )

    E_r1_on_r1 = float(g_r1.evidence_score(r1_only))
    E_full_on_r1 = float(g_r1.evidence_score(full))

    assert 0.0 <= E_r1_on_r1 <= 1.0
    assert 0.0 <= E_full_on_r1 <= 1.0

    # Longer-than-graph should be penalized relative to the matching shorter concept.
    assert E_r1_on_r1 > E_full_on_r1

    print("\n[concept coverage cases]")
    print(
        f"full graph:     E_full={E_full:.4f}, "
        f"E_partial={E_partial:.4f}, E_irrelevant={E_irrelevant:.4f}"
    )
    print(
        f"r1-only graph:  E_r1_on_r1={E_r1_on_r1:.4f}, "
        f"E_full_on_r1={E_full_on_r1:.4f}"
    )


# ============================================================
# SECTION 8: Multiple generating concepts (mixtures)
# ============================================================

def test_sensitivity_multiple_generating_concepts_mixture_table():
    """
    Plant two distinct concepts and measure each concept's evidence response
    to mixture proportions.

    Hypothesis: Evidence should scale with number of matching instances.
    """
    sch = make_schema()

    A = base_concept_r1r2(gamma=5.0, tau=1.0)
    B = base_concept_rxry(gamma=5.0, tau=1.0)

    grid = [(a, b) for a in [0, 1, 2, 4] for b in [0, 1, 2, 4]]
    rows: List[Dict[str, Any]] = []

    for a, b in grid:
        planted = []
        if a > 0:
            planted.append((A, dict(
                rng=np.random.default_rng(0),
                instances=int(a),
                rel_pick="mode",
                time_noise_std=0.0,
                feat_noise_std=0.0,
            )))
        if b > 0:
            planted.append((B, dict(
                rng=np.random.default_rng(1),
                instances=int(b),
                rel_pick="mode",
                time_noise_std=0.0,
                feat_noise_std=0.0,
            )))

        # Neutral root to isolate effect of planted paths
        g = build_graph(
            sch,
            root_time=0.0,
            root_feat=np.zeros_like(A.mu[0]),
            planted=planted,
        )

        EA = score(g, A)
        EB = score(g, B)

        rows.append(
            dict(
                instances_A=int(a),
                instances_B=int(b),
                evidence_A=float(EA),
                evidence_B=float(EB),
                gamma=5.0,
                tau=1.0,
            )
        )

    out = write_csv("mixture_two_concepts_instances_grid.csv", rows)

    for r in rows:
        assert 0.0 <= r["evidence_A"] <= 1.0
        assert 0.0 <= r["evidence_B"] <= 1.0

    rows_b0 = sorted(
        [r for r in rows if r["instances_B"] == 0],
        key=lambda d: d["instances_A"],
    )
    EA_b0 = np.array([r["evidence_A"] for r in rows_b0], dtype=float)
    assert bool(np.all(EA_b0[1:] >= EA_b0[:-1] - 1e-10))

    print(f"\n[multiple concepts] wrote {out}")
    print(
        "instances_A, instances_B -> (EA, EB):",
        [
            (r["instances_A"], r["instances_B"],
             round(r["evidence_A"], 4), round(r["evidence_B"], 4))
            for r in rows
        ],
    )
