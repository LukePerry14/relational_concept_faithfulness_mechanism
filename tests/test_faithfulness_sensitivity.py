# ============================================================
# Test Suite: Faithfulness Evidence Score Sensitivity Analysis
# ============================================================
# This module contains characterization-oriented tests for the evidence-score
# sensitivity metric described in Strategy_discussion.pdf (Eq. 8–13).
#
# Purpose:
#   - Probe the expressiveness/sensitivity of the evidence scoring function
#   - Sweep controlled parameters (time, features, relations, noise)
#   - Record results to CSV for inspection and analysis
#   - Verify monotonicity properties (sanity checks only)
#
# Outputs: CSV files written to <this file's dir>/evidence_reports/
#
# Execution notes (order & side-effects):
#   - PyTest collects the functions below in module order and executes any
#   - test_* functions it finds. Several tests perform parameter sweeps and
#     write CSV files into `REPORT_DIR` for offline inspection. The CSV outputs
#     are side-effects of these tests; they are intentional and useful for
#     sensitivity analysis. Running `pytest -q -s` will both execute the
#     assertions and produce the CSV artifacts.
#
#   - Randomness: tests that rely on sampling set explicit RNG seeds to make
#     behavior reproducible. Tests that perform large-sample sampling use
#     deterministic seeds to produce stable counts for assertions.
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
# Basic validation tests to ensure setup is correct.
# ============================================================

def test_smoke_concept_normalizes_rel_probs_rows():
    """
    Verify that concept relation probability rows sum to 1.0 (valid probability distribution).
    """
    c = base_concept_r1r2()
    rs = c.rel_probs.sum(axis=1)
    assert np.allclose(rs, 1.0)


def test_smoke_evidence_in_range():
    """
    Verify that evidence score is always in valid range [0, 1].
    """
    sch = make_schema()
    c = base_concept_r1r2(gamma=3.0, tau=1.0)
    
    # Build graph with one perfect r1->r2 instance
    g = build_graph(
        sch,
        root_time=0.0,
        root_feat=c.feature_centroid[0],
        planted=[(c, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
    )
    
    # Score the concept against itself
    E = score(g, c)
    
    # Check range
    assert 0.0 <= E <= 1.0


# ============================================================
# SECTION 2: Sensitivity tests - Time component
# ============================================================
# Tests that probe how evidence responds to temporal mismatches.
# ============================================================

def test_sensitivity_time_similarity_sweep_writes_curve():
    """
    Sweep time offset and measure evidence degradation.
    
    Hypothesis: As temporal mismatch increases, evidence should decrease
    (assuming gamma > 0, tau > 0).
    
    Method:
        1. Generate evidence with time-shifted concept (same structure/features)
        2. Score against original concept
        3. Record evidence at each offset
        4. Verify non-increasing monotonicity
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=5.0, tau=1.0)

    # Sweep time offsets from 0 to 4 seconds
    offsets = np.linspace(0.0, 4.0, 9)
    rows: List[Dict[str, Any]] = []

    for off in offsets:
        # Create concept with shifted time deltas
        C_gen = concept_template(
            name=f"C_gen_time_off_{off:.2f}",
            rel_probs=C.rel_probs,  # Same relation preferences
            time_deltas=[C.time_deltas[0] + off, C.time_deltas[1] + off],  # Shifted times
            feature_centroid=C.feature_centroid,  # Same features
            gamma=C.gamma,
            tau=C.tau,
        )
        
        # Build graph with shifted concept, score with original
        g = build_graph(
            sch,
            root_time=0.0,
            root_feat=C.feature_centroid[0],
            planted=[(C_gen, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
        )
        E = score(g, C)
        
        # Record result
        rows.append(dict(kind="time_offset", offset=float(off), evidence=E, gamma=C.gamma, tau=C.tau))

    # Write results to CSV
    # Columns: kind, offset, evidence, gamma, tau
    out = write_csv("time_similarity_offset_sweep.csv", rows)
    
    # Verify monotonicity property
    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert nonincreasing(ev, atol=1e-8)

    # Print summary
    print(f"\n[time similarity] wrote {out}")
    print("offset -> evidence:", [(float(r["offset"]), float(r["evidence"])) for r in rows])


def test_sensitivity_path_noise_accumulation_writes_curve():
    """
    Measure evidence scores as we accumulate increasingly noisy paths based on the same concept.
    
    Hypothesis: Evidence should scale with number of matching instances, but degrade
    as noise increases. This tests robustness to accumulated noisy evidence.
    
    Method:
        1. Build graph with one perfect r1->r2 instance
        2. Iteratively add more instances with increasing time noise
        3. Score against original noise-free concept
        4. Record evidence at each accumulation step
        5. Verify evidence behavior under noise accumulation
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=5.0, tau=1.0)

    # Sweep noise levels from 0 to 1.0 seconds
    noise_levels = np.linspace(0.0, 1.0, 9)
    rows: List[Dict[str, Any]] = []

    for noise in noise_levels:
        # Build graph with one perfect instance + noisy instances
        planted = [
            (C, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode", time_noise_std=0.0, feat_noise_std=0.0)),
            (C, dict(rng=np.random.default_rng(1), instances=1, rel_pick="mode", time_noise_std=float(noise), feat_noise_std=0.0))
        ]
        
        g = build_graph(
            sch,
            root_time=0.0,
            root_feat=C.feature_centroid[0],
            planted=planted,
        )
        
        # Score noise-free concept against accumulated noisy graph
        E = score(g, C)
        
        # Record result
        rows.append(dict(kind="accumulated_noise", noise_std=float(noise), evidence=E, gamma=C.gamma, tau=C.tau))

    # Write results to CSV
    # Columns: kind, noise_std, evidence, gamma, tau
    out = write_csv("path_noise_accumulation_sweep.csv", rows)

    # Print summary
    print(f"\n[path noise accumulation] wrote {out}")
    print("noise_std -> evidence:", [(float(r["noise_std"]), float(r["evidence"])) for r in rows])
    
    
# ============================================================
# SECTION 3: Sensitivity tests - Feature component
# ============================================================
# Tests that probe how evidence responds to feature centroid mismatches.
# ============================================================

def test_sensitivity_feature_centroid_shift_sweep_writes_curve():
    """
    Sweep feature centroid shift and measure evidence degradation.
    
    Hypothesis: As feature mismatch increases, evidence should decrease.
    
    Method:
        1. Generate evidence with shifted feature centroids (same structure/time)
        2. Score against original concept
        3. Record evidence at each shift distance
        4. Verify non-increasing monotonicity
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=5.0, tau=1.0)

    # Direction vector for consistent shift (normalized diagonal)
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)

    # Sweep shift distances from 0 to 2.0 units
    shifts = np.linspace(0.0, 2.0, 9)
    rows: List[Dict[str, Any]] = []

    for s in shifts:
        # Shift hop-1 and hop-2 centroids in the same direction
        feature_centroid_shifted = [C.feature_centroid[0]]  # Keep root centroid
        feature_centroid_shifted.append(C.feature_centroid[1] + s * direction)  # Shift hop 1
        feature_centroid_shifted.append(C.feature_centroid[2] + s * direction)  # Shift hop 2

        # Create concept with shifted features
        C_gen = concept_template(
            name=f"C_gen_feat_shift_{s:.2f}",
            rel_probs=C.rel_probs,  # Same relation preferences
            time_deltas=C.time_deltas,  # Same times
            feature_centroid=feature_centroid_shifted,  # Shifted features
            gamma=C.gamma,
            tau=C.tau,
        )
        
        # Build graph with shifted concept, score with original
        g = build_graph(
            sch,
            root_time=0.0,
            root_feat=C.feature_centroid[0],
            planted=[(C_gen, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
        )
        E = score(g, C)
        
        # Record result
        rows.append(dict(kind="feat_shift", shift=float(s), evidence=E, gamma=C.gamma, tau=C.tau))

    # Write results to CSV
    # Columns: kind, shift, evidence, gamma, tau
    out = write_csv("feature_centroid_shift_sweep.csv", rows)
    
    # Verify monotonicity property
    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert nonincreasing(ev, atol=1e-8)

    # Print summary
    print(f"\n[feature centroid] wrote {out}")
    print("shift -> evidence:", [(float(r["shift"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 4: Sensitivity tests - Relation component
# ============================================================
# Tests that probe how evidence responds to relation probability changes.
# ============================================================

def test_sensitivity_relation_similarity_sweep_writes_curve():
    """
    Sweep relation probability mass and measure evidence response.
    
    Hypothesis: As probability mass on observed relations increases, evidence increases.
    
    Method:
        1. Fix observed path (r1->r2) with perfect time/features
        2. Vary concept's relation probability mass on r1 and r2
        3. Score concept against graph
        4. Record evidence at each probability level
        5. Verify non-decreasing monotonicity
    """
    sch = make_schema()

    # Build graph: one perfect r1->r2 instance with perfect time/features
    C_planted = base_concept_r1r2(gamma=5.0, tau=1.0)
    g = build_graph(
        sch,
        root_time=0.0,
        root_feat=C_planted.feature_centroid[0],
        planted=[(C_planted, dict(rng=np.random.default_rng(0), instances=1, rel_pick="mode"))],
    )

    # Sweep probability mass from 0.05 to 0.95
    ps = np.linspace(0.05, 0.95, 10)
    rows: List[Dict[str, Any]] = []

    for p in ps:
        # Create concept with varying probability mass
        # Hop 0: [p for r1, 0, 1-p for rx, 0]
        # Hop 1: [0, p for r2, 0, 1-p for ry]
        rel_probs = np.array([
            [p, 0.0, 1.0 - p, 0.0],
            [0.0, p, 0.0, 1.0 - p],
        ], dtype=float)

        C_score = concept_template(
            name=f"C_score_rel_p_{p:.2f}",
            rel_probs=rel_probs,  # Varying rel probs
            time_deltas=C_planted.time_deltas,  # Same times as planted
            feature_centroid=C_planted.feature_centroid,  # Same features as planted
            gamma=C_planted.gamma,
            tau=C_planted.tau,
        )
        
        # Score the concept against fixed graph
        E = score(g, C_score)

        # Record result with product p*p for reference
        rows.append(dict(kind="rel_mass", p=float(p), p_product=float(p * p), evidence=E, gamma=C_score.gamma, tau=C_score.tau))

    # Write results to CSV
    # Columns: kind, p, p_product, evidence, gamma, tau
    out = write_csv("relation_similarity_mass_sweep.csv", rows)

    # Verify monotonicity: evidence should increase with relation mass
    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert bool(np.all(ev[1:] >= ev[:-1] - 1e-10))

    # Print summary
    print(f"\n[relation similarity] wrote {out}")
    print("p -> evidence:", [(float(r["p"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 5: Sensitivity tests - Noise component
# ============================================================
# Tests that probe how evidence responds to generation noise with varying gamma.
# ============================================================

@pytest.mark.parametrize("gamma", [1.0, 5.0, 15.0])
def test_sensitivity_noise_sweep_time_and_feature_writes_curve(gamma: float):
    """
    Sweep generation noise (time and feature) across different gamma values.
    
    Hypothesis: As noise increases, evidence should decrease. Different gamma
    values should show different sensitivity curves (higher gamma = sharper).
    
    Method:
        1. For each gamma value, sweep noise standard deviation
        2. Generate evidence with noisy instances (fixed RNG seed)
        3. Score against original noise-free concept
        4. Record evidence at each noise level
        5. Verify non-increasing monotonicity
    """
    sch = make_schema()
    C = base_concept_r1r2(gamma=gamma, tau=1.0)

    # Sweep noise std from 0 to 1.0
    noise = np.linspace(0.0, 1.0, 9)
    rows: List[Dict[str, Any]] = []

    for n in noise:
        # Build graph with noisy evidence instances
        g = build_graph(
            sch,
            root_time=0.0,
            root_feat=C.feature_centroid[0],
            planted=[(C, dict(
                rng=np.random.default_rng(0),
                instances=1,
                rel_pick="mode",
                time_noise_std=float(n),  # Add time noise
                feat_noise_std=float(n)   # Add feature noise
            ))],
        )
        
        # Score noise-free concept against noisy graph
        E = score(g, C)
        
        # Record result
        rows.append(dict(kind="noise", noise_std=float(n), evidence=E, gamma=C.gamma, tau=C.tau))

    # Write results to CSV
    # Columns: kind, noise_std, evidence, gamma, tau
    out = write_csv(f"noise_sweep_gamma_{gamma:.1f}.csv", rows)

    # Verify monotonicity: evidence should not increase with noise
    ev = np.array([r["evidence"] for r in rows], dtype=float)
    assert nonincreasing(ev, atol=1e-8)

    # Print summary
    print(f"\n[noise sweep] gamma={gamma} wrote {out}")
    print("noise -> evidence:", [(float(r["noise_std"]), float(r["evidence"])) for r in rows])


# ============================================================
# SECTION 6: Sensitivity tests - Multiple concepts (mixtures)
# ============================================================
# Tests that probe evidence behavior with multiple generating concepts.
# ============================================================

def test_sensitivity_multiple_generating_concepts_mixture_table():
    """
    Plant two distinct concepts and measure each concept's evidence response
    to mixture proportions.
    
    Hypothesis: Evidence should scale with number of matching instances.
    Potential failure modes:
        - Evidence inflation from unrelated branching paths
        - Cross-concept confusion if time/features are close but relations differ
    
    Method:
        1. Create two orthogonal concepts (r1r2 and rxry)
        2. Sweep grid of instance counts for each concept
        3. Build graph at each grid point with neutral root
        4. Measure evidence for both concepts
        5. Verify that evidence scales with matching instances
    """
    sch = make_schema()
    
    # Concept A: prefers r1->r2
    A = base_concept_r1r2(gamma=5.0, tau=1.0)
    
    # Concept B: prefers rx->ry (orthogonal)
    B = base_concept_rxry(gamma=5.0, tau=1.0)

    # Create grid of instance counts
    grid = [(a, b) for a in [0, 1, 2, 4] for b in [0, 1, 2, 4]]
    rows: List[Dict[str, Any]] = []

    for a, b in grid:
        # Build planting list based on instance counts
        planted = []
        if a > 0:
            planted.append((A, dict(
                rng=np.random.default_rng(0),
                instances=int(a),
                rel_pick="mode",
                time_noise_std=0.0,
                feat_noise_std=0.0
            )))
        if b > 0:
            planted.append((B, dict(
                rng=np.random.default_rng(1),
                instances=int(b),
                rel_pick="mode",
                time_noise_std=0.0,
                feat_noise_std=0.0
            )))

        # Build graph with neutral root (intentional mismatch to isolate effect)
        g = build_graph(
            sch,
            root_time=0.0,
            root_feat=np.array([0.0, 0.0]),
            planted=planted,
        )

        # Score both concepts
        EA = score(g, A)
        EB = score(g, B)

        # Record result
        rows.append(dict(instances_A=int(a), instances_B=int(b), evidence_A=EA, evidence_B=EB, gamma=5.0, tau=1.0))

    # Write results to CSV
    # Columns: instances_A, instances_B, evidence_A, evidence_B, gamma, tau
    out = write_csv("mixture_two_concepts_instances_grid.csv", rows)

    # Verify evidence is in valid range [0, 1]
    for r in rows:
        assert 0.0 <= r["evidence_A"] <= 1.0
        assert 0.0 <= r["evidence_B"] <= 1.0

    # Verify monotonicity on A-axis when B is fixed at 0
    # (increasing A instances should not decrease EA)
    rows_b0 = sorted([r for r in rows if r["instances_B"] == 0], key=lambda d: d["instances_A"])
    EA_b0 = np.array([r["evidence_A"] for r in rows_b0], dtype=float)
    assert bool(np.all(EA_b0[1:] >= EA_b0[:-1] - 1e-10))

    # Print summary
    print(f"\n[multiple concepts] wrote {out}")
    print("instances_A, instances_B -> (EA, EB):",
          [(r["instances_A"], r["instances_B"], round(r["evidence_A"], 4), round(r["evidence_B"], 4)) for r in rows])
