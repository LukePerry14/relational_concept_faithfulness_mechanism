import numpy as np
import pytest
from numpy.testing import assert_allclose

from faithfulness_poc import Schema, Concept, Subgraph


# -----------------------
# Helpers
# -----------------------

def make_schema():
    return Schema(
        root_type="ROOT",
        transitions={
            ("ROOT", "r1"): "A",
            ("ROOT", "rx"): "A",
            ("A", "r2"): "B",
            ("A", "ry"): "B",
        },
    )

def make_concept_onehot(tau=0.0, gamma=10.0):
    rel_types = ["r1", "r2", "rx", "ry"]
    rel_probs = np.array([
        [1.0, 0.0, 0.0, 0.0],  # hop0: r1
        [0.0, 1.0, 0.0, 0.0],  # hop1: r2
    ], dtype=float)

    return Concept(
        name="C_onehot",
        rel_types=rel_types,
        rel_probs=rel_probs,
        time_deltas=[2.0, 5.0],
        feature_centroid=[np.array([1.0, 0.0]),
                 np.array([0.0, 1.0]),
                 np.array([1.0, 1.0])],
        gamma=gamma,
        tau=tau,
    )

def make_concept_soft(tau=0.0, gamma=10.0):
    rel_types = ["r1", "r2", "rx", "ry"]
    rel_probs = np.array([
        [0.8, 0.0, 0.2, 0.0],  # hop0: mostly r1, sometimes rx
        [0.0, 0.7, 0.0, 0.3],  # hop1: mostly r2, sometimes ry
    ], dtype=float)

    return Concept(
        name="C_soft",
        rel_types=rel_types,
        rel_probs=rel_probs,
        time_deltas=[2.0, 5.0],
        feature_centroid=[np.array([1.0, 0.0]),
                 np.array([0.0, 1.0]),
                 np.array([1.0, 1.0])],
        gamma=gamma,
        tau=tau,
    )


# -----------------------
# Concept validity tests
# -----------------------

@pytest.mark.parametrize("rel_probs", [
    np.array([1.0, 0.0, 0.0]),              # wrong ndim
    np.array([[1.0, 0.0], [0.5, 0.5]]),     # wrong R if rel_types length mismatched
])


def test_concept_invalid_rel_probs_shape(rel_probs):
    rel_types = ["r1", "r2", "rx", "ry"]
    with pytest.raises(ValueError):
        Concept(
            name="bad",
            rel_types=rel_types,
            rel_probs=rel_probs,
            time_deltas=[1.0, 1.0],
            feature_centroid=[np.zeros(2), np.zeros(2), np.zeros(2)],
        )

def test_concept_rejects_negative_probs():
    rel_types = ["r1", "r2"]
    rel_probs = np.array([[1.0, -0.1]], dtype=float)
    with pytest.raises(ValueError):
        Concept(
            name="badneg",
            rel_types=rel_types,
            rel_probs=rel_probs,
            time_deltas=[1.0],
            feature_centroid=[np.zeros(2), np.zeros(2)],
        )

def test_concept_rejects_zero_row():
    rel_types = ["r1", "r2"]
    rel_probs = np.array([[0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError):
        Concept(
            name="badzero",
            rel_types=rel_types,
            rel_probs=rel_probs,
            time_deltas=[1.0],
            feature_centroid=[np.zeros(2), np.zeros(2)],
        )


# -----------------------
# Schema tests
# -----------------------

def test_schema_check_valid():
    sch = make_schema()
    sch.check("ROOT", "r1", "A")
    sch.check("A", "r2", "B")

def test_schema_check_invalid_relation():
    sch = make_schema()
    with pytest.raises(ValueError):
        sch.check("ROOT", "nope", "A")

def test_schema_check_wrong_destination_type():
    sch = make_schema()
    with pytest.raises(ValueError):
        sch.check("ROOT", "r1", "B")


# -----------------------
# Graph construction tests
# -----------------------

def test_create_root_sets_type_and_is_stable():
    sch = make_schema()
    g = Subgraph(schema=sch)

    root_feat = np.array([9.0, 9.0])
    rid = g.create_root(time=0.0, feat=root_feat)

    assert g.nodes[rid].node_type == "ROOT"
    assert_allclose(g.nodes[rid].feature_embedding, root_feat)

def test_add_evidence_does_not_mutate_root_feat():
    sch = make_schema()
    c = make_concept_onehot()
    g = Subgraph(schema=sch)

    root_feat = np.array([9.0, 9.0])
    g.create_root(time=0.0, feat=root_feat)

    g.add_evidence(c, rng=np.random.default_rng(0), instances=1, rel_pick="mode")
    assert_allclose(g.nodes[g.root].feature_embedding, root_feat)

def test_add_evidence_adds_expected_edges_for_mode_onehot():
    sch = make_schema()
    c = make_concept_onehot()
    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=c.feature_centroid[0])

    g.add_evidence(c, rng=np.random.default_rng(0), instances=1, rel_pick="mode")

    # Expect exactly L edges in a chain from root
    assert len(g.edges) == c.L()
    rels = [e.rel for e in g.edges]
    assert rels == ["r1", "r2"]


# -----------------------
# Path enumeration tests
# -----------------------

def test_sample_paths_exact_length_counts_branching():
    sch = make_schema()
    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=np.zeros(2))

    # Manually create two children at hop1, then one child each at hop2 => 2 paths length 2
    # Use add_evidence twice with different mode concepts
    cA = make_concept_onehot()
    cB = make_concept_onehot()

    # Modify cB to choose rx then ry (one-hot)
    rel_types = cB.rel_types
    rp = np.array([
        [0.0, 0.0, 1.0, 0.0],  # rx
        [0.0, 0.0, 0.0, 1.0],  # ry
    ], dtype=float)
    cB = Concept(
        name="C_rxry",
        rel_types=rel_types,
        rel_probs=rp,
        time_deltas=[2.0, 5.0],
        feature_centroid=cB.feature_centroid,
        gamma=cB.gamma,
        tau=cB.tau,
    )

    g.add_evidence(cA, rng=np.random.default_rng(1), instances=1, rel_pick="mode")
    g.add_evidence(cB, rng=np.random.default_rng(2), instances=1, rel_pick="mode")

    paths = g.sample_paths(L=2)
    assert len(paths) == 2
    rel_seqs = sorted([p.rel_seq for p in paths])
    assert rel_seqs == [["r1", "r2"], ["rx", "ry"]]


# -----------------------
# Evidence correctness tests
# -----------------------

def test_evidence_perfect_match_tau0_is_1():
    sch = make_schema()
    c = make_concept_onehot(tau=0.0, gamma=10.0)

    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=c.feature_centroid[0])
    g.add_evidence(c, rng=np.random.default_rng(0), time_noise_std=0.0, feat_noise_std=0.0, instances=1, rel_pick="mode")

    E = g.evidence_score(c)
    assert_allclose(E, 1.0)

@pytest.mark.parametrize("tau", [0.1, 1.0, 10.0])
def test_evidence_matches_closed_form_for_single_perfect_match(tau):
    sch = make_schema()
    c = make_concept_onehot(tau=tau, gamma=10.0)

    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=c.feature_centroid[0])
    g.add_evidence(c, rng=np.random.default_rng(0), time_noise_std=0.0, feat_noise_std=0.0, instances=1, rel_pick="mode")

    # one-hot + perfect TF => S = 1, so E = 1 / (1 + tau)
    E = g.evidence_score(c)
    assert_allclose(E, 1.0 / (1.0 + tau))

def test_evidence_in_range():
    sch = make_schema()
    c = make_concept_soft(tau=0.5, gamma=3.0)

    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=np.array([10.0, -10.0]))  # mismatch on purpose
    g.add_evidence(c, rng=np.random.default_rng(0), time_noise_std=0.5, feat_noise_std=0.5, instances=1, rel_pick="sample")

    E = g.evidence_score(c)
    assert 0.0 <= E <= 1.0

def test_evidence_monotonic_in_number_of_matching_instances():
    sch = make_schema()
    c = make_concept_onehot(tau=1.0, gamma=10.0)

    g1 = Subgraph(schema=sch)
    g1.create_root(time=0.0, feat=c.feature_centroid[0])
    g1.add_evidence(c, rng=np.random.default_rng(0), instances=1, rel_pick="mode")
    E1 = g1.evidence_score(c)

    g2 = Subgraph(schema=sch)
    g2.create_root(time=0.0, feat=c.feature_centroid[0])
    g2.add_evidence(c, rng=np.random.default_rng(0), instances=3, rel_pick="mode")
    E2 = g2.evidence_score(c)

    assert E2 > E1

def test_gamma_zero_neutralizes_time_feature_mismatch():
    sch = make_schema()
    c = make_concept_onehot(tau=0.0, gamma=0.0)  # gamma=0 => exp(0)=1 regardless of mismatch

    g = Subgraph(schema=sch)
    g.create_root(time=123.0, feat=np.array([999.0, -999.0]))  # deliberately mismatched
    g.add_evidence(c, rng=np.random.default_rng(0), time_noise_std=0.0, feat_noise_std=0.0, instances=1, rel_pick="mode")

    E = g.evidence_score(c)
    # structural match is 1, TF becomes 1 (gamma=0), tau=0 => E=1
    assert_allclose(E, 1.0)

def test_soft_structural_similarity_affects_evidence():
    sch = make_schema()
    c = make_concept_soft(tau=0.0, gamma=10.0)

    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=c.feature_centroid[0])

    # Plant using mode => relations r1 then r2, with probs 0.8 and 0.7
    g.add_evidence(c, rng=np.random.default_rng(0), time_noise_std=0.0, feat_noise_std=0.0, instances=1, rel_pick="mode")
    E = g.evidence_score(c)

    # For perfect TF and tau=0, E should be 1 as long as any match exists? No:
    # E uses sum S_ij; for tau=0 it's s_sum/s_sum = 1 if s_sum>0.
    # So for tau=0, ANY positive similarity yields E=1.
    # To make structural weights observable, set tau>0.
    c_tau = make_concept_soft(tau=1.0, gamma=10.0)
    g2 = Subgraph(schema=sch)
    g2.create_root(time=0.0, feat=c_tau.feature_centroid[0])
    g2.add_evidence(c_tau, rng=np.random.default_rng(0), time_noise_std=0.0, feat_noise_std=0.0, instances=1, rel_pick="mode")
    E2 = g2.evidence_score(c_tau)

    # For one path: s_sum = 0.8*0.7 = 0.56 => E = 0.56 / (0.56 + 1)
    assert_allclose(E2, 0.56 / 1.56, rtol=1e-6)

def test_expressiveness_two_concepts_different_rel_probs_rank_correctly():
    sch = make_schema()

    # Graph contains r1->r2 path (mode of both), but concept A assigns high prob to r1,r2 while concept B assigns low.
    rel_types = ["r1", "r2", "rx", "ry"]

    A = Concept(
        name="A",
        rel_types=rel_types,
        rel_probs=np.array([[0.9, 0.0, 0.1, 0.0],
                            [0.0, 0.9, 0.0, 0.1]]),
        time_deltas=[2.0, 5.0],
        feature_centroid=[np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])],
        gamma=10.0,
        tau=1.0,
    )
    B = Concept(
        name="B",
        rel_types=rel_types,
        rel_probs=np.array([[0.2, 0.0, 0.8, 0.0],
                            [0.0, 0.2, 0.0, 0.8]]),
        time_deltas=[2.0, 5.0],
        feature_centroid=A.feature_centroid,
        gamma=10.0,
        tau=1.0,
    )

    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=A.feature_centroid[0])
    g.add_evidence(A, rng=np.random.default_rng(0), instances=1, rel_pick="mode")  # plants r1,r2

    EA = g.evidence_score(A)
    EB = g.evidence_score(B)
    assert EA > EB


# -----------------------
# Stochastic generation tests (kept stable)
# -----------------------

def test_rel_pick_sample_biases_toward_high_prob():
    sch = make_schema()
    c = make_concept_soft(tau=1.0, gamma=10.0)

    g = Subgraph(schema=sch)
    g.create_root(time=0.0, feat=c.feature_centroid[0])

    rng = np.random.default_rng(123)
    # plant many 1-hop paths to observe hop0 sampling
    # temporarily use L=1 concept (simplify)
    rel_types = c.rel_types
    rel_probs = np.array([[0.85, 0.0, 0.15, 0.0]], dtype=float)  # r1 vs rx
    c1 = Concept(
        name="C1hop",
        rel_types=rel_types,
        rel_probs=rel_probs,
        time_deltas=[1.0],
        feature_centroid=[c.feature_centroid[0], c.feature_centroid[1]],
        gamma=10.0,
        tau=1.0,
    )

    g.add_evidence(c1, rng=rng, instances=400, rel_pick="sample")

    # Count rels on edges from root
    root_edges = [e.rel for e in g.edges if e.source == g.root]
    r1 = sum(1 for r in root_edges if r == "r1")
    rx = sum(1 for r in root_edges if r == "rx")

    # With seed fixed and N=400, r1 should exceed rx robustly
    assert r1 > rx
