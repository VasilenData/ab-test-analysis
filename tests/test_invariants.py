"""Seeded algebraic invariants and invalid-input regressions for public statistical APIs."""
import numpy as np
import pytest

from src.abtest import (
    aa_false_positive_rate,
    bayes_prob_b_better,
    bootstrap_diff,
    holm,
    mde,
    sample_size,
    srm_check,
    two_prop_test,
)
from src.sequential import (
    always_valid_p,
    gs_boundaries,
    msprt_normal,
    spend_obf,
    spend_pocock,
)


def test_swapping_arms_reverses_effect_but_preserves_two_sided_evidence():
    rng = np.random.default_rng(102)
    for _ in range(200):
        na, nb = (int(n) for n in rng.integers(2, 2000, size=2))
        xa, xb = int(rng.integers(0, na + 1)), int(rng.integers(0, nb + 1))
        forward, reverse = two_prop_test(xa, na, xb, nb), two_prop_test(xb, nb, xa, na)
        assert 0 <= forward["p_value"] <= 1
        assert forward["p_value"] == pytest.approx(reverse["p_value"])
        assert forward["diff"] == pytest.approx(-reverse["diff"])
        assert forward["z"] == pytest.approx(-reverse["z"])
        if forward["ci_low"] is not None:
            assert forward["ci_low"] == pytest.approx(-reverse["ci_high"])
        assert srm_check(na, nb, 0.3) == pytest.approx(srm_check(nb, na, 0.7), abs=1e-12)


def test_holm_is_permutation_equivariant_and_monotone_in_alpha():
    rng = np.random.default_rng(103)
    for size in range(1, 31):
        p = rng.uniform(0, 0.1, size)
        order = rng.permutation(size)
        decisions = np.array(holm(p))
        assert np.array_equal(np.array(holm(p[order])), decisions[order])
        assert np.all(decisions <= np.array(holm(p, alpha=0.1)))


def test_bootstrap_translation_changes_only_the_effect_origin():
    rng = np.random.default_rng(104)
    a, b = rng.normal(size=20), rng.normal(size=30)
    lo, hi = bootstrap_diff(a, b, n_boot=50)
    shifted = bootstrap_diff(a + 10, b + 13, n_boot=50)
    assert shifted == pytest.approx((lo + 3, hi + 3))


def test_zero_and_infinite_likelihood_evidence_has_defined_limits():
    assert np.array_equal(always_valid_p([0, 1, 2, np.inf, 0]), [1, 1, 0.5, 0, 0])
    for spending in (spend_obf, spend_pocock):
        curve = spending(np.linspace(0, 1, 101))
        assert curve[0] == 0 and curve[-1] == pytest.approx(0.05)
        assert np.all(np.diff(curve) >= 0)


@pytest.mark.parametrize("call", [
    lambda: srm_check(0, 0), lambda: srm_check(-1, 10), lambda: srm_check(1.5, 10),
    lambda: srm_check(10, 10, expected_ratio=0),
    lambda: mde(0.2, 0), lambda: mde(1, 100), lambda: mde(0.2, 100, power=1),
    lambda: sample_size(0.2, 0), lambda: sample_size(0.9, 0.2),
    lambda: holm([0.01, np.nan]), lambda: holm([-0.1]), lambda: holm([1.1]),
    lambda: bootstrap_diff([], [1]), lambda: bootstrap_diff([1], [np.inf]),
    lambda: bootstrap_diff([1], [2], n_boot=0),
    lambda: bayes_prob_b_better(11, 10, 2, 10), lambda: bayes_prob_b_better(1, 10, 2, 10, draws=0),
    lambda: aa_false_positive_rate([0.2, 0.4]), lambda: aa_false_positive_rate([1]),
    lambda: aa_false_positive_rate([1, 0], sims=0),
    lambda: always_valid_p([-1, 2]), lambda: always_valid_p([1, np.nan]),
    lambda: spend_obf([-0.1]), lambda: spend_pocock([1.1]),
    lambda: msprt_normal(1, 0, 0.1), lambda: msprt_normal(np.nan, 1, 0.1),
    lambda: gs_boundaries(0), lambda: gs_boundaries(2, grid=2),
    lambda: gs_boundaries(2, spending=lambda t, alpha: np.array([0.04, 0.01])),
])
def test_invalid_inputs_cannot_produce_apparently_valid_inference(call):
    with pytest.raises(ValueError):
        call()
