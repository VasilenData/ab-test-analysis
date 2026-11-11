import numpy as np
import pytest

from src.abtest import *


def test_srm_balanced_passes():
    assert srm_check(5000, 5010) > 0.5

def test_srm_broken_split_flags():
    assert srm_check(5000, 6000) < 0.001

def test_two_prop_known_values():
    r = two_prop_test(450, 1000, 500, 1000)
    assert abs(r["diff"] - 0.05) < 1e-12
    assert r["p_value"] < 0.05 and r["ci_low"] < 0.05 < r["ci_high"]

def test_two_prop_null():
    assert two_prop_test(500, 1000, 500, 1000)["p_value"] > 0.99

def test_sample_size_matches_textbook():
    # 10% -> 12% at 80% power, 5% alpha is about 3,800 per arm
    assert 3700 < sample_size(0.10, 0.02) < 3900

def test_mde_inverts_sample_size():
    n = sample_size(0.2, 0.01)
    assert abs(mde(0.2, n) - 0.01) < 5e-4

def test_holm():
    assert holm([0.001, 0.04, 0.5]) == [True, False, False]
    assert holm([0.04, 0.03]) == [False, False]

def test_bootstrap_ci_covers_true_diff():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 2000), rng.normal(0.3, 1, 2000)
    lo, hi = bootstrap_diff(a, b, n_boot=500)
    assert lo < 0.3 < hi

def test_bayes_clear_winner():
    assert bayes_prob_b_better(100, 1000, 200, 1000) > 0.999

def test_aa_false_positive_rate_near_alpha():
    y = (np.random.default_rng(2).random(20000) < 0.3).astype(int)
    assert 0.02 < aa_false_positive_rate(y, sims=400) < 0.09


def test_zero_control_conversions_have_undefined_relative_lift():
    r = two_prop_test(0, 100, 10, 100)
    assert r["diff"] == 0.1 and r["rel_lift"] is None and 0 < r["p_value"] < 0.01


@pytest.mark.parametrize("successes", [0, 100])
def test_constant_equal_arms_do_not_crash_or_claim_zero_uncertainty(successes):
    r = two_prop_test(successes, 100, successes, 100)
    assert r["p_value"] == 1 and r["z"] == 0
    assert r["ci_low"] is None and r["ci_high"] is None


@pytest.mark.parametrize("args", [(1, 0, 1, 10), (11, 10, 1, 10), (-1, 10, 1, 10),
                                  (1.5, 10, 1, 10), (True, 10, 1, 10)])
def test_invalid_conversion_counts_fail_clearly(args):
    with pytest.raises(ValueError, match="counts"):
        two_prop_test(*args)


def _z_p(xa, na, xb, nb):
    return two_prop_test(xa, na, xb, nb)["p_value"]


def test_fisher_known_value():
    # Tea-tasting table [[3, 1], [1, 3]]: two-sided p = 34/70
    assert fisher_exact_test(3, 4, 1, 4) == pytest.approx(34 / 70)


def test_fisher_identical_arms_p_one():
    assert fisher_exact_test(5, 20, 5, 20) == pytest.approx(1.0)


@pytest.mark.parametrize("args", [(-1, 10, 1, 10), (11, 10, 1, 10), (1, 0, 1, 10), (True, 10, 1, 10)])
def test_fisher_invalid_counts(args):
    with pytest.raises(ValueError):
        fisher_exact_test(*args)


def test_fisher_never_exceeds_alpha_on_small_samples():
    for p in (0.05, 0.1, 0.3, 0.5):
        assert exact_type_i_error(15, 15, p, fisher_exact_test) <= 0.05 + 1e-12


def test_z_test_is_anticonservative_at_small_n():
    # Exact enumeration: with n=15 per arm and p=0.3 the pooled z-test rejects (5.5%)
    # a true null more than the nominal 5%, though at other p it is conservative (0.5% to 4.6%).
    worst = max(exact_type_i_error(15, 15, p, _z_p) for p in (0.05, 0.1, 0.3, 0.5))
    assert worst > 0.05


def test_z_test_near_nominal_at_large_n():
    assert exact_type_i_error(300, 300, 0.3, _z_p) == pytest.approx(0.05, abs=0.01)


def test_exact_type_i_error_trivial_tests():
    assert exact_type_i_error(5, 5, 0.4, lambda *a: 0.0) == pytest.approx(1.0)
    assert exact_type_i_error(5, 5, 0.4, lambda *a: 1.0) == 0.0


@pytest.mark.parametrize("kw", [{"p": 0.0}, {"p": 1.0}, {"n_a": 0}, {"alpha": 1.5}])
def test_exact_type_i_error_validation(kw):
    args = {"n_a": 5, "n_b": 5, "p": 0.3, "test": _z_p, "alpha": 0.05} | kw
    with pytest.raises(ValueError):
        exact_type_i_error(**args)
