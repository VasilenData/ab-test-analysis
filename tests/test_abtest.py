import numpy as np

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
