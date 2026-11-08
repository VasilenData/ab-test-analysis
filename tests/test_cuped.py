import numpy as np
import pytest
from scipy import stats

from src.cuped import cuped_adjust, cuped_test, diff_in_means


def _sim(rng, n, rho, effect):
    x = rng.standard_normal(2 * n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(2 * n)
    t = rng.permutation(2 * n) < n
    y = y + effect * t
    return y[~t], x[~t], y[t], x[t]


def test_variance_ratio_matches_one_minus_rho_squared():
    rng = np.random.default_rng(0)
    ratios = [cuped_test(*_sim(rng, 4000, 0.7, 0.2))["variance_ratio"] for _ in range(60)]
    assert abs(np.mean(ratios) - 0.51) < 0.03


def test_cuped_is_unbiased():
    rng = np.random.default_rng(1)
    est = [cuped_test(*_sim(rng, 3000, 0.5, 0.3))["diff"] for _ in range(300)]
    assert abs(np.mean(est) - 0.3) < 0.01


def test_independent_covariate_changes_nothing_much():
    rng = np.random.default_rng(2)
    r = cuped_test(*_sim(rng, 4000, 0.0, 0.2))
    assert 0.98 < r["variance_ratio"] <= 1.001


def test_adjust_preserves_mean_and_reports_theta():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000)
    y = 2 * x + rng.standard_normal(5000)
    adj, theta = cuped_adjust(y, x)
    assert abs(theta - 2) < 0.1 and abs(adj.mean() - y.mean()) < 1e-9 and adj.var() < y.var()


def test_ci_covers_true_effect_about_95_percent():
    rng = np.random.default_rng(4)
    hit = 0
    for _ in range(400):
        r = cuped_test(*_sim(rng, 2000, 0.6, 0.3))
        hit += r["ci_low"] <= 0.3 <= r["ci_high"]
    assert 0.92 < hit / 400 < 0.98


def test_post_treatment_covariate_biases_the_estimate():
    rng = np.random.default_rng(5)
    n, est = 4000, []
    for _ in range(100):
        t = rng.permutation(2 * n) < n
        x = rng.standard_normal(2 * n) + 0.5 * t          # treatment changes the covariate
        y = 0.5 * x + rng.standard_normal(2 * n) + 0.3 * t
        est.append(cuped_test(y[~t], x[~t], y[t], x[t])["diff"])
    assert abs(np.mean(est) - 0.55) > 0.2                  # total effect is 0.55; CUPED misses part of it


def test_diff_in_means_known_values():
    r = diff_in_means(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0, 5.0]))
    assert abs(r["diff"] - 1.0) < 1e-12 and r["ci_low"] < 1 < r["ci_high"]


def test_small_unequal_samples_match_welch_t_inference():
    a, b = np.array([1., 2., 7.]), np.array([3., 4., 5., 6., 7.])
    expected = stats.ttest_ind(b, a, equal_var=False)
    result = diff_in_means(a, b)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["ci_low"] == pytest.approx(expected.confidence_interval().low)
    assert result["ci_high"] == pytest.approx(expected.confidence_interval().high)


def test_cuped_rejects_covariates_misaligned_between_arms():
    with pytest.raises(ValueError, match="align"):
        cuped_test([1, 2], [1], [3, 4], [2, 3, 4])


@pytest.mark.parametrize("a,b", [([1], [2, 3]), ([1, float('nan')], [2, 3]), ([1, 1], [2, 2])])
def test_welch_rejects_undefined_inference(a, b):
    with pytest.raises(ValueError):
        diff_in_means(a, b)
