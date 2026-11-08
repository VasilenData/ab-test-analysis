"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

Adjusts each outcome with a covariate measured BEFORE assignment: y_adj = y - theta * (x - mean(x)),
theta = cov(x, y) / var(x), estimated on the pooled data. Variance falls by about rho^2.
The covariate must be pre-treatment; a covariate the treatment can change biases the estimate.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (adjusted outcome, theta) using pooled data from both arms."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    var_x = x.var(ddof=1)
    theta = 0.0 if var_x == 0 else float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    return y - theta * (x - x.mean()), theta


def diff_in_means(y_a: np.ndarray, y_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch difference (b - a) with confidence interval."""
    d = float(np.mean(y_b) - np.mean(y_a))
    se = float(np.sqrt(np.var(y_a, ddof=1) / len(y_a) + np.var(y_b, ddof=1) / len(y_b)))
    q = stats.norm.ppf(1 - alpha / 2)
    return {"diff": d, "se": se, "ci_low": d - q * se, "ci_high": d + q * se,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(d / se))))}


def cuped_test(y_a, x_a, y_b, x_b, alpha: float = 0.05) -> dict:
    """Difference in means after CUPED adjustment, plus the variance reduction achieved."""
    y = np.concatenate([y_a, y_b])
    x = np.concatenate([x_a, x_b])
    adj, theta = cuped_adjust(y, x)
    res = diff_in_means(adj[: len(y_a)], adj[len(y_a):], alpha)
    raw = diff_in_means(y_a, y_b, alpha)
    res["theta"] = theta
    res["variance_ratio"] = res["se"] ** 2 / raw["se"] ** 2
    res["raw_se"] = raw["se"]
    return res
