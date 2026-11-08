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
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x) or len(y) < 2:
        raise ValueError("outcomes and covariates must be equal-length vectors with at least two rows")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("outcomes and covariates must be finite")
    var_x = x.var(ddof=1)
    theta = 0.0 if var_x == 0 else float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    return y - theta * (x - x.mean()), theta


def diff_in_means(y_a: np.ndarray, y_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch difference (b - a) with confidence interval."""
    y_a, y_b = np.asarray(y_a, float), np.asarray(y_b, float)
    if any(y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all() for y in (y_a, y_b)):
        raise ValueError("each arm must contain at least two finite observations")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    d = float(np.mean(y_b) - np.mean(y_a))
    va, vb = np.var(y_a, ddof=1) / len(y_a), np.var(y_b, ddof=1) / len(y_b)
    se = float(np.sqrt(va + vb))
    if se == 0:
        raise ValueError("Welch inference requires nonzero variance in at least one arm")
    df = float((va + vb) ** 2 / (va ** 2 / (len(y_a) - 1) + vb ** 2 / (len(y_b) - 1)))
    q = stats.t.ppf(1 - alpha / 2, df)
    return {"diff": d, "se": se, "ci_low": d - q * se, "ci_high": d + q * se,
            "df": df, "p_value": float(2 * stats.t.sf(abs(d / se), df))}


def cuped_test(y_a, x_a, y_b, x_b, alpha: float = 0.05) -> dict:
    """Difference in means after CUPED adjustment, plus the variance reduction achieved."""
    if len(y_a) != len(x_a) or len(y_b) != len(x_b):
        raise ValueError("each arm's covariate must align with its outcomes")
    y = np.concatenate([y_a, y_b])
    x = np.concatenate([x_a, x_b])
    adj, theta = cuped_adjust(y, x)
    res = diff_in_means(adj[: len(y_a)], adj[len(y_a):], alpha)
    raw = diff_in_means(y_a, y_b, alpha)
    res["theta"] = theta
    res["variance_ratio"] = res["se"] ** 2 / raw["se"] ** 2
    res["raw_se"] = raw["se"]
    return res
