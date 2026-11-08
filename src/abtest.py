"""Small, tested A/B-test toolkit: SRM, proportion tests, power, bootstrap, Bayesian, A/A check."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _alpha(alpha: float) -> None:
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _planning(p_base: float, alpha: float, power: float) -> None:
    _alpha(alpha)
    if not 0 < p_base < 1 or not 0.5 < power < 1:
        raise ValueError("normal-approximation planning requires 0 < baseline < 1 and 0.5 < power < 1")


def _counts(x: int, n: int) -> None:
    if (isinstance(x, (bool, np.bool_)) or isinstance(n, (bool, np.bool_))
            or not isinstance(x, (int, np.integer)) or not isinstance(n, (int, np.integer))
            or n <= 0 or not 0 <= x <= n):
        raise ValueError("counts must be integers with n > 0 and 0 <= successes <= n")


def srm_check(n_a: int, n_b: int, expected_ratio: float = 0.5) -> float:
    """Sample-ratio-mismatch chi-square p-value. p < 0.001 means the split is broken."""
    total = n_a + n_b
    _counts(n_a, total)
    _counts(n_b, total)
    if not 0 < expected_ratio < 1:
        raise ValueError("expected allocation must be strictly between 0 and 1")
    exp = [total * expected_ratio, total * (1 - expected_ratio)]
    return float(stats.chisquare([n_a, n_b], exp).pvalue)


def two_prop_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Two-sided pooled z-test, unpooled Wald CI for the absolute lift (b - a)."""
    _counts(x_a, n_a)
    _counts(x_b, n_b)
    _alpha(alpha)
    p_a, p_b = x_a / n_a, x_b / n_b
    pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    q = stats.norm.ppf(1 - alpha / 2)
    d = p_b - p_a
    # Zero observed variance does not establish zero population uncertainty.
    # The normal CI is unavailable for two constant arms; relative lift is
    # undefined when the control conversion rate is zero.
    return {"p_a": p_a, "p_b": p_b, "diff": d,
            "ci_low": d - q * se if se > 0 else None,
            "ci_high": d + q * se if se > 0 else None,
            "rel_lift": d / p_a if p_a > 0 else None, "z": float(z), "p_value": float(p)}


def mde(p_base: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable absolute effect for a two-sided two-proportion test."""
    _planning(p_base, alpha, power)
    _positive_integer(n_per_arm, "n_per_arm")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za + zb) * np.sqrt(2 * p_base * (1 - p_base) / n_per_arm))


def sample_size(p_base: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users per arm needed to detect an absolute lift of `delta`."""
    _planning(p_base, alpha, power)
    if not np.isfinite(delta) or delta == 0 or not 0 < p_base + delta < 1:
        raise ValueError("effect must be nonzero and leave the treatment rate strictly between 0 and 1")
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p2 = p_base + delta
    var = p_base * (1 - p_base) + p2 * (1 - p2)
    return int(np.ceil((za + zb) ** 2 * var / delta ** 2))


def holm(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which hypotheses are rejected."""
    _alpha(alpha)
    pvals = np.asarray(pvals, dtype=float)
    if pvals.ndim != 1 or not np.isfinite(pvals).all() or ((pvals < 0) | (pvals > 1)).any():
        raise ValueError("p-values must be a finite vector in [0, 1]")
    order = np.argsort(pvals)
    reject = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - rank):
            reject[i] = True
        else:
            break
    return reject


def bootstrap_diff(a: np.ndarray, b: np.ndarray, stat=np.mean, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap CI for stat(b) - stat(a)."""
    _positive_integer(n_boot, "n_boot")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all() for x in (a, b)):
        raise ValueError("bootstrap arms must be nonempty finite vectors")
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = stat(rng.choice(b, len(b))) - stat(rng.choice(a, len(a)))
    if not np.isfinite(d).all():
        raise ValueError("bootstrap statistic must return finite values")
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bayes_prob_b_better(x_a: int, n_a: int, x_b: int, n_b: int, draws: int = 200_000, seed: int = 0) -> float:
    """P(rate_b > rate_a) under flat Beta(1,1) priors."""
    _counts(x_a, n_a)
    _counts(x_b, n_b)
    _positive_integer(draws, "draws")
    rng = np.random.default_rng(seed)
    a = rng.beta(1 + x_a, 1 + n_a - x_a, draws)
    b = rng.beta(1 + x_b, 1 + n_b - x_b, draws)
    return float((b > a).mean())


def aa_false_positive_rate(outcome: np.ndarray, sims: int = 1000, alpha: float = 0.05, seed: int = 0) -> float:
    """Randomly split one population into two fake arms; share of 'significant' results should be ~alpha."""
    _positive_integer(sims, "sims")
    _alpha(alpha)
    outcome = np.asarray(outcome)
    if outcome.ndim != 1 or len(outcome) < 2 or not np.isin(outcome, [0, 1]).all():
        raise ValueError("A/A outcomes must be a binary vector with at least two observations")
    rng = np.random.default_rng(seed)
    n, hits = len(outcome), 0
    for _ in range(sims):
        m = rng.permutation(n) < n // 2
        xa, xb = outcome[m].sum(), outcome[~m].sum()
        hits += two_prop_test(int(xa), int(m.sum()), int(xb), int((~m).sum()))["p_value"] < alpha
    return hits / sims
