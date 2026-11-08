"""Sequential testing: group-sequential boundaries (Lan-DeMets alpha spending) and always-valid mSPRT.

Peeking at a fixed-sample test inflates false positives. These tools let you look repeatedly and keep the
error rate at alpha.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats


def spend_obf(t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Lan-DeMets O'Brien-Fleming-type cumulative alpha spent at information fraction t (two-sided)."""
    t = np.asarray(t, dtype=float)
    return 2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))


def spend_pocock(t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Lan-DeMets Pocock-type cumulative alpha spent at information fraction t."""
    t = np.asarray(t, dtype=float)
    return alpha * np.log(1 + (np.e - 1) * t)


def _survive(fy: np.ndarray, ys: np.ndarray, c: float, tk: float) -> float:
    """Probability mass of the density `fy` inside |B| < c * sqrt(tk)."""
    return float(fy[np.abs(ys) < c * np.sqrt(tk)].sum() * (ys[1] - ys[0]))


def gs_boundaries(looks: int, alpha: float = 0.05, spending=spend_obf, grid: int = 3001) -> np.ndarray:
    """Two-sided z-boundaries for `looks` equally spaced looks.

    Solved by numerical integration of the Brownian-motion density through the continuation region
    (Armitage-McPherson-Rowe recursion), so each look spends exactly its share of alpha.
    """
    t = np.arange(1, looks + 1) / looks
    cum = spending(t, alpha)
    spend = np.diff(np.concatenate([[0.0], cum]))
    bounds = np.empty(looks)
    prev_x = prev_f = None
    survive_prev = 1.0
    for k in range(looks):
        tk = float(t[k])
        ys = np.linspace(-7 * np.sqrt(tk), 7 * np.sqrt(tk), grid)
        if k == 0:
            fy = stats.norm.pdf(ys, scale=np.sqrt(tk))
        else:
            dt = tk - float(t[k - 1])
            kern = stats.norm.pdf(ys[:, None], loc=prev_x[None, :], scale=np.sqrt(dt))
            fy = (kern * prev_f[None, :]).sum(axis=1) * (prev_x[1] - prev_x[0])
        target = survive_prev - float(spend[k])
        bounds[k] = optimize.brentq(lambda c, f=fy, y=ys, tt=tk, tg=target: _survive(f, y, c, tt) - tg,
                                    0.3, 9.0, xtol=1e-6)
        keep = np.abs(ys) < bounds[k] * np.sqrt(tk)
        prev_x, prev_f = ys[keep], fy[keep]
        survive_prev = _survive(fy, ys, bounds[k], tk)
    return bounds


def msprt_normal(theta_hat: float, var_hat: float, tau: float) -> float:
    """Mixture-SPRT likelihood ratio for H0: theta = 0 with a N(0, tau^2) mixing prior.

    theta_hat: observed difference, var_hat: its estimated variance. Reject when ratio >= 1/alpha;
    the running minimum of 1/ratio is an always-valid p-value.
    """
    v, t2 = var_hat, tau**2
    return float(np.sqrt(v / (v + t2)) * np.exp(t2 * theta_hat**2 / (2 * v * (v + t2))))


def always_valid_p(ratios: np.ndarray) -> np.ndarray:
    """Running minimum of min(1, 1/ratio): valid at any stopping time."""
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / np.asarray(ratios, dtype=float)))
