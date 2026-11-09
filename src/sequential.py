"""Sequential testing: group-sequential boundaries (Lan-DeMets alpha spending) and a normal-mixture mSPRT approximation.

Peeking at a fixed-sample test inflates false positives. These tools let you look repeatedly and keep the
error rate near alpha under the simulated assumptions.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .abtest import _alpha, _positive_integer


def _fractions(t, alpha):
    _alpha(alpha)
    t = np.asarray(t, dtype=float)
    if not np.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("information fractions must be finite and in [0, 1]")
    return t


def spend_obf(t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Lan-DeMets O'Brien-Fleming-type cumulative alpha spent at information fraction t (two-sided)."""
    t = _fractions(t, alpha)
    z = np.divide(stats.norm.ppf(1 - alpha / 2), np.sqrt(t), out=np.full_like(t, np.inf), where=t > 0)
    return 2 * stats.norm.sf(z)


def spend_pocock(t: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Lan-DeMets Pocock-type cumulative alpha spent at information fraction t."""
    t = _fractions(t, alpha)
    return alpha * np.log(1 + (np.e - 1) * t)


def _survive(fy: np.ndarray, ys: np.ndarray, c: float, tk: float) -> float:
    """Probability mass of the density `fy` inside |B| < c * sqrt(tk)."""
    return float(fy[np.abs(ys) < c * np.sqrt(tk)].sum() * (ys[1] - ys[0]))


def _look_times(looks) -> np.ndarray:
    """Information fractions for the looks: an int means equally spaced, a sequence is used as given."""
    if np.ndim(looks) == 0:
        _positive_integer(looks, "looks")
        return np.arange(1, looks + 1) / looks
    t = np.asarray(looks, dtype=float)
    if (t.ndim != 1 or not len(t) or not np.isfinite(t).all()
            or t[0] <= 0 or t[-1] > 1 or (np.diff(t) <= 0).any()):
        raise ValueError("look times must be a nonempty strictly increasing vector in (0, 1]")
    return t


def gs_boundaries(looks, alpha: float = 0.05, spending=spend_obf, grid: int = 3001) -> np.ndarray:
    """Two-sided z-boundaries for the looks.

    `looks` is either a number of equally spaced looks or a strictly increasing sequence of information
    fractions in (0, 1] (for example [0.1, 0.5, 1.0] for unequal spacing). The fraction is the share of the
    planned sample (or events) observed at the look; the spending function is evaluated at those fractions.

    Solved by numerical integration of the Brownian-motion density through the continuation region
    (Armitage-McPherson-Rowe recursion), so each look spends exactly its share of alpha.
    """
    t = _look_times(looks)
    looks = len(t)
    _positive_integer(grid, "grid")
    _alpha(alpha)
    if grid < 101:
        raise ValueError("numerical integration requires at least 101 grid points")
    cum = spending(t, alpha)
    cum = np.asarray(cum, dtype=float)
    if (cum.shape != t.shape or not np.isfinite(cum).all() or (cum < 0).any()
            or (cum > alpha + 1e-12).any() or (np.diff(cum) < 0).any()):
        raise ValueError("cumulative spending must match the looks, increase and stay within alpha")
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
    with plug-in estimated variance, the running minimum is an approximation, not an exact guarantee.
    """
    if not np.isfinite([theta_hat, var_hat, tau]).all() or var_hat <= 0 or tau <= 0:
        raise ValueError("effect must be finite; variance and prior width must be finite and positive")
    v, t2 = var_hat, tau**2
    return float(np.sqrt(v / (v + t2)) * np.exp(t2 * theta_hat**2 / (2 * v * (v + t2))))


def always_valid_p(ratios: np.ndarray) -> np.ndarray:
    """Running minimum of min(1, 1/ratio); validity depends on the input likelihood-ratio process."""
    ratios = np.asarray(ratios, dtype=float)
    if ratios.ndim != 1 or not len(ratios) or np.isnan(ratios).any() or (ratios < 0).any():
        raise ValueError("likelihood ratios must be a nonempty nonnegative vector without NaN")
    inverse = np.divide(1.0, ratios, out=np.full_like(ratios, np.inf), where=ratios > 0)
    return np.minimum.accumulate(np.minimum(1.0, inverse))
