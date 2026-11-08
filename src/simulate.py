"""Simulation study: does sequential testing hold the error rate, and does CUPED cut variance?

Run: python -m src.simulate   (writes reports/sequential.json, reports/cuped.json, reports/sequential.png)
"""
import json

import numpy as np
import pandas as pd

from src.abtest import two_prop_test
from src.cuped import cuped_test
from src.sequential import (
    always_valid_p,
    gs_boundaries,
    msprt_normal,
    spend_obf,
    spend_pocock,
)

ALPHA = 0.05


def _looks_z(rng, p_a, p_b, n, looks, sims):
    """Cumulative two-proportion z at equally spaced looks, plus theta/var for mSPRT."""
    step = n // looks
    xa = rng.binomial(step, p_a, (sims, looks)).cumsum(axis=1)
    xb = rng.binomial(step, p_b, (sims, looks)).cumsum(axis=1)
    ns = step * np.arange(1, looks + 1)
    pa, pb = xa / ns, xb / ns
    pool = (xa + xb) / (2 * ns)
    se0 = np.sqrt(np.maximum(pool * (1 - pool) * 2 / ns, 1e-12))
    z = (pb - pa) / se0
    se1 = np.sqrt(np.maximum(pa * (1 - pa) / ns + pb * (1 - pb) / ns, 1e-12))
    return z, pb - pa, se1**2, ns


def _stop_frac(hit: np.ndarray, ns: np.ndarray, n: int) -> float:
    """Average share of the planned sample used before the first rejection (all of it if none)."""
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), hit.shape[1] - 1)
    return float((ns[first] / n).mean())


def sequential_study(p=0.19, effect=0.0082, n=20_000, looks=10, sims=4000, seed=7, tau=0.01):
    rng = np.random.default_rng(seed)
    obf = gs_boundaries(looks, ALPHA, spend_obf)
    poc = gs_boundaries(looks, ALPHA, spend_pocock)
    crit = 1.959964
    out = {"n_per_arm_max": n, "looks": looks, "sims": sims, "baseline": p,
           "obf_bounds": obf.round(3).tolist(), "pocock_bounds": poc.round(3).tolist()}
    for label, eff in (("null", 0.0), ("effect", effect)):
        z, th, var, ns = _looks_z(rng, p, p + eff, n, looks, sims)
        rej_fixed = np.abs(z[:, -1]) > crit
        rej_naive = (np.abs(z) > crit).any(axis=1)
        hit_obf, hit_poc = np.abs(z) > obf, np.abs(z) > poc
        lam = np.array([[msprt_normal(th[i, j], var[i, j], tau) for j in range(looks)] for i in range(sims)])
        hit_m = lam >= 1 / ALPHA

        out[label] = {
            "fixed_sample_final_look": float(rej_fixed.mean()),
            "naive_peeking_any_look": float(rej_naive.mean()),
            "obf": float(hit_obf.any(axis=1).mean()),
            "pocock": float(hit_poc.any(axis=1).mean()),
            "msprt": float(hit_m.any(axis=1).mean()),
            "avg_sample_fraction_at_stop": {"obf": _stop_frac(hit_obf, ns, n),
                                            "pocock": _stop_frac(hit_poc, ns, n),
                                            "msprt": _stop_frac(hit_m, ns, n)}}
    return out


def cuped_study(n=5000, effect=0.30, sims=2000, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        est_raw, est_adj, cov_hit, vr = [], [], 0, []
        for _ in range(sims):
            x = rng.standard_normal(2 * n)
            y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(2 * n)
            treat = rng.permutation(2 * n) < n
            y = y + effect * treat
            ya, yb, xa, xb = y[~treat], y[treat], x[~treat], x[treat]
            r = cuped_test(ya, xa, yb, xb)
            est_raw.append(r["raw_se"]); est_adj.append(r["diff"]); vr.append(r["variance_ratio"])
            cov_hit += r["ci_low"] <= effect <= r["ci_high"]
        rows.append({"rho": rho, "mean_estimate": float(np.mean(est_adj)), "true_effect": effect,
                     "variance_ratio_observed": float(np.mean(vr)), "variance_ratio_theory": 1 - rho**2,
                     "ci_coverage": cov_hit / sims,
                     "sample_size_needed_vs_raw": float(np.mean(vr))})
    # leakage demo: covariate that the treatment itself shifts (post-treatment) biases the estimate
    leak = []
    for _ in range(500):
        treat = rng.permutation(2 * n) < n
        x = rng.standard_normal(2 * n) + 0.5 * treat       # affected by treatment: NOT pre-experiment
        y = 0.5 * x + rng.standard_normal(2 * n) + effect * treat
        r = cuped_test(y[~treat], x[~treat], y[treat], x[treat])
        leak.append(r["diff"])
    return {"by_correlation": rows, "post_treatment_covariate": {
        "true_effect": effect + 0.25, "note": "total effect is direct + via covariate (0.30 + 0.25)",
        "cuped_estimate_mean": float(np.mean(leak)), "bias_vs_total": float(np.mean(leak) - (effect + 0.25))}}


def cookie_cats_illustration(path="data/cookie_cats.csv", looks=10, seed=3, tau=0.01):
    """Replay Cookie Cats day-7 retention in RANDOM order (no timestamps exist) to show stopping behavior."""
    df = pd.read_csv(path).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    obf = gs_boundaries(looks, ALPHA, spend_obf)
    a = (df.version == "gate_30").values
    r = df.retention_7.astype(int).values
    rows = []
    for k in range(1, looks + 1):
        m = int(len(df) * k / looks)
        aa, bb = a[:m], ~a[:m]
        res = two_prop_test(int(r[:m][aa].sum()), int(aa.sum()), int(r[:m][bb].sum()), int(bb.sum()))
        se = (res["ci_high"] - res["ci_low"]) / (2 * 1.959964)
        lam = msprt_normal(res["diff"], se**2, tau)
        rows.append({"look": k, "players": m, "diff_pp": round(res["diff"] * 100, 3), "z": round(res["z"], 3),
                     "obf_bound": round(float(obf[k - 1]), 3), "obf_stop": bool(abs(res["z"]) > obf[k - 1]),
                     "naive_stop": bool(abs(res["z"]) > 1.959964), "msprt_ratio": round(lam, 2),
                     "msprt_stop": bool(lam >= 1 / ALPHA)})
    pv = always_valid_p(np.array([x["msprt_ratio"] for x in rows]))
    for x, p in zip(rows, pv):
        x["always_valid_p"] = round(float(p), 4)
    return rows


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    seq, cup, cc = sequential_study(), cuped_study(), cookie_cats_illustration()
    _write("reports/sequential.json", seq)
    _write("reports/cuped.json", cup)
    _write("reports/cookie_cats_sequential.json", cc)
    print(json.dumps({"sequential": seq, "cuped": cup, "cookie_cats": cc}, indent=1))
