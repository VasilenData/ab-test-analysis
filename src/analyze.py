"""Cookie Cats gate_30 vs gate_40 experiment analysis. Run: python -m src.analyze"""
import json

import pandas as pd

from src.abtest import *

df = pd.read_csv("data/cookie_cats.csv")
a, b = df[df.version == "gate_30"], df[df.version == "gate_40"]
out = {"rows": len(df), "n_a": len(a), "n_b": len(b), "unique_users": int(df.userid.nunique())}
out["srm_p"] = srm_check(len(a), len(b))
out["bot_like_max_rounds"] = int(df.sum_gamerounds.max())
out["zero_round_share"] = float((df.sum_gamerounds == 0).mean())

res = {}
for m in ["retention_1", "retention_7"]:
    r = two_prop_test(int(a[m].sum()), len(a), int(b[m].sum()), len(b))
    r["bayes_p_b_better"] = bayes_prob_b_better(int(a[m].sum()), len(a), int(b[m].sum()), len(b))
    r["aa_false_positive"] = aa_false_positive_rate(df[m].astype(int).values, sims=500)
    res[m] = r
hp = holm([res["retention_1"]["p_value"], res["retention_7"]["p_value"]])
res["retention_1"]["holm_reject"], res["retention_7"]["holm_reject"] = hp
out["metrics"] = res

# engagement (rounds): winsorize at p99 and compare with bootstrap, plus Mann-Whitney
cap = df.sum_gamerounds.quantile(0.99)
wa, wb = a.sum_gamerounds.clip(upper=cap).values, b.sum_gamerounds.clip(upper=cap).values
out["rounds"] = {"cap_p99": float(cap), "mean_a": float(wa.mean()), "mean_b": float(wb.mean()),
                 "boot_ci_diff": bootstrap_diff(wa, wb, n_boot=1000),
                 "mannwhitney_p": float(stats.mannwhitneyu(a.sum_gamerounds, b.sum_gamerounds).pvalue)}
out["power"] = {"mde_ret7_abs": mde(float(a.retention_7.mean()), len(a)),
                "n_needed_for_0.5pp_ret7": sample_size(float(a.retention_7.mean()), 0.005)}
# segment: players who never reached the gate (<30 rounds) cannot be affected by it
seg = {}
for name, mask in {"under_30_rounds": df.sum_gamerounds < 30, "30_plus_rounds": df.sum_gamerounds >= 30}.items():
    s = df[mask]; sa, sb = s[s.version == "gate_30"], s[s.version == "gate_40"]
    seg[name] = {"n": len(s), **two_prop_test(int(sa.retention_7.sum()), len(sa), int(sb.retention_7.sum()), len(sb))}
out["segments_ret7"] = seg
with open("reports/results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(json.dumps(out, indent=2, default=float))
