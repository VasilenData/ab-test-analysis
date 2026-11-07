import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("reports/results.json") as f:
    r = json.load(f)["metrics"]
fig, ax = plt.subplots(figsize=(7, 2.8), facecolor="#111315")
ax.set_facecolor("#111315")
for i, (k, lab) in enumerate([("retention_1", "Day-1 retention"), ("retention_7", "Day-7 retention")]):
    m = r[k]
    ax.errorbar(m["diff"] * 100, i, xerr=[[ (m["diff"]-m["ci_low"])*100], [(m["ci_high"]-m["diff"])*100]],
                fmt="o", color="#7EE0B8", ecolor="#7EE0B8", capsize=4)
ax.axvline(0, color="#A3ACB3", lw=1, ls="--")
ax.set_yticks([0, 1]); ax.set_yticklabels(["Day-1 retention", "Day-7 retention"], color="#ECEFF1")
ax.set_xlabel("gate_40 minus gate_30, percentage points (95% CI)", color="#A3ACB3")
ax.tick_params(colors="#A3ACB3")
for s in ax.spines.values(): s.set_color("#2B3035")
ax.set_ylim(-0.6, 1.6); plt.tight_layout()
plt.savefig("reports/lift.png", dpi=160, facecolor="#111315")
