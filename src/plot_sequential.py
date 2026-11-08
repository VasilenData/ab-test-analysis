import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, MUTED, MINT, AMBER, BLUE = "#111315", "#ECEFF1", "#A3ACB3", "#7EE0B8", "#F0B86E", "#8FB4FF"

with open("reports/cookie_cats_sequential.json") as f:
    cc = json.load(f)
with open("reports/sequential.json") as f:
    seq = json.load(f)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=BG, gridspec_kw={"width_ratios": [1.4, 1]})
for ax in (a, b):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#2B3035")
looks = [r["look"] for r in cc]
a.plot(looks, [r["obf_bound"] for r in cc], color=AMBER, lw=2, label="O'Brien-Fleming bound")
a.axhline(1.96, color=MUTED, ls="--", lw=1, label="Naive 1.96 line")
a.plot(looks, [abs(r["z"]) for r in cc], color=MINT, lw=2, marker="o", label="Cookie Cats |z| (random order)")
a.set_xlabel("Look (each adds 9,019 players)", color=MUTED)
a.set_ylabel("|z|", color=MUTED)
a.set_title("Replay: naive peeking stops at look 4, OBF at look 8", color=INK, fontsize=10, loc="left")
a.set_ylim(0, 6.5)
a.legend(facecolor=BG, edgecolor="#2B3035", labelcolor=INK, fontsize=8, loc="upper right")
n = seq["null"]
names = ["Fixed sample\n(one look)", "Naive peeking\n(10 looks)", "O'Brien-\nFleming", "Pocock", "mSPRT"]
vals = [n["fixed_sample_final_look"], n["naive_peeking_any_look"], n["obf"], n["pocock"], n["msprt"]]
bars = b.bar(names, [v * 100 for v in vals], color=[BLUE, AMBER, MINT, MINT, MINT])
b.axhline(5, color=MUTED, ls="--", lw=1)
for r, v in zip(bars, vals):
    b.text(r.get_x() + r.get_width() / 2, v * 100 + 0.4, f"{v * 100:.1f}%", ha="center", color=INK, fontsize=8)
b.set_ylabel("False positives, no true effect (%)", color=MUTED)
b.set_title("Simulated error rate, 4,000 runs", color=INK, fontsize=10, loc="left")
b.tick_params(axis="x", labelsize=7)
plt.tight_layout()
plt.savefig("reports/sequential.png", dpi=150, facecolor=BG)
