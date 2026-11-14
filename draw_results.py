"""Final comparison figure for the README: melanoma AUROC, mean ± std over 3 seeds (from notes/step9_results.csv)."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = pd.read_csv("notes/step9_results.csv")
order = [("sequential", "Ours: sequential CBM\n(explains itself)", "#2f855a"),
         ("joint λ=1", "Joint CBM, label weight 1\n(paper's training)", "#6b46c1"),
         ("joint λ=5", "Joint CBM, label weight 5", "#9f7aea"),
         ("black box", "Black box\n(same pre-training, no explanation)", "#555555")]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (key, label, col) in enumerate(order[::-1]):
    q = R[R.method == key]
    for j, (split, dy, marker) in enumerate([("valid", 0.17, "o"), ("test", -0.17, "s")]):
        v = q[f"{split}_mel"].values
        ax.errorbar(v.mean(), i + dy, xerr=v.std(), fmt=marker, color=col, ms=8, capsize=3, lw=1.6,
                    mfc=col if split == "test" else "white", mec=col)
        ax.scatter(v, [i + dy] * len(v), s=14, color=col, alpha=.45, zorder=1)
ax.axvline(0.926, color="#999999", ls="--", lw=1)
ax.text(0.9235, 3.62, "ceiling: expert-labelled signs\n→ linear classifier (test)", fontsize=8, color="#777777", ha="right", va="top")
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[1] for o in order[::-1]], fontsize=9)
ax.set_xlim(0.78, 0.93); ax.set_ylim(-0.6, 3.75)
ax.set_xlabel("melanoma AUROC   (dots = individual seeds, bar = ± 1 std)")
ax.plot([], [], "o", mfc="white", mec="k", color="k", label="validation (used for all decisions)")
ax.plot([], [], "s", color="k", label="test (395 cases, 101 melanomas)")
ax.legend(loc="upper left", fontsize=8, frameon=False)
ax.grid(axis="x", alpha=.3)
ax.set_title("Explainable chain vs. black box, same data and pre-training, 3 seeds", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig("notes/results.png", dpi=130)
print("saved notes/results.png")
