"""نمودار معماری کل سامانه‌ی ساخته‌شده → notes/architecture.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 9.4))
ax.set_xlim(0, 160); ax.set_ylim(0, 94); ax.axis("off")

COL = {"data": ("#f2f2f2", "#555555"), "p1": ("#dbe9f8", "#2b6cb0"), "p2": ("#d7f0e4", "#2f855a"),
       "brain": ("#eaf6ee", "#2f855a"), "p3": ("#fde8d0", "#c05621"), "out": ("#ece6f7", "#6b46c1"),
       "grey": ("#ffffff", "#8a8a8a")}


def box(x, y, w, h, title, lines, kind, dashed=False, tsize=10.5, lsize=9.3):
    face, edge = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=face, ec=edge, lw=1.8, ls="--" if dashed else "-"))
    ax.text(x + 1.4, y + h - 1.3, title, fontsize=tsize, fontweight="bold", color=edge, va="top", ha="left")
    for i, t in enumerate(lines):
        ax.text(x + 1.4, y + h - 4.3 - i * 2.45, t, fontsize=lsize, color="#222222", va="top", ha="left")


def arrow(p, q, color="#333333", ls="-", lw=1.8, rad=0.0, label=None, lpos=None, lsize=8.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, fontsize=lsize, color=color, style="italic", ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))


ax.text(80, 92.3, "Light CARE-MD: architecture of what we built", fontsize=15, fontweight="bold", ha="center")

# ───── datasets ─────
box(4, 76, 62, 12, "HAM10000", ["10,015 dermoscopic images · lesion masks · 7 diagnoses",
                                 "split by lesion_id (no lesion in both parts)"], "data")
box(94, 76, 62, 12, "Derm7pt  (clinical 7-point checklist)", ["1,011 cases · 7 expert-labelled signs · melanoma yes / no",
                                                              "official split: 413 train · 203 valid · 395 test"], "data")

# ───── main inference path ─────
box(2, 36, 16, 26, "Image", ["dermoscopic", "768 × 512"], "data")
box(22, 36, 30, 26, "PHASE 1 · Observation", ["Attention U-Net", "(ResNet18 encoder)", "→ lesion mask",
                                              "→ biggest blob + 15% margin", "→ square crop 224×224"], "p1")
box(56, 36, 40, 26, "PHASE 2 · Interpretation", ["the 'Eye' = ResNet18 backbone", "ImageNet → HAM10000 → Derm7pt",
                                                              "512-d image feature", "7 linear heads, one per sign",
                                                              "→ 19 concept probabilities"], "p2")
box(100, 36, 24, 26, "the 'Brain'", ["Linear 19 → 1", "melanoma score z", "threshold chosen on", "valid (recall ≥ 90%)"], "brain")
box(128, 36, 30, 26, "OUTPUT  (explanation)", ["diagnosis + score", "top signs the eye saw", "3 similar training cases"], "out")

for (a, b) in [((18, 49), (22, 49)), ((52, 49), (56, 49)), ((96, 49), (100, 49)), ((124, 49), (128, 49))]:
    arrow(a, b)

# ───── Phase 3 ─────
box(56, 10, 40, 20, "PHASE 3 · Reference", ["memory bank: 413 training cases", "similarity: cosine / L1 / triplet",
                                           "→ 3 most similar cases"], "p3")
arrow((76, 36), (76, 30), color=COL["p3"][1])
ax.plot([96, 143], [20, 20], color=COL["p3"][1], lw=1.8)
arrow((143, 20), (143, 36), color=COL["p3"][1])

# ───── Phase 4 (not done) and baseline ─────
box(22, 10, 30, 20, "PHASE 4 · Validation", ["NOT DONE YET", "stability of signs and", "neighbours under small", "image changes"], "grey", dashed=True)
arrow((52, 20), (56, 20), color=COL["grey"][1], ls="--")
box(2, 1.5, 156, 5.5, "", [], "grey", dashed=True)
ax.text(3.4, 4.25, "Baseline for comparison:  black box", fontsize=9.5, fontweight="bold", color=COL["grey"][1], va="center", ha="left")
ax.text(40, 4.25, "image → ResNet18 (same pre-training) → melanoma score.  Same training, same 3 seeds, no signs, no explanation.",
        fontsize=9.2, color="#222222", va="center", ha="left")

# ───── training-time arrows (dotted) ─────
T = "#6b46c1"
arrow((22, 76), (34, 62), T, ":", 1.7, label="lesion masks", lpos=(19, 69))
arrow((58, 76), (66, 62), T, ":", 1.7, label="10k diagnoses\n(pre-training)", lpos=(55, 69))
arrow((100, 76), (88, 62), T, ":", 1.7, label="413 train cases:\n7 signs", lpos=(100, 69))
arrow((120, 76), (112, 62), T, ":", 1.7, label="melanoma labels +\neye's honest opinions\n(5-fold)", lpos=(124, 69))
ax.text(158, 90.3, "dotted arrows = used only during training", fontsize=8.6, color=T, style="italic", ha="right")

fig.savefig("notes/architecture.png", dpi=130, bbox_inches="tight", facecolor="white")
print("saved notes/architecture.png")
