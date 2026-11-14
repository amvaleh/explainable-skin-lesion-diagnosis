"""نمودار لایه‌های شبکه‌ها با اندازه‌ی واقعی تنسورها و تعداد وزن‌ها (اعداد از خود مدل‌ها گرفته شده) → notes/layers.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

fig, axes = plt.subplots(3, 1, figsize=(16, 16.5), gridspec_kw={"height_ratios": [4.6, 3.7, 6.6], "hspace": 0.12})
GREEN, GREEN_L, BLUE, BLUE_L, ORANGE, GREY, VIOLET = "#2f855a", "#d7f0e4", "#2b6cb0", "#dbe9f8", "#c05621", "#6a6a6a", "#6b46c1"


def rect(ax, cx, cy, w, h, face, edge, lw=1.6, dashed=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=face, ec=edge, lw=lw, ls="--" if dashed else "-"))


def arr(ax, p, q, color="#333333", lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def panel_title(ax, text, sub):
    ax.text(0.5, 1.0, text, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    ax.text(0.5, 0.965, "", transform=ax.transAxes)
    ax.text(159.5, ax.get_ylim()[1], sub, fontsize=9.5, color=GREY, ha="right", va="bottom", style="italic")


# ═════════════════ پنل ۱: ستون فقرات ResNet18 ═════════════════
ax = axes[0]; ax.set_xlim(0, 160); ax.set_ylim(-6, 47); ax.axis("off")
stages = [  # name, shape, params, h, w
    ("Input", "3 × 224 × 224", "", 26, 4),
    ("conv1 7×7/2\n+ BN + ReLU", "64 × 112 × 112", "9,536", 22, 9),
    ("max-pool 3×3/2", "64 × 56 × 56", "0", 14, 9),
    ("layer1", "64 × 56 × 56", "147,968", 14, 9),
    ("layer2", "128 × 28 × 28", "525,568", 9, 13),
    ("layer3", "256 × 14 × 14", "2,099,712", 6, 17),
    ("layer4", "512 × 7 × 7", "8,393,728", 4, 21),
    ("global\naverage pool", "512-d vector", "0", 14, 3),
]
shade = ["#f2f2f2", "#e6f4ec", "#e6f4ec", "#cfeadb", "#b9dfca", "#9fd2b7", "#82c3a2", "#f2f2f2"]
x, CY, GAP = 3.0, 28, 5.6
centers = []
for (name, shp, par, h, w), fc in zip(stages, shade):
    cx = x + w / 2
    rect(ax, cx, CY, w, h, fc, GREEN)
    lines = name.split("\n") + [shp] + ([f"{par} weights"] if par else [])
    ax.text(cx, 11.5, "\n".join(lines), fontsize=8.4, ha="center", va="top", linespacing=1.45)
    centers.append((cx, w))
    x += w + GAP
for i in range(len(centers) - 1):
    arr(ax, (centers[i][0] + centers[i][1] / 2, CY), (centers[i + 1][0] - centers[i + 1][1] / 2, CY))
x1, x2 = centers[3][0] - centers[3][1] / 2, centers[6][0] + centers[6][1] / 2
ax.plot([x1, x2], [39.5, 39.5], color=GREEN, lw=1.6)
ax.plot([x1, x1], [39.5, 38.3], color=GREEN, lw=1.6); ax.plot([x2, x2], [39.5, 38.3], color=GREEN, lw=1.6)
ax.text((x1 + x2) / 2, 40.6, "4 'layers', each = 2 residual blocks   ·   spatial size halves, channels double",
        fontsize=9.2, color=GREEN, ha="center", va="bottom", fontweight="bold")
ax.text(3, -1.2, "ResNet18 backbone: 11,176,512 weights.  Starts from ImageNet, then fine-tuned on HAM10000 (10k lesions), then on Derm7pt.",
        fontsize=9.2, color="#222222", va="center")
ax.text(3, -4.2, "Blocks are drawn with height ∝ spatial size and width ∝ number of channels.", fontsize=8.6, color=GREY, va="center", style="italic")

# ───── داخل یک بلوک باقی‌مانده ─────
ix, iw = 145, 16
rect(ax, ix, 28, iw + 12, 38, "#ffffff", GREY, lw=1.2, dashed=True)
ax.text(ix, 45.6, "inside each residual block", fontsize=8.8, ha="center", va="top", color=GREY, fontweight="bold")
items = ["conv 3×3", "BatchNorm", "ReLU", "conv 3×3", "BatchNorm", "+  skip", "ReLU"]
ys = [39.8 - i * 4.7 for i in range(len(items))]
for t, yy in zip(items, ys):
    rect(ax, ix, yy, iw, 3.2, GREEN_L if t != "+  skip" else "#fde8d0", GREEN if t != "+  skip" else ORANGE, lw=1.2)
    ax.text(ix, yy, t, fontsize=8.4, ha="center", va="center")
for a, b in zip(ys[:-1], ys[1:]):
    arr(ax, (ix, a - 1.6), (ix, b + 1.6), lw=1.2)
sk = ix + iw / 2 + 2.6
ax.plot([ix, sk], [ys[0] + 3.0, ys[0] + 3.0], color=ORANGE, lw=1.4)
ax.plot([ix, ix], [ys[0] + 3.0, ys[0] + 1.6], color=ORANGE, lw=1.4)
ax.plot([sk, sk], [ys[0] + 3.0, ys[5]], color=ORANGE, lw=1.4)
arr(ax, (sk, ys[5]), (ix + iw / 2, ys[5]), color=ORANGE, lw=1.4)

# ═════════════════ پنل ۲: سرها + مغز ═════════════════
ax = axes[1]; ax.set_xlim(0, 160); ax.set_ylim(0, 37); ax.axis("off")
heads = [("pigment network", 3, "absent / typical / atypical"), ("streaks", 3, "absent / regular / irregular"),
         ("pigmentation", 3, "absent / regular / irregular"), ("regression structures", 2, "absent / present"),
         ("dots & globules", 3, "absent / regular / irregular"), ("blue-whitish veil", 2, "absent / present"),
         ("vascular structures", 3, "absent / benign / irregular")]
rect(ax, 5.5, 17, 4.5, 24, "#f2f2f2", GREEN)
ax.text(5.5, 3.0, "512-d\nfeature", fontsize=8.6, ha="center", va="top", linespacing=1.4)
hx, hw, hh = 44, 56, 3.3
hy = [29.5 - i * 4.35 for i in range(7)]
for (nm, k, lv), yy in zip(heads, hy):
    rect(ax, hx, yy, hw, hh, GREEN_L, GREEN, lw=1.3)
    ax.text(hx - hw / 2 + 1.2, yy, f"{nm}:  Linear 512 → {k}", fontsize=8.2, va="center", ha="left")
    ax.text(hx + hw / 2 - 1.0, yy, lv, fontsize=7.4, va="center", ha="right", color=GREY)
    arr(ax, (7.8, 17 + (yy - 17) * 0.45), (hx - hw / 2, yy), lw=1.0, color=GREEN)
    arr(ax, (hx + hw / 2, yy), (77.0, 17 + (yy - 17) * 0.9), lw=1.0, color=GREEN)
ax.text(hx - 2, 33.2, "7 heads, one per clinical sign:  9,747 weights in total", fontsize=9.2, ha="center", va="bottom", color=GREEN, fontweight="bold")
rect(ax, 79, 17, 4.0, 26, "#f2f2f2", GREEN)
ax.text(79, 3.0, "19 concept\nprobabilities", fontsize=8.6, ha="center", va="top", linespacing=1.4)
arr(ax, (81, 17), (94, 17), lw=1.8)
rect(ax, 105, 17, 22, 14, "#eaf6ee", GREEN, lw=2)
ax.text(105, 21.6, "the 'Brain'", fontsize=10, ha="center", fontweight="bold", color=GREEN)
ax.text(105, 17.4, "Linear 19 → 1", fontsize=9, ha="center")
ax.text(105, 13.6, "20 weights", fontsize=8.6, ha="center", color=GREY)
arr(ax, (116, 17), (124, 17), lw=1.8)
rect(ax, 130, 17, 12, 8, "#ffffff", GREEN, lw=1.6)
ax.text(130, 17, "score z", fontsize=9.5, ha="center", va="center", fontweight="bold")
arr(ax, (136, 17), (144, 17), lw=1.8)
rect(ax, 151.5, 17, 15, 14, "#ece6f7", VIOLET, lw=2)
ax.text(151.5, 21.4, "z ≥ t", fontsize=10, ha="center", fontweight="bold", color=VIOLET)
ax.text(151.5, 17.6, "→ melanoma", fontsize=8.6, ha="center")
ax.text(151.5, 13.9, "t from validation", fontsize=7.6, ha="center", color=GREY)
ax.text(105, 7.2, "readable: every weight says how much one sign pushes\ntoward melanoma (positive) or away (negative)",
        fontsize=8.6, ha="center", va="top", color=GREY, style="italic", linespacing=1.4)

# ═════════════════ پنل ۳: Attention U-Net ═════════════════
ax = axes[2]; ax.set_xlim(0, 160); ax.set_ylim(0, 67); ax.axis("off")
LV = [58, 46.5, 35, 23.5, 12]
BW, BH = 13.5, 8.6
X = lambda c: 8 + c * 15.5 + (3.0 if c == 9 else 0)
blocks = {  # col, level, name, shape, params, kind
    "stem": (0, 0, "stem (conv1)", "64 × 128 × 128", "9,536", "enc"),
    "enc1": (1, 1, "enc1 (layer1)", "64 × 64 × 64", "147,968", "enc"),
    "enc2": (2, 2, "enc2 (layer2)", "128 × 32 × 32", "525,568", "enc"),
    "enc3": (3, 3, "enc3 (layer3)", "256 × 16 × 16", "2,099,712", "enc"),
    "enc4": (4, 4, "enc4 (layer4)", "512 × 8 × 8", "8,393,728", "enc"),
    "up4": (5, 3, "up4", "256 × 16 × 16", "2,459,009", "dec"),
    "up3": (6, 2, "up3", "128 × 32 × 32", "615,105", "dec"),
    "up2": (7, 1, "up2", "64 × 64 × 64", "153,953", "dec"),
    "up1": (8, 0, "up1", "32 × 128 × 128", "50,401", "dec"),
    "head": (9, 0, "head 1×1", "1 × 128 × 128", "33", "out"),
}
for k, (c, l, nm, shp, par, kind) in blocks.items():
    fc, ec = {"enc": (GREEN_L, GREEN), "dec": (BLUE_L, BLUE), "out": ("#ece6f7", VIOLET)}[kind]
    rect(ax, X(c), LV[l], BW, BH, fc, ec)
    ax.text(X(c), LV[l] + 2.2, nm, fontsize=8.4, ha="center", va="center", fontweight="bold", color=ec)
    ax.text(X(c), LV[l] - 0.4, shp, fontsize=8.2, ha="center", va="center")
    ax.text(X(c), LV[l] - 2.9, f"{par}", fontsize=7.8, ha="center", va="center", color=GREY)
# مسیر اصلی
chain = ["stem", "enc1", "enc2", "enc3", "enc4", "up4", "up3", "up2", "up1", "head"]
for a, b in zip(chain[:-1], chain[1:]):
    (ca, la), (cb, lb) = blocks[a][:2], blocks[b][:2]
    if la == lb:
        arr(ax, (X(ca) + BW / 2, LV[la]), (X(cb) - BW / 2, LV[lb]), lw=1.7)
    else:
        sgn = -1 if lb > la else 1      # پایین رفتن یا بالا رفتن
        arr(ax, (X(ca) + BW / 2 - 1, LV[la] + sgn * BH / 2), (X(cb) - BW / 2 + 1, LV[lb] - sgn * BH / 2), lw=1.7,
            color=GREEN if sgn == -1 else BLUE)
# پل‌ها با دروازه‌ی توجه
for e, d in [("stem", "up1"), ("enc1", "up2"), ("enc2", "up3"), ("enc3", "up4")]:
    (ce, le), (cd, ld) = blocks[e][:2], blocks[d][:2]
    gx = X(cd) - BW / 2 - 4.6
    ax.plot([X(ce) + BW / 2, gx - 2.6], [LV[le], LV[le]], color=ORANGE, lw=1.5, ls=(0, (5, 2)))
    ax.add_patch(Ellipse((gx, LV[le]), 5.2, 4.2, fc="#fde8d0", ec=ORANGE, lw=1.6))
    ax.text(gx, LV[le], "AG", fontsize=8.2, ha="center", va="center", fontweight="bold", color=ORANGE)
    arr(ax, (gx + 2.6, LV[le]), (X(cd) - BW / 2, LV[le]), color=ORANGE, lw=1.5)
ax.text(X(0) - 6.5, LV[0] + 7.6, "input  3 × 256 × 256", fontsize=8.6, color="#222222", ha="left")
arr(ax, (X(0) - 5.5, LV[0] + 6.5), (X(0) - 2.5, LV[0] + 4.4), lw=1.4)
ax.text(X(9) + 6.2, LV[0] - 6.8, "then upsampled to 256 × 256\n= lesion probability for every pixel",
        fontsize=8.2, color=VIOLET, ha="right", va="top", linespacing=1.4)
ax.text(2, 28.5, "AG = attention gate on each skip connection:", fontsize=9, fontweight="bold", color=ORANGE, va="top")
ax.text(2, 25.2, "α = σ( ψ( ReLU( Wx·skip + Wg·decoder ) ) )\ngated skip = α ⊙ skip", fontsize=8.8, va="top", linespacing=1.5)
ax.text(2, 18.4, "α is a 0–1 map: it switches off skin, hair and ruler\nbefore the skip is merged into the decoder.\n(gates total 133,892 weights)",
        fontsize=8.4, color=GREY, va="top", linespacing=1.5)
ax.text(2, 2.6, "green = encoder (ResNet18, pre-trained)    blue = decoder    orange = skip connection with gate",
        fontsize=8.4, color=GREY, va="center", style="italic")

axes[0].text(0.0, 1.02, "A · The 'Eye', part 1: image → 512-d feature  (ResNet18, input already cropped around the lesion)",
             transform=axes[0].transAxes, fontsize=12.5, fontweight="bold", va="bottom", ha="left")
axes[1].text(0.0, 1.02, "B · The 'Eye', part 2 and the 'Brain': feature → 19 probabilities → melanoma score",
             transform=axes[1].transAxes, fontsize=12.5, fontweight="bold", va="bottom", ha="left")
axes[2].text(0.0, 1.02, "C · Phase 1 network: Attention U-Net  (14,455,013 weights), draws the lesion outline",
             transform=axes[2].transAxes, fontsize=12.5, fontweight="bold", va="bottom", ha="left")
fig.savefig("notes/layers.png", dpi=110, bbox_inches="tight", facecolor="white")
print("saved notes/layers.png")
