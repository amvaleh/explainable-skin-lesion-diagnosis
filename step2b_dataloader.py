"""
قدم ۲-ب — رساندن عکس‌ها به مدل.

هنوز هیچ مدلی آموزش نمی‌بیند. فقط خط تولید داده را می‌سازیم:
    فایل عکس ← تنسور ← دسته‌ی ۱۶تایی ← (بعداً) مدل
"""
import time
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from derm7pt_data import (CRIT, LEVELS, MEAN, STD, load_table,
                          Derm7ptDataset, train_transform, eval_transform)

torch.manual_seed(0)
BATCH = 16

# ───────────── ۱. جدول و برچسب‌ها ─────────────
df = load_table()
print("هر مفهوم به شماره‌ی حالت تبدیل می‌شود (نه one-hot):")
for c in CRIT:
    print(f"  {c:<22} " + "  ".join(f"{i}={lvl}" for i, lvl in enumerate(LEVELS[c])))

print("\nتعداد هر حالت در بخش train:")
tr = df[df.split == "train"]
for c in CRIT:
    counts = tr[c + "_id"].value_counts().sort_index()
    print(f"  {c:<22} " + "  ".join(f"{LEVELS[c][i]}:{n}" for i, n in counts.items()))

# ───────────── ۲. Dataset: هر عضو یک بیمار ─────────────
ds = {
    "train": Derm7ptDataset(df, "train", train_transform),   # با افزایش داده
    "valid": Derm7ptDataset(df, "valid", eval_transform),    # بدون افزایش داده
    "test":  Derm7ptDataset(df, "test",  eval_transform),
}
print("\nتعداد بیمار در هر بخش:", {s: len(d) for s, d in ds.items()})

x, concepts, y = ds["train"][0]
print("\nیک عضو از Dataset (بیمار اول train):")
print(f"  image    shape {tuple(x.shape)}")
print(f"  concepts {concepts.tolist()}  ← هفت عدد، یکی برای هر معیار")
print(f"  y        {y.item()}")

# ───────────── ۳. DataLoader: بیماران را دسته‌دسته می‌آورد ─────────────
loader = {
    "train": DataLoader(ds["train"], batch_size=BATCH, shuffle=True),    # هر دور ترتیب تازه
    "valid": DataLoader(ds["valid"], batch_size=BATCH, shuffle=False),
    "test":  DataLoader(ds["test"],  batch_size=BATCH, shuffle=False),
}
print(f"\nتعداد دسته در هر دور: " + ", ".join(f"{s}={len(l)}" for s, l in loader.items()))

xb, cb, yb = next(iter(loader["train"]))
print("\nیک دسته:")
print(f"  images   shape {tuple(xb.shape)}   = (batch, channels, height, width)")
print(f"  concepts shape {tuple(cb.shape)}          = (batch, 7 concepts)")
print(f"  y        shape {tuple(yb.shape)}             = (batch,)")
print(f"  y values {yb.tolist()}")

# ───────────── ۴. زمان یک دور کامل روی train ─────────────
t0 = time.time()
n = 0
for xb_, cb_, yb_ in loader["train"]:
    n += len(yb_)
print(f"\nیک دور کامل train: {n} عکس در {time.time() - t0:.1f} ثانیه")

# ───────────── ۵. نمایش همان دسته با برچسب‌هایش ─────────────
mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)
short = {"pigment_network": "PN", "streaks": "STR", "pigmentation": "PIG",
         "regression_structures": "RS", "dots_and_globules": "DG",
         "blue_whitish_veil": "BWV", "vascular_structures": "VS"}

fig, ax = plt.subplots(2, 4, figsize=(16, 9.2))
for k, a in enumerate(ax.ravel()):
    img = (xb[k] * std + mean).clamp(0, 1).permute(1, 2, 0)   # برگرداندن نرمال‌سازی فقط برای نمایش
    a.imshow(img)
    a.axis("off")
    lines = [f"{short[c]}: {LEVELS[c][cb[k, j]]}" for j, c in enumerate(CRIT)
             if cb[k, j] != 0]
    label = "MELANOMA" if yb[k] == 1 else "non-melanoma"
    a.set_title(label, fontsize=11, fontweight="bold",
                color="#b2182b" if yb[k] == 1 else "#1d4f91")
    a.text(0.02, 0.02, "\n".join(lines) if lines else "all absent",
           transform=a.transAxes, fontsize=8.5, va="bottom",
           bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
fig.suptitle("first 8 images of one training batch (after augmentation)", fontsize=12)
fig.tight_layout()
fig.savefig("notes/step2b_batch.png", dpi=110)
print("saved notes/step2b_batch.png")
