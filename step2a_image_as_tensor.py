"""
قدم ۲-الف — عکس برای کامپیوتر چیست؟

هیچ مدلی اینجا آموزش نمی‌بیند. فقط یک عکس را باز می‌کنیم،
به عدد تبدیلش می‌کنیم، و می‌بینیم «افزایش داده» چه شکلی است.
"""
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "data/derm7pt/release_v0"
torch.manual_seed(0)

# ───────────── ۱. یک بیمار را انتخاب و عکسش را باز کن ─────────────
df = pd.read_csv(f"{R}/meta/meta.csv")
row = df[df.diagnosis.str.contains("melanoma")].iloc[3]
img = Image.open(f"{R}/images/{row.derm}").convert("RGB")

print(f"case_num  : {row.case_num}")
print(f"diagnosis : {row.diagnosis}")
print(f"file      : {row.derm}")
print(f"size      : {img.size[0]} x {img.size[1]} pixels  (width x height)")

# ───────────── ۲. عکس → تنسور (آرایه‌ی عدد) ─────────────
x = T.ToTensor()(img)          # هر پیکسل سه عدد بین ۰ و ۱ می‌شود: قرمز، سبز، آبی
print(f"\ntensor shape : {tuple(x.shape)}   = (channels, height, width)")
print(f"total numbers: {x.numel():,}")
print(f"min / max    : {x.min():.3f} / {x.max():.3f}")

h, w = x.shape[1] // 2, x.shape[2] // 2
print(f"\nیک پیکسل در مرکز عکس (سطر {h}، ستون {w}):")
print(f"  red={x[0, h, w]:.3f}  green={x[1, h, w]:.3f}  blue={x[2, h, w]:.3f}")
print("\nیک تکه‌ی ۳×۳ از کانال قرمز، دور همان پیکسل:")
print(x[0, h-1:h+2, w-1:w+2].numpy().round(3))

# ───────────── ۳. آماده‌سازی برای شبکه‌ی آماده ─────────────
# شبکه‌ای که از آن استفاده خواهیم کرد روی عکس‌های ۲۲۴×۲۲۴ با این میانگین و
# انحراف معیار آموزش دیده، پس باید عکس خودمان را به همان شکل درآوریم.
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

prepare = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
xp = prepare(img)
print(f"\nafter prepare: shape {tuple(xp.shape)},  numbers {xp.numel():,}")
print(f"min / max    : {xp.min():.3f} / {xp.max():.3f}   (دیگر بین ۰ و ۱ نیست)")

# ───────────── ۴. افزایش داده: از یک عکس، چند نسخه ─────────────
# هر بار که این تابع صدا زده شود، یک نسخه‌ی تصادفی متفاوت می‌سازد.
augment = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),   # برش تصادفی و بزرگ‌نمایی
    T.RandomHorizontalFlip(),                     # آینه‌ی افقی
    T.RandomVerticalFlip(),                       # آینه‌ی عمودی
    T.RandomRotation(180),                        # چرخش تا ۱۸۰ درجه
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),  # نور و کنتراست؛ رنگ‌مایه دست‌نخورده
])

fig, ax = plt.subplots(2, 5, figsize=(15, 6.4))
ax[0, 0].imshow(img.resize((224, 224)))
ax[0, 0].set_title("original (resized)", fontsize=11, fontweight="bold")
for k, a in enumerate(ax.ravel()[1:], start=1):
    a.imshow(augment(img))
    a.set_title(f"augmented #{k}", fontsize=10)
for a in ax.ravel():
    a.axis("off")
fig.suptitle(f"case {row.case_num} - {row.diagnosis}: one image, nine random versions", fontsize=12)
fig.tight_layout()
fig.savefig("notes/step2a_augmentation.png", dpi=120)
print("\nsaved notes/step2a_augmentation.png")
