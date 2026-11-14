"""
قدم ۵-ب — بریدن عکس‌های Derm7pt دور خال.

برای هر یک از ۱۰۱۱ عکس درموسکوپی:
  ۱. شبکه‌ی قدم ۵-الف نقشه‌ی احتمال «خال بودن» هر پیکسل را می‌دهد   (معادله‌ی ۲ مقاله)
  ۲. آستانه‌ی ۰٫۵ → ماسک                                               (معادله‌ی ۳ مقاله)
  ۳. فقط بزرگ‌ترین تکه‌ی پیوسته‌ی ماسک نگه داشته می‌شود (تکه‌های پراکنده‌ی خط‌کش و قاب حذف)
  ۴. کادر دور آن تکه + ۱۵٪ حاشیه، مربع‌شده → برش عکس اصلی            (الگوریتم ۱، خط ۴ تا ۱۰)
  ۵. اگر ماسک خالی یا خیلی کوچک بود، کل عکس نگه داشته می‌شود

برش‌ها در data/derm7pt_crops/ ذخیره می‌شوند و یک برگه‌ی نمونه برای بررسی چشمی ساخته می‌شود.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from derm7pt_data import ROOT, load_table
from step5a_train_segmentation import AttentionUNet, seg_eval_tf, DEVICE
from torchvision import tv_tensors

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/step5a_attention_unet.pt"
OUT = "data/derm7pt_crops"
MARGIN, MIN_AREA = 0.15, 0.005


def crop_box(mask, margin=MARGIN):
    """کادر مربعی دور بزرگ‌ترین تکه‌ی ماسک، با حاشیه. None اگر ماسک خالی یا خیلی کوچک باشد."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return None, 0.0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    frac = keep.mean()
    if frac < MIN_AREA:
        return None, frac
    ys, xs = np.where(keep)
    H, W = mask.shape
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margin)
    side = min(side, H, W)
    y0 = int(np.clip(cy - side / 2, 0, H - side)); x0 = int(np.clip(cx - side / 2, 0, W - side))
    return (x0, y0, x0 + int(side), y0 + int(side)), frac


def main():
    os.makedirs(OUT, exist_ok=True)
    model = AttentionUNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    df = load_table()
    rows, keep_masks = [], {}
    for i, r in df.iterrows():
        img = Image.open(f"{ROOT}/images/{r.derm}").convert("RGB")
        W, H = img.size
        x = tv_tensors.Image(torch.from_numpy(np.asarray(img)).permute(2, 0, 1))
        x = seg_eval_tf(x).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))
            prob = torch.nn.functional.interpolate(prob, size=(H, W), mode="bilinear")[0, 0].cpu().numpy()
        mask = prob >= 0.5
        box, frac = crop_box(mask)
        fallback = box is None
        if fallback:
            box = (0, 0, W, H)
        img.crop(box).save(f"{OUT}/{r.case_num}.jpg", quality=95)
        rows.append({"case_num": r.case_num, "split": r.split, "x0": box[0], "y0": box[1],
                     "x1": box[2], "y1": box[3], "lesion_area_frac": round(frac, 4), "fallback_full_image": fallback})
        keep_masks[r.case_num] = mask
        if (len(rows)) % 100 == 0 or len(rows) == len(df):
            print(f"  {len(rows):4d}/{len(df)} images cropped", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv("notes/step5b_crop_boxes.csv", index=False)
    print(f"\nfallback to full image (no lesion found): {int(t.fallback_full_image.sum())} of {len(t)}")
    print(f"lesion covers on average {100 * t.lesion_area_frac.mean():.0f}% of the original image")
    side = (t.x1 - t.x0)
    print(f"crop side: median {int(side.median())} px (original images are 768×512)")

    # برگه‌ی نمونه: عکس اصلی با مرز پیدا‌شده و کادر برش، و کنارش برش نهایی
    rng = np.random.default_rng(0)
    pick = rng.choice(len(df), 12, replace=False)
    fig, ax = plt.subplots(4, 6, figsize=(18, 11.5))
    for k, i in enumerate(pick):
        r, b = df.iloc[i], t.iloc[i]
        a1, a2 = ax.ravel()[2 * k], ax.ravel()[2 * k + 1]
        a1.imshow(Image.open(f"{ROOT}/images/{r.derm}").convert("RGB"))
        a1.contour(keep_masks[r.case_num], levels=[0.5], colors="cyan", linewidths=1.2)
        a1.add_patch(patches.Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0,
                                       fill=False, edgecolor="yellow", linewidth=2))
        a1.set_title(f"case {r.case_num}: lesion boundary + crop box", fontsize=9)
        a2.imshow(Image.open(f"{OUT}/{r.case_num}.jpg"))
        a2.set_title("what the eye will see" + ("  (FALLBACK)" if b.fallback_full_image else ""), fontsize=9)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("Derm7pt: cyan = boundary found by the network (trained only on HAM10000), yellow = crop", fontsize=12)
    fig.tight_layout()
    fig.savefig("notes/step5b_crops_sample.png", dpi=90)
    print("saved notes/step5b_crops_sample.png  and  notes/step5b_crop_boxes.csv")


if __name__ == "__main__":
    main()
