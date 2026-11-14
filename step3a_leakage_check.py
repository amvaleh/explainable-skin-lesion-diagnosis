"""
قدم ۳-الف — بررسی تقلب: آیا عکسی هم در HAM10000 هست و هم در Derm7pt؟

اگر بیماری از امتحان Derm7pt قبلاً در HAM10000 دیده شده باشد، پیش‌آموزش روی
HAM10000 نتیجه را تقلبی می‌کند. برای هر عکس یک «اثر انگشت» ۶۴ بیتی می‌سازیم
(dHash) و اثر انگشت‌های دو مجموعه را با هم مقایسه می‌کنیم.

محدودیت: این روش عکس‌های یکسان یا تغییر اندازه‌داده را پیدا می‌کند، ولی اگر
عکسی بریده یا چرخانده شده باشد، ممکن است پیدا نشود.
"""
import glob
import os
import time
import numpy as np
from PIL import Image

from derm7pt_data import ROOT, load_table


def dhash(path, size=8):
    """عکس → ۶۴ بیت: کوچک کن، خاکستری کن، هر پیکسل را با همسایه‌ی راستش مقایسه کن."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def hash_all(paths, name):
    t0 = time.time()
    out = np.zeros((len(paths), 8), dtype=np.uint8)
    for i, p in enumerate(paths):
        out[i] = dhash(p)
        if (i + 1) % 1000 == 0 or i + 1 == len(paths):
            print(f"  {name}: {i + 1:5d}/{len(paths)} hashed  ({time.time() - t0:4.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ham_paths = sorted(glob.glob("data/ham10000/images/*.jpg"))
    df = load_table()
    d7_paths = [f"{ROOT}/images/{p}" for p in df.derm]

    print(f"HAM10000 images: {len(ham_paths)}   Derm7pt dermoscopic images: {len(d7_paths)}\n")
    H = hash_all(ham_paths, "HAM10000")
    D = hash_all(d7_paths, "Derm7pt ")

    # فاصله‌ی همینگ: در چند بیت از ۶۴ بیت فرق دارند؟ صفر یعنی یکسان.
    Hb = np.unpackbits(H, axis=1).astype(bool)
    Db = np.unpackbits(D, axis=1).astype(bool)
    print("\ncomparing every Derm7pt image with every HAM10000 image ...", flush=True)
    dist = np.zeros((len(D), len(H)), dtype=np.uint8)
    for i in range(len(D)):
        dist[i] = (Hb != Db[i]).sum(1)
    nearest = dist.min(1)

    print("\nnearest HAM10000 match for each Derm7pt image (bits different out of 64):")
    for lo, hi, label in [(0, 0, "identical"), (1, 5, "near-duplicate"),
                          (6, 10, "suspicious"), (11, 64, "different")]:
        n = int(((nearest >= lo) & (nearest <= hi)).sum())
        print(f"  {lo:2d}–{hi:2d} bits  {label:<15} {n:5d}")

    # ثبت موارد مشکوک برای بررسی چشمی
    sus = np.where(nearest <= 10)[0]
    os.makedirs("notes", exist_ok=True)
    with open("notes/step3a_leakage_candidates.csv", "w") as f:
        f.write("case_num,split,derm_image,ham_image,bits\n")
        for i in sus:
            j = int(dist[i].argmin())
            f.write(f"{df.case_num.iloc[i]},{df.split.iloc[i]},{d7_paths[i]},{ham_paths[j]},{nearest[i]}\n")
    print(f"\n{len(sus)} candidate pairs written to notes/step3a_leakage_candidates.csv")
    if len(sus) == 0:
        print("VERDICT: no duplicates found → safe to pretrain on HAM10000")
    else:
        by_split = df.split.iloc[sus].value_counts().to_dict()
        print(f"VERDICT: check these pairs by eye before pretraining. by split: {by_split}")
