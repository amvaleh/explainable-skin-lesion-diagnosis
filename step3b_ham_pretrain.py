"""
قدم ۳-ب — تنظیم دقیق ۱: شبکه‌ی آماده (ImageNet) با ده هزار خال HAM10000 تمرین می‌کند.

کار تمرینی: تشخیص ۷ نوع ضایعه. این تشخیص را بعداً لازم نداریم؛ هدف این است که
«بدنه»ی شبکه به دیدن پوست عادت کند. در پایان فقط بدنه ذخیره می‌شود و سرِ تشخیص
دور ریخته می‌شود.

تقسیم بر اساس lesion_id است تا عکس‌های یک ضایعه هم در آموزش و هم در اعتبارسنجی نیفتند.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

from derm7pt_data import train_transform, eval_transform

QUICK = os.environ.get("QUICK") == "1"          # اجرای آزمایشی کوچک
EPOCHS = 1 if QUICK else 8
BATCH, LR = 32, 1e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


class HamDataset(Dataset):
    def __init__(self, table, transform):
        self.t = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        r = self.t.iloc[i]
        x = self.transform(Image.open(f"data/ham10000/images/{r.image_id}.jpg").convert("RGB"))
        return x, CLASSES.index(r.dx)


def main():
    torch.manual_seed(0)
    os.makedirs("checkpoints", exist_ok=True)

    meta = pd.read_csv("data/ham10000/raw/HAM10000_metadata.tab", sep="\t")
    for c in meta.select_dtypes(include=["object", "string"]):
        meta[c] = meta[c].str.strip('"')

    # یک هفتم ضایعه‌ها برای اعتبارسنجی، با حفظ نسبت کلاس‌ها، بدون اشتراک ضایعه
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=0)
    tr_idx, va_idx = next(sgkf.split(meta, meta.dx, groups=meta.lesion_id))
    tr, va = meta.iloc[tr_idx], meta.iloc[va_idx]
    assert not set(tr.lesion_id) & set(va.lesion_id)
    if QUICK:
        tr, va = tr.sample(320, random_state=0), va.sample(160, random_state=0)
    print(f"train: {len(tr)} images / {tr.lesion_id.nunique()} lesions   "
          f"valid: {len(va)} images / {va.lesion_id.nunique()} lesions   (no lesion in both)")

    kw = dict(batch_size=BATCH, num_workers=4, persistent_workers=True)
    loader_tr = DataLoader(HamDataset(tr, train_transform), shuffle=True, **kw)
    loader_va = DataLoader(HamDataset(va, eval_transform), shuffle=False, **kw)

    counts = np.bincount([CLASSES.index(d) for d in tr.dx], minlength=7).astype(np.float32)
    w = counts.sum() / (7 * counts)
    print("class weights: " + "  ".join(f"{c}:{x:.2f}" for c, x in zip(CLASSES, w)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 7)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    best, t0 = float("inf"), time.time()
    n_batches = len(loader_tr)
    print(f"\n─── fine-tune #1 on HAM10000: {EPOCHS} epochs × {n_batches} batches ───", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run, n = 0.0, 0
        for b, (x, y) in enumerate(loader_tr, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * len(x); n += len(x)
            if b % 50 == 0 or b == n_batches:
                bar = "█" * int(20 * b / n_batches) + "·" * (20 - int(20 * b / n_batches))
                print(f"  epoch {epoch}/{EPOCHS} [{bar}] batch {b:3d}/{n_batches}  "
                      f"loss {run / n:.3f}  ({time.time() - t0:4.0f}s)", flush=True)

        model.eval()
        preds, trues, vloss, vn = [], [], 0.0, 0
        with torch.no_grad():
            for x, y in loader_va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                vloss += loss_fn(out, y).item() * len(x); vn += len(x)
                preds.append(out.argmax(1).cpu()); trues.append(y.cpu())
        vloss /= vn
        bacc = balanced_accuracy_score(torch.cat(trues), torch.cat(preds))
        mark = ""
        if vloss < best:
            best = vloss
            backbone = {k: v for k, v in model.state_dict().items() if not k.startswith("fc.")}
            torch.save(backbone, "checkpoints/step3b_ham_backbone.pt")
            mark = "  ← best, backbone saved"
        print(f"epoch {epoch}/{EPOCHS} done | train loss {run / n:.3f} | valid loss {vloss:.3f} "
              f"| valid balanced acc (7 classes) {bacc:.3f}{mark}\n", flush=True)

    print(f"finished in {(time.time() - t0) / 60:.1f} min → checkpoints/step3b_ham_backbone.pt")


if __name__ == "__main__":
    main()
