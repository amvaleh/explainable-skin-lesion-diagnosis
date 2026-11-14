"""
قدم ۲-ج — آموزش مدلی که از روی عکس درموسکوپی، هفت مفهوم بالینی را حدس بزند.

    عکس → شبکه‌ی آماده (ResNet18) → بردار ۵۱۲تایی → هفت سر خطی → هفت پیش‌بینی

هنوز تشخیص ملانوما نمی‌دهیم؛ فقط مفاهیم. وصل کردن به تشخیص در قدم ۲-د است.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from derm7pt_data import (CRIT, LEVELS, N_LEVELS, load_table,
                          Derm7ptDataset, train_transform, eval_transform)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH, EPOCHS, LR = 16, 30, 1e-4
os.makedirs("checkpoints", exist_ok=True)

# ───────────── ۱. داده ─────────────
df = load_table()
ds = {"train": Derm7ptDataset(df, "train", train_transform),
      "valid": Derm7ptDataset(df, "valid", eval_transform),
      "test":  Derm7ptDataset(df, "test",  eval_transform)}
loader = {s: DataLoader(d, batch_size=BATCH, shuffle=(s == "train")) for s, d in ds.items()}


# ───────────── ۲. مدل ─────────────
class ConceptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()            # لایه‌ی آخرِ مخصوص ImageNet را برمی‌داریم
        self.heads = nn.ModuleList([nn.Linear(512, k) for k in N_LEVELS])   # هفت سر

    def forward(self, x):
        f = self.backbone(x)                         # (batch, 512)
        return [head(f) for head in self.heads]      # فهرست ۷تایی از (batch, تعداد حالت)


model = ConceptModel().to(DEVICE)
n_backbone = sum(p.numel() for p in model.backbone.parameters())
n_heads = sum(p.numel() for p in model.heads.parameters())
print(f"device: {DEVICE}")
print(f"parameters — backbone: {n_backbone:,}   heads: {n_heads:,}")

# ───────────── ۳. تابع خطا: یکی برای هر مفهوم، با وزن کلاس جداگانه ─────────────
train_tab = df[df.split == "train"]
loss_fns = []
print("\nclass weights per concept (rarer level → larger weight):")
for c, k in zip(CRIT, N_LEVELS):
    counts = np.bincount(train_tab[c + "_id"], minlength=k).astype(np.float32)
    w = counts.sum() / (k * counts)
    loss_fns.append(nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE)))
    print(f"  {c:<22} " + "  ".join(f"{LEVELS[c][i]}:{w[i]:.2f}" for i in range(k)))


def total_loss(outputs, concepts):
    """جمع خطای هفت مفهوم."""
    return sum(fn(out, concepts[:, j]) for j, (fn, out) in enumerate(zip(loss_fns, outputs)))


optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)


@torch.no_grad()
def predict(split):
    """احتمال هر حالتِ هر مفهوم برای همه‌ی بیماران یک بخش، به‌علاوه‌ی میانگین خطا."""
    model.eval()
    probs = [[] for _ in CRIT]
    total, n = 0.0, 0
    for x, concepts, _ in loader[split]:
        x, concepts = x.to(DEVICE), concepts.to(DEVICE)
        out = model(x)
        total += total_loss(out, concepts).item() * len(x)
        n += len(x)
        for j, o in enumerate(out):
            probs[j].append(torch.softmax(o, 1).cpu())
    return [torch.cat(p).numpy() for p in probs], total / n


# ───────────── ۴. حلقه‌ی آموزش ─────────────
print(f"\n─── training: {EPOCHS} epochs × {len(loader['train'])} batches ───")
best_valid, best_epoch = float("inf"), -1
t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    running, n = 0.0, 0
    for x, concepts, _ in loader["train"]:
        x, concepts = x.to(DEVICE), concepts.to(DEVICE)
        optimizer.zero_grad()
        loss = total_loss(model(x), concepts)
        loss.backward()
        optimizer.step()
        running += loss.item() * len(x)
        n += len(x)

    _, valid_loss = predict("valid")
    mark = ""
    if valid_loss < best_valid:
        best_valid, best_epoch = valid_loss, epoch
        torch.save(model.state_dict(), "checkpoints/step2c_concepts.pt")
        mark = "  ← best"
    print(f"epoch {epoch:2d} | train loss {running / n:.3f} | valid loss {valid_loss:.3f} "
          f"| {time.time() - t_start:5.0f}s{mark}", flush=True)

print(f"\nbest epoch: {best_epoch}  (valid loss {best_valid:.3f})")
model.load_state_dict(torch.load("checkpoints/step2c_concepts.pt", map_location=DEVICE))

# ───────────── ۵. سنجش روی test ─────────────
probs, test_loss = predict("test")
test = df[df.split == "test"].reset_index(drop=True)

print(f"\n─── concept prediction on {len(test)} test images ───")
print(f"{'concept':<22}{'majority':>10}{'accuracy':>10}{'bal.acc':>9}{'AUROC':>8}")
print("-" * 59)
rows = []
for j, c in enumerate(CRIT):
    true = test[c + "_id"].values
    pred = probs[j].argmax(1)
    majority = (true == np.bincount(train_tab[c + "_id"]).argmax()).mean()
    acc = (pred == true).mean()
    bacc = balanced_accuracy_score(true, pred)
    if N_LEVELS[j] == 2:
        auc = roc_auc_score(true, probs[j][:, 1])
    else:
        auc = roc_auc_score(true, probs[j], multi_class="ovr", average="macro")
    rows.append((c, majority, acc, bacc, auc))
    print(f"{c:<22}{majority:>10.3f}{acc:>10.3f}{bacc:>9.3f}{auc:>8.3f}")
m = np.array([r[1:] for r in rows]).mean(0)
print("-" * 59)
print(f"{'mean':<22}{m[0]:>10.3f}{m[1]:>10.3f}{m[2]:>9.3f}{m[3]:>8.3f}")

# ───────────── ۶. ذخیره‌ی پیش‌بینی‌ها برای قدم ۲-د ─────────────
out = test[["case_num", "y"] + [c + "_id" for c in CRIT]].copy()
for j, c in enumerate(CRIT):
    for i, lvl in enumerate(LEVELS[c]):
        out[f"p_{c}_{lvl}"] = probs[j][:, i]
out.to_csv("step2c_test_predictions.csv", index=False)
print("\nsaved step2c_test_predictions.csv  and  checkpoints/step2c_concepts.pt")
