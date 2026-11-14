"""
قدم ۴-الف — مقایسه‌ی منصفانه‌ی جعبه‌سیاه: آیا تمرین روی ده هزار خال به جعبه‌سیاه هم کمک می‌کند؟

    جعبه‌سیاه قدیم : بدنه‌ی ImageNet            → عکس ← ملانوما
    جعبه‌سیاه جدید : بدنه‌ی ImageNet + HAM10000 → عکس ← ملانوما

هر کدام ۳ بار. همان تنظیمات قدم ۲-د. تصمیم فقط با valid؛ test فقط گزارش می‌شود.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score

from derm7pt_data import load_table, Derm7ptDataset, train_transform, eval_transform

QUICK = os.environ.get("QUICK") == "1"
EPOCHS = 2 if QUICK else 30
SEEDS = [0] if QUICK else [0, 1, 2]
ARMS = ["imagenet", "ham"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HAM_BACKBONE = "checkpoints/step3b_ham_backbone.pt"


def make_model(arm):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if arm == "ham":
        missing, unexpected = m.load_state_dict(torch.load(HAM_BACKBONE, map_location="cpu"), strict=False)
        assert not unexpected and all(k.startswith("fc.") for k in missing)
    m.fc = nn.Linear(512, 1)
    return m.to(DEVICE)


def run_one(arm, seed, loader, pos_weight):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_model(arm)
    lf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    @torch.no_grad()
    def scores(split):
        model.eval()
        zs, ys, tl, n = [], [], 0.0, 0
        for x, _, y in loader[split]:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            z = model(x).squeeze(1)
            tl += lf(z, y).item() * len(x); n += len(x)
            zs.append(z.cpu()); ys.append(y.cpu())
        return torch.cat(zs).numpy(), torch.cat(ys).numpy(), tl / n

    best, best_ep, best_state, t0 = float("inf"), 0, None, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, _, y in loader["train"]:
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            opt.zero_grad(); lf(model(x).squeeze(1), y).backward(); opt.step()
        _, _, vl = scores("valid")
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep:2d}/{EPOCHS}  valid loss {vl:.3f}{'  ← best' if best_ep == ep else ''}", flush=True)
    model.load_state_dict(best_state)
    r = {"arm": arm, "seed": seed, "best_epoch": best_ep, "minutes": (time.time() - t0) / 60}
    for split in ["valid", "test"]:
        z, y, _ = scores(split)
        r[f"{split}_auroc"] = roc_auc_score(y, z)
    return r


def main():
    df = load_table()
    ds = {"train": Derm7ptDataset(df, "train", train_transform),
          "valid": Derm7ptDataset(df, "valid", eval_transform),
          "test":  Derm7ptDataset(df, "test",  eval_transform)}
    loader = {s: DataLoader(d, batch_size=16, shuffle=(s == "train"),
                            num_workers=4, persistent_workers=True) for s, d in ds.items()}
    ytr = df[df.split == "train"].y
    pos_weight = torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()], dtype=torch.float32, device=DEVICE)

    results = []
    runs = [(s, a) for s in SEEDS for a in ARMS]
    for i, (seed, arm) in enumerate(runs, 1):
        label = "OLD black box (ImageNet only)" if arm == "imagenet" else "NEW black box (ImageNet + HAM10000)"
        print(f"\n━━━ run {i}/{len(runs)}: {label}, seed {seed} ━━━", flush=True)
        r = run_one(arm, seed, loader, pos_weight)
        results.append(r)
        print(f"  → best epoch {r['best_epoch']} | VALID melanoma AUROC {r['valid_auroc']:.3f} | {r['minutes']:.1f} min", flush=True)
        pd.DataFrame(results).to_csv("notes/step4a_results.csv", index=False)

    R = pd.DataFrame(results)
    print("\n\n════════════ SUMMARY: black box, melanoma AUROC (mean ± std over seeds) ════════════")
    print(f"{'':12}{'OLD':>16}{'NEW':>16}")
    for split in ["valid", "test"]:
        cells = [f"{R[R.arm == a][f'{split}_auroc'].mean():.3f} ± {R[R.arm == a][f'{split}_auroc'].std(ddof=0):.3f}" for a in ARMS]
        print(f"{split.upper():<12}{cells[0]:>16}{cells[1]:>16}")
    print("\nsaved notes/step4a_results.csv")


if __name__ == "__main__":
    main()
