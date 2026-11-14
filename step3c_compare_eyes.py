"""
قدم ۳-ج — تنظیم دقیق ۲ و مقایسه‌ی منصفانه‌ی دو «چشم»:

    چشم قدیم : بدنه‌ی ImageNet            → تنظیم دقیق روی ۴۱۳ خال Derm7pt
    چشم جدید : بدنه‌ی ImageNet + HAM10000 → تنظیم دقیق روی ۴۱۳ خال Derm7pt

هر کدام ۳ بار با شروع تصادفی متفاوت. همه‌چیز دیگر یکسان است.
تصمیم «کدام بهتر است» فقط با بخش valid گرفته می‌شود. عدد test هم چاپ می‌شود
ولی هیچ انتخابی بر اساس آن انجام نمی‌شود.

برای هر اجرا دو چیز سنجیده می‌شود:
  • نمره‌ی چشم : میانگین AUROC هفت نشانه
  • نمره‌ی زنجیره: AUROC ملانوما وقتی نشانه‌های حدس‌زده به «مغز» داده شوند
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

from derm7pt_data import (CRIT, LEVELS, N_LEVELS, load_table,
                          Derm7ptDataset, train_transform, eval_transform)

QUICK = os.environ.get("QUICK") == "1"
EPOCHS = 2 if QUICK else 30
SEEDS = [0] if QUICK else [0, 1, 2]
ARMS = ["imagenet", "ham"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HAM_BACKBONE = "checkpoints/step3b_ham_backbone.pt"
COLS = [(c, l) for c in CRIT for l in LEVELS[c]]


def onehot(table):
    return np.stack([(table[c + "_id"].values == LEVELS[c].index(l)).astype(np.float32)
                     for c, l in COLS], axis=1)


def train_brain(df):
    """«مغز» = همان طبقه‌بند خطی گام اول، روی برچسب پزشک. یک بار، برای همه‌ی اجراها."""
    torch.manual_seed(0)
    X = {s: torch.tensor(onehot(df[df.split == s])) for s in ["train", "valid"]}
    y = {s: torch.tensor(df[df.split == s].y.values, dtype=torch.float32) for s in ["train", "valid"]}
    lf = nn.BCEWithLogitsLoss(pos_weight=(y["train"] == 0).sum() / (y["train"] == 1).sum())
    clf = nn.Linear(len(COLS), 1)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    best = (float("inf"), None)
    for _ in range(301):
        opt.zero_grad(); lf(clf(X["train"]).squeeze(1), y["train"]).backward(); opt.step()
        with torch.no_grad():
            v = lf(clf(X["valid"]).squeeze(1), y["valid"]).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in clf.state_dict().items()})
    clf.load_state_dict(best[1])
    return lambda P: clf(torch.tensor(P, dtype=torch.float32)).squeeze(1).detach().numpy()


class ConceptModel(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        if arm == "ham":
            missing, unexpected = self.backbone.load_state_dict(
                torch.load(HAM_BACKBONE, map_location="cpu"), strict=False)
            assert not unexpected and all(k.startswith("fc.") for k in missing)
        self.heads = nn.ModuleList([nn.Linear(512, k) for k in N_LEVELS])

    def forward(self, x):
        f = self.backbone(x)
        return [h(f) for h in self.heads]


def run_one(arm, seed, df, loader, loss_fns, brain):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ConceptModel(arm).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    total = lambda out, c: sum(fn(o, c[:, j]) for j, (fn, o) in enumerate(zip(loss_fns, out)))

    @torch.no_grad()
    def predict(split):
        model.eval()
        probs, tl, n = [[] for _ in CRIT], 0.0, 0
        for x, c, _ in loader[split]:
            x, c = x.to(DEVICE), c.to(DEVICE)
            out = model(x)
            tl += total(out, c).item() * len(x); n += len(x)
            for j, o in enumerate(out):
                probs[j].append(torch.softmax(o, 1).cpu())
        return [torch.cat(p).numpy() for p in probs], tl / n

    best, best_ep, best_state, t0 = float("inf"), 0, None, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, c, _ in loader["train"]:
            x, c = x.to(DEVICE), c.to(DEVICE)
            opt.zero_grad(); total(model(x), c).backward(); opt.step()
        _, vl = predict("valid")
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {ep:2d}/{EPOCHS}  valid loss {vl:.3f}{'  ← best' if best_ep == ep else ''}", flush=True)
    model.load_state_dict(best_state)

    result = {"arm": arm, "seed": seed, "best_epoch": best_ep, "minutes": (time.time() - t0) / 60}
    for split in ["valid", "test"]:
        probs, _ = predict(split)
        tab = df[df.split == split]
        aucs = []
        for j, c in enumerate(CRIT):
            true = tab[c + "_id"].values
            a = (roc_auc_score(true, probs[j][:, 1]) if N_LEVELS[j] == 2 else
                 roc_auc_score(true, probs[j], multi_class="ovr", average="macro"))
            result[f"{split}_{c}"] = a
            aucs.append(a)
        result[f"{split}_eye"] = float(np.mean(aucs))
        P = np.concatenate(probs, axis=1)          # همان ترتیب COLS
        result[f"{split}_chain"] = roc_auc_score(tab.y.values, brain(P))
    return result


def main():
    if "ham" in ARMS and not os.path.exists(HAM_BACKBONE):
        raise SystemExit(f"missing {HAM_BACKBONE} — run step3b_ham_pretrain.py first")
    df = load_table()
    ds = {"train": Derm7ptDataset(df, "train", train_transform),
          "valid": Derm7ptDataset(df, "valid", eval_transform),
          "test":  Derm7ptDataset(df, "test",  eval_transform)}
    loader = {s: DataLoader(d, batch_size=16, shuffle=(s == "train"),
                            num_workers=4, persistent_workers=True) for s, d in ds.items()}
    tr = df[df.split == "train"]
    loss_fns = []
    for c, k in zip(CRIT, N_LEVELS):
        cnt = np.bincount(tr[c + "_id"], minlength=k).astype(np.float32)
        loss_fns.append(nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (k * cnt), device=DEVICE)))
    brain = train_brain(df)

    results = []
    n_runs = len(ARMS) * len(SEEDS)
    for i, (seed, arm) in enumerate([(s, a) for s in SEEDS for a in ARMS], 1):
        label = "OLD eye (ImageNet only)" if arm == "imagenet" else "NEW eye (ImageNet + HAM10000)"
        print(f"\n━━━ run {i}/{n_runs}: {label}, seed {seed} ━━━", flush=True)
        r = run_one(arm, seed, df, loader, loss_fns, brain)
        results.append(r)
        print(f"  → best epoch {r['best_epoch']} | VALID eye {r['valid_eye']:.3f}  chain {r['valid_chain']:.3f} "
              f"| {r['minutes']:.1f} min", flush=True)
        pd.DataFrame(results).to_csv("notes/step3c_results.csv", index=False)

    R = pd.DataFrame(results)
    print("\n\n════════════ SUMMARY (mean ± std over seeds) ════════════")
    print(f"{'':32}{'OLD eye':>16}{'NEW eye':>16}")
    for key, name in [("valid_eye", "VALID  eye (7 signs AUROC)"), ("valid_chain", "VALID  chain (melanoma AUROC)"),
                      ("test_eye", "TEST   eye (7 signs AUROC)"), ("test_chain", "TEST   chain (melanoma AUROC)")]:
        cells = [f"{R[R.arm == a][key].mean():.3f} ± {R[R.arm == a][key].std(ddof=0):.3f}" for a in ARMS]
        print(f"{name:<32}{cells[0]:>16}{cells[1]:>16}")
    print("\nper sign, VALID AUROC (mean over seeds):")
    for c in CRIT:
        o, n = R[R.arm == "imagenet"][f"valid_{c}"].mean(), R[R.arm == "ham"][f"valid_{c}"].mean()
        print(f"  {c:<24}{o:>8.3f}{n:>8.3f}   {'+' if n > o else ''}{n - o:.3f}")
    print("\nsaved notes/step3c_results.csv")


if __name__ == "__main__":
    main()
