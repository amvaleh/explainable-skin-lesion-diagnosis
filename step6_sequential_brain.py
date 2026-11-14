"""
قدم ۶ — آموزش «مغز» روی خروجی واقعی «چشم» (sequential CBM).

    مغز قدیم: طبقه‌بند خطی روی برچسب پزشک        (independent CBM — همان قدم ۱)
    مغز جدید: طبقه‌بند خطی روی نظر چشم            (sequential CBM)

نظر چشم روی عکس‌های آموزشی باید «بی‌تقلب» باشد، پس پنج‌قسمتی (cross-fitting):
برای هر قسمت، چشم روی چهار قسمت دیگر آموزش می‌بیند و فقط روی این قسمت نظر می‌دهد.

چشم: بهترین نسخه‌ی فعلی — بدنه‌ی HAM10000 + عکس بریده‌ی دور خال.
هر دو مغز روی همان پیش‌بینی‌های چشم برای valid/test سنجیده می‌شوند، پس مقایسه جفت‌جفت است.
تصمیم فقط با valid؛ test فقط گزارش می‌شود.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from derm7pt_data import CRIT, LEVELS, N_LEVELS, load_table, train_transform, eval_transform
from step3c_compare_eyes import ConceptModel, onehot, COLS, DEVICE
from step5c_compare_crop import CropDataset

QUICK = os.environ.get("QUICK") == "1"
SEEDS = [0] if QUICK else [0, 1, 2]
FOLDS = 2 if QUICK else 5
MAX_EPOCHS, PATIENCE = (2, 1) if QUICK else (30, 8)


def loader(df, split, train):
    ds = CropDataset(df, split, train_transform if train else eval_transform)
    return DataLoader(ds, batch_size=16, shuffle=train, num_workers=0)   # بدون کارگر کمکی: در مک راه‌اندازی دوباره‌ی کارگرها کندتر از خواندن برش‌های کوچک است


def train_eye(df, train_split, seed, tag):
    """چشم را روی ردیف‌های train_split آموزش بده؛ توقف زودهنگام با خطای valid رسمی."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr = df[df.split == train_split]
    loss_fns = []
    for c, k in zip(CRIT, N_LEVELS):
        cnt = np.bincount(tr[c + "_id"], minlength=k).astype(np.float32) + 1.0   # +۱ تا قسمت بی‌نمونه صفر نشود
        loss_fns.append(nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (k * cnt), device=DEVICE)))
    total = lambda out, c: sum(fn(o, c[:, j]) for j, (fn, o) in enumerate(zip(loss_fns, out)))

    model = ConceptModel("ham").to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lt, lv = loader(df, train_split, True), loader(df, "valid", False)
    best, best_ep, best_state, t0 = float("inf"), 0, None, time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, c, _ in lt:
            x, c = x.to(DEVICE), c.to(DEVICE)
            opt.zero_grad(); total(model(x), c).backward(); opt.step()
        model.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for x, c, _ in lv:
                x, c = x.to(DEVICE), c.to(DEVICE)
                vl += total(model(x), c).item() * len(x); n += len(x)
        vl /= n
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - best_ep >= PATIENCE:
            break
    model.load_state_dict(best_state)
    print(f"    {tag:<34} {len(tr):3d} images → best epoch {best_ep:2d} (stopped at {ep}) "
          f"| {time.time() - t0:4.0f}s", flush=True)
    return model


@torch.no_grad()
def eye_opinion(model, df, split):
    """نظر چشم: ماتریس (بیماران × ۱۹) از احتمال‌ها، به ترتیب COLS."""
    model.eval()
    out = []
    for x, _, _ in loader(df, split, False):
        probs = [torch.softmax(o, 1).cpu() for o in model(x.to(DEVICE))]
        out.append(torch.cat(probs, 1))
    return torch.cat(out).numpy()


def train_brain(Xtr, ytr, Xva, yva):
    """طبقه‌بند خطی تک‌خروجی، مثل قدم ۱. انتخاب بهترین دوره با خطای valid."""
    torch.manual_seed(0)
    Xtr, Xva = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    ytr, yva = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    lf = nn.BCEWithLogitsLoss(pos_weight=(ytr == 0).sum() / (ytr == 1).sum())
    clf = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    best = (float("inf"), None)
    for _ in range(301):
        opt.zero_grad(); lf(clf(Xtr).squeeze(1), ytr).backward(); opt.step()
        with torch.no_grad():
            v = lf(clf(Xva).squeeze(1), yva).item()
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in clf.state_dict().items()})
    clf.load_state_dict(best[1])
    return lambda X: clf(torch.tensor(X, dtype=torch.float32)).squeeze(1).detach().numpy()


def scores(z, y):
    p = (z > 0).astype(int)
    TP = int(((p == 1) & (y == 1)).sum()); FN = int(((p == 0) & (y == 1)).sum())
    TN = int(((p == 0) & (y == 0)).sum()); FP = int(((p == 1) & (y == 0)).sum())
    return {"auroc": roc_auc_score(y, z), "recall": TP / (TP + FN), "spec": TN / (TN + FP), "FN": FN, "FP": FP}


def main():
    df = load_table()
    y = {s: df[df.split == s].y.values for s in ["train", "valid", "test"]}
    doctor = {s: onehot(df[df.split == s]) for s in ["train", "valid", "test"]}
    brain_doctor = train_brain(doctor["train"], y["train"], doctor["valid"], y["valid"])

    train_idx = df.index[df.split == "train"]
    results = []
    for seed in SEEDS:
        print(f"\n━━━ seed {seed}: building out-of-fold eye opinions on the {len(train_idx)} training images ━━━", flush=True)
        oof = np.zeros((len(train_idx), len(COLS)), dtype=np.float32)
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for f, (fit, held) in enumerate(skf.split(train_idx, y["train"]), 1):
            d = df.copy()
            d.loc[train_idx[fit], "split"] = "fold_fit"
            d.loc[train_idx[held], "split"] = "fold_held"
            m = train_eye(d, "fold_fit", seed, f"fold {f}/{FOLDS}: eye trained without part {f}")
            oof[held] = eye_opinion(m, d, "fold_held")

        print(f"  final eye on all training images (gives valid/test opinions):", flush=True)
        m = train_eye(df, "train", seed, "final eye")
        eye = {s: eye_opinion(m, df, s) for s in ["valid", "test"]}

        brain_seq = train_brain(oof, y["train"], eye["valid"], y["valid"])
        for name, brain in [("doctor-trained brain", brain_doctor), ("eye-trained brain", brain_seq)]:
            r = {"seed": seed, "brain": name}
            for s in ["valid", "test"]:
                for k, v in scores(brain(eye[s]), y[s]).items():
                    r[f"{s}_{k}"] = v
            results.append(r)
            print(f"  → {name:<22} VALID AUROC {r['valid_auroc']:.3f} | recall {r['valid_recall']:.3f} "
                  f"spec {r['valid_spec']:.3f} | missed {r['valid_FN']:2d}  false alarms {r['valid_FP']:2d}", flush=True)
        pd.DataFrame(results).to_csv("notes/step6_results.csv", index=False)

    R = pd.DataFrame(results)
    names = ["doctor-trained brain", "eye-trained brain"]
    print("\n\n════════════ SUMMARY (mean ± std over seeds) ════════════")
    print(f"{'':26}{'doctor-trained':>18}{'eye-trained':>18}")
    for s in ["valid", "test"]:
        for k, fmt in [("auroc", ".3f"), ("recall", ".3f"), ("spec", ".3f"), ("FN", ".1f"), ("FP", ".1f")]:
            cells = [f"{R[R.brain == b][f'{s}_{k}'].mean():{fmt}} ± {R[R.brain == b][f'{s}_{k}'].std(ddof=0):{fmt}}"
                     for b in names]
            print(f"{s.upper() + '  ' + k:<26}{cells[0]:>18}{cells[1]:>18}")
        print()
    d = (R[R.brain == names[1]].sort_values("seed").valid_auroc.values
         - R[R.brain == names[0]].sort_values("seed").valid_auroc.values)
    print(f"paired VALID AUROC: eye-trained − doctor-trained = {np.round(d, 3)}  (eye-trained wins {int((d > 0).sum())}/{len(d)})")
    print("reference: black box with HAM pretraining, VALID AUROC 0.832 ± 0.007")
    print("\nsaved notes/step6_results.csv")


if __name__ == "__main__":
    main()
