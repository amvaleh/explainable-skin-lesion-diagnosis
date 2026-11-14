"""
قدم ۷ — تنظیم مرز تصمیم (threshold) برای رسیدن به حساسیت هدف.

مشکل گام ۶: با مرز صفر، مدل حدود ۳۱ ملانوما از ۱۰۱ تا را از دست می‌دهد.
روش: مرز را فقط روی valid انتخاب می‌کنیم — بزرگ‌ترین مرزی که دست‌کم «هدف» درصد ملانوماهای valid
را بگیرد — و همان مرز را بدون تغییر روی test به کار می‌بریم. test برای انتخاب مرز استفاده نمی‌شود.

سه مدل، سه تکرار، هر سه با بدنه‌ی HAM10000:
  seq    : چشم (عکس بریده) + مغز آموزش‌دیده با نظر واقعی چشم     ← مدل توضیح‌دهنده‌ی ما
  doctor : همان چشم + مغز آموزش‌دیده با یادداشت پزشک
  black  : عکس کامل → ملانوما، بدون نشانه                          ← رقیب بدون توضیح

نظر چشم و نمره‌ها ذخیره می‌شوند: notes/step7_scores.npz
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import step6_sequential_brain as s6
from derm7pt_data import load_table, Derm7ptDataset, train_transform, eval_transform
from step3c_compare_eyes import onehot, DEVICE

TARGETS = [0.85, 0.90, 0.95]


def pick_threshold(z_valid, y_valid, target):
    """بزرگ‌ترین مرزی که دست‌کم target از ملانوماهای valid را بگیرد (z >= t یعنی ملانوما)."""
    pos = np.sort(z_valid[y_valid == 1])
    k = int(np.floor((1 - target) * len(pos)))
    return pos[k]


def at_threshold(z, y, t):
    p = z >= t
    TP = int((p & (y == 1)).sum()); FN = int((~p & (y == 1)).sum())
    TN = int((~p & (y == 0)).sum()); FP = int((p & (y == 0)).sum())
    return {"recall": TP / (TP + FN), "spec": TN / (TN + FP), "FN": FN, "FP": FP}


def black_box_scores(df, seed):
    """جعبه‌ی سیاه با بدنه‌ی HAM: عکس کامل → یک امتیاز ملانوما. توقف زودهنگام با خطای valid."""
    torch.manual_seed(seed); np.random.seed(seed)
    ds = {s: Derm7ptDataset(df, s, train_transform if s == "train" else eval_transform)
          for s in ["train", "valid", "test"]}
    ld = {s: DataLoader(d, batch_size=16, shuffle=(s == "train"), num_workers=0) for s, d in ds.items()}
    ytr = df[df.split == "train"].y
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(ytr == 0).sum() / (ytr == 1).sum()],
                                                      dtype=torch.float32, device=DEVICE))
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    missing, unexpected = m.load_state_dict(torch.load("checkpoints/step3b_ham_backbone.pt", map_location="cpu"), strict=False)
    assert not unexpected and all(k.startswith("fc.") for k in missing)
    m.fc = nn.Linear(512, 1)
    m = m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-5)

    @torch.no_grad()
    def run(split):
        m.eval(); zs, tl, n = [], 0.0, 0
        for x, _, y in ld[split]:
            z = m(x.to(DEVICE)).squeeze(1)
            tl += lf(z, y.float().to(DEVICE)).item() * len(x); n += len(x)
            zs.append(z.cpu())
        return torch.cat(zs).numpy(), tl / n

    best, best_ep, state = float("inf"), 0, None
    for ep in range(1, s6.MAX_EPOCHS + 1):
        m.train()
        for x, _, y in ld["train"]:
            opt.zero_grad(); lf(m(x.to(DEVICE)).squeeze(1), y.float().to(DEVICE)).backward(); opt.step()
        _, vl = run("valid")
        if vl < best:
            best, best_ep, state = vl, ep, {k: v.detach().clone() for k, v in m.state_dict().items()}
        elif ep - best_ep >= s6.PATIENCE:
            break
    m.load_state_dict(state)
    print(f"    black box: best epoch {best_ep} (stopped at {ep})", flush=True)
    return {"valid": run("valid")[0], "test": run("test")[0]}


def main():
    df = load_table()
    y = {s: df[df.split == s].y.values for s in ["train", "valid", "test"]}
    doctor = {s: onehot(df[df.split == s]) for s in ["train", "valid", "test"]}
    brain_doctor = s6.train_brain(doctor["train"], y["train"], doctor["valid"], y["valid"])
    train_idx = df.index[df.split == "train"]

    rows, saved = [], {}
    for seed in s6.SEEDS:
        t0 = time.time()
        print(f"\n━━━ seed {seed}: eye opinions (5-fold out-of-fold on training images) ━━━", flush=True)
        oof = np.zeros((len(train_idx), doctor["train"].shape[1]), dtype=np.float32)
        skf = StratifiedKFold(n_splits=s6.FOLDS, shuffle=True, random_state=seed)
        for f, (fit, held) in enumerate(skf.split(train_idx, y["train"]), 1):
            d = df.copy()
            d.loc[train_idx[fit], "split"] = "fold_fit"
            d.loc[train_idx[held], "split"] = "fold_held"
            oof[held] = s6.eye_opinion(s6.train_eye(d, "fold_fit", seed, f"fold {f}/{s6.FOLDS}"), d, "fold_held")
        eye_model = s6.train_eye(df, "train", seed, "final eye")
        eye = {s: s6.eye_opinion(eye_model, df, s) for s in ["valid", "test"]}
        brain_seq = s6.train_brain(oof, y["train"], eye["valid"], y["valid"])

        print("  black box ...", flush=True)
        bb = black_box_scores(df, seed)

        scores = {
            "seq":    {s: brain_seq(eye[s]) for s in ["valid", "test"]},
            "doctor": {s: brain_doctor(eye[s]) for s in ["valid", "test"]},
            "black":  bb,
        }
        for name, sc in scores.items():
            saved[f"{name}_seed{seed}_valid"], saved[f"{name}_seed{seed}_test"] = sc["valid"], sc["test"]
            auc_v, auc_t = roc_auc_score(y["valid"], sc["valid"]), roc_auc_score(y["test"], sc["test"])
            for target in TARGETS:
                t = pick_threshold(sc["valid"], y["valid"], target)
                v, te = at_threshold(sc["valid"], y["valid"], t), at_threshold(sc["test"], y["test"], t)
                rows.append({"seed": seed, "model": name, "target": target, "threshold": t,
                             "valid_auroc": auc_v, "test_auroc": auc_t,
                             **{f"valid_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in te.items()}})
            r = [x for x in rows if x["seed"] == seed and x["model"] == name and x["target"] == 0.90][0]
            print(f"  → {name:<7} target 90%: valid recall {r['valid_recall']:.2f} spec {r['valid_spec']:.2f} | "
                  f"TEST recall {r['test_recall']:.2f} spec {r['test_spec']:.2f}  missed {r['test_FN']:2d}  "
                  f"false alarms {r['test_FP']:3d}", flush=True)
        pd.DataFrame(rows).to_csv("notes/step7_results.csv", index=False)
        np.savez("notes/step7_scores.npz", **saved)
        print(f"  seed {seed} took {(time.time() - t0) / 60:.1f} min", flush=True)

    R = pd.DataFrame(rows)
    names = {"seq": "OUR chain (explains)", "doctor": "chain, doctor-trained brain", "black": "black box (no explanation)"}
    print("\n\n════════════ SUMMARY: threshold chosen on VALID, applied to TEST (mean over seeds) ════════════")
    for target in TARGETS:
        print(f"\n— target recall on valid = {target:.0%} —")
        print(f"{'':30}{'valid recall':>13}{'TEST recall':>13}{'TEST spec':>11}{'missed/101':>12}{'false alarms/294':>18}")
        for k, nm in names.items():
            q = R[(R.model == k) & (R.target == target)]
            print(f"{nm:<30}{q.valid_recall.mean():>13.2f}{q.test_recall.mean():>13.2f}{q.test_spec.mean():>11.2f}"
                  f"{q.test_FN.mean():>12.1f}{q.test_FP.mean():>18.1f}")
    print("\nTEST AUROC (threshold-free):  " + "   ".join(
        f"{names[k].split(' (')[0]}: {R[(R.model == k) & (R.target == 0.9)].test_auroc.mean():.3f}" for k in names))
    print("\nreference (step 2-D, doctor's 7-point rule with the DOCTOR's own concepts, TEST): "
          "recall 0.95, spec 0.80, missed 5, false alarms 58")
    print("saved notes/step7_results.csv and notes/step7_scores.npz")


if __name__ == "__main__":
    main()
