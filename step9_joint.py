"""
قدم ۹ — آموزش هم‌زمان (joint CBM، معادله‌ی ۴ مقاله): چشم و مغز با هم، مشتق خطای تشخیص تا چشم می‌رسد.

    L = Σ_j w_j · CE(نشانه‌ی j)  +  λ · BCE(مغز(۱۹ احتمال), ملانوما)

مقایسه (همان داده، همان بدنه‌ی HAM، عکس بریده، ۳ تکرار، تصمیم فقط با valid):
  seq     : ترتیبی، همان قدم ۷ (نمره‌های ذخیره‌شده‌ی notes/step7_scores.npz)
  joint λ : هم‌زمان با λ = 1 و ۵
  black   : جعبه‌سیاه، همان قدم ۷

سنجه‌ها: AUROC ملانوما، AUROC هفت نشانه (هزینه‌ی تفسیرپذیری)، و حساسیت/هشدار اشتباه روی test
با مرزی که فقط روی valid برای حساسیت ۹۰٪ انتخاب شده.
برای «AUROC نشانه‌ها»ی روش ترتیبی، چشم نهایی هر تکرار دوباره آموزش داده می‌شود (همان کد گام ۶).
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import step6_sequential_brain as s6
from derm7pt_data import CRIT, LEVELS, N_LEVELS, load_table
from step3c_compare_eyes import ConceptModel, DEVICE
from step7_threshold import pick_threshold, at_threshold

LAMBDAS = [1.0, 5.0]
SEEDS = s6.SEEDS
BRAIN_LR = 5e-2      # همان نرخ یادگیری مغز در قدم ۶، تا مقایسه منصفانه باشد
OFF = np.concatenate([[0], np.cumsum(N_LEVELS)])


def concept_auroc(P, tab):
    """میانگین AUROC هفت نشانه از ماتریس ۱۹ احتمال."""
    out = []
    for j, c in enumerate(CRIT):
        pj, true = P[:, OFF[j]:OFF[j + 1]], tab[c + "_id"].values
        out.append(roc_auc_score(true, pj[:, 1]) if N_LEVELS[j] == 2
                   else roc_auc_score(true, pj, multi_class="ovr", average="macro"))
    return float(np.mean(out))


def train_joint(df, seed, lam):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = df[df.split == "train"]
    loss_fns = []
    for c, k in zip(CRIT, N_LEVELS):
        cnt = np.bincount(tr[c + "_id"], minlength=k).astype(np.float32) + 1.0
        loss_fns.append(nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (k * cnt), device=DEVICE)))
    pw = torch.tensor([(tr.y == 0).sum() / (tr.y == 1).sum()], dtype=torch.float32, device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    eye, brain = ConceptModel("ham").to(DEVICE), nn.Linear(int(OFF[-1]), 1).to(DEVICE)
    opt = torch.optim.Adam([{"params": eye.parameters(), "lr": 1e-4, "weight_decay": 1e-5},
                            {"params": brain.parameters(), "lr": BRAIN_LR}])

    def forward(x):
        out = eye(x)
        P = torch.cat([torch.softmax(o, 1) for o in out], 1)       # ۱۹ احتمال نشانه
        return out, P, brain(P).squeeze(1)

    def losses(out, c, z, y):
        lc = sum(fn(o, c[:, j]) for j, (fn, o) in enumerate(zip(loss_fns, out)))
        return lc, bce(z, y.float())

    lt, lv = s6.loader(df, "train", True), s6.loader(df, "valid", False)
    best, best_ep, state, t0 = float("inf"), 0, None, time.time()
    for ep in range(1, s6.MAX_EPOCHS + 1):
        eye.train()
        for x, c, y in lt:
            x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
            out, _, z = forward(x)
            lc, lb = losses(out, c, z, y)
            opt.zero_grad(); (lc + lam * lb).backward(); opt.step()
        eye.eval(); v, n = 0.0, 0
        with torch.no_grad():
            for x, c, y in lv:
                x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                out, _, z = forward(x)
                lc, lb = losses(out, c, z, y)
                v += (lc + lam * lb).item() * len(x); n += len(x)
        v /= n
        if v < best:
            best, best_ep = v, ep
            state = ({k: t.detach().clone() for k, t in eye.state_dict().items()},
                     {k: t.detach().clone() for k, t in brain.state_dict().items()})
        elif ep - best_ep >= s6.PATIENCE:
            break
    eye.load_state_dict(state[0]); brain.load_state_dict(state[1])
    print(f"    joint λ={lam:g}: best epoch {best_ep} (stopped at {ep}) | {time.time() - t0:.0f}s", flush=True)

    @torch.no_grad()
    def run(split):
        eye.eval(); Ps, zs = [], []
        for x, _, _ in s6.loader(df, split, False):
            _, P, z = forward(x.to(DEVICE))
            Ps.append(P.cpu()); zs.append(z.cpu())
        return torch.cat(Ps).numpy(), torch.cat(zs).numpy()

    return {s: run(s) for s in ["valid", "test"]}, brain.weight.detach().cpu().numpy()[0]


def main():
    df = load_table()
    y = {s: df[df.split == s].y.values for s in ["valid", "test"]}
    tab = {s: df[df.split == s] for s in ["valid", "test"]}
    saved = np.load("notes/step7_scores.npz")
    rows, weights = [], {}

    def record(seed, method, z, P):
        r = {"seed": seed, "method": method}
        for s in ["valid", "test"]:
            r[f"{s}_mel"] = roc_auc_score(y[s], z[s])
            r[f"{s}_concepts"] = concept_auroc(P[s], tab[s]) if P else np.nan
        t = pick_threshold(z["valid"], y["valid"], 0.90)
        te = at_threshold(z["test"], y["test"], t)
        r.update({"test_recall": te["recall"], "test_FN": te["FN"], "test_FP": te["FP"]})
        rows.append(r)
        cs = f"{r['valid_concepts']:.3f}" if P else "  —  "
        print(f"  → {method:<14} VALID melanoma {r['valid_mel']:.3f} | concepts {cs} | "
              f"TEST @90%: recall {te['recall']:.2f} missed {te['FN']:2d} false alarms {te['FP']:3d}", flush=True)

    for seed in SEEDS:
        t0 = time.time()
        print(f"\n━━━ seed {seed} ━━━", flush=True)
        eye = s6.train_eye(df, "train", seed, "sequential eye (for concept AUROC)")
        Pseq = {s: s6.eye_opinion(eye, df, s) for s in ["valid", "test"]}
        z = {s: saved[f"seq_seed{seed}_{s}"] for s in ["valid", "test"]}
        record(seed, "sequential", z, Pseq)
        for lam in LAMBDAS:
            res, w = train_joint(df, seed, lam)
            weights[(seed, lam)] = w
            record(seed, f"joint λ={lam:g}", {s: res[s][1] for s in res}, {s: res[s][0] for s in res})
        record(seed, "black box", {s: saved[f"black_seed{seed}_{s}"] for s in ["valid", "test"]}, None)
        print(f"  seed {seed}: {(time.time() - t0) / 60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv("notes/step9_results.csv", index=False)

    R = pd.DataFrame(rows)
    methods = list(dict.fromkeys(R.method))
    print("\n\n════════════ SUMMARY (mean ± std over seeds) ════════════")
    print(f"{'':16}{'VALID melanoma':>17}{'TEST melanoma':>16}{'VALID signs':>14}{'TEST signs':>13}"
          f"{'recall@90':>11}{'missed':>8}{'false al.':>10}")
    for m in methods:
        q = R[R.method == m]
        f = lambda k: f"{q[k].mean():.3f}±{q[k].std(ddof=0):.3f}"
        sg = (lambda k: f(k)) if q.valid_concepts.notna().all() else (lambda k: "—")
        print(f"{m:<16}{f('valid_mel'):>17}{f('test_mel'):>16}{sg('valid_concepts'):>14}{sg('test_concepts'):>13}"
              f"{q.test_recall.mean():>11.2f}{q.test_FN.mean():>8.1f}{q.test_FP.mean():>10.1f}")
    seq = R[R.method == "sequential"].sort_values("seed")
    for m in methods:
        if m.startswith("joint"):
            q = R[R.method == m].sort_values("seed")
            d = q.valid_mel.values - seq.valid_mel.values
            dc = q.valid_concepts.values - seq.valid_concepts.values
            print(f"paired VALID, {m} − sequential: melanoma {np.round(d, 3)} (wins {int((d > 0).sum())}/{len(d)}) | "
                  f"signs {np.round(dc, 3)} (wins {int((dc > 0).sum())}/{len(dc)})")
    print("\nbrain weights of joint models, seed 0 (positive = toward melanoma):")
    names = [f"{c}:{l}" for c in CRIT for l in LEVELS[c]]
    for lam in LAMBDAS:
        w = weights[(SEEDS[0], lam)]
        top = np.argsort(-w)
        print(f"  λ={lam:g}  top+: " + ", ".join(f"{names[i]} {w[i]:+.2f}" for i in top[:3])
              + "   top−: " + ", ".join(f"{names[i]} {w[i]:+.2f}" for i in top[-3:]))
    print("\nsaved notes/step9_results.csv")


if __name__ == "__main__":
    main()
