"""
قدم ۸ — مرحله‌ی سوم مقاله (Reference): ارجاع به نمونه‌های مشابه.

بانک مرجع: ۴۱۳ خال آموزشی. پرس‌وجو: بیماران valid/test. چهار تعریف «شباهت» می‌سنجیم:
  A  شباهت کسینوسی روی ۱۹ احتمال نشانه (همان روش مقاله)
  B  فاصله‌ی L1 روی همان ۱۹ احتمال (اصلاح ضعف کسینوس روی بردار نامنفی)
  C  شباهت کسینوسی روی ۵۱۲ ویژگی عکس
  D  ۵۱۲ ویژگی → ۶۴ بعد، آموزش با triplet loss روی گروه تشخیص (مثل مقاله)

سنجه‌ها (هر سه با مقایسه‌ی «۳ همسایه‌ی نزدیک»):
  group     : چند درصد همسایه‌ها هم‌گروه بیمارند (ملانوما / خال / بقیه)
  signs     : چند نشانه از ۷ نشانه‌ی «برچسب پزشک» بین بیمار و همسایه یکی است
  kNN AUROC : رأی ۱۰ همسایه برای تشخیص ملانوما

انتخاب روش فقط با valid. سه تکرار. در پایان، گزارش تصویری برای چند بیمار test.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import step6_sequential_brain as s6
from derm7pt_data import CRIT, LEVELS, load_table
from step3c_compare_eyes import onehot, DEVICE

QUICK = os.environ.get("QUICK") == "1"
SEEDS = [0] if QUICK else [0, 1, 2]
TRIPLET_STEPS = 30 if QUICK else 300
METHODS = ["A cosine on concepts (paper)", "B L1 on concepts",
           "C cosine on image features", "D triplet embedding (paper)"]
SHORT = {"pigment_network": "network", "streaks": "streaks", "pigmentation": "pigment.",
         "regression_structures": "regression", "dots_and_globules": "dots",
         "blue_whitish_veil": "blue veil", "vascular_structures": "vessels"}


def group_of(diag):
    return 0 if "melanoma" in diag else (1 if "nevus" in diag else 2)      # ملانوما / خال / بقیه


GROUP_NAME = ["melanoma", "nevus", "other"]


@torch.no_grad()
def extract(model, df, split):
    """برای هر بیمار: ۱۹ احتمال نشانه و بردار ۵۱۲تایی ویژگی عکس."""
    model.eval()
    probs, feats = [], []
    for x, _, _ in s6.loader(df, split, False):
        f = model.backbone(x.to(DEVICE))
        out = [h(f) for h in model.heads]
        probs.append(torch.cat([torch.softmax(o, 1) for o in out], 1).cpu())
        feats.append(f.cpu())
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def l2n(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-9)


def similarity(method, Q, B):
    if method.startswith("B"):
        return -cdist(Q, B, "cityblock")
    return l2n(Q) @ l2n(B).T


def train_projection(Ftr, gtr, Fva, gva, seed):
    """ویژگی ۵۱۲ → ۶۴ بعد با triplet loss؛ توقف زودهنگام با هم‌گروهی valid."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    prep = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    Xtr, Xva = prep(Ftr), prep(Fva)
    proj = nn.Linear(512, 64)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3, weight_decay=1e-3)
    members = [np.where(gtr == g)[0] for g in range(3)]
    emb = lambda X: F.normalize(proj(X), dim=1)
    best, best_state = -1.0, None
    for step in range(1, TRIPLET_STEPS + 1):
        a = rng.integers(0, len(gtr), 128)
        p = np.array([rng.choice(members[gtr[i]]) for i in a])
        n = np.array([rng.choice(np.concatenate([members[g] for g in range(3) if g != gtr[i]])) for i in a])
        loss = F.triplet_margin_loss(emb(Xtr[a]), emb(Xtr[p]), emb(Xtr[n]), margin=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 10 == 0:
            with torch.no_grad():
                S = (emb(Xva) @ emb(Xtr).T).numpy()
            g3 = (gtr[np.argsort(-S, 1)[:, :3]] == gva[:, None]).mean()
            if g3 > best:
                best, best_state = g3, {k: v.clone() for k, v in proj.state_dict().items()}
    proj.load_state_dict(best_state)
    return lambda A: emb(prep(A)).detach().numpy()


def retrieval_scores(S, g_q, y_q, conc_q, g_b, y_b, conc_b):
    top = np.argsort(-S, 1)
    t3, t10 = top[:, :3], top[:, :10]
    return {"group": (g_b[t3] == g_q[:, None]).mean(),
            "signs": (conc_b[t3] == conc_q[:, None, :]).mean(),
            "knn_auroc": roc_auc_score(y_q, y_b[t10].mean(1))}


def main():
    df = load_table()
    df["group"] = df.diagnosis.map(group_of)
    sp = {s: df[df.split == s] for s in ["train", "valid", "test"]}
    g = {s: sp[s].group.values for s in sp}
    y = {s: sp[s].y.values for s in sp}
    conc = {s: sp[s][[c + "_id" for c in CRIT]].values for s in sp}
    base = (conc["train"][None] == conc["valid"][:, None]).mean()
    print(f"bank: {len(sp['train'])} training cases   groups: "
          + "  ".join(f"{GROUP_NAME[k]} {int((g['train'] == k).sum())}" for k in range(3)))
    print(f"chance level for 'signs agree' (random neighbours): {base:.3f}\n", flush=True)

    train_idx = df.index[df.split == "train"]
    doctor = {s: onehot(sp[s]) for s in sp}
    rows, keep = [], {}
    for seed in SEEDS:
        t0 = time.time()
        print(f"━━━ seed {seed} ━━━", flush=True)
        if seed == 0:                                   # مغز فقط برای گزارش تصویری لازم است
            oof = np.zeros((len(train_idx), doctor["train"].shape[1]), dtype=np.float32)
            skf = StratifiedKFold(n_splits=s6.FOLDS, shuffle=True, random_state=seed)
            for f, (fit, held) in enumerate(skf.split(train_idx, y["train"]), 1):
                d = df.copy()
                d.loc[train_idx[fit], "split"] = "fold_fit"; d.loc[train_idx[held], "split"] = "fold_held"
                oof[held] = s6.eye_opinion(s6.train_eye(d, "fold_fit", seed, f"fold {f}/{s6.FOLDS}"), d, "fold_held")
        eye = s6.train_eye(df, "train", seed, "final eye")
        P, Fe = {}, {}
        for s in sp:
            P[s], Fe[s] = extract(eye, df, s)
        emb = train_projection(Fe["train"], g["train"], Fe["valid"], g["valid"], seed)
        spaces = {METHODS[0]: P, METHODS[1]: P, METHODS[2]: Fe, METHODS[3]: {s: emb(Fe[s]) for s in sp}}
        for m in METHODS:
            for s in ["valid", "test"]:
                S = similarity(m, spaces[m][s], spaces[m]["train"])
                rows.append({"seed": seed, "method": m, "split": s,
                             **retrieval_scores(S, g[s], y[s], conc[s], g["train"], y["train"], conc["train"])})
        if seed == 0:
            brain = s6.train_brain(oof, y["train"], P["valid"], y["valid"])
            keep = {"P": P, "Fe": Fe, "spaces": spaces, "z_test": brain(P["test"])}
            np.savez("notes/step8_eye_outputs.npz", **{f"P_{s}": P[s] for s in sp},
                     **{f"F_{s}": Fe[s] for s in sp}, z_test=keep["z_test"])
        print(f"  seed {seed}: {(time.time() - t0) / 60:.1f} min", flush=True)
        pd.DataFrame(rows).to_csv("notes/step8_results.csv", index=False)

    R = pd.DataFrame(rows)
    print("\n\n════════════ SUMMARY: 3 nearest cases, bank = 413 training cases (mean ± std over seeds) ════════════")
    print(f"chance level for 'signs agree': {base:.3f}")
    for s in ["valid", "test"]:
        print(f"\n— {s.upper()} —")
        print(f"{'method':<32}{'group agree':>14}{'signs agree':>14}{'kNN AUROC':>14}")
        for m in METHODS:
            q = R[(R.method == m) & (R.split == s)]
            print(f"{m:<32}" + "".join(f"{q[k].mean():>9.3f}±{q[k].std(ddof=0):.3f}" for k in ["group", "signs", "knn_auroc"]))
    v = R[R.split == "valid"].groupby("method").group.mean()
    best = v.idxmax()
    print(f"\nbest on VALID by group agreement: {best}")
    print("reference: our chain's own melanoma AUROC on test = 0.835")

    # ───── گزارش تصویری برای ۶ بیمار test ─────
    z, S = keep["z_test"], similarity(best, keep["spaces"][best]["test"], keep["spaces"][best]["train"])
    yt, Pt = y["test"], keep["P"]["test"]
    order = np.argsort(-z)
    pick = ([i for i in order if yt[i] == 1][:2] + [[i for i in order if yt[i] == 1][-1]]
            + [[i for i in order if yt[i] == 0][-1], [i for i in order if yt[i] == 0][-2]]
            + [[i for i in order if yt[i] == 0][0]])
    tags = ["melanoma, flagged", "melanoma, flagged", "melanoma, MISSED",
            "benign, correct", "benign, correct", "benign, FALSE ALARM"]
    col = {c: [i for i, (cc, _) in enumerate([(c, l) for c in CRIT for l in LEVELS[c]]) if cc == c] for c in CRIT}

    fig, ax = plt.subplots(6, 4, figsize=(14, 20))
    print("\n─── example reports (test cases) ───")
    for r, (i, tag) in enumerate(zip(pick, tags)):
        q = sp["test"].iloc[i]
        seen = []
        for c in CRIT:
            pr = Pt[i][col[c]]
            k = int(pr.argmax())
            if k != 0:
                seen.append((pr[k], f"{SHORT[c]}: {LEVELS[c][k]} ({pr[k]:.2f})"))
        seen = [t for _, t in sorted(seen, reverse=True)[:3]] or ["no sign found"]
        nb = np.argsort(-S[i])[:3]
        ax[r, 0].imshow(Image.open(f"data/derm7pt_crops/{q.case_num}.jpg"))
        ax[r, 0].set_title(f"QUERY #{q.case_num}  [{tag}]\ntrue: {q.diagnosis}\nscore {z[i]:+.1f}  eye sees: " + "; ".join(seen),
                           fontsize=8, loc="left")
        lines = []
        for k, j in enumerate(nb):
            b = sp["train"].iloc[j]
            agree = int((conc["train"][j] == conc["test"][i]).sum())
            ax[r, k + 1].imshow(Image.open(f"data/derm7pt_crops/{b.case_num}.jpg"))
            ax[r, k + 1].set_title(f"similar #{b.case_num}  sim {S[i, j]:.2f}\n{b.diagnosis}\n{agree}/7 signs agree with doctor",
                                   fontsize=8, loc="left")
            lines.append(f"#{b.case_num} ({b.diagnosis}, {agree}/7)")
        print(f"#{q.case_num} [{tag}] true={q.diagnosis} | score {z[i]:+.1f} | eye sees: {'; '.join(seen)}")
        print(f"      similar cases: {'  '.join(lines)}")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"Phase 3 — Reference: three most similar training cases  (similarity = {best})", fontsize=12)
    fig.tight_layout()
    fig.savefig("notes/step8_reference_examples.png", dpi=75)
    print("\nsaved notes/step8_reference_examples.png, notes/step8_results.csv, notes/step8_eye_outputs.npz")


if __name__ == "__main__":
    main()
