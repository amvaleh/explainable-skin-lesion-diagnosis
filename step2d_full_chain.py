"""
قدم ۲-د — زنجیره‌ی کامل: عکس ← مفاهیم حدس‌زده ← تشخیص ملانوما.

سه مدل را روی همان ۳۹۵ بیمار test مقایسه می‌کنیم:
  ۱. سقف       : طبقه‌بند خطی روی برچسب پزشک
  ۲. زنجیره    : همان طبقه‌بند، روی مفاهیمی که مدل قدم ۲-ج از عکس حدس زده
  ۳. جعبه‌سیاه : ResNet18 که مستقیم از عکس ملانوما را پیش‌بینی می‌کند (بدون مفاهیم)
به‌علاوه‌ی قاعده‌ی هفت‌نقطه‌ای، هم با برچسب پزشک و هم با مفاهیم حدس‌زده.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score

from derm7pt_data import (CRIT, LEVELS, load_table,
                          Derm7ptDataset, train_transform, eval_transform)

torch.manual_seed(0)
np.random.seed(0)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df = load_table()
test = df[df.split == "test"].reset_index(drop=True)
y_test = test.y.values

# ستون‌های بردار مفهومی، به ترتیب ثابت LEVELS (۱۹ ستون)
COLS = [(c, lvl) for c in CRIT for lvl in LEVELS[c]]


def onehot_from_ids(table):
    """برچسب‌های شماره‌ای → بردار ۱۹تایی صفر و یک."""
    return np.stack([(table[c + "_id"].values == LEVELS[c].index(lvl)).astype(np.float32)
                     for c, lvl in COLS], axis=1)


def metrics(z, y):
    """z: امتیاز ملانوما (مثبت = ملانوما). خروجی: سنجه‌ها."""
    pred = (z > 0).astype(int)
    TP = ((pred == 1) & (y == 1)).sum(); FN = ((pred == 0) & (y == 1)).sum()
    TN = ((pred == 0) & (y == 0)).sum(); FP = ((pred == 1) & (y == 0)).sum()
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN)
    return dict(acc=(TP + TN) / len(y), recall=rec, spec=TN / (TN + FP), prec=prec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                auroc=roc_auc_score(y, z), FN=int(FN), FP=int(FP))


# ───────────── ۱. طبقه‌بند خطی روی برچسب پزشک (همان قدم ۱) ─────────────
Xg = {s: torch.tensor(onehot_from_ids(df[df.split == s])) for s in ["train", "valid", "test"]}
yg = {s: torch.tensor(df[df.split == s].y.values, dtype=torch.float32) for s in ["train", "valid", "test"]}
pos_weight = torch.tensor([(yg["train"] == 0).sum() / (yg["train"] == 1).sum()])
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

clf = nn.Linear(len(COLS), 1)
opt = torch.optim.Adam(clf.parameters(), lr=0.05)
best, best_state = float("inf"), None
for epoch in range(301):
    opt.zero_grad()
    loss_fn(clf(Xg["train"]).squeeze(1), yg["train"]).backward()
    opt.step()
    with torch.no_grad():
        vl = loss_fn(clf(Xg["valid"]).squeeze(1), yg["valid"]).item()
    if vl < best:
        best, best_state = vl, {k: v.clone() for k, v in clf.state_dict().items()}
clf.load_state_dict(best_state)


def classify(X):
    with torch.no_grad():
        return clf(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


# ───────────── ۲. مفاهیم حدس‌زده از قدم ۲-ج ─────────────
pred = pd.read_csv("step2c_test_predictions.csv")
assert (pred.case_num.values == test.case_num.values).all()

X_soft = np.stack([pred[f"p_{c}_{lvl}"].values for c, lvl in COLS], axis=1)   # احتمال‌ها
pred_ids = pd.DataFrame({c + "_id": np.argmax(
    np.stack([pred[f"p_{c}_{lvl}"].values for lvl in LEVELS[c]], 1), 1) for c in CRIT})
X_hard = onehot_from_ids(pred_ids)                                              # فقط محتمل‌ترین حالت


# ───────────── ۳. قاعده‌ی هفت‌نقطه‌ای ─────────────
def seven_point(ids):
    """امتیاز قاعده از روی شماره‌ی حالت‌ها. معیار اصلی ۲ امتیاز، فرعی ۱ امتیاز."""
    L = {c: LEVELS[c] for c in CRIT}
    s = 2 * (ids["pigment_network_id"] == L["pigment_network"].index("atypical"))
    s += 2 * (ids["blue_whitish_veil_id"] == L["blue_whitish_veil"].index("present"))
    s += 2 * (ids["vascular_structures_id"] == L["vascular_structures"].index("irregular"))
    s += 1 * (ids["streaks_id"] == L["streaks"].index("irregular"))
    s += 1 * (ids["pigmentation_id"] == L["pigmentation"].index("irregular"))
    s += 1 * (ids["dots_and_globules_id"] == L["dots_and_globules"].index("irregular"))
    s += 1 * (ids["regression_structures_id"] == L["regression_structures"].index("present"))
    return s.values.astype(float)


# امتیاز منهای ۲٫۵ ⇒ مثبت یعنی امتیاز ≥ ۳ (همان آستانه‌ی بالینی)
z_rule_gt = seven_point(test) - 2.5
z_rule_pred = seven_point(pred_ids) - 2.5

# ───────────── ۴. جعبه‌سیاه: عکس → ملانوما، مستقیم ─────────────
print("training black-box ResNet18 (image → melanoma) ...", flush=True)
ds = {"train": Derm7ptDataset(df, "train", train_transform),
      "valid": Derm7ptDataset(df, "valid", eval_transform),
      "test":  Derm7ptDataset(df, "test",  eval_transform)}
loader = {s: DataLoader(d, batch_size=16, shuffle=(s == "train")) for s, d in ds.items()}

bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
bb.fc = nn.Linear(512, 1)
bb = bb.to(DEVICE)
bb_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
bb_opt = torch.optim.Adam(bb.parameters(), lr=1e-4, weight_decay=1e-5)


@torch.no_grad()
def bb_scores(split):
    bb.eval()
    zs, total, n = [], 0.0, 0
    for x, _, y in loader[split]:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        z = bb(x).squeeze(1)
        total += bb_loss(z, y).item() * len(x); n += len(x)
        zs.append(z.cpu())
    return torch.cat(zs).numpy(), total / n


best, best_epoch, t0 = float("inf"), 0, time.time()
for epoch in range(1, 31):
    bb.train()
    for x, _, y in loader["train"]:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        bb_opt.zero_grad()
        bb_loss(bb(x).squeeze(1), y).backward()
        bb_opt.step()
    _, vl = bb_scores("valid")
    if vl < best:
        best, best_epoch = vl, epoch
        torch.save(bb.state_dict(), "checkpoints/step2d_blackbox.pt")
print(f"black-box best epoch {best_epoch} (valid loss {best:.3f}), {time.time() - t0:.0f}s\n")
bb.load_state_dict(torch.load("checkpoints/step2d_blackbox.pt", map_location=DEVICE))
z_bb, _ = bb_scores("test")

# ───────────── ۵. جدول نهایی ─────────────
rows = [
    ("1  ceiling: doctor's concepts → linear", classify(onehot_from_ids(test))),
    ("2a chain: predicted concepts (soft)",    classify(X_soft)),
    ("2b chain: predicted concepts (hard)",    classify(X_hard)),
    ("3  black-box: image → melanoma",         z_bb),
    ("R1 7-point rule, doctor's concepts",     z_rule_gt),
    ("R2 7-point rule, predicted concepts",    z_rule_pred),
]
print(f"{'model':<40}{'acc':>7}{'recall':>8}{'spec':>7}{'prec':>7}{'F1':>7}{'AUROC':>7}{'FN':>5}{'FP':>5}")
print("-" * 93)
for name, z in rows:
    m = metrics(z, y_test)
    print(f"{name:<40}{m['acc']:>7.3f}{m['recall']:>8.3f}{m['spec']:>7.3f}{m['prec']:>7.3f}"
          f"{m['f1']:>7.3f}{m['auroc']:>7.3f}{m['FN']:>5}{m['FP']:>5}")
print(f"\n(test: {len(y_test)} cases, {y_test.sum()} melanoma)")
