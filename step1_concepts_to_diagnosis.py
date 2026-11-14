"""
قدم ۱ — از مفاهیم به تشخیص، بدون تصویر.

سؤال: اگر هفت مفهوم بالینی را دقیقاً درست بدانیم (برچسب پزشک)،
یک طبقه‌بند خطی چقدر خوب ملانوما را تشخیص می‌دهد؟
این عدد «سقف کارایی» گلوگاه مفهومی است.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

R = "data/derm7pt/release_v0"
torch.manual_seed(0)
np.random.seed(0)

# ───────────────────────── ۱. خواندن داده ─────────────────────────
df = pd.read_csv(f"{R}/meta/meta.csv")

# جدول ادغام در فایل مشترک derm7pt_data.py تعریف شده تا همه‌ی گام‌ها از یک نسخه استفاده کنند
from derm7pt_data import MERGE
CRIT = list(MERGE)

for c, mapping in MERGE.items():
    df[c + "_m"] = df[c].map(mapping)
    assert df[c + "_m"].notna().all(), f"برچسب نگاشت‌نشده در {c}"

# برچسب هدف: ملانوما در برابر بقیه
df["y"] = df.diagnosis.str.contains("melanoma").astype(int)

# ───────────── ۲. تبدیل مفاهیم به عدد (one-hot) ─────────────
# هر مفهوم چند کلاس دارد؛ هر کلاس یک ستون صفر/یک می‌شود.
X = pd.get_dummies(df[[c + "_m" for c in CRIT]], prefix=CRIT).astype(np.float32)
feature_names = list(X.columns)
print(f"ابعاد بردار مفهومی: {X.shape[1]}")
print("ستون‌ها:", feature_names, "\n")

# ───────────── ۳. تقسیم رسمی (شماره‌ی سطر، نه case_num) ─────────────
idx = {s: pd.read_csv(f"{R}/meta/{s}_indexes.csv").iloc[:, 0].values
       for s in ["train", "valid", "test"]}

# ───── ۳.۵ ذخیره و نمایش جدول نهایی پس از پیش‌پردازش ─────
split = pd.Series("?", index=df.index, name="split")
for s, v in idx.items():
    split.iloc[v] = s

table = pd.concat([
    df[["case_num", "diagnosis"]],
    split,
    df.y.rename("y_melanoma"),
    df[[c + "_m" for c in CRIT]],   # مفاهیم پس از ادغام (متنی، برای خواندن آدم)
    X.astype(int),                  # همان مفاهیم به‌صورت ۱۹ ستون صفر/یک (ورودی مدل)
], axis=1)
table.to_csv("step1_preprocessed.csv", index=False)

print(f"\nجدول نهایی ذخیره شد → step1_preprocessed.csv  (ابعاد {table.shape[0]}×{table.shape[1]})")
print("\n─── پنج سطر اول جدول (فقط ستون‌های خوانا) ───")
readable = table[["case_num", "split", "y_melanoma"] + [c + "_m" for c in CRIT]]
print(readable.head().to_string(index=False))
print("\n─── همان پنج سطر، ۱۹ ستون عددی که واقعاً وارد مدل می‌شود ───")
print(X.astype(int).head().to_string(index=False))
print()
Xt = {s: torch.tensor(X.iloc[v].values) for s, v in idx.items()}
yt = {s: torch.tensor(df.y.iloc[v].values) for s, v in idx.items()}
for s in idx:
    n = len(idx[s]); pos = int(yt[s].sum())
    print(f"{s:5s}: {n:4d} نمونه — melanoma {pos:3d} ({pos/n:.0%})")

# ───────────── ۴. مدل: یک لایه‌ی خطی با یک خروجی ─────────────
# خروجی یک عدد z است:  z > 0 یعنی ملانوما ،  z < 0 یعنی خوش‌خیم
# (همان رگرسیون لجستیک: ۱۹ وزن + ۱ عدد ثابت = ۲۰ پارامتر)
model = nn.Linear(X.shape[1], 1)

# وزن‌دهی کلاس: ملانوما کم‌تعدادتر است، پس اشتباه روی آن گران‌تر
n_pos = int(yt["train"].sum())
n_neg = len(yt["train"]) - n_pos
pos_weight = torch.tensor([n_neg / n_pos])
print(f"\npos_weight (melanoma): {pos_weight.item():.2f}   ({n_neg} non-melanoma / {n_pos} melanoma)")

# تابع خطا: اول z را با تابع لجستیک به احتمال تبدیل می‌کند، بعد -ln می‌گیرد
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


def forward(split):
    """خروجی مدل برای یک بخش داده: بردار z به طول تعداد بیماران."""
    return model(Xt[split]).squeeze(1)


# ───────────── ۵. حلقه‌ی آموزش — قلب یادگیری عمیق ─────────────
print("\n─── آموزش ───")
best_valid_loss, best_state, best_epoch = float("inf"), None, 0
for epoch in range(301):
    model.train()
    optimizer.zero_grad()                          # الف) مشتق‌های قبلی را صفر کن
    z = forward("train")                           # ب) پیش‌بینی جلو برو
    loss = loss_fn(z, yt["train"].float())         # ج) چقدر بد بود؟
    loss.backward()                                # د) مشتق خطا نسبت به هر وزن
    optimizer.step()                               # ه) وزن‌ها را کمی اصلاح کن

    # هر دور روی valid بسنج و بهترین لحظه را نگه دار
    model.eval()
    with torch.no_grad():
        zv = forward("valid")
        valid_loss = loss_fn(zv, yt["valid"].float()).item()
    if valid_loss < best_valid_loss:
        best_valid_loss, best_epoch = valid_loss, epoch
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 30 == 0:
        tr = ((z > 0).long() == yt["train"]).float().mean()
        va = ((zv > 0).long() == yt["valid"]).float().mean()
        print(f"epoch {epoch:3d} | train loss {loss.item():.4f} | valid loss {valid_loss:.4f} "
              f"| train acc {tr:.3f} | valid acc {va:.3f}")

model.load_state_dict(best_state)      # بهترین مدل بر اساس valid، نه test
print(f"بهترین مدل: epoch {best_epoch}  (valid loss {best_valid_loss:.4f})")

# ───────────── ۶. ارزیابی روی بخش آزمون ─────────────
model.eval()
with torch.no_grad():
    pred = (forward("test") > 0).long()            # علامت z تصمیم می‌گیرد
y_true = yt["test"]
TP = int(((pred == 1) & (y_true == 1)).sum()); FN = int(((pred == 0) & (y_true == 1)).sum())
TN = int(((pred == 0) & (y_true == 0)).sum()); FP = int(((pred == 1) & (y_true == 0)).sum())
acc = (TP + TN) / len(y_true)
recall = TP / (TP + FN)        # sensitivity — چند درصد ملانوماها را گرفت؟ (حیاتی‌ترین معیار)
spec = TN / (TN + FP)
prec = TP / (TP + FP) if TP + FP else 0.0
f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0

print("\n─── نتیجه روی ۳۹۵ مورد test ───")
print(f"confusion matrix :  TP={TP}  FN={FN}  TN={TN}  FP={FP}")
print(f"accuracy         : {acc:.3f}")
print(f"recall (sensitivity): {recall:.3f}   ← چند درصد ملانوماها تشخیص داده شد")
print(f"specificity      : {spec:.3f}")
print(f"precision        : {prec:.3f}")
print(f"F1               : {f1:.3f}")

# ───────────── ۷. مقایسه با خودِ قاعده‌ی بالینی ─────────────
# قاعده‌ی هفت‌نقطه‌ای: امتیاز ۳ یا بیشتر = مشکوک به ملانوما
s7 = df.seven_point_score.iloc[idx["test"]].values >= 3
yt_np = y_true.numpy()
rule_acc = (s7 == yt_np).mean()
rule_sens = (s7 & (yt_np == 1)).sum() / (yt_np == 1).sum()
rule_spec = (~s7 & (yt_np == 0)).sum() / (yt_np == 0).sum()
print(f"\nbaseline — قاعده‌ی کلاسیک ۷-نقطه‌ای (امتیاز ≥ ۳):")
print(f"accuracy {rule_acc:.3f} | recall {rule_sens:.3f} | specificity {rule_spec:.3f}")

# ───────────── ۸. وزن‌های یادگرفته‌شده = تفسیرپذیری ─────────────
# فقط یک ردیف وزن داریم: مثبت یعنی به سمت ملانوما، منفی یعنی به سمت خوش‌خیم
w = model.weight[0].detach().numpy()
print(f"\nعدد ثابت (bias): {model.bias.item():+.2f}")
order = np.argsort(-w)
print("\n─── چه چیزی مدل را به سمت «ملانوما» می‌برد؟ ───")
for i in order[:6]:
    print(f"  {w[i]:+.2f}  {feature_names[i]}")
print("─── و چه چیزی به سمت «خوش‌خیم»؟ ───")
for i in order[-5:]:
    print(f"  {w[i]:+.2f}  {feature_names[i]}")
