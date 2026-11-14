"""
قدم ۵-ج — آیا زوم روی خال به چشم کمک می‌کند؟

    چشم با عکس کامل  : بدنه‌ی HAM10000 → تنظیم دقیق روی عکس کامل Derm7pt
    چشم با عکس بریده : بدنه‌ی HAM10000 → تنظیم دقیق روی برش‌های قدم ۵-ب

هر دو از بهترین بدنه‌ی فعلی (قدم ۳-ب) شروع می‌کنند. هر کدام ۳ بار.
تصمیم فقط با valid؛ test فقط گزارش می‌شود.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from derm7pt_data import CRIT, N_LEVELS, load_table, Derm7ptDataset, train_transform, eval_transform
import step3c_compare_eyes as s3

CROPS = "data/derm7pt_crops"
ARMS = ["full", "crop"]


class CropDataset(Derm7ptDataset):
    def __getitem__(self, i):
        row = self.t.iloc[i]
        x = self.transform(Image.open(f"{CROPS}/{row.case_num}.jpg").convert("RGB"))
        concepts = torch.tensor([row[c + "_id"] for c in CRIT], dtype=torch.long)
        return x, concepts, torch.tensor(row.y, dtype=torch.long)


def main():
    df = load_table()
    kw = dict(batch_size=16, num_workers=4, persistent_workers=True)
    loaders = {}
    for arm, D in [("full", Derm7ptDataset), ("crop", CropDataset)]:
        loaders[arm] = {s: DataLoader(D(df, s, train_transform if s == "train" else eval_transform),
                                      shuffle=(s == "train"), **kw) for s in ["train", "valid", "test"]}
    tr = df[df.split == "train"]
    loss_fns = []
    for c, k in zip(CRIT, N_LEVELS):
        cnt = np.bincount(tr[c + "_id"], minlength=k).astype(np.float32)
        loss_fns.append(nn.CrossEntropyLoss(weight=torch.tensor(cnt.sum() / (k * cnt), device=s3.DEVICE)))
    brain = s3.train_brain(df)

    results = []
    runs = [(s, a) for s in s3.SEEDS for a in ARMS]
    for i, (seed, arm) in enumerate(runs, 1):
        label = "eye on FULL image" if arm == "full" else "eye on CROPPED lesion"
        print(f"\n━━━ run {i}/{len(runs)}: {label} (HAM backbone), seed {seed} ━━━", flush=True)
        r = s3.run_one("ham", seed, df, loaders[arm], loss_fns, brain)
        r["arm"] = arm
        results.append(r)
        print(f"  → best epoch {r['best_epoch']} | VALID eye {r['valid_eye']:.3f}  chain {r['valid_chain']:.3f} "
              f"| {r['minutes']:.1f} min", flush=True)
        pd.DataFrame(results).to_csv("notes/step5c_results.csv", index=False)

    R = pd.DataFrame(results)
    print("\n\n════════════ SUMMARY (mean ± std over seeds) ════════════")
    print(f"{'':32}{'FULL image':>16}{'CROPPED':>16}")
    for key, name in [("valid_eye", "VALID  eye (7 signs AUROC)"), ("valid_chain", "VALID  chain (melanoma AUROC)"),
                      ("test_eye", "TEST   eye (7 signs AUROC)"), ("test_chain", "TEST   chain (melanoma AUROC)")]:
        cells = [f"{R[R.arm == a][key].mean():.3f} ± {R[R.arm == a][key].std(ddof=0):.3f}" for a in ARMS]
        print(f"{name:<32}{cells[0]:>16}{cells[1]:>16}")
    print("\nper sign, VALID AUROC (mean over seeds):")
    for c in CRIT:
        f, k = R[R.arm == "full"][f"valid_{c}"].mean(), R[R.arm == "crop"][f"valid_{c}"].mean()
        print(f"  {c:<24}{f:>8.3f}{k:>8.3f}   {'+' if k > f else ''}{k - f:.3f}")
    for key in ["valid_eye", "valid_chain"]:
        d = R[R.arm == "crop"].sort_values("seed")[key].values - R[R.arm == "full"].sort_values("seed")[key].values
        print(f"paired {key}: crop − full = {np.round(d, 3)}  (crop wins {int((d > 0).sum())}/{len(d)})")
    print("\nsaved notes/step5c_results.csv")


if __name__ == "__main__":
    main()
