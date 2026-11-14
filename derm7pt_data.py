"""
تعاریف مشترک داده‌ی Derm7pt — جدول ادغام، ترتیب حالت‌ها، تقسیم رسمی، و Dataset.
همه‌ی گام‌ها از همین فایل استفاده می‌کنند تا تعریف داده فقط یک جا باشد.
"""
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = "data/derm7pt/release_v0"

# جدول ادغام: برچسب‌های کم‌تعداد در گروه‌های بالینی معنادار جمع می‌شوند.
# نام کلاس‌ها عمداً همان واژگان بالینی اصلی است.
MERGE = {
    "pigment_network":       {"absent": "absent", "typical": "typical", "atypical": "atypical"},
    "streaks":               {"absent": "absent", "regular": "regular", "irregular": "irregular"},
    "pigmentation":          {"absent": "absent",
                              "diffuse regular": "regular", "localized regular": "regular",
                              "diffuse irregular": "irregular", "localized irregular": "irregular"},
    "regression_structures": {"absent": "absent",
                              "blue areas": "present", "white areas": "present",
                              "combinations": "present"},
    "dots_and_globules":     {"absent": "absent", "regular": "regular", "irregular": "irregular"},
    "blue_whitish_veil":     {"absent": "absent", "present": "present"},
    "vascular_structures":   {"absent": "absent",
                              "comma": "benign", "hairpin": "benign",
                              "arborizing": "benign", "wreath": "benign",
                              "dotted": "irregular", "linear irregular": "irregular",
                              "within regression": "irregular"},
}
CRIT = list(MERGE)

# ترتیب ثابت حالت‌ها برای هر معیار: شماره‌ی هر حالت = جایگاهش در این فهرست.
# حالت «absent» همیشه شماره‌ی ۰ است.
LEVELS = {
    "pigment_network":       ["absent", "typical", "atypical"],
    "streaks":               ["absent", "regular", "irregular"],
    "pigmentation":          ["absent", "regular", "irregular"],
    "regression_structures": ["absent", "present"],
    "dots_and_globules":     ["absent", "regular", "irregular"],
    "blue_whitish_veil":     ["absent", "present"],
    "vascular_structures":   ["absent", "benign", "irregular"],
}
N_LEVELS = [len(LEVELS[c]) for c in CRIT]   # [3, 3, 3, 2, 3, 2, 3]

# نرمال‌سازی شبکه‌های آموزش‌دیده روی ImageNet
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 224


def load_table():
    """جدول ۱۰۱۱ بیمار با مفاهیم ادغام‌شده، شماره‌ی هر حالت، برچسب ملانوما و بخش رسمی."""
    df = pd.read_csv(f"{ROOT}/meta/meta.csv")
    for c, mapping in MERGE.items():
        df[c + "_m"] = df[c].map(mapping)
        assert df[c + "_m"].notna().all(), f"unmapped label in {c}"
        df[c + "_id"] = df[c + "_m"].map({lvl: i for i, lvl in enumerate(LEVELS[c])})
    df["y"] = df.diagnosis.str.contains("melanoma").astype(int)

    # فایل‌های تقسیم رسمی شماره‌ی سطر را نگه می‌دارند، نه case_num
    df["split"] = ""
    for s in ["train", "valid", "test"]:
        rows = pd.read_csv(f"{ROOT}/meta/{s}_indexes.csv").iloc[:, 0].values
        df.loc[df.index[rows], "split"] = s
    assert (df.split != "").all()
    return df


train_transform = T.Compose([
    T.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(180),
    # رنگ‌مایه دست‌نخورده می‌ماند، چون حجاب آبی‌ـ‌سفید و پسرفت آبی خودشان رنگ‌اند
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

eval_transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class Derm7ptDataset(Dataset):
    """هر عضو: (تصویر درموسکوپی، شماره‌ی حالت ۷ مفهوم، برچسب ملانوما)."""

    def __init__(self, table, split, transform):
        self.t = table[table.split == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        row = self.t.iloc[i]
        img = Image.open(f"{ROOT}/images/{row.derm}").convert("RGB")
        x = self.transform(img)
        concepts = torch.tensor([row[c + "_id"] for c in CRIT], dtype=torch.long)
        y = torch.tensor(row.y, dtype=torch.long)
        return x, concepts, y
