"""
قدم ۵-الف — فاز اول مقاله (Observation): شبکه‌ای که مرز خال را پیدا می‌کند.

معماری: Attention U-Net با رمزگذار ResNet18.
    رمزگذار عکس را کوچک می‌کند تا بفهمد چه می‌بیند،
    رمزگشا دوباره بزرگش می‌کند تا بگوید کجاست،
    دروازه‌ی توجه روی هر پل، پوست و مو و خط‌کش را خاموش می‌کند.

داده: ده هزار عکس HAM10000 با مرز خالشان. همان تقسیم بر اساس lesion_id که در قدم ۳-ب.
خروجی: برای هر پیکسل، احتمال اینکه جزء خال باشد.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, tv_tensors
from torchvision.transforms import v2
from sklearn.model_selection import StratifiedGroupKFold

from derm7pt_data import MEAN, STD

QUICK = os.environ.get("QUICK") == "1"
EPOCHS = 1 if QUICK else 6
SIZE, BATCH, LR = 256, 16, 3e-4
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# تبدیل‌ها روی عکس و ماسک «با هم» اعمال می‌شوند: اگر عکس بچرخد، ماسک هم همان‌قدر می‌چرخد
seg_train_tf = v2.Compose([
    v2.RandomResizedCrop(SIZE, scale=(0.6, 1.0), antialias=True),
    v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(180),
    v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD),
])
seg_eval_tf = v2.Compose([
    v2.Resize((SIZE, SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD),
])


class SegDataset(Dataset):
    def __init__(self, table, transform):
        self.t = table.reset_index(drop=True)
        self.tf = transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        iid = self.t.image_id.iloc[i]
        img = tv_tensors.Image(torch.from_numpy(np.asarray(
            Image.open(f"data/ham10000/images/{iid}.jpg").convert("RGB"))).permute(2, 0, 1))
        m = np.asarray(Image.open(f"data/ham10000/masks/{iid}_segmentation.png").convert("L")) > 127
        mask = tv_tensors.Mask(torch.from_numpy(m.astype(np.uint8)))
        img, mask = self.tf(img, mask)
        return img, mask.float().unsqueeze(0)


# ───────────── معماری ─────────────
class AttentionGate(nn.Module):
    """α = σ(ψ(ReLU(W_x·x + W_g·g)))  ؛  خروجی = α ⊙ x"""
    def __init__(self, ch_x, ch_g, ch_mid):
        super().__init__()
        self.wx = nn.Conv2d(ch_x, ch_mid, 1)
        self.wg = nn.Conv2d(ch_g, ch_mid, 1)
        self.psi = nn.Conv2d(ch_mid, 1, 1)

    def forward(self, x, g):
        alpha = torch.sigmoid(self.psi(F.relu(self.wx(x) + self.wg(g))))
        return x * alpha


def conv_block(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True),
        nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.gate = AttentionGate(ch_skip, ch_in, max(ch_skip // 2, 16))
        self.conv = conv_block(ch_in + ch_skip, ch_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.gate(skip, x)], dim=1))


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # ۶۴ کانال، ۱/۲ اندازه
        self.pool = r.maxpool
        self.enc1, self.enc2, self.enc3, self.enc4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)                 # 1/2
        s1 = self.enc1(self.pool(s0))     # 1/4
        s2 = self.enc2(s1)                # 1/8
        s3 = self.enc3(s2)                # 1/16
        s4 = self.enc4(s3)                # 1/32
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(d)               # (batch, 1, H, W) — امتیاز خام هر پیکسل


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def dice_iou(pred, target):
    """برای هر عکس جداگانه، با آستانه‌ی ۰٫۵. همان معادلات ۶ و ۷ مقاله."""
    inter = (pred * target).sum((1, 2, 3))
    union = ((pred + target) > 0).float().sum((1, 2, 3))
    dice = (2 * inter) / (pred.sum((1, 2, 3)) + target.sum((1, 2, 3))).clamp(min=1)
    return dice, inter / union.clamp(min=1)


def ham_split():
    meta = pd.read_csv("data/ham10000/raw/HAM10000_metadata.tab", sep="\t")
    for c in meta.select_dtypes(include=["object", "string"]):
        meta[c] = meta[c].str.strip('"')
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=0)
    tr, va = next(sgkf.split(meta, meta.dx, groups=meta.lesion_id))
    return meta.iloc[tr], meta.iloc[va]


def main():
    torch.manual_seed(0)
    os.makedirs("checkpoints", exist_ok=True)
    tr, va = ham_split()
    if QUICK:
        tr, va = tr.sample(160, random_state=0), va.sample(64, random_state=0)
    print(f"train: {len(tr)} images   valid: {len(va)} images   (split by lesion_id, same as step 3-B)")

    kw = dict(batch_size=BATCH, num_workers=4, persistent_workers=True)
    lt = DataLoader(SegDataset(tr, seg_train_tf), shuffle=True, **kw)
    lv = DataLoader(SegDataset(va, seg_eval_tf), shuffle=False, **kw)

    model = AttentionUNet().to(DEVICE)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bce = nn.BCEWithLogitsLoss()

    best, t0, nb = float("inf"), time.time(), len(lt)
    print(f"\n─── training Attention U-Net: {EPOCHS} epochs × {nb} batches ───", flush=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run, n = 0.0, 0
        for b, (x, m) in enumerate(lt, 1):
            x, m = x.to(DEVICE), m.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = bce(out, m) + dice_loss(out, m)
            loss.backward()
            opt.step()
            run += loss.item() * len(x); n += len(x)
            if b % 50 == 0 or b == nb:
                bar = "█" * int(20 * b / nb) + "·" * (20 - int(20 * b / nb))
                print(f"  epoch {epoch}/{EPOCHS} [{bar}] batch {b:3d}/{nb}  loss {run / n:.3f}  "
                      f"({time.time() - t0:4.0f}s)", flush=True)

        model.eval()
        vl, vn, dices, ious = 0.0, 0, [], []
        with torch.no_grad():
            for x, m in lv:
                x, m = x.to(DEVICE), m.to(DEVICE)
                out = model(x)
                vl += (bce(out, m) + dice_loss(out, m)).item() * len(x); vn += len(x)
                d, i = dice_iou((torch.sigmoid(out) > 0.5).float(), m)
                dices.append(d.cpu()); ious.append(i.cpu())
        vl /= vn
        sched.step(vl)
        dice, iou = torch.cat(dices).mean().item(), torch.cat(ious).mean().item()
        mark = ""
        if vl < best:
            best = vl
            torch.save(model.state_dict(), "checkpoints/step5a_attention_unet.pt")
            mark = "  ← best, saved"
        print(f"epoch {epoch}/{EPOCHS} done | train loss {run / n:.3f} | valid loss {vl:.3f} "
              f"| valid Dice {dice:.3f}  IoU {iou:.3f}{mark}\n", flush=True)

    print(f"finished in {(time.time() - t0) / 60:.1f} min → checkpoints/step5a_attention_unet.pt")
    print("reference: paper reports Dice 0.82 (Attention U-Net) and 0.85 (CARE-MD) on ISIC 2018")


if __name__ == "__main__":
    main()
