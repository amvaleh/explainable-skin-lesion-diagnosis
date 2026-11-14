#!/bin/bash
# اجرای کامل قدم ۵: آموزش شبکه‌ی مرز خال ← بریدن عکس‌های Derm7pt ← مقایسه‌ی عکس کامل و بریده
set -e
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
mkdir -p logs
echo "════════ STEP 5-A: train Attention U-Net on 10,000 HAM10000 lesion masks ════════"
.venv/bin/python step5a_train_segmentation.py 2>&1 | tee logs/step5a.log
echo; echo "════════ STEP 5-B: find the lesion in every Derm7pt image and crop around it ════════"
.venv/bin/python step5b_crop_derm7pt.py 2>&1 | tee logs/step5b.log
echo; echo "════════ STEP 5-C: eye on FULL image vs eye on CROPPED lesion (3 seeds each) ════════"
.venv/bin/python step5c_compare_crop.py 2>&1 | tee logs/step5c.log
echo; echo "ALL DONE"
