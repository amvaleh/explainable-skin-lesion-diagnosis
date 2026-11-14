#!/bin/bash
# اجرای کامل قدم ۳: بررسی تقلب ← تنظیم دقیق ۱ روی HAM10000 ← مقایسه‌ی دو چشم
set -e
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
mkdir -p logs
echo "════════ STEP 3-A: leakage check ════════"
.venv/bin/python step3a_leakage_check.py 2>&1 | tee logs/step3a.log
echo; echo "════════ STEP 3-B: fine-tune #1 on 10,000 HAM10000 lesions ════════"
.venv/bin/python step3b_ham_pretrain.py 2>&1 | tee logs/step3b.log
echo; echo "════════ STEP 3-C: fine-tune #2 + fair comparison (old eye vs new eye, 3 seeds each) ════════"
.venv/bin/python step3c_compare_eyes.py 2>&1 | tee logs/step3c.log
echo; echo "ALL DONE"
