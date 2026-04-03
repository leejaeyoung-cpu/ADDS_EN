"""
nnU-Net 5-Fold Ensemble Training for Dataset011_ColonMasked

이 스크립트는 fold 1-4를 순차적으로 학습합니다 (fold 0은 이미 완료).
각 fold는 약 3-8시간 소요됩니다.

환경변수:
  nnUNet_raw = C:\nnUNet_data\nnUNet_raw (하지만 실제 data는 F:\ADDS 아래)
  nnUNet_preprocessed = f:/ADDS/nnUNet_preprocessed
  nnUNet_results = F:\ADDS\nnUNet_results
"""

import subprocess
import sys
import os
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set environment
os.environ["nnUNet_raw"] = "C:/nnUNet_data/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "F:/ADDS/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "F:/ADDS/nnUNet_results"

DATASET_ID = 11
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
CONFIG = "3d_fullres"

# Check which folds are done
results_dir = Path(os.environ["nnUNet_results"])
base = results_dir / f"Dataset0{DATASET_ID}_ColonMasked" / f"{TRAINER}__{PLANS}__{CONFIG}"

folds_done = []
folds_todo = []

for fold in range(5):
    fold_dir = base / f"fold_{fold}"
    ckpt = fold_dir / "checkpoint_final.pth"
    if ckpt.exists():
        folds_done.append(fold)
        logger.info(f"  fold_{fold}: DONE (checkpoint_final.pth exists)")
    else:
        folds_todo.append(fold)
        logger.info(f"  fold_{fold}: TODO")

if not folds_todo:
    logger.info("All 5 folds are already trained!")
    sys.exit(0)

logger.info(f"\nTraining folds: {folds_todo}")
logger.info(f"  Dataset: {DATASET_ID} (ColonMasked)")
logger.info(f"  Config: {CONFIG}")
logger.info(f"  Estimated time: ~{len(folds_todo) * 4}h total")

total_start = time.time()

for fold in folds_todo:
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting fold {fold}")
    logger.info(f"{'='*60}")

    t0 = time.time()

    cmd = [
        sys.executable, "-m", "nnunetv2.run_training",
        str(DATASET_ID), CONFIG, str(fold),
        "--npz",
    ]

    logger.info(f"  Command: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        env={**os.environ},
        capture_output=False,
        text=True,
    )

    elapsed = (time.time() - t0) / 3600
    if proc.returncode == 0:
        logger.info(f"  fold {fold} completed in {elapsed:.1f}h")
    else:
        logger.error(f"  fold {fold} FAILED (return code {proc.returncode})")
        logger.error(f"  Stopping sequential training.")
        break

total_h = (time.time() - total_start) / 3600
logger.info(f"\nTotal time: {total_h:.1f}h")

# Summary
for fold in range(5):
    ckpt = base / f"fold_{fold}" / "checkpoint_final.pth"
    status = "DONE" if ckpt.exists() else "MISSING"
    logger.info(f"  fold_{fold}: {status}")
