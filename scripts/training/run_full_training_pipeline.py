"""
run_full_training_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 (자기지도 MAE) → Phase 2 (지도학습 분할) 자동 연속 실행

Phase 1이 이미 실행 중이라면:
  - Phase 1 best 체크포인트가 생성될 때까지 대기
  - Phase 1 프로세스 종료 감지 후 Phase 2 자동 시작

실행:
    python run_full_training_pipeline.py

Author : ADDS Research Team
Date   : 2026-03-17
"""
import os
import sys
import time
import subprocess
import logging
from pathlib import Path

# Force UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

ROOT          = Path(__file__).parent
CKPT_DIR      = ROOT / 'checkpoints'
PHASE1_BEST   = CKPT_DIR / 'swin5ch_phase1_best.pth'
PHASE2_BEST   = CKPT_DIR / 'swin5ch_phase2_best.pth'
PREPROCESSED  = ROOT / 'preprocessed'
TRAIN_SCRIPT  = ROOT / 'train_swin_unetr_5ch.py'

SEP = '=' * 64

def wait_for_phase1(poll_sec: int = 60, timeout_hrs: float = 12.0):
    """Phase 1 best checkpoint가 생성될 때까지 대기."""
    deadline = time.time() + timeout_hrs * 3600
    print(f'\n{SEP}')
    print(f'  Waiting for Phase 1 to complete...')
    print(f'  Polling: {PHASE1_BEST}')
    print(f'  Timeout: {timeout_hrs}h')
    print(SEP)

    last_mtime = PHASE1_BEST.stat().st_mtime if PHASE1_BEST.exists() else 0

    while time.time() < deadline:
        if PHASE1_BEST.exists():
            cur_mtime = PHASE1_BEST.stat().st_mtime
            if cur_mtime > last_mtime:
                last_mtime = cur_mtime
                logger.info(f"  Phase1 checkpoint updated: {time.strftime('%H:%M:%S')}")

        # Check for history file — Phase1 complete indicator
        hist_path = CKPT_DIR / 'history_phase1.json'
        if hist_path.exists():
            import json
            try:
                with open(hist_path) as f:
                    history = json.load(f)
                if history:
                    last_epoch = history[-1].get('epoch', 0)
                    last_val   = history[-1].get('val_loss', 999)
                    logger.info(f"  Phase1 history: epoch={last_epoch}, val_loss={last_val:.4f}")
                    # If last epoch >= target (50) or early stopped
                    if last_epoch >= 50 or (len(history) > 1 and last_epoch == history[-2].get('epoch', 0)):
                        logger.info(f"  Phase1 complete at epoch {last_epoch}")
                        return last_epoch, last_val
            except Exception:
                pass

        # Also detect completion via lock file or process exit
        # Simple approach: if checkpoint exists and training script not in psutil
        try:
            import psutil
            running = any(
                'train_swin_unetr_5ch' in ' '.join(p.cmdline())
                for p in psutil.process_iter(['cmdline'])
                if p.pid != os.getpid()
            )
            if not running and PHASE1_BEST.exists():
                logger.info("  Phase1 training process ended.")
                import json
                try:
                    with open(CKPT_DIR / 'history_phase1.json') as f:
                        history = json.load(f)
                    return history[-1].get('epoch', '?'), history[-1].get('val_loss', '?')
                except Exception:
                    return '?', '?'
        except ImportError:
            pass

        time.sleep(poll_sec)

    raise TimeoutError(f"Phase 1 did not complete within {timeout_hrs} hours")


def run_phase2(phase1_checkpoint: str, epochs: int = 100):
    """Phase 2 지도학습 실행."""
    print(f'\n{SEP}')
    print(f'  Starting Phase 2 — Supervised Segmentation')
    print(f'  Resume from: {phase1_checkpoint}')
    print(f'  Epochs: {epochs}')
    print(SEP + '\n')

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--phase', '2',
        '--epochs', str(epochs),
        '--batch-size', '1',
        '--workers', '0',
        '--lr', '3e-5',          # Lower LR for fine-tuning
        '--resume', phase1_checkpoint,
        '--early-stop', '20',
        '--preprocessed', str(PREPROCESSED),
        '--checkpoint-dir', str(CKPT_DIR),
    ]

    logger.info(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def main():
    print(f'\n{SEP}')
    print(f'  ADDS CT Swin-UNETR — Full Training Pipeline')
    print(f'  Phase 1: Self-supervised MAE  (50 epochs)')
    print(f'  Phase 2: Supervised Seg.       (100 epochs)')
    print(SEP)

    # ── Phase 1 완료 대기 ─────────────────────────────────────────
    if PHASE1_BEST.exists():
        import json
        hist = CKPT_DIR / 'history_phase1.json'
        if hist.exists():
            with open(hist) as f:
                h = json.load(f)
            last_epoch = h[-1].get('epoch', 0) if h else 0
            if last_epoch >= 50:
                logger.info(f"  Phase1 already done (epoch={last_epoch}). Skipping to Phase2.")
                ep, vl = last_epoch, h[-1].get('val_loss', '?')
            else:
                logger.info(f"  Phase1 in progress (epoch={last_epoch}/50). Waiting...")
                ep, vl = wait_for_phase1()
        else:
            logger.info("  Phase1 checkpoint exists — assuming complete.")
            ep, vl = '?', '?'
    else:
        logger.info("  Phase1 checkpoint not found yet. Waiting for Phase1 to produce it...")
        ep, vl = wait_for_phase1()

    print(f'\n  Phase 1 complete: epoch={ep}, val_loss={vl}')
    print(f'  Checkpoint: {PHASE1_BEST}')

    # ── Phase 2 실행 ──────────────────────────────────────────────
    rc = run_phase2(str(PHASE1_BEST), epochs=100)
    if rc == 0:
        print(f'\n  Phase 2 complete!')
        print(f'  Best model: {PHASE2_BEST}')
        print(f'\n  To evaluate:')
        print(f'    python evaluate_swin_unetr.py --checkpoint {PHASE2_BEST}')
    else:
        print(f'\n  Phase 2 exited with code {rc}')

    print(f'\n{SEP}\n')


if __name__ == '__main__':
    main()
