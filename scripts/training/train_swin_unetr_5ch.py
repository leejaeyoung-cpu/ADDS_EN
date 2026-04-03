"""
train_swin_unetr_5ch.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5-Channel Swin-UNETR Fine-tuning for KRAS CRC Tumor Detection

데이터:  f:/ADDS/preprocessed/tensors/*.pt  (1,021개 96³ 패치)
모델:    MONAI SwinUNETR (in_channels=5, feature_size=48)
목표:    자기지도 마스크 재건(pretraining) + 이진 종양 탐지 분류 헤드

학습 전략:
  Phase 1 — Self-supervised masked autoencoder (50 epochs)
            => 5채널 입력 → 재건 MSE loss
  Phase 2 — Supervised binary segmentation (100 epochs)
            => 5채널 입력 → Dice+CE loss (종양 mask 라벨 사용)

실행:
    # Phase 1 (self-supervised)
    python train_swin_unetr_5ch.py --phase 1 --epochs 50

    # Phase 2 (supervised, 기존 Phase1 체크포인트 이어서)
    python train_swin_unetr_5ch.py --phase 2 --epochs 100 --resume checkpoints/swin5ch_phase1_best.pth

Author : ADDS Research Team
Date   : 2026-03-17
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Force UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. Dataset
# ═══════════════════════════════════════════════════════════════════

class CT5ChannelDataset(Dataset):
    """
    Load preprocessed 96^3 5-channel CT patches from .pt files.
    Phase 1: returns (image, image) for self-supervised reconstruction
    Phase 2: returns (image, label) — label from segmentation mask .pt if available
    """

    def __init__(
        self,
        split_json: str,
        phase: int = 1,
        aug_prob: float = 0.5,
    ):
        self.phase = phase
        self.aug_prob = aug_prob

        with open(split_json) as f:
            self.entries = json.load(f)

        logger.info(f"Dataset[{Path(split_json).stem}]: {len(self.entries)} patches (phase={phase})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        data = torch.load(entry['path'], weights_only=True)
        image = data['image'].float()  # (5, 96, 96, 96)

        # Augmentation
        image = self._augment(image)

        if self.phase == 1:
            # Self-supervised: masked reconstruction target
            masked, mask = self._random_mask(image, mask_ratio=0.50)
            return masked, image, mask   # input, target, mask

        else:
            # Phase 2: binary segmentation — label from meta or zeros
            label = torch.zeros(1, *image.shape[1:], dtype=torch.float32)
            return image, label

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Light 3D augmentations."""
        import random
        if random.random() < self.aug_prob:
            # Random flip along D/H/W axes (indices 1,2,3 for 4D tensor C,D,H,W)
            for axis in [1, 2, 3]:
                if random.random() < 0.5:
                    x = torch.flip(x, [axis])
        if random.random() < self.aug_prob * 0.5:
            # Gaussian noise
            noise = torch.randn_like(x) * 0.05
            x = x + noise
        return x

    def _random_mask(
        self,
        x: torch.Tensor,
        patch_size: int = 16,
        mask_ratio: float = 0.50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Block-wise random masking for masked autoencoder pretraining.
        Returns: (masked_image, binary_mask)
        """
        C, D, H, W = x.shape
        n_d = D // patch_size
        n_h = H // patch_size
        n_w = W // patch_size
        n_patches = n_d * n_h * n_w

        n_mask = int(n_patches * mask_ratio)
        mask_idx = torch.randperm(n_patches)[:n_mask]

        mask = torch.zeros(n_patches, dtype=torch.bool)
        mask[mask_idx] = True
        mask_3d = mask.reshape(n_d, n_h, n_w)

        masked = x.clone()
        for i in range(n_d):
            for j in range(n_h):
                for k in range(n_w):
                    if mask_3d[i, j, k]:
                        z0, y0, x0 = i * patch_size, j * patch_size, k * patch_size
                        masked[:, z0:z0+patch_size, y0:y0+patch_size, x0:x0+patch_size] = 0.0

        mask_vol = mask_3d.repeat_interleave(patch_size, dim=0) \
                          .repeat_interleave(patch_size, dim=1) \
                          .repeat_interleave(patch_size, dim=2)  # (D, H, W)
        return masked, mask_vol.unsqueeze(0)  # (1, D, H, W)


# ═══════════════════════════════════════════════════════════════════
# 2. Model
# ═══════════════════════════════════════════════════════════════════

def build_model(phase: int, device: torch.device, checkpoint: Optional[str] = None):
    """
    Build 5-channel Swin-UNETR.

    Phase 1: decoder head outputs 5 channels (reconstruction)
    Phase 2: decoder head outputs 1 channel (binary seg logit)
    """
    from monai.networks.nets import SwinUNETR

    out_ch = 5 if phase == 1 else 1

    model = SwinUNETR(
        in_channels=5,
        out_channels=out_ch,
        feature_size=48,
        window_size=7,          # 96 / 2^5 = ~3, window=7 covers it
        use_checkpoint=True,    # Gradient checkpointing saves ~40% VRAM
        spatial_dims=3,
    )

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)

        if phase == 2 and out_ch != 5:
            # Strip final decoder head (out_channels mismatch)
            state = {k: v for k, v in state.items() if 'out' not in k}
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info(f"Phase2 load: {len(missing)} missing, {len(unexpected)} unexpected keys")
        else:
            model.load_state_dict(state, strict=True)
            logger.info(f"Checkpoint loaded: {checkpoint}")

    return model.to(device)


# ═══════════════════════════════════════════════════════════════════
# 3. Loss Functions
# ═══════════════════════════════════════════════════════════════════

class MaskedMSELoss(nn.Module):
    """MSE loss only on masked regions (Phase 1)."""
    def forward(self, pred, target, mask):
        # mask: (B, 1, D, H, W)  1=masked
        diff = (pred - target) ** 2
        masked_loss = (diff * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)
        return masked_loss


class DiceBCELoss(nn.Module):
    """Dice + BCE combination (Phase 2 binary segmentation)."""
    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        # Soft Dice
        pred = torch.sigmoid(logits)
        inter = (pred * targets).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        dice_loss = 1.0 - (2.0 * inter + 1e-5) / (union + 1e-5)
        dice_loss = dice_loss.mean()
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss


# ═══════════════════════════════════════════════════════════════════
# 4. Training Loop
# ═══════════════════════════════════════════════════════════════════

class Swin5CHTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        phase: int,
        checkpoint_dir: str,
        use_amp: bool = True,
        grad_accum: int = 2,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.phase = phase
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp
        self.grad_accum = grad_accum
        self.scaler = GradScaler(enabled=use_amp)
        self.best_metric = float('inf') if phase == 1 else 0.0
        self.history: List[dict] = []

    # ---------------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            if self.phase == 1:
                masked, target, mask = batch
                masked = masked.to(self.device)
                target = target.to(self.device)
                mask   = mask.to(self.device)

                with autocast(enabled=self.use_amp):
                    pred  = self.model(masked)
                    loss  = self.criterion(pred, target, mask)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                with autocast(enabled=self.use_amp):
                    pred = self.model(images)
                    loss = self.criterion(pred, labels)

            loss = loss / self.grad_accum
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum

            if step % 10 == 0:
                logger.info(
                    f"  E{epoch} [{step}/{len(loader)}] "
                    f"loss={loss.item()*self.grad_accum:.4f}"
                )

        return {'loss': total_loss / len(loader), 'epoch': epoch}

    # ---------------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0

        for batch in loader:
            if self.phase == 1:
                masked, target, mask = batch
                masked = masked.to(self.device)
                target = target.to(self.device)
                mask   = mask.to(self.device)
                with autocast(enabled=self.use_amp):
                    pred = self.model(masked)
                    loss = self.criterion(pred, target, mask)
                total_loss += loss.item()
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                with autocast(enabled=self.use_amp):
                    pred = self.model(images)
                    loss = self.criterion(pred, labels)
                dice = self._dice(torch.sigmoid(pred), labels)
                total_loss += loss.item()
                total_dice += dice

        avg_loss = total_loss / max(len(loader), 1)
        avg_dice = total_dice / max(len(loader), 1)

        logger.info(f"  Val E{epoch}: loss={avg_loss:.4f}"
                    + (f" dice={avg_dice:.4f}" if self.phase == 2 else ""))
        return {'loss': avg_loss, 'dice': avg_dice, 'epoch': epoch}

    # ---------------------------------------------------------------
    def _dice(self, pred: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
        pred_bin = (pred > thr).float()
        inter = (pred_bin * target).sum()
        union = pred_bin.sum() + target.sum()
        return (2.0 * inter / (union + 1e-5)).item()

    # ---------------------------------------------------------------
    def save(self, epoch: int, metrics: dict, tag: str = ''):
        fname = f"swin5ch_phase{self.phase}_E{epoch:04d}{tag}.pth"
        path = self.ckpt_dir / fname
        torch.save({
            'epoch': epoch,
            'phase': self.phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, path)
        logger.info(f"  Checkpoint -> {path}")
        return path

    # ---------------------------------------------------------------
    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        scheduler=None,
        early_stop: int = 15,
    ):
        patience = 0
        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            train_m = self.train_epoch(train_loader, epoch)
            val_m   = self.validate(val_loader, epoch)

            if scheduler:
                scheduler.step()

            # Track best
            metric_key = 'loss' if self.phase == 1 else 'dice'
            metric_val = val_m[metric_key]
            better = (metric_val < self.best_metric) if self.phase == 1 else (metric_val > self.best_metric)

            if better:
                self.best_metric = metric_val
                best_path = self.save(epoch, val_m, tag='_best')
                # Also write latest best link
                link = self.ckpt_dir / f'swin5ch_phase{self.phase}_best.pth'
                import shutil
                shutil.copy2(best_path, link)
                patience = 0
            else:
                patience += 1

            elapsed = time.time() - t0
            SEP = '-' * 56
            print(f"\n{SEP}")
            print(f"  Phase{self.phase} Epoch {epoch}/{num_epochs}  ({elapsed:.1f}s)")
            print(f"  Train loss : {train_m['loss']:.4f}")
            print(f"  Val   loss : {val_m['loss']:.4f}")
            if self.phase == 2:
                print(f"  Val   dice : {val_m['dice']:.4f}  (best={self.best_metric:.4f})")
            print(f"  Patience   : {patience}/{early_stop}")
            print(SEP)

            self.history.append({'epoch': epoch, **train_m, 'val_loss': val_m['loss'],
                                 'val_dice': val_m.get('dice', 0)})

            # Save every 10 epochs
            if epoch % 10 == 0:
                self.save(epoch, val_m)

            if patience >= early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        print(f"\n  Training Phase {self.phase} complete.")
        print(f"  Best {'loss' if self.phase == 1 else 'Dice'}: {self.best_metric:.4f}")
        print(f"  Checkpoints: {self.ckpt_dir}")

        # Save history JSON
        hist_path = self.ckpt_dir / f'history_phase{self.phase}.json'
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"  History: {hist_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. Entry Point
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='5-ch Swin-UNETR Training')
    p.add_argument('--preprocessed', default=r'f:\ADDS\preprocessed',
                   help='Preprocessed root (contains splits/ and tensors/)')
    p.add_argument('--checkpoint-dir', default=r'f:\ADDS\checkpoints',
                   help='Output checkpoint directory')
    p.add_argument('--phase', type=int, default=1, choices=[1, 2],
                   help='1=self-supervised (MAE), 2=supervised segmentation')
    p.add_argument('--epochs', type=int, default=50,
                   help='Training epochs (default: 50 for phase1, 100 for phase2)')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size (default: 1, increase if VRAM allows)')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--workers', type=int, default=0,
                   help='DataLoader workers (0=main thread, safe on Windows)')
    p.add_argument('--no-amp', action='store_true',
                   help='Disable automatic mixed precision')
    p.add_argument('--early-stop', type=int, default=15,
                   help='Early stopping patience (epochs)')
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device  : {device}')
    if torch.cuda.is_available():
        print(f'  GPU     : {torch.cuda.get_device_name(0)}')
        print(f'  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    # ── Datasets ─────────────────────────────────────────────────
    splits_dir = Path(args.preprocessed) / 'splits'
    train_ds = CT5ChannelDataset(str(splits_dir / 'train.json'), phase=args.phase, aug_prob=0.6)
    val_ds   = CT5ChannelDataset(str(splits_dir / 'val.json'),   phase=args.phase, aug_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())

    print(f'  Train   : {len(train_ds)} patches')
    print(f'  Val     : {len(val_ds)} patches')

    # ── Model ────────────────────────────────────────────────────
    model = build_model(args.phase, device, checkpoint=args.resume)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model   : SwinUNETR-5ch ({n_params/1e6:.1f}M params)')

    # ── Optimizer / Criterion ────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    if args.phase == 1:
        criterion = MaskedMSELoss()
    else:
        criterion = DiceBCELoss(dice_weight=0.6)

    # ── Trainer ──────────────────────────────────────────────────
    trainer = Swin5CHTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        phase=args.phase,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
        grad_accum=2,
    )

    print(f'\n  Phase   : {args.phase} ({"Self-supervised MAE" if args.phase==1 else "Supervised Segmentation"})')
    print(f'  Epochs  : {args.epochs}')
    print(f'  LR      : {args.lr}')
    print(f'  AMP     : {not args.no_amp}')

    # ── Run ──────────────────────────────────────────────────────
    trainer.run(
        train_loader, val_loader,
        num_epochs=args.epochs,
        scheduler=scheduler,
        early_stop=args.early_stop,
    )


if __name__ == '__main__':
    main()
