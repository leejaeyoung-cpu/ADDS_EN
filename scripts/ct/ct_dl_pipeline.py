"""
CT Advanced Preprocessing + Deep Learning Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
End-to-end pipeline:
  Phase 1: Apply advanced preprocessing to real CT data
  Phase 2: Train 5-channel Swin-UNETR tumor detector
  Phase 3: Evaluate and iterate

Author: ADDS Research Team
Date  : 2026-03-17
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import gc
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
import json
import time

# MONAI imports
import monai
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, ToTensord,
    EnsureTyped, AsDiscrete,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.inferers import sliding_window_inference

# Local imports
from src.medical_imaging.ct_advanced_preprocessing import (
    AdvancedCTPreprocessor, WINDOW_PRESETS
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Data
    'ct_path': 'F:/ADDS/nnUNet_data/nnUNet_raw/Dataset100_CRC_CT/imagesTr/CRC_CT_001_0000.nii.gz',
    'label_path': 'F:/ADDS/nnUNet_data/nnUNet_raw/Dataset100_CRC_CT/labelsTr/CRC_CT_001.nii.gz',
    'output_dir': 'F:/ADDS/outputs/ct_dl_pipeline',

    # Preprocessing
    'skip_heavy_stages': True,  # Skip N4/NLM on normalized data

    # Tumor label mapping: which organ labels are "tumor-relevant"
    # organ_13 = colon (largest organ, 0.18%), organ_4/5 = kidney-like
    'tumor_labels': [13],       # colon as primary target
    'context_labels': [4, 5, 9, 11, 14, 15, 20],  # surrounding organs

    # Patch extraction
    'patch_size': (64, 64, 64),
    'num_patches_per_volume': 200,  # total patches to extract
    'fg_ratio': 0.5,               # 50% patches contain foreground

    # Multi-window channels (applied on normalized data)
    'n_channels': 5,

    # Model
    'feature_size': 24,
    'in_channels': 5,
    'out_channels': 2,

    # Training
    'epochs': 100,
    'batch_size': 1,
    'accum_steps': 4,  # gradient accumulation
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 20,

    # Targets
    'target_dice': 0.60,
}


# ═══════════════════════════════════════════════════════════════
# Phase 1: Preprocessing + Data Preparation
# ═══════════════════════════════════════════════════════════════

class CTDataPreparer:
    """Prepare CT data for DL training."""

    def __init__(self, config: dict):
        self.cfg = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Load CT volume and segmentation labels."""
        logger.info("Loading CT volume...")
        ct_img = sitk.ReadImage(self.cfg['ct_path'])
        ct_vol = sitk.GetArrayFromImage(ct_img).astype(np.float32)
        spacing = tuple(reversed(ct_img.GetSpacing()))

        logger.info("Loading labels...")
        lbl_img = sitk.ReadImage(self.cfg['label_path'])
        lbl_vol = sitk.GetArrayFromImage(lbl_img).astype(np.float32)

        logger.info(f"  CT shape: {ct_vol.shape}, range: [{ct_vol.min():.3f}, {ct_vol.max():.3f}]")
        logger.info(f"  Labels shape: {lbl_vol.shape}, unique: {len(np.unique(lbl_vol))}")

        return ct_vol, lbl_vol, spacing

    def create_binary_mask(self, labels: np.ndarray) -> np.ndarray:
        """Convert multi-class labels to binary tumor mask."""
        tumor_mask = np.zeros_like(labels, dtype=np.uint8)

        # Primary tumor labels
        for lbl in self.cfg['tumor_labels']:
            count = int((labels == lbl).sum())
            tumor_mask[labels == lbl] = 1
            logger.info(f"  Tumor label {lbl}: {count:,} voxels")

        total_fg = int(tumor_mask.sum())
        logger.info(f"  Total tumor voxels: {total_fg:,} ({total_fg/tumor_mask.size*100:.4f}%)")

        return tumor_mask

    def apply_multi_window(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply 5-channel multi-window to normalized [0,1] volume.
        Since data is already normalized, we simulate different contrast
        ranges by gamma/sigmoid transforms instead of HU windows.
        """
        channels = []

        # Ch 1: Identity (original soft tissue window)
        channels.append(volume.copy())

        # Ch 2: Low-contrast enhance (liver/lesion — gamma < 1)
        channels.append(np.power(np.clip(volume, 0, 1), 0.5).astype(np.float32))

        # Ch 3: High-contrast enhance (colon — gamma > 1)
        channels.append(np.power(np.clip(volume, 0, 1), 2.0).astype(np.float32))

        # Ch 4: Edge-enhanced (Laplacian-like)
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(volume, sigma=1.0)
        edge = np.clip(volume - blurred + 0.5, 0, 1).astype(np.float32)
        channels.append(edge)

        # Ch 5: Local contrast (CLAHE-like, per-slice)
        import cv2
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_vol = np.zeros_like(volume)
        for z in range(volume.shape[0]):
            slc_u8 = (volume[z] * 255).astype(np.uint8)
            clahe_vol[z] = clahe.apply(slc_u8).astype(np.float32) / 255.0
        channels.append(clahe_vol)

        multi = np.stack(channels, axis=0)  # (5, D, H, W)
        logger.info(f"  Multi-window tensor shape: {multi.shape}")
        return multi

    def extract_patches(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
    ) -> list:
        """
        Extract balanced patches for training.

        Args:
            volume: (C, D, H, W) multi-channel volume
            mask: (D, H, W) binary mask

        Returns:
            list of {'image': (C,pD,pH,pW), 'label': (pD,pH,pW)} dicts
        """
        C, D, H, W = volume.shape
        pD, pH, pW = self.cfg['patch_size']
        n_total = self.cfg['num_patches_per_volume']
        n_fg = int(n_total * self.cfg['fg_ratio'])
        n_bg = n_total - n_fg

        patches = []

        # Foreground-centered patches
        fg_coords = np.argwhere(mask > 0)
        logger.info(f"  Extracting {n_fg} foreground + {n_bg} background patches...")

        if len(fg_coords) > 0:
            for _ in range(n_fg):
                idx = np.random.randint(len(fg_coords))
                cz, cy, cx = fg_coords[idx]

                z0 = max(0, min(cz - pD // 2, D - pD))
                y0 = max(0, min(cy - pH // 2, H - pH))
                x0 = max(0, min(cx - pW // 2, W - pW))

                img_patch = volume[:, z0:z0+pD, y0:y0+pH, x0:x0+pW].copy()
                lbl_patch = mask[z0:z0+pD, y0:y0+pH, x0:x0+pW].copy()

                patches.append({
                    'image': img_patch.astype(np.float32),
                    'label': lbl_patch.astype(np.int64),
                })

        # Background patches
        for _ in range(n_bg):
            z0 = np.random.randint(0, max(1, D - pD))
            y0 = np.random.randint(0, max(1, H - pH))
            x0 = np.random.randint(0, max(1, W - pW))

            img_patch = volume[:, z0:z0+pD, y0:y0+pH, x0:x0+pW].copy()
            lbl_patch = mask[z0:z0+pD, y0:y0+pH, x0:x0+pW].copy()

            patches.append({
                'image': img_patch.astype(np.float32),
                'label': lbl_patch.astype(np.int64),
            })

        np.random.shuffle(patches)
        logger.info(f"  Total patches: {len(patches)}")

        # Validate
        fg_patches = sum(1 for p in patches if p['label'].sum() > 0)
        logger.info(f"  Patches with foreground: {fg_patches}/{len(patches)}")

        return patches

    def prepare(self) -> Tuple[list, list, dict]:
        """Full data preparation pipeline."""
        logger.info("=" * 60)
        logger.info("  PHASE 1: Data Preparation")
        logger.info("=" * 60)
        t0 = time.time()

        # Load
        ct_vol, lbl_vol, spacing = self.load_data()

        # Binary mask
        logger.info("Creating binary tumor mask...")
        tumor_mask = self.create_binary_mask(lbl_vol)

        # Multi-window
        logger.info("Applying multi-window transforms...")
        multi_vol = self.apply_multi_window(ct_vol)

        # Body isolation (simple threshold on normalized data)
        from src.medical_imaging.ct_advanced_preprocessing import AdvancedCTPreprocessor
        logger.info("Applying body isolation...")
        body_mask = (ct_vol > 0.05).astype(np.uint8)  # simple threshold for [0,1] data

        # QC stats
        body_pct = body_mask.sum() / body_mask.size * 100
        tumor_pct = tumor_mask.sum() / tumor_mask.size * 100
        stats = {
            'volume_shape': list(ct_vol.shape),
            'body_coverage_pct': round(body_pct, 2),
            'tumor_coverage_pct': round(tumor_pct, 4),
            'multi_window_shape': list(multi_vol.shape),
        }
        logger.info(f"  Body coverage: {body_pct:.1f}%")
        logger.info(f"  Tumor coverage: {tumor_pct:.4f}%")

        # Extract patches
        logger.info("Extracting patches...")
        patches = self.extract_patches(multi_vol, tumor_mask)

        # Train/Val split (80/20)
        split_idx = int(len(patches) * 0.8)
        train_patches = patches[:split_idx]
        val_patches = patches[split_idx:]

        logger.info(f"  Train: {len(train_patches)}, Val: {len(val_patches)}")

        elapsed = time.time() - t0
        logger.info(f"  Phase 1 complete in {elapsed:.1f}s")
        stats['prep_time_s'] = round(elapsed, 1)

        return train_patches, val_patches, stats


# ═══════════════════════════════════════════════════════════════
# Phase 2: Deep Learning Training
# ═══════════════════════════════════════════════════════════════

class PatchDataset(torch.utils.data.Dataset):
    """Simple patch dataset from pre-extracted patches."""

    def __init__(self, patches: list, augment: bool = False):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx]
        image = torch.from_numpy(p['image'])   # (C, D, H, W)
        label = torch.from_numpy(p['label']).long()  # (D, H, W)

        if self.augment:
            # Random flip (each axis, 50%)
            for axis in [1, 2, 3]:  # D, H, W
                if torch.rand(1).item() > 0.5:
                    image = torch.flip(image, [axis])
                    label = torch.flip(label, [axis - 1])

            # Random intensity shift
            if torch.rand(1).item() > 0.5:
                shift = (torch.rand(1).item() - 0.5) * 0.2
                image = image + shift
                image = torch.clamp(image, 0, 1)

            # Random Gaussian noise
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(image) * 0.02
                image = image + noise
                image = torch.clamp(image, 0, 1)

        return image, label


class SwinUNETRTrainer:
    """Train 5-channel Swin-UNETR for tumor segmentation."""

    def __init__(self, config: dict):
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Device: {self.device}")

    def create_model(self) -> nn.Module:
        """Create Swin-UNETR model."""
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        model = SwinUNETR(
            in_channels=self.cfg['in_channels'],
            out_channels=self.cfg['out_channels'],
            feature_size=self.cfg['feature_size'],
            use_checkpoint=True,
            spatial_dims=3,
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Model: Swin-UNETR")
        logger.info(f"  Total params: {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,}")
        logger.info(f"  Input channels: {self.cfg['in_channels']}")
        logger.info(f"  Output classes: {self.cfg['out_channels']}")

        return model.to(self.device)

    def train(
        self,
        train_patches: list,
        val_patches: list,
    ) -> Dict:
        """
        Full training loop with evaluation.

        Returns:
            results: dict with best_dice, loss_history, etc.
        """
        logger.info("=" * 60)
        logger.info("  PHASE 2: Deep Learning Training")
        logger.info("=" * 60)

        # Create model
        model = self.create_model()

        # Datasets
        train_ds = PatchDataset(train_patches, augment=True)
        val_ds = PatchDataset(val_patches, augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg['batch_size'],
            shuffle=True, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg['batch_size'],
            shuffle=False, num_workers=0, pin_memory=True,
        )

        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")

        # Loss, optimizer, scheduler
        loss_fn = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay'],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg['epochs'], eta_min=1e-6,
        )

        scaler = GradScaler('cuda')
        accum_steps = self.cfg.get('accum_steps', 4)

        # Metrics
        dice_metric = DiceMetric(include_background=False, reduction='mean')
        post_pred = AsDiscrete(argmax=True, to_onehot=self.cfg['out_channels'])
        post_label = AsDiscrete(to_onehot=self.cfg['out_channels'])

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'lr': [],
        }

        best_dice = 0.0
        best_epoch = 0
        patience_counter = 0

        t_start = time.time()

        for epoch in range(1, self.cfg['epochs'] + 1):
            t_epoch = time.time()

            # ── Train ──
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            optimizer.zero_grad()
            for step_i, (batch_images, batch_labels) in enumerate(train_loader):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)  # (B,1,D,H,W)

                with autocast('cuda'):
                    outputs = model(batch_images)
                    loss = loss_fn(outputs, batch_labels) / accum_steps

                scaler.scale(loss).backward()

                if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accum_steps
                n_batches += 1

                # Free memory
                del outputs, loss
                torch.cuda.empty_cache()

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # ── Validate ──
            model.eval()
            val_loss_sum = 0.0
            val_n = 0

            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.to(self.device).unsqueeze(1)

                    with autocast('cuda'):
                        outputs = model(batch_images)
                        v_loss = loss_fn(outputs, batch_labels)

                    val_loss_sum += v_loss.item()
                    val_n += 1

                    # Dice computation
                    outputs_onehot = [post_pred(o) for o in monai.data.decollate_batch(outputs)]
                    labels_onehot = [post_label(l) for l in monai.data.decollate_batch(batch_labels)]
                    dice_metric(outputs_onehot, labels_onehot)

                    del outputs, v_loss
                    torch.cuda.empty_cache()

            avg_val_loss = val_loss_sum / max(val_n, 1)
            val_dice = float(dice_metric.aggregate().item())
            dice_metric.reset()

            # Step scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            history['train_loss'].append(round(avg_train_loss, 4))
            history['val_loss'].append(round(avg_val_loss, 4))
            history['val_dice'].append(round(val_dice, 4))
            history['lr'].append(round(current_lr, 8))

            epoch_time = time.time() - t_epoch

            # Best model
            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), self.output_dir / 'best_model.pth')
                marker = ' ** BEST **'
            else:
                patience_counter += 1
                marker = ''

            # Log every 5 epochs or if best
            if epoch % 5 == 0 or epoch == 1 or marker:
                logger.info(
                    f"  Epoch {epoch:3d}/{self.cfg['epochs']} | "
                    f"TrLoss={avg_train_loss:.4f} | "
                    f"VaLoss={avg_val_loss:.4f} | "
                    f"Dice={val_dice:.4f} | "
                    f"LR={current_lr:.6f} | "
                    f"{epoch_time:.1f}s{marker}"
                )

            # Early stopping
            if patience_counter >= self.cfg['early_stopping_patience']:
                logger.info(f"  Early stopping at epoch {epoch} (patience={self.cfg['early_stopping_patience']})")
                break

        total_time = time.time() - t_start

        logger.info("=" * 60)
        logger.info(f"  Training complete in {total_time:.0f}s")
        logger.info(f"  Best Dice: {best_dice:.4f} (epoch {best_epoch})")
        logger.info("=" * 60)

        # Final evaluation
        eval_results = self._final_evaluation(model, val_loader, post_pred, post_label)

        results = {
            'best_dice': round(best_dice, 4),
            'best_epoch': best_epoch,
            'total_epochs': epoch,
            'total_time_s': round(total_time, 1),
            'history': history,
            **eval_results,
        }

        # Save results
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _final_evaluation(
        self, model, val_loader, post_pred, post_label
    ) -> Dict:
        """Compute final detailed metrics on validation set."""
        model.eval()

        dice_metric = DiceMetric(include_background=False, reduction='mean')

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(self.device)

                with autocast('cuda'):
                    outputs = model(batch_images)

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = batch_labels.numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

                # Dice
                outputs_onehot = [post_pred(o) for o in monai.data.decollate_batch(outputs)]
                labels_onehot = [post_label(l.unsqueeze(0).to(self.device)) for l in monai.data.decollate_batch(batch_labels)]
                dice_metric(outputs_onehot, labels_onehot)

        final_dice = float(dice_metric.aggregate().item())
        dice_metric.reset()

        # Precision, Recall, F1
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        tp = int(((all_preds == 1) & (all_labels == 1)).sum())
        fp = int(((all_preds == 1) & (all_labels == 0)).sum())
        fn = int(((all_preds == 0) & (all_labels == 1)).sum())
        tn = int(((all_preds == 0) & (all_labels == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        logger.info("  Final Evaluation:")
        logger.info(f"    Dice      : {final_dice:.4f}")
        logger.info(f"    Precision : {precision:.4f}")
        logger.info(f"    Recall    : {recall:.4f}")
        logger.info(f"    F1        : {f1:.4f}")
        logger.info(f"    TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}")

        return {
            'final_dice': round(final_dice, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        }


# ═══════════════════════════════════════════════════════════════
# Phase 3: Iterative Training
# ═══════════════════════════════════════════════════════════════

def run_pipeline(config: dict = None) -> Dict:
    """Run the full pipeline with iterative improvement."""
    config = config or CONFIG

    logger.info("*" * 60)
    logger.info("  CT ADVANCED PREPROCESSING + DL PIPELINE")
    logger.info("*" * 60)

    all_results = []

    # Phase 1: Data preparation
    preparer = CTDataPreparer(config)
    train_patches, val_patches, prep_stats = preparer.prepare()

    # Phase 2 + 3: Train and iterate
    round_configs = [
        # Round 1: Baseline
        {'round': 1, 'label': 'Baseline', 'epochs': 50,
         'lr': 1e-4, 'batch_size': 1, 'num_patches_per_volume': 200},

        # Round 2: More aggressive augmentation + longer training
        {'round': 2, 'label': 'Extended', 'epochs': 100,
         'lr': 5e-5, 'batch_size': 1, 'num_patches_per_volume': 400},

        # Round 3: Fine-tune with lower LR
        {'round': 3, 'label': 'Fine-tune', 'epochs': 100,
         'lr': 1e-5, 'batch_size': 1, 'num_patches_per_volume': 400},
    ]

    best_overall_dice = 0.0

    for rcfg in round_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"  ROUND {rcfg['round']}: {rcfg['label']}")
        logger.info(f"{'='*60}")

        # Update config
        round_config = {**config, **rcfg}

        # Re-extract patches if count changed
        if rcfg['num_patches_per_volume'] != config.get('num_patches_per_volume', 200):
            config_copy = {**config, 'num_patches_per_volume': rcfg['num_patches_per_volume']}
            preparer_r = CTDataPreparer(config_copy)
            ct_vol, lbl_vol, spacing = preparer_r.load_data()
            tumor_mask = preparer_r.create_binary_mask(lbl_vol)
            multi_vol = preparer_r.apply_multi_window(ct_vol)
            all_patches = preparer_r.extract_patches(multi_vol, tumor_mask)
            split = int(len(all_patches) * 0.8)
            train_patches = all_patches[:split]
            val_patches = all_patches[split:]

        trainer = SwinUNETRTrainer(round_config)
        results = trainer.train(train_patches, val_patches)
        results['round'] = rcfg['round']
        results['label'] = rcfg['label']
        all_results.append(results)

        best_dice = results['best_dice']
        if best_dice > best_overall_dice:
            best_overall_dice = best_dice

        # Check target
        if best_dice >= config['target_dice']:
            logger.info(f"  TARGET REACHED: Dice={best_dice:.4f} >= {config['target_dice']}")
            break
        else:
            logger.info(f"  Dice={best_dice:.4f} < target {config['target_dice']}, continuing...")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE SUMMARY")
    logger.info("=" * 60)
    for r in all_results:
        logger.info(f"  Round {r['round']} ({r['label']}): "
                     f"Dice={r['best_dice']:.4f}, "
                     f"F1={r.get('f1', 0):.4f}, "
                     f"Epochs={r['total_epochs']}, "
                     f"Time={r['total_time_s']}s")
    logger.info(f"  Best overall Dice: {best_overall_dice:.4f}")

    if best_overall_dice >= config['target_dice']:
        logger.info(f"  [OK] TARGET MET")
    else:
        logger.info(f"  [!!] TARGET NOT MET (need {config['target_dice']})")

    logger.info("=" * 60)

    # Save summary
    summary = {
        'prep_stats': prep_stats,
        'rounds': all_results,
        'best_overall_dice': round(best_overall_dice, 4),
        'target_met': best_overall_dice >= config['target_dice'],
    }
    with open(Path(config['output_dir']) / 'pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    # Run Round 1 only first to validate
    quick_config = {**CONFIG, 'epochs': 30, 'num_patches_per_volume': 100}
    run_pipeline(quick_config)
