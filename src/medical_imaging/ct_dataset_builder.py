"""
CT Dataset Builder — Swin-UNETR / nnU-Net Training Dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전처리 결과(preprocessed.nii.gz, multi_window_5ch.npy)를
96³ 패치로 분할하여 PyTorch .pt 텐서로 저장합니다.

출력:
  {output_root}/tensors/
    ├── {series_id}_patch_{i:04d}.pt   → dict{'image':(5,96,96,96), 'meta':dict}
  {output_root}/manifest.json          → 전체 데이터셋 메타데이터
  {output_root}/splits/
    ├── train.json
    └── val.json

Author : ADDS Research Team
Date   : 2026-03-17
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Patch Extractor
# ═══════════════════════════════════════════════════════════════════

class CTDatasetBuilder:
    """
    Converts batch-preprocessed CT results into training-ready patches.

    Patch strategy:
      - Slide a 96×96×96 window over the 5-ch isotropic volume
      - Stride = patch_size // 2   (50% overlap for boundary coverage)
      - Patches with body_mask coverage < min_body_ratio are discarded
      - Z-score normalize each patch independently

    Usage:
        builder = CTDatasetBuilder(output_root='f:/ADDS/preprocessed')
        builder.build_all(qc_pass_only=True, patch_size=96, val_ratio=0.2)
    """

    def __init__(
        self,
        output_root: str,
        patch_size: int = 96,
        stride_ratio: float = 0.5,
        min_body_ratio: float = 0.30,
    ):
        self.output_root = Path(output_root)
        self.tensor_dir = self.output_root / 'tensors'
        self.tensor_dir.mkdir(parents=True, exist_ok=True)

        (self.output_root / 'splits').mkdir(exist_ok=True)

        self.patch_size = patch_size
        self.stride = max(1, int(patch_size * stride_ratio))
        self.min_body_ratio = min_body_ratio

    # ---------------------------------------------------------------
    def build_all(
        self,
        qc_pass_only: bool = True,
        val_ratio: float = 0.2,
        seed: int = 42,
        overwrite: bool = False,
    ) -> Dict:
        """
        Scan output_root for qc_report.json → build patches → write manifest.
        """
        random.seed(seed)

        # Collect all processed series
        qc_files = sorted(self.output_root.rglob('qc_report.json'))
        logger.info(f"Found {len(qc_files)} processed series")

        manifest_entries = []
        total_patches = 0

        for qc_file in qc_files:
            series_dir = qc_file.parent

            with open(qc_file) as f:
                qc = json.load(f)

            verdict = qc.get('qc_verdict', 'UNKNOWN')
            series_id = qc.get('series_id', series_dir.name)

            if qc_pass_only and verdict == 'FAIL':
                logger.info(f"  Skipping {series_id} — QC FAIL")
                continue

            # Load 5-channel npy (preferred) or fall back to isotropic.nii.gz
            npy_path = series_dir / 'multi_window_5ch.npy'
            iso_path = series_dir / 'isotropic.nii.gz'
            mask_path = series_dir / 'body_mask.nii.gz'

            if npy_path.exists():
                multi_ch = np.load(str(npy_path))  # (5, D, H, W)
            elif iso_path.exists():
                multi_ch = self._nifti_to_5ch(str(iso_path))
            else:
                logger.warning(f"  No volume found for {series_id}, skipping")
                continue

            if multi_ch.ndim != 4 or multi_ch.shape[0] != 5:
                logger.warning(f"  {series_id}: unexpected shape {multi_ch.shape}")
                continue

            # Load body mask for filtering
            body_mask = None
            if mask_path.exists():
                try:
                    import SimpleITK as sitk
                    bm_img = sitk.ReadImage(str(mask_path))
                    body_mask = sitk.GetArrayFromImage(bm_img).astype(np.uint8)
                except Exception:
                    pass

            # Extract patches
            patches = self._extract_patches(multi_ch, body_mask, series_id, overwrite)
            total_patches += len(patches)

            for patch_path in patches:
                manifest_entries.append({
                    'path': str(patch_path),
                    'series_id': series_id,
                    'patient_id': qc.get('patient_id', ''),
                    'qc_verdict': verdict,
                    'spacing_mm': qc.get('isotropic_shape', []),
                    'input_type': qc.get('input_type', ''),
                })

        # Train/val split
        random.shuffle(manifest_entries)
        n_val = max(1, int(len(manifest_entries) * val_ratio))
        val_entries = manifest_entries[:n_val]
        train_entries = manifest_entries[n_val:]

        manifest = {
            'total_patches': total_patches,
            'train_patches': len(train_entries),
            'val_patches': len(val_entries),
            'patch_size': self.patch_size,
            'channels': 5,
            'qc_pass_only': qc_pass_only,
            'entries': manifest_entries,
        }

        manifest_path = self.output_root / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with open(self.output_root / 'splits' / 'train.json', 'w') as f:
            json.dump(train_entries, f, indent=2)
        with open(self.output_root / 'splits' / 'val.json', 'w') as f:
            json.dump(val_entries, f, indent=2)

        print(f"\n  Dataset built:")
        print(f"    Total patches  : {total_patches}")
        print(f"    Train          : {len(train_entries)}")
        print(f"    Val            : {len(val_entries)}")
        print(f"    Manifest       : {manifest_path}")

        return manifest

    # ---------------------------------------------------------------
    def _extract_patches(
        self,
        volume: np.ndarray,   # (5, D, H, W)
        body_mask: Optional[np.ndarray],  # (D, H, W) uint8
        series_id: str,
        overwrite: bool,
    ) -> List[Path]:
        """Slide window over volume and save valid patches."""
        import torch

        P = self.patch_size
        S = self.stride
        _, D, H, W = volume.shape

        patch_paths = []
        patch_idx = 0

        for z in range(0, max(1, D - P + 1), S):
            for y in range(0, max(1, H - P + 1), S):
                for x in range(0, max(1, W - P + 1), S):
                    z_end = min(z + P, D)
                    y_end = min(y + P, H)
                    x_end = min(x + P, W)

                    patch = volume[:, z:z_end, y:y_end, x:x_end]

                    # Pad if boundary patch is smaller
                    if patch.shape[1:] != (P, P, P):
                        pad_d = P - patch.shape[1]
                        pad_h = P - patch.shape[2]
                        pad_w = P - patch.shape[3]
                        patch = np.pad(
                            patch,
                            ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                            mode='constant', constant_values=0
                        )

                    # Body mask filtering
                    if body_mask is not None:
                        bm_patch = body_mask[z:z_end, y:y_end, x:x_end]
                        if bm_patch.size > 0:
                            body_ratio = bm_patch.mean()
                            if body_ratio < self.min_body_ratio:
                                continue

                    # Z-score normalize per patch
                    patch_mean = patch.mean()
                    patch_std = max(patch.std(), 1e-6)
                    patch_norm = ((patch - patch_mean) / patch_std).astype(np.float32)

                    # Save
                    out_name = f"{series_id}_patch_{patch_idx:04d}.pt"
                    out_path = self.tensor_dir / out_name

                    if not out_path.exists() or overwrite:
                        tensor_data = {
                            'image': torch.from_numpy(patch_norm),
                            'meta': {
                                'series_id': series_id,
                                'patch_idx': patch_idx,
                                'bbox': [z, y, x, z_end, y_end, x_end],
                                'mean': float(patch_mean),
                                'std': float(patch_std),
                            }
                        }
                        torch.save(tensor_data, str(out_path))

                    patch_paths.append(out_path)
                    patch_idx += 1

        logger.info(f"  {series_id}: {patch_idx} patches extracted")
        return patch_paths

    # ---------------------------------------------------------------
    def _nifti_to_5ch(self, nifti_path: str) -> np.ndarray:
        """Convert isotropic NIfTI → 5-channel using window presets."""
        import SimpleITK as sitk

        img = sitk.ReadImage(nifti_path)
        vol = sitk.GetArrayFromImage(img).astype(np.float32)

        WINDOWS = [
            (40, 400),   # soft tissue
            (60, 150),   # liver
            (50, 350),   # colon
            (-600, 1500),# lung
            (400, 1800), # bone
        ]
        channels = []
        for center, width in WINDOWS:
            lo = center - width / 2
            hi = center + width / 2
            ch = np.clip(vol, lo, hi)
            ch = (ch - lo) / (hi - lo)
            channels.append(ch.astype(np.float32))

        return np.stack(channels, axis=0)  # (5, D, H, W)


# ═══════════════════════════════════════════════════════════════════
# PyTorch Dataset wrapper
# ═══════════════════════════════════════════════════════════════════

class CTPreprocessedDataset:
    """
    Lightweight PyTorch-compatible Dataset from manifest splits.

    Usage:
        from torch.utils.data import DataLoader
        ds = CTPreprocessedDataset('f:/ADDS/preprocessed', split='train')
        loader = DataLoader(ds, batch_size=2, shuffle=True)
    """

    def __init__(self, root: str, split: str = 'train'):
        split_file = Path(root) / 'splits' / f'{split}.json'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            self.entries = json.load(f)
        logger.info(f"Dataset[{split}]: {len(self.entries)} patches")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        import torch
        entry = self.entries[idx]
        data = torch.load(entry['path'], weights_only=True)
        return data['image'], data['meta']


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s │ %(levelname)-5s │ %(message)s')
    builder = CTDatasetBuilder(r'f:\ADDS\preprocessed')
    builder.build_all(qc_pass_only=False)
