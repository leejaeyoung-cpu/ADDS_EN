"""
CT Batch Preprocessor — ADDS Pipeline Orchestrator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
자동으로 CTdata1 / CTdata2 하위 DICOM·NIfTI를 탐색하여
AdvancedCTPreprocessor 10단계 파이프라인을 배치 실행합니다.

출력:
  {output_root}/{patient_id}/{series_name}/
    ├── preprocessed.nii.gz     (denoised, body-masked 볼륨)
    ├── isotropic.nii.gz        (등방성 리샘플링 볼륨)
    ├── multi_window_5ch.npy    (5채널 융합 텐서)
    ├── qc_report.json          (QC 메트릭)
    └── annotation.json         (복부 ROI 어노테이션)

Author : ADDS Research Team
Date   : 2026-03-17
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

class CTSeriesInfo:
    """Represents a single CT series to be processed."""

    def __init__(
        self,
        series_id: str,
        patient_id: str,
        source_dir: str,
        input_type: str,        # 'dicom' | 'nifti'
        file_paths: List[str],  # DICOM files or single NIfTI path
        series_label: str = '',
    ):
        self.series_id = series_id
        self.patient_id = patient_id
        self.source_dir = source_dir
        self.input_type = input_type
        self.file_paths = file_paths
        self.series_label = series_label

    def __repr__(self):
        return (f"CTSeriesInfo(id={self.series_id!r}, type={self.input_type}, "
                f"files={len(self.file_paths)})")


# ═══════════════════════════════════════════════════════════════════
# Source Scanner
# ═══════════════════════════════════════════════════════════════════

def scan_ct_sources(
    source_dirs: List[str],
    extensions: Tuple[str, ...] = ('.dcm', '.nii', '.nii.gz'),
) -> List[CTSeriesInfo]:
    """
    Recursively scan directories for CT data.

    Splits DICOM files by:
      - prefix pattern (e.g. '0930 pre', '1223 post')
      - or by flat directory (each directory = one series)

    NIfTI files are treated as one series each.
    """
    series_list: List[CTSeriesInfo] = []

    for src_dir in source_dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            logger.warning(f"Source directory not found: {src_dir}")
            continue

        # --- NIfTI files -----------------------------------------------
        nifti_files = (
            list(src_path.rglob('*.nii.gz')) +
            list(src_path.rglob('*.nii'))
        )

        # Exclude segmentation, mask, label, and per-organ NIfTIs
        EXCLUDE_STEMS = {
            'mask', 'segmentation', 'tumor', 'label', 'pred',
            'body_mask', 'raw', 'clahe', 'preprocessed', 'isotropic',
        }
        # Known TotalSegmentator organ names (first part before underscore)
        ORGAN_PREFIXES = {
            'liver', 'spleen', 'kidney', 'colon', 'small', 'stomach',
            'pancreas', 'gallbladder', 'duodenum', 'esophagus', 'aorta',
            'lung', 'heart', 'brain', 'femur', 'hip', 'humerus',
            'vertebrae', 'rib', 'sacrum', 'scapula', 'skull', 'sternum',
            'iliac', 'gluteus', 'iliopsoas', 'trachea', 'thyroid',
            'prostate', 'urinary', 'pulmonary', 'portal', 'inferior',
            'superior', 'brachiocephalic', 'subclavian', 'autochthon',
            'clavicula', 'costal', 'adrenal', 'atrial', 'common',
            'spinal', 'kidney_cyst',
        }

        filtered_nifti = []
        for nf in nifti_files:
            stem_lower = nf.stem.lower().replace('.nii', '')
            # Skip if in excluded stem list
            if any(x in stem_lower for x in EXCLUDE_STEMS):
                continue
            # Skip if parent directory is named 'segmentation'
            if 'segmentation' in [p.name.lower() for p in nf.parents]:
                continue
            # Skip known organ-level files (TotalSegmentator output)
            first_word = stem_lower.split('_')[0]
            if first_word in ORGAN_PREFIXES:
                continue
            filtered_nifti.append(nf)

        for nf in filtered_nifti:
            patient_id = src_path.name
            series_id = nf.stem.replace('.nii', '')
            series_list.append(CTSeriesInfo(
                series_id=series_id,
                patient_id=patient_id,
                source_dir=str(src_path),
                input_type='nifti',
                file_paths=[str(nf)],
                series_label=series_id,
            ))

        # --- DICOM files -----------------------------------------------
        # Group by prefix (text before the last digit block)
        dcm_files = sorted(src_path.glob('*.dcm'))
        if dcm_files:
            groups: Dict[str, List[str]] = {}
            for dcm in dcm_files:
                # Extract prefix: e.g. '0930 pre', '1223 post'
                m = re.match(r'^(.+?)(\d+)\.dcm$', dcm.name)
                prefix = m.group(1).strip() if m else 'series'
                groups.setdefault(prefix, []).append(str(dcm))

            for prefix, files in groups.items():
                safe_prefix = re.sub(r'[^\w]', '_', prefix).strip('_')
                series_id = f"{src_path.name}_{safe_prefix}"
                series_list.append(CTSeriesInfo(
                    series_id=series_id,
                    patient_id=src_path.name,
                    source_dir=str(src_path),
                    input_type='dicom',
                    file_paths=sorted(files),
                    series_label=prefix,
                ))

        # --- Sub-directory DICOM (each sub-dir = one series) -----------
        for sub in sorted(src_path.iterdir()):
            if not sub.is_dir():
                continue
            sub_dcm = sorted(sub.glob('*.dcm'))
            if not sub_dcm:
                continue
            series_id = f"{src_path.name}_{sub.name}"
            series_list.append(CTSeriesInfo(
                series_id=series_id,
                patient_id=src_path.name,
                source_dir=str(sub),
                input_type='dicom',
                file_paths=[str(f) for f in sub_dcm],
                series_label=sub.name,
            ))

    # Deduplicate by series_id
    seen = set()
    deduped = []
    for s in series_list:
        if s.series_id not in seen:
            seen.add(s.series_id)
            deduped.append(s)

    logger.info(f"Found {len(deduped)} CT series across {len(source_dirs)} source dir(s)")
    return deduped


# ═══════════════════════════════════════════════════════════════════
# DICOM → NIfTI Converter
# ═══════════════════════════════════════════════════════════════════

def dicom_series_to_nifti(
    dicom_files: List[str],
    output_path: Optional[str] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float], Optional[np.ndarray]]:
    """
    Load a DICOM series → 3D numpy volume (Z,Y,X) + spacing.

    Uses SimpleITK for robust multi-frame handling.
    Returns: (volume_hu, spacing_zyx, slice_positions)
    """
    if not dicom_files:
        raise ValueError("No DICOM files provided")

    # Sort by filename number
    def _sort_key(p):
        m = re.search(r'(\d+)', Path(p).stem)
        return int(m.group(1)) if m else 0

    sorted_files = sorted(dicom_files, key=_sort_key)

    try:
        # Use SimpleITK ImageSeriesReader
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        sitk_img = reader.Execute()
    except Exception:
        # Fallback: read file by file
        logger.warning("SeriesReader failed — falling back to per-slice loading")
        sitk_img = _load_dicom_fallback(sorted_files)

    volume_arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # Z,Y,X
    spacing_xyz = sitk_img.GetSpacing()  # (x, y, z)
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))

    # Collect Z positions
    try:
        positions = []
        for i in range(reader.GetSize()):
            meta = reader.GetMetaData(i, '0020|1041') if reader.HasMetaDataKey(i, '0020|1041') else ''
            positions.append(float(meta) if meta else i * spacing_zyx[0])
        slice_positions = np.array(positions)
    except Exception:
        slice_positions = None

    if output_path:
        sitk.WriteImage(sitk_img, output_path, True)
        logger.info(f"  Saved NIfTI → {output_path}")

    logger.info(f"  DICOM loaded: shape={volume_arr.shape}, spacing={spacing_zyx}")
    return volume_arr, spacing_zyx, slice_positions


def _load_dicom_fallback(files: List[str]):
    """Per-slice fallback using pydicom when SimpleITK fails."""
    import pydicom
    from collections import Counter

    slices = []
    shapes = []
    for f in files:
        try:
            dcm = pydicom.dcmread(f, force=True)
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            px = dcm.pixel_array.astype(np.float32) * slope + intercept
            slices.append(px)
            shapes.append(px.shape)
        except Exception as e:
            logger.debug(f"Skipping {f}: {e}")

    if not slices:
        raise RuntimeError("All DICOM files failed to load")

    # Majority voting for shape consistency
    shape_counts = Counter(shapes)
    majority_shape = shape_counts.most_common(1)[0][0]
    n_skipped = sum(1 for s in shapes if s != majority_shape)
    if n_skipped > 0:
        logger.warning(f"  Skipping {n_skipped} slices with non-majority shape "
                       f"(majority={majority_shape})")

    valid_slices = [s for s, sh in zip(slices, shapes) if sh == majority_shape]
    if not valid_slices:
        raise RuntimeError("No valid slices after majority-shape filtering")

    arr = np.stack(valid_slices, axis=0)
    img = sitk.GetImageFromArray(arr)
    return img


# ═══════════════════════════════════════════════════════════════════
# Batch Processor
# ═══════════════════════════════════════════════════════════════════

class BatchPreprocessor:
    """
    Orchestrates 10-stage preprocessing across all CT series.

    Usage:
        bp = BatchPreprocessor(output_root='f:/ADDS/preprocessed')
        series = scan_ct_sources(['CTdata1', 'CTdata2'])
        results = bp.run_all(series, max_workers=2)
    """

    def __init__(
        self,
        output_root: str,
        skip_stages: Optional[List[int]] = None,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        overwrite: bool = False,
    ):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.skip_stages = skip_stages or []
        self.target_spacing = target_spacing
        self.overwrite = overwrite

    # ---------------------------------------------------------------
    def run_all(
        self,
        series_list: List[CTSeriesInfo],
        max_workers: int = 1,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, dict]:
        """
        Process all series. Returns summary dict keyed by series_id.
        """
        if limit:
            series_list = series_list[:limit]

        if dry_run:
            print("\n=== DRY RUN — no files will be written ===")
            for s in series_list:
                print(f"  [{s.input_type.upper():5s}] {s.series_id} ({len(s.file_paths)} files)")
            print(f"\nTotal: {len(series_list)} series")
            return {}

        summary = {}
        print(f"\n{'='*60}")
        print(f"  ADDS Batch CT Preprocessor — {len(series_list)} series")
        print(f"  Output: {self.output_root}")
        print(f"  Skip stages: {self.skip_stages or 'none'}")
        print(f"{'='*60}\n")

        if max_workers == 1:
            for i, series in enumerate(series_list, 1):
                print(f"  [{i}/{len(series_list)}] {series.series_id}")
                result = self._process_one(series)
                summary[series.series_id] = result
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_one, s): s
                    for s in series_list
                }
                for i, fut in enumerate(as_completed(futures), 1):
                    s = futures[fut]
                    try:
                        result = fut.result()
                        summary[s.series_id] = result
                        print(f"  [{i}/{len(series_list)}] ✓ {s.series_id} — {result.get('qc_verdict', '?')}")
                    except Exception as e:
                        summary[s.series_id] = {'status': 'ERROR', 'error': str(e)}
                        logger.error(f"  [{i}] ✗ {s.series_id}: {e}")

        # Write global summary
        summary_path = self.output_root / 'batch_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  Batch summary → {summary_path}")

        return summary

    # ---------------------------------------------------------------
    def _process_one(self, series: CTSeriesInfo) -> dict:
        """Process a single CTSeriesInfo through the full 10-stage pipeline."""
        from src.medical_imaging.ct_advanced_preprocessing import AdvancedCTPreprocessor

        out_dir = self.output_root / series.patient_id / series.series_id
        out_dir.mkdir(parents=True, exist_ok=True)

        qc_path = out_dir / 'qc_report.json'
        if qc_path.exists() and not self.overwrite:
            logger.info(f"  Skipping {series.series_id} (already processed)")
            with open(qc_path) as f:
                return json.load(f)

        t0 = time.time()

        # ── Load volume ──────────────────────────────────────────
        try:
            if series.input_type == 'dicom':
                volume, spacing, slice_positions = dicom_series_to_nifti(
                    series.file_paths,
                    output_path=str(out_dir / 'raw.nii.gz'),
                )
            else:  # nifti
                nii_path = series.file_paths[0]
                sitk_img = sitk.ReadImage(nii_path)
                volume = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                sp = sitk_img.GetSpacing()
                spacing = (float(sp[2]), float(sp[1]), float(sp[0]))
                slice_positions = None
        except Exception as e:
            logger.error(f"Load failed for {series.series_id}: {e}")
            return {'status': 'ERROR', 'error': f'load: {e}'}

        # ── Load segmentation mask if available ──────────────────
        seg_candidates = [
            Path(series.source_dir).parent / 'CTdata1' / 'segmentation.nii',
            Path(series.source_dir) / 'segmentation.nii',
            Path(series.source_dir) / 'segmentation_resampled.nii.gz',
        ]
        seg_mask = None
        for sc in seg_candidates:
            if sc.exists():
                try:
                    seg_img = sitk.ReadImage(str(sc))
                    seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.int16)
                    if seg_arr.shape == volume.shape:
                        seg_mask = seg_arr
                        logger.info(f"  Segmentation mask loaded: {sc}")
                except Exception:
                    pass
                break

        # ── Run 10-stage pipeline ────────────────────────────────
        preprocessor = AdvancedCTPreprocessor(
            skip_stages=self.skip_stages,
            target_spacing=self.target_spacing,
        )
        try:
            result = preprocessor.run(
                volume,
                spacing=spacing,
                slice_positions=slice_positions,
                segmentation_mask=seg_mask,
            )
        except Exception as e:
            logger.error(f"Pipeline failed for {series.series_id}: {e}")
            return {'status': 'ERROR', 'error': f'pipeline: {e}'}

        # ── Save outputs ─────────────────────────────────────────
        def _save_nifti(arr: np.ndarray, path: Path, spacing_xyz: tuple):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing(spacing_xyz)
            sitk.WriteImage(img, str(path), True)

        sp_xyz = (spacing[2], spacing[1], spacing[0])
        iso_xyz = (result.isotropic_spacing[2],
                   result.isotropic_spacing[1],
                   result.isotropic_spacing[0])

        _save_nifti(result.volume_denoised,   out_dir / 'preprocessed.nii.gz', sp_xyz)
        _save_nifti(result.volume_isotropic,  out_dir / 'isotropic.nii.gz',    iso_xyz)
        _save_nifti(result.volume_clahe,      out_dir / 'clahe.nii.gz',        sp_xyz)
        _save_nifti(result.body_mask.astype(np.uint8), out_dir / 'body_mask.nii.gz', sp_xyz)

        np.save(str(out_dir / 'multi_window_5ch.npy'), result.multi_window)

        qc = result.quality.to_dict()
        qc['stage_timings'] = result.stage_timings
        qc['series_id'] = series.series_id
        qc['patient_id'] = series.patient_id
        qc['input_type'] = series.input_type
        qc['volume_shape'] = list(volume.shape)
        qc['spacing_mm'] = list(spacing)
        qc['isotropic_shape'] = list(result.volume_isotropic.shape)
        qc['status'] = 'OK'
        qc['wall_time_s'] = round(time.time() - t0, 2)

        with open(qc_path, 'w', encoding='utf-8') as f:
            json.dump(qc, f, indent=2, ensure_ascii=False)

        ann_path = out_dir / 'annotation.json'
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(result.abdomen_annotation, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  {series.series_id} → QC:{qc['qc_verdict']} "
            f"in {qc['wall_time_s']:.1f}s"
        )
        return qc


# ═══════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════

def run_batch(
    source_dirs: List[str],
    output_root: str,
    skip_stages: Optional[List[int]] = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_workers: int = 1,
    limit: Optional[int] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """One-call batch preprocessing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s │ %(levelname)-5s │ %(message)s',
        datefmt='%H:%M:%S',
    )
    series = scan_ct_sources(source_dirs)
    bp = BatchPreprocessor(
        output_root=output_root,
        skip_stages=skip_stages,
        target_spacing=target_spacing,
        overwrite=overwrite,
    )
    return bp.run_all(series, max_workers=max_workers, limit=limit, dry_run=dry_run)


if __name__ == '__main__':
    # Quick dry-run test
    run_batch(
        source_dirs=[r'f:\ADDS\CTdata1', r'f:\ADDS\CTdata2'],
        output_root=r'f:\ADDS\preprocessed',
        dry_run=True,
    )
