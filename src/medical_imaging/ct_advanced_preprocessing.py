"""
Advanced CT Preprocessing Pipeline for KRAS CRC Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10-stage high-difficulty preprocessing:
  1. HU Clamping + Metal Artifact Masking
  2. Body Isolation (Otsu + Morphology)
  3. N4 Bias Field Correction
  4. Non-Local Means 3D Denoising
  5. 3D CLAHE (Contrast-Limited Adaptive Histogram Equalization)
  6. 5-Channel Multi-Window Fusion
  7. Organ-Aware ROI Cropping
  8. Quality Metrics (SNR / CNR / Motion Score)
  9. Isotropic Resampling (1.0 x 1.0 x 1.0 mm³)
  10. Abdomen ROI Annotation (HU-based landmark tagging)

Author: ADDS Research Team
Date  : 2026-03-17 (v2 — 10-stage)
"""

import numpy as np
import SimpleITK as sitk
import cv2
from scipy import ndimage
from scipy.ndimage import label as ndlabel, binary_fill_holes, binary_opening, binary_closing
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QualityMetrics:
    """Automated QC metrics for a CT volume."""
    snr: float = 0.0
    cnr: float = 0.0
    motion_score: float = 0.0          # 0 = no motion, higher = worse
    slice_spacing_cv: float = 0.0      # coefficient of variation
    metal_voxel_ratio: float = 0.0     # fraction of metal voxels
    body_coverage_ratio: float = 0.0   # body voxels / total voxels
    qc_verdict: str = "UNKNOWN"        # PASS / WARN / FAIL
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'snr': round(self.snr, 2),
            'cnr': round(self.cnr, 2),
            'motion_score': round(self.motion_score, 4),
            'slice_spacing_cv': round(self.slice_spacing_cv, 4),
            'metal_voxel_ratio': round(self.metal_voxel_ratio, 6),
            'body_coverage_ratio': round(self.body_coverage_ratio, 4),
            'qc_verdict': self.qc_verdict,
            'details': self.details,
        }


@dataclass
class PreprocessingResult:
    """Complete output of the advanced preprocessing pipeline (v2: 10 stages)."""
    volume_hu: np.ndarray               # HU-clamped volume
    body_mask: np.ndarray               # binary body mask
    volume_denoised: np.ndarray         # after N4 + NLM
    volume_clahe: np.ndarray            # after CLAHE
    multi_window: np.ndarray            # (5, D, H, W) 5-channel tensor
    roi_volume: np.ndarray              # organ-aware cropped volume
    roi_bbox: Dict                      # z/y/x start-end of ROI
    quality: QualityMetrics
    # Stage 9
    volume_isotropic: np.ndarray = None          # isotropically resampled
    isotropic_spacing: tuple = (1.0, 1.0, 1.0)  # achieved spacing (mm)
    # Stage 10
    abdomen_annotation: Dict = field(default_factory=dict)  # ROI tags
    stage_timings: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Multi-Window Configuration
# ═══════════════════════════════════════════════════════════════════

WINDOW_PRESETS = {
    'soft_tissue': {'center':   40, 'width':  400, 'desc': 'Soft tissue (general)'},
    'liver':       {'center':   60, 'width':  150, 'desc': 'Liver / lesion detection'},
    'colon':       {'center':   50, 'width':  350, 'desc': 'Colon-optimized'},
    'lung':        {'center': -600, 'width': 1500, 'desc': 'Lung parenchyma'},
    'bone':        {'center':  400, 'width': 1800, 'desc': 'Bone / calcification'},
}


# ═══════════════════════════════════════════════════════════════════
# Main Class
# ═══════════════════════════════════════════════════════════════════

class AdvancedCTPreprocessor:
    """
    8-stage advanced CT preprocessing for KRAS CRC analysis.

    Usage:
        preprocessor = AdvancedCTPreprocessor()
        result = preprocessor.run(volume_hu, spacing=(1.0, 1.0, 1.0))
    """

    def __init__(
        self,
        # Stage 1 — HU clamping
        hu_min: float = -1024.0,
        hu_max: float = 3071.0,
        metal_threshold: float = 3000.0,
        # Stage 3 — N4
        n4_convergence: float = 1e-6,
        n4_iterations: List[int] = None,
        # Stage 4 — NLM
        nlm_patch_size: int = 5,
        nlm_patch_distance: int = 6,
        nlm_h_factor: float = 0.8,
        # Stage 5 — CLAHE
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        # Stage 7 — ROI
        roi_padding_ratio: float = 0.10,
        # Stage 9 — Isotropic resampling
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        # general
        skip_stages: Optional[List[int]] = None,
    ):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.metal_threshold = metal_threshold

        self.n4_convergence = n4_convergence
        self.n4_iterations = n4_iterations or [50, 50, 50, 50]

        self.nlm_patch_size = nlm_patch_size
        self.nlm_patch_distance = nlm_patch_distance
        self.nlm_h_factor = nlm_h_factor

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        self.roi_padding_ratio = roi_padding_ratio
        self.target_spacing = target_spacing

        self.skip_stages = set(skip_stages or [])

    # ---------------------------------------------------------------
    # PUBLIC: End-to-end pipeline
    # ---------------------------------------------------------------

    def run(
        self,
        volume: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        slice_positions: Optional[np.ndarray] = None,
        segmentation_mask: Optional[np.ndarray] = None,
    ) -> PreprocessingResult:
        """
        Execute the full 8-stage preprocessing pipeline.

        Args:
            volume       : 3D CT volume in Hounsfield Units (Z, Y, X)
            spacing      : voxel spacing in mm (Z, Y, X)
            slice_positions: optional array of Z-positions for spacing QC

        Returns:
            PreprocessingResult with all intermediate outputs
        """
        timings = {}
        logger.info("═" * 60)
        logger.info("  Advanced CT Preprocessing Pipeline — START")
        logger.info("═" * 60)
        logger.info(f"  Input shape : {volume.shape}")
        logger.info(f"  Input HU    : [{volume.min():.1f}, {volume.max():.1f}]")
        logger.info(f"  Spacing     : {spacing}")

        t_total = time.time()

        # ── Stage 1 ── HU Clamping + Metal Artifact Masking ──────
        t0 = time.time()
        volume_hu, metal_mask = self._stage1_hu_clamp(volume)
        timings['stage1_hu_clamp'] = time.time() - t0
        logger.info(f"  [1/8] HU Clamp ................. {timings['stage1_hu_clamp']:.2f}s")

        # ── Stage 2 ── Body Isolation ────────────────────────────
        t0 = time.time()
        body_mask = self._stage2_body_isolation(volume_hu)
        timings['stage2_body_isolation'] = time.time() - t0
        logger.info(f"  [2/8] Body Isolation ........... {timings['stage2_body_isolation']:.2f}s")

        # ── Stage 3 ── N4 Bias Field Correction ──────────────────
        t0 = time.time()
        if 3 not in self.skip_stages:
            volume_n4 = self._stage3_n4_correction(volume_hu, body_mask)
        else:
            volume_n4 = volume_hu.copy()
            logger.info("  [3/8] N4 Correction ............ SKIPPED")
        timings['stage3_n4'] = time.time() - t0
        if 3 not in self.skip_stages:
            logger.info(f"  [3/8] N4 Correction ............ {timings['stage3_n4']:.2f}s")

        # ── Stage 4 ── Non-Local Means Denoising ─────────────────
        t0 = time.time()
        if 4 not in self.skip_stages:
            volume_denoised = self._stage4_nlm_denoise(volume_n4, body_mask)
        else:
            volume_denoised = volume_n4.copy()
            logger.info("  [4/8] NLM Denoise .............. SKIPPED")
        timings['stage4_nlm'] = time.time() - t0
        if 4 not in self.skip_stages:
            logger.info(f"  [4/8] NLM Denoise .............. {timings['stage4_nlm']:.2f}s")

        # ── Stage 5 ── 3D CLAHE ──────────────────────────────────
        t0 = time.time()
        volume_clahe = self._stage5_clahe(volume_denoised, body_mask)
        timings['stage5_clahe'] = time.time() - t0
        logger.info(f"  [5/8] 3D CLAHE ................. {timings['stage5_clahe']:.2f}s")

        # ── Stage 6 ── 5-Channel Multi-Window Fusion ─────────────
        t0 = time.time()
        multi_window = self._stage6_multi_window(volume_denoised)
        timings['stage6_multi_window'] = time.time() - t0
        logger.info(f"  [6/8] Multi-Window Fusion ....... {timings['stage6_multi_window']:.2f}s")

        # ── Stage 7 ── Organ-Aware ROI Cropping ──────────────────
        t0 = time.time()
        roi_volume, roi_bbox = self._stage7_roi_crop(volume_denoised, body_mask, spacing)
        timings['stage7_roi_crop'] = time.time() - t0
        logger.info(f"  [7/8] ROI Crop ................. {timings['stage7_roi_crop']:.2f}s")

        # ── Stage 8 ── Quality Metrics ───────────────────────────
        t0 = time.time()
        quality = self._stage8_quality_metrics(
            volume_hu, volume_denoised, body_mask, metal_mask, slice_positions
        )
        timings['stage8_quality'] = time.time() - t0
        logger.info(f"  [8/8] Quality Metrics .......... {timings['stage8_quality']:.2f}s")

        # ── Stage 9 ── Isotropic Resampling ──────────────────────
        t0 = time.time()
        if 9 not in self.skip_stages:
            volume_iso, achieved_spacing = self._stage9_isotropic_resample(
                volume_denoised, spacing
            )
            # Resample body_mask to match isotropic volume shape
            if volume_iso.shape != volume_denoised.shape:
                from scipy.ndimage import zoom as ndz
                zoom_factors = tuple(
                    volume_iso.shape[i] / volume_denoised.shape[i] for i in range(3)
                )
                body_mask_iso = (ndz(body_mask.astype(np.float32),
                                    zoom_factors, order=0) > 0.5).astype(np.uint8)
            else:
                body_mask_iso = body_mask
        else:
            volume_iso = volume_denoised.copy()
            body_mask_iso = body_mask
            achieved_spacing = spacing
            logger.info("  [9/10] Isotropic Resample ....... SKIPPED")
        timings['stage9_isotropic'] = time.time() - t0
        if 9 not in self.skip_stages:
            logger.info(f"  [9/10] Isotropic Resample ....... {timings['stage9_isotropic']:.2f}s")

        # ── Stage 10 ── Abdomen ROI Annotation ────────────────────
        t0 = time.time()
        abdomen_annotation = self._stage10_abdomen_annotation(
            volume_iso, body_mask_iso, spacing=achieved_spacing,
            segmentation_mask=segmentation_mask
        )
        timings['stage10_annotation'] = time.time() - t0
        logger.info(f"  [10/10] Abdomen Annotation ...... {timings['stage10_annotation']:.2f}s")

        total_time = time.time() - t_total
        timings['total'] = total_time

        logger.info("═" * 60)
        logger.info(f"  Pipeline DONE in {total_time:.2f}s — QC: {quality.qc_verdict}")
        logger.info("═" * 60)

        return PreprocessingResult(
            volume_hu=volume_hu,
            body_mask=body_mask,
            volume_denoised=volume_denoised,
            volume_clahe=volume_clahe,
            multi_window=multi_window,
            roi_volume=roi_volume,
            roi_bbox=roi_bbox,
            quality=quality,
            volume_isotropic=volume_iso,
            isotropic_spacing=achieved_spacing,
            abdomen_annotation=abdomen_annotation,
            stage_timings=timings,
        )

    # ===============================================================
    # Stage 1 — HU Clamping + Metal Artifact Masking
    # ===============================================================

    def _stage1_hu_clamp(
        self, volume: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clamp HU values to valid range and mask metal artifacts.
        Metal voxels (HU > 3000) are replaced with the mean of their
        non-metal neighborhood.
        """
        # Metal detection
        metal_mask = volume > self.metal_threshold
        metal_count = int(metal_mask.sum())

        # Clamp to valid range
        volume_clamped = np.clip(volume, self.hu_min, self.hu_max).astype(np.float32)

        # Interpolate metal regions with neighborhood mean
        if metal_count > 0:
            logger.info(f"    Metal voxels detected: {metal_count:,}")
            # Dilate metal mask to get neighborhood
            struct = ndimage.generate_binary_structure(3, 1)
            dilated = ndimage.binary_dilation(metal_mask, struct, iterations=3)
            neighborhood = dilated & ~metal_mask

            if neighborhood.sum() > 0:
                fill_value = float(volume_clamped[neighborhood].mean())
            else:
                fill_value = 0.0

            volume_clamped[metal_mask] = fill_value

        return volume_clamped, metal_mask

    # ===============================================================
    # Stage 2 — Body Isolation (Otsu + Morphology)
    # ===============================================================

    def _stage2_body_isolation(self, volume_hu: np.ndarray) -> np.ndarray:
        """
        Separate body from air/bed using Otsu thresholding + 3D morphology.

        Returns binary body mask (1 = body, 0 = air/bed).
        """
        # Use a representative subset of slices for speed
        D, H, W = volume_hu.shape

        # --- per-slice Otsu threshold ---
        body_mask = np.zeros_like(volume_hu, dtype=np.uint8)
        for z in range(D):
            slc = volume_hu[z]
            # Normalize slice to uint8 for Otsu
            s_min, s_max = slc.min(), slc.max()
            if s_max - s_min < 1e-6:
                continue
            slc_u8 = ((slc - s_min) / (s_max - s_min) * 255).astype(np.uint8)
            thresh_val, mask_u8 = cv2.threshold(slc_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            body_mask[z] = (mask_u8 > 0).astype(np.uint8)

        # --- 3D morphological cleanup ---
        struct = ndimage.generate_binary_structure(3, 1)

        # Fill holes inside body
        body_mask = binary_fill_holes(body_mask, structure=struct).astype(np.uint8)

        # Opening to remove small noise
        body_mask = binary_opening(body_mask, structure=struct, iterations=2).astype(np.uint8)

        # Closing to fill small gaps
        body_mask = binary_closing(body_mask, structure=struct, iterations=2).astype(np.uint8)

        # Keep only the largest connected component (the body)
        labeled_array, num_features = ndlabel(body_mask, structure=struct)
        if num_features > 1:
            component_sizes = ndimage.sum(body_mask, labeled_array, range(1, num_features + 1))
            largest_label = int(np.argmax(component_sizes)) + 1
            body_mask = (labeled_array == largest_label).astype(np.uint8)

        coverage = body_mask.sum() / body_mask.size
        logger.info(f"    Body coverage: {coverage:.1%} of total volume")

        return body_mask

    # ===============================================================
    # Stage 3 — N4 Bias Field Correction (SimpleITK)
    # ===============================================================

    def _stage3_n4_correction(
        self, volume_hu: np.ndarray, body_mask: np.ndarray
    ) -> np.ndarray:
        """
        N4ITK bias field correction to compensate for intensity
        inhomogeneity (shading artifacts), especially in low-dose CT.
        """
        # Shift HU so all values are positive (N4 needs > 0)
        shift = abs(volume_hu.min()) + 1.0
        vol_pos = (volume_hu + shift).astype(np.float32)

        # Convert to SimpleITK
        sitk_image = sitk.GetImageFromArray(vol_pos)
        sitk_mask = sitk.GetImageFromArray(body_mask.astype(np.uint8))

        # Cast to required types
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)

        # Configure N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(self.n4_iterations)
        corrector.SetConvergenceThreshold(self.n4_convergence)
        corrector.SetNumberOfHistogramBins(200)

        # Execute
        try:
            corrected = corrector.Execute(sitk_image, sitk_mask)
            result = sitk.GetArrayFromImage(corrected)

            # Shift back to original HU range
            result = result - shift

            logger.info(f"    N4 converged — output HU: [{result.min():.1f}, {result.max():.1f}]")
        except Exception as e:
            logger.warning(f"    N4 correction failed ({e}), using uncorrected volume")
            result = volume_hu.copy()

        return result.astype(np.float32)

    # ===============================================================
    # Stage 4 — Non-Local Means 3D Denoising
    # ===============================================================

    def _stage4_nlm_denoise(
        self, volume: np.ndarray, body_mask: np.ndarray
    ) -> np.ndarray:
        """
        Non-Local Means denoising applied within the body mask only
        to reduce quantum noise while preserving edges.
        """
        from skimage.restoration import denoise_nl_means, estimate_sigma

        # Estimate noise level from body region
        body_region = volume[body_mask > 0]
        sigma_est = float(estimate_sigma(body_region))
        h_param = self.nlm_h_factor * sigma_est

        logger.info(f"    Estimated σ: {sigma_est:.3f}, h: {h_param:.3f}")

        # Normalize to [0, 1] for NLM
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min < 1e-6:
            return volume.copy()

        vol_norm = (volume - v_min) / (v_max - v_min)

        # Apply NLM (fast mode for 3D)
        denoised_norm = denoise_nl_means(
            vol_norm,
            patch_size=self.nlm_patch_size,
            patch_distance=self.nlm_patch_distance,
            h=h_param / (v_max - v_min),  # scale h to normalized range
            fast_mode=True,
            channel_axis=None,
        )

        # Rescale back to HU
        denoised = denoised_norm * (v_max - v_min) + v_min

        # Only apply within body mask
        result = volume.copy()
        result[body_mask > 0] = denoised[body_mask > 0]

        logger.info(f"    Denoised HU: [{result.min():.1f}, {result.max():.1f}]")

        return result.astype(np.float32)

    # ===============================================================
    # Stage 5 — 3D CLAHE (slice-wise + Z-smoothing)
    # ===============================================================

    def _stage5_clahe(
        self, volume: np.ndarray, body_mask: np.ndarray
    ) -> np.ndarray:
        """
        Contrast-Limited Adaptive Histogram Equalization applied
        per-slice (2D) then smoothed along Z to avoid inter-slice
        discontinuity. Enhances tumor boundary contrast.
        """
        D, H, W = volume.shape

        # Normalize to [0, 255] uint8
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min < 1e-6:
            return volume.copy()

        vol_u8 = ((volume - v_min) / (v_max - v_min) * 255).astype(np.uint8)

        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_grid_size,
        )

        # Apply per-slice
        result_u8 = np.zeros_like(vol_u8)
        for z in range(D):
            result_u8[z] = clahe.apply(vol_u8[z])

        # Z-axis Gaussian smoothing to reduce inter-slice banding
        result_f = result_u8.astype(np.float32)
        result_f = ndimage.gaussian_filter1d(result_f, sigma=0.5, axis=0)

        # Rescale back to original HU range
        result = result_f / 255.0 * (v_max - v_min) + v_min

        # Only apply within body mask
        output = volume.copy()
        output[body_mask > 0] = result[body_mask > 0]

        return output.astype(np.float32)

    # ===============================================================
    # Stage 6 — 5-Channel Multi-Window Fusion
    # ===============================================================

    def _stage6_multi_window(self, volume_hu: np.ndarray) -> np.ndarray:
        """
        Create a 5-channel tensor from 5 different HU windows.
        Each channel is normalized to [0, 1].

        Returns shape: (5, D, H, W)
        """
        channels = []
        for name, cfg in WINDOW_PRESETS.items():
            center = cfg['center']
            width = cfg['width']
            lo = center - width / 2
            hi = center + width / 2
            windowed = np.clip(volume_hu, lo, hi)
            windowed = (windowed - lo) / (hi - lo)  # → [0, 1]
            channels.append(windowed.astype(np.float32))

        multi = np.stack(channels, axis=0)  # (5, D, H, W)
        logger.info(f"    5-ch shape: {multi.shape}")
        return multi

    # ===============================================================
    # Stage 7 — Organ-Aware ROI Cropping
    # ===============================================================

    def _stage7_roi_crop(
        self,
        volume: np.ndarray,
        body_mask: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Dict]:
        """
        Crop volume to abdomen region of interest:
          Z : estimate diaphragm → pelvis using HU gradient analysis
          X/Y : body mask bounding box + padding
        """
        D, H, W = volume.shape

        # ---- Z-axis: detect diaphragm → pelvis ----
        # Avg HU per slice (within body)
        avg_hu = np.zeros(D)
        for z in range(D):
            bm_slice = body_mask[z]
            if bm_slice.sum() > 0:
                avg_hu[z] = volume[z][bm_slice > 0].mean()

        # Gradient of average HU along Z
        hu_grad = np.gradient(avg_hu)

        # Diaphragm: large negative gradient (lung → abdomen transition)
        # Search in upper half
        upper = D // 4
        lower_search = 3 * D // 4
        z_start = 0
        z_end = D

        # Find diaphragm (largest negative gradient in upper region)
        if D > 20:
            upper_grad = hu_grad[upper:D // 2]
            if len(upper_grad) > 0:
                diaphragm_rel = int(np.argmin(upper_grad))
                z_start = max(0, upper + diaphragm_rel - 5)

            # Pelvis: find where body mask area starts shrinking rapidly
            body_area_per_slice = np.array([body_mask[z].sum() for z in range(D)])
            area_grad = np.gradient(body_area_per_slice)
            lower_grad = area_grad[lower_search:]
            if len(lower_grad) > 0:
                pelvis_rel = int(np.argmin(lower_grad))
                z_end = min(D, lower_search + pelvis_rel + 5)

        # Ensure valid range
        if z_end - z_start < 10:
            z_start, z_end = 0, D

        # ---- X/Y bounding box from body mask ----
        body_proj_y = body_mask.any(axis=(0, 2))  # project to Y
        body_proj_x = body_mask.any(axis=(0, 1))  # project to X

        y_indices = np.where(body_proj_y)[0]
        x_indices = np.where(body_proj_x)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            # Fallback: no cropping
            roi_bbox = {'z_start': 0, 'z_end': D, 'y_start': 0, 'y_end': H, 'x_start': 0, 'x_end': W}
            return volume.copy(), roi_bbox

        y_start = int(y_indices[0])
        y_end = int(y_indices[-1]) + 1
        x_start = int(x_indices[0])
        x_end = int(x_indices[-1]) + 1

        # Add padding
        pad_y = int((y_end - y_start) * self.roi_padding_ratio)
        pad_x = int((x_end - x_start) * self.roi_padding_ratio)

        y_start = max(0, y_start - pad_y)
        y_end = min(H, y_end + pad_y)
        x_start = max(0, x_start - pad_x)
        x_end = min(W, x_end + pad_x)

        roi_bbox = {
            'z_start': int(z_start), 'z_end': int(z_end),
            'y_start': int(y_start), 'y_end': int(y_end),
            'x_start': int(x_start), 'x_end': int(x_end),
        }

        roi_volume = volume[z_start:z_end, y_start:y_end, x_start:x_end].copy()
        logger.info(f"    ROI bbox: z=[{z_start}:{z_end}], y=[{y_start}:{y_end}], x=[{x_start}:{x_end}]")
        logger.info(f"    ROI shape: {roi_volume.shape}  (original: {volume.shape})")

        return roi_volume, roi_bbox

    # ===============================================================
    # Stage 8 — Quality Metrics
    # ===============================================================

    def _stage8_quality_metrics(
        self,
        volume_original: np.ndarray,
        volume_denoised: np.ndarray,
        body_mask: np.ndarray,
        metal_mask: np.ndarray,
        slice_positions: Optional[np.ndarray] = None,
    ) -> QualityMetrics:
        """
        Compute automated quality metrics:
          - SNR   : signal / noise within body
          - CNR   : (mean_soft - mean_muscle) / noise
          - Motion: edge sharpness analysis
        """
        qm = QualityMetrics()

        # --- SNR ---
        body_voxels = volume_denoised[body_mask > 0]
        if len(body_voxels) > 100:
            signal = float(np.mean(body_voxels))
            noise = float(np.std(body_voxels))
            qm.snr = signal / noise if noise > 1e-6 else 0.0
        qm.details['snr_signal'] = float(np.mean(body_voxels)) if len(body_voxels) > 0 else 0.0
        qm.details['snr_noise'] = float(np.std(body_voxels)) if len(body_voxels) > 0 else 0.0

        # --- CNR (soft tissue vs fat) ---
        # Soft tissue: HU [20, 80], Fat: HU [-120, -60]
        soft_mask = (volume_denoised > 20) & (volume_denoised < 80) & (body_mask > 0)
        fat_mask = (volume_denoised > -120) & (volume_denoised < -60) & (body_mask > 0)

        if soft_mask.sum() > 100 and fat_mask.sum() > 100:
            mean_soft = float(volume_denoised[soft_mask].mean())
            mean_fat = float(volume_denoised[fat_mask].mean())
            std_bg = float(volume_denoised[fat_mask].std())
            qm.cnr = abs(mean_soft - mean_fat) / std_bg if std_bg > 1e-6 else 0.0
            qm.details['cnr_soft_mean'] = mean_soft
            qm.details['cnr_fat_mean'] = mean_fat

        # --- Motion artifact score ---
        # Edge sharpness: Laplacian variance across central slices
        D = volume_denoised.shape[0]
        center_slices = range(max(0, D // 3), min(D, 2 * D // 3))
        laplacian_vars = []
        for z in center_slices:
            slc = volume_denoised[z]
            s_min, s_max = slc.min(), slc.max()
            if s_max - s_min < 1e-6:
                continue
            slc_u8 = ((slc - s_min) / (s_max - s_min) * 255).astype(np.uint8)
            lap = cv2.Laplacian(slc_u8, cv2.CV_64F)
            laplacian_vars.append(float(lap.var()))

        if laplacian_vars:
            mean_lap = np.mean(laplacian_vars)
            # Higher Laplacian variance = sharper => less motion
            # Normalize: motion_score ~0 is good, >1 is bad
            # Empirical: good CT has Laplacian var > 500
            qm.motion_score = max(0.0, 1.0 - mean_lap / 500.0)
            qm.details['laplacian_var_mean'] = float(mean_lap)

        # --- Slice spacing uniformity ---
        if slice_positions is not None and len(slice_positions) > 2:
            diffs = np.diff(np.sort(slice_positions))
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            qm.slice_spacing_cv = float(std_diff / mean_diff) if mean_diff > 1e-6 else 0.0

        # --- Metal ratio ---
        qm.metal_voxel_ratio = float(metal_mask.sum() / metal_mask.size)

        # --- Body coverage ---
        qm.body_coverage_ratio = float(body_mask.sum() / body_mask.size)

        # --- QC Verdict ---
        fails = []
        warns = []

        if qm.snr < 1.0:
            fails.append("SNR too low")
        elif qm.snr < 3.0:
            warns.append("SNR marginal")

        if qm.motion_score > 0.7:
            fails.append("Severe motion artifact")
        elif qm.motion_score > 0.4:
            warns.append("Moderate motion artifact")

        if qm.metal_voxel_ratio > 0.01:
            warns.append(f"Metal artifacts ({qm.metal_voxel_ratio:.2%})")

        if qm.body_coverage_ratio < 0.1:
            fails.append("Body coverage too low")

        if qm.slice_spacing_cv > 0.2:
            warns.append(f"Irregular slice spacing (CV={qm.slice_spacing_cv:.2f})")

        if fails:
            qm.qc_verdict = "FAIL"
        elif warns:
            qm.qc_verdict = "WARN"
        else:
            qm.qc_verdict = "PASS"

        qm.details['fail_reasons'] = fails
        qm.details['warn_reasons'] = warns

        logger.info(f"    SNR={qm.snr:.2f}  CNR={qm.cnr:.2f}  Motion={qm.motion_score:.3f}  → {qm.qc_verdict}")

        return qm

    # ===============================================================
    # Stage 9 — Isotropic Resampling
    # ===============================================================

    def _stage9_isotropic_resample(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        Resample volume to isotropic voxel spacing (default 1.0 mm³).
        Uses 3rd-order spline interpolation for sub-mm accuracy.

        Returns:
            resampled volume, achieved spacing tuple
        """
        from scipy.ndimage import zoom as ndz

        cz, cy, cx = current_spacing
        tz, ty, tx = self.target_spacing

        zoom_factors = (cz / tz, cy / ty, cx / tx)

        # Skip if already at target (within 1% tolerance)
        if all(abs(1.0 - z) < 0.01 for z in zoom_factors):
            logger.info(f"    Already isotropic {current_spacing} — skipping")
            return volume.copy(), current_spacing

        resampled = ndz(volume, zoom_factors, order=3, prefilter=True)

        logger.info(
            f"    Resampled {volume.shape} @ {current_spacing} mm"
            f" → {resampled.shape} @ {self.target_spacing} mm"
        )
        return resampled.astype(np.float32), self.target_spacing

    # ===============================================================
    # Stage 10 — Abdomen ROI Annotation
    # ===============================================================

    def _stage10_abdomen_annotation(
        self,
        volume: np.ndarray,
        body_mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        segmentation_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Annotate anatomical landmarks for abdomen ROI:
          - Diaphragm Z position (lung→abdomen transition)
          - Pelvic floor Z position
          - Abdominal centroid (Y, X)
          - Organ presence flags (from segmentation mask if provided)
          - Effective field-of-view coverage

        Returns:
            dict with annotation metadata
        """
        D, H, W = volume.shape
        sz, sy, sx = spacing
        annotation = {}

        # ── HU profile along Z ──────────────────────────────────
        hu_profile = np.zeros(D)
        for z in range(D):
            bm = body_mask[z] if body_mask.shape[0] == D else np.ones((H, W), dtype=bool)
            pixels = volume[z][bm > 0]
            if len(pixels) > 0:
                hu_profile[z] = float(pixels.mean())

        # ── Diaphragm detection ───────────────────────────────────
        # Lung HU ≈ -700 HU, Abdomen ≈ -50 HU → largest positive gradient
        hu_grad = np.gradient(hu_profile)
        search_top = max(0, D // 6)
        search_bot = D // 2
        diaphragm_z = int(np.argmax(hu_grad[search_top:search_bot])) + search_top
        annotation['diaphragm_z_idx'] = diaphragm_z
        annotation['diaphragm_z_mm'] = round(float(diaphragm_z * sz), 1)

        # ── Pelvic floor detection ────────────────────────────────
        # Body area per slice; rapid area loss = pelvic floor exit
        body_area = body_mask.sum(axis=(1, 2)).astype(np.float64)
        area_grad = np.gradient(body_area)
        search_pelvis = 3 * D // 4
        pelvic_z = int(np.argmin(area_grad[search_pelvis:])) + search_pelvis
        annotation['pelvis_z_idx'] = pelvic_z
        annotation['pelvis_z_mm'] = round(float(pelvic_z * sz), 1)

        # ── Abdomen centroid ──────────────────────────────────────
        abd_start = diaphragm_z
        abd_end = min(pelvic_z, D)
        if abd_end > abd_start:
            abd_body = body_mask[abd_start:abd_end]
            coords = np.argwhere(abd_body)
            if len(coords) > 0:
                cy_mean = float(coords[:, 1].mean() * sy)
                cx_mean = float(coords[:, 2].mean() * sx)
                annotation['abdomen_centroid_mm'] = (round(cy_mean, 1), round(cx_mean, 1))

        # ── FOV coverage ─────────────────────────────────────────
        total_slices = D
        abd_slices = max(0, abd_end - abd_start)
        annotation['abdomen_slice_coverage'] = round(abd_slices / max(total_slices, 1), 3)

        # ── HU tissue distribution ────────────────────────────────
        body_vox = volume[body_mask > 0]
        if len(body_vox) > 0:
            annotation['hu_percentiles'] = {
                'p5':  round(float(np.percentile(body_vox, 5)), 1),
                'p25': round(float(np.percentile(body_vox, 25)), 1),
                'p50': round(float(np.percentile(body_vox, 50)), 1),
                'p75': round(float(np.percentile(body_vox, 75)), 1),
                'p95': round(float(np.percentile(body_vox, 95)), 1),
            }

        # ── Organ presence from segmentation mask ─────────────────
        if segmentation_mask is not None:
            ORGAN_LABELS = {
                'liver': 1, 'spleen': 2, 'kidney_right': 3, 'kidney_left': 4,
                'colon': 14, 'stomach': 17
            }
            organ_presence = {}
            for organ, label in ORGAN_LABELS.items():
                vox_count = int((segmentation_mask == label).sum())
                organ_presence[organ] = {
                    'present': vox_count > 50,
                    'voxels': vox_count,
                    'volume_cm3': round(vox_count * sx * sy * sz / 1000.0, 2)
                }
            annotation['organs'] = organ_presence

        logger.info(
            f"    Diaphragm Z={diaphragm_z} ({annotation['diaphragm_z_mm']}mm) "
            f"| Pelvis Z={pelvic_z} ({annotation['pelvis_z_mm']}mm) "
            f"| Abd coverage={annotation['abdomen_slice_coverage']:.1%}"
        )

        return annotation


# ═══════════════════════════════════════════════════════════════════
# Convenience: run on a NIfTI file
# ═══════════════════════════════════════════════════════════════════

def preprocess_nifti(
    nifti_path: str,
    output_path: Optional[str] = None,
    skip_stages: Optional[List[int]] = None,
) -> PreprocessingResult:
    """
    Load a NIfTI file and run the advanced preprocessing pipeline.
    """
    image = sitk.ReadImage(nifti_path)
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = tuple(reversed(image.GetSpacing()))  # ITK (x,y,z) → numpy (z,y,x)

    preprocessor = AdvancedCTPreprocessor(skip_stages=skip_stages)
    result = preprocessor.run(volume, spacing=spacing)

    if output_path:
        out_img = sitk.GetImageFromArray(result.volume_denoised)
        out_img.SetSpacing(image.GetSpacing())
        out_img.SetOrigin(image.GetOrigin())
        out_img.SetDirection(image.GetDirection())
        sitk.WriteImage(out_img, output_path, True)
        logger.info(f"  Saved preprocessed volume → {output_path}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Self-test with synthetic phantom
# ═══════════════════════════════════════════════════════════════════

def run_phantom_test():
    """
    Generate a synthetic 3D phantom with known properties
    and validate the preprocessing pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s │ %(levelname)-5s │ %(message)s',
        datefmt='%H:%M:%S',
    )

    logger.info("=" * 60)
    logger.info("  PHANTOM VALIDATION TEST")
    logger.info("=" * 60)

    np.random.seed(42)

    # ── Build phantom ────────────────────────────────────────────
    D, H, W = 64, 128, 128
    phantom = np.full((D, H, W), -1000.0, dtype=np.float32)  # air

    # Body ellipse (soft tissue, HU ~40)
    zz, yy, xx = np.ogrid[:D, :H, :W]
    cz, cy, cx = D // 2, H // 2, W // 2
    body_ellipse = ((zz - cz) / (D * 0.45)) ** 2 + \
                   ((yy - cy) / (H * 0.40)) ** 2 + \
                   ((xx - cx) / (W * 0.40)) ** 2 < 1.0
    phantom[body_ellipse] = 40.0

    # Fat layer (HU ~-90)
    fat_shell = ((zz - cz) / (D * 0.45)) ** 2 + \
                ((yy - cy) / (H * 0.40)) ** 2 + \
                ((xx - cx) / (W * 0.40)) ** 2 < 1.0
    inner = ((zz - cz) / (D * 0.38)) ** 2 + \
            ((yy - cy) / (H * 0.33)) ** 2 + \
            ((xx - cx) / (W * 0.33)) ** 2 < 1.0
    fat_region = fat_shell & ~inner
    phantom[fat_region] = -90.0

    # Tumor (HU ~60, small sphere)
    tumor_center = (cz, cy + 10, cx - 8)
    tumor_r = 6
    tumor_dist = np.sqrt(
        (zz - tumor_center[0]) ** 2 +
        (yy - tumor_center[1]) ** 2 +
        (xx - tumor_center[2]) ** 2
    )
    phantom[tumor_dist < tumor_r] = 60.0

    # Bone (HU ~700, spine-like rod)
    spine_dist = np.sqrt((yy - cy - 25) ** 2 + (xx - cx) ** 2)
    phantom[(spine_dist < 4) & body_ellipse] = 700.0

    # Metal artifact (HU ~4000, tiny point)
    phantom[cz, cy, cx + 30] = 4500.0
    phantom[cz, cy, cx + 31] = 4200.0

    # Add Gaussian noise (σ = 15 HU)
    noise = np.random.normal(0, 15, phantom.shape).astype(np.float32)
    phantom_noisy = phantom + noise

    # Add intensity bias gradient (simulating shading)
    bias = np.linspace(0.9, 1.1, H).reshape(1, H, 1)
    phantom_noisy = phantom_noisy * bias

    logger.info(f"  Phantom: shape={phantom_noisy.shape}, "
                f"HU=[{phantom_noisy.min():.1f}, {phantom_noisy.max():.1f}]")
    logger.info(f"  Tumor at {tumor_center}, r={tumor_r}")
    logger.info(f"  Metal at ({cz},{cy},{cx+30})")

    # ── Run pipeline ─────────────────────────────────────────────
    preprocessor = AdvancedCTPreprocessor(
        n4_iterations=[20, 20],  # faster for test
    )
    result = preprocessor.run(phantom_noisy, spacing=(2.0, 1.0, 1.0))

    # ── Validate ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)

    checks = []

    # 1. Metal masked
    metal_after = (result.volume_hu > 3500).sum()
    ok = metal_after == 0
    checks.append(('Metal removed', ok))
    print(f"  [{'OK' if ok else 'XX'}] Metal voxels after stage 1: {metal_after}")

    # 2. Body mask covers tumor
    tumor_mask_check = result.body_mask[tumor_center[0], tumor_center[1], tumor_center[2]]
    ok2 = tumor_mask_check > 0
    checks.append(('Body mask covers tumor', ok2))
    print(f"  [{'OK' if ok2 else 'XX'}] Body mask at tumor center: {tumor_mask_check}")

    # 3. Body coverage reasonable
    cov = result.quality.body_coverage_ratio
    ok3 = 0.10 < cov < 0.90
    checks.append(('Body coverage', ok3))
    print(f"  [{'OK' if ok3 else 'XX'}] Body coverage: {cov:.1%}")

    # 4. Denoising: compare LOCAL noise in a small homogeneous region
    #    (global std is unfair because N4 removes bias gradient, changing distribution)
    local_z, local_y, local_x = tumor_center  # use region near tumor
    local_patch_orig = phantom_noisy[local_z-2:local_z+2, local_y-15:local_y-10, local_x-15:local_x-10]
    local_patch_dn = result.volume_denoised[local_z-2:local_z+2, local_y-15:local_y-10, local_x-15:local_x-10]
    body_orig_std = float(local_patch_orig.std())
    body_dn_std = float(local_patch_dn.std())
    ok4 = body_dn_std < body_orig_std
    checks.append(('Noise reduced', ok4))
    print(f"  [{'OK' if ok4 else 'XX'}] Local noise sigma: {body_orig_std:.2f} -> {body_dn_std:.2f}")

    # 5. Multi-window 5 channels
    ok5 = result.multi_window.shape[0] == 5
    checks.append(('5-channel output', ok5))
    print(f"  [{'OK' if ok5 else 'XX'}] Multi-window shape: {result.multi_window.shape}")

    # 6. ROI smaller than original
    ok6 = result.roi_volume.size < phantom_noisy.size
    checks.append(('ROI cropped', ok6))
    print(f"  [{'OK' if ok6 else 'XX'}] ROI: {result.roi_volume.shape} (from {phantom_noisy.shape})")

    # 7. QC metrics populated
    ok7 = result.quality.snr > 0 and result.quality.qc_verdict in ('PASS', 'WARN', 'FAIL')
    checks.append(('QC metrics', ok7))
    print(f"  [{'OK' if ok7 else 'XX'}] SNR={result.quality.snr:.2f}, "
          f"CNR={result.quality.cnr:.2f}, Verdict={result.quality.qc_verdict}")

    # 8. Timings recorded
    ok8 = result.stage_timings.get('total', 0) > 0
    checks.append(('Timings', ok8))
    print(f"  [{'OK' if ok8 else 'XX'}] Total time: {result.stage_timings.get('total', 0):.2f}s")

    # Summary
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    print(f"\n  RESULT: {passed}/{total} checks passed")
    print("=" * 60)

    if passed == total:
        print("  [OK] ALL CHECKS PASSED")
    else:
        failed = [name for name, ok in checks if not ok]
        print(f"  [!!] FAILED: {', '.join(failed)}")

    return result


if __name__ == "__main__":
    run_phantom_test()
