"""
Modal 1: Cellpose Cell Image Feature Extractor
----------------------------------------------
Extracts 128-dim feature vector from Cellpose-processed cell images.
Uses morphological features: cell density, viability proxy, size distribution,
circularity, and intensity statistics.
"""

import numpy as np
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CellposeFeatureExtractor:
    """
    Extracts interpretable features from Cellpose cell segmentation output.
    Simulates deep CNN embeddings when actual image files not available.
    
    Feature vector (128-dim):
        [0:16]   Cell density / confluence metrics
        [16:32]  Viability proxy (area, perimeter stats)
        [32:48]  Morphology (circularity, elongation, solidity)
        [48:64]  Intensity statistics (mean, std, skew per channel)
        [64:80]  Spatial distribution (nearest-neighbor distances)
        [80:96]  Treatment response markers (PrPc expression proxy)
        [96:112] Apoptosis markers (nuclear fragmentation proxy)
        [112:128] Interaction features (cell-cell contact)
    """
    FEATURE_DIM = 128

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        logger.info("CellposeFeatureExtractor initialized (feature_dim=128)")

    def extract_from_mask(self, mask: np.ndarray,
                          intensity: np.ndarray = None) -> np.ndarray:
        """
        Extract features from a real Cellpose segmentation mask.
        mask: 2D array of cell labels (0=background, i=cell_i)
        intensity: optional 2D or 3D intensity image
        """
        n_cells = int(mask.max())
        if n_cells == 0:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        feats = np.zeros(self.FEATURE_DIM, dtype=np.float32)
        areas, perims, circs = [], [], []
        for i in range(1, n_cells + 1):
            cell_pixels = (mask == i)
            area = cell_pixels.sum()
            areas.append(area)
            # Approximate perimeter
            from scipy.ndimage import binary_erosion
            interior = binary_erosion(cell_pixels)
            perim = (cell_pixels & ~interior).sum()
            perims.append(perim)
            circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
            circs.append(circ)

        areas  = np.array(areas,  dtype=np.float32)
        perims = np.array(perims, dtype=np.float32)
        circs  = np.array(circs,  dtype=np.float32)

        total_pix = mask.size
        # [0:16] Density
        feats[0]  = n_cells / (total_pix / 1e4)
        feats[1]  = areas.mean()
        feats[2]  = areas.std()
        feats[3]  = np.percentile(areas, 25)
        feats[4]  = np.percentile(areas, 75)
        # [16:32] Viability (large cells = healthy)
        feats[16] = (areas > areas.mean()).mean()
        feats[17] = perims.mean()
        feats[18] = perims.std()
        # [32:48] Morphology
        feats[32] = circs.mean()
        feats[33] = circs.std()
        feats[34] = (circs > 0.7).mean()  # fraction of round cells
        # [48:64] Intensity (if available)
        if intensity is not None:
            for c_idx in range(min(4, intensity.shape[-1] if intensity.ndim == 3 else 1)):
                ch = intensity[..., c_idx] if intensity.ndim == 3 else intensity
                feats[48 + c_idx * 4 + 0] = ch[mask > 0].mean()
                feats[48 + c_idx * 4 + 1] = ch[mask > 0].std()
                feats[48 + c_idx * 4 + 2] = ch[mask > 0].max()
                feats[48 + c_idx * 4 + 3] = (ch[mask > 0] > ch[mask > 0].mean()).mean()

        return feats

    def simulate_features(self, n_samples: int,
                          pritamab_treated: bool = False,
                          prpc_high: bool = True,
                          kras_allele: str = "G12D",
                          chemo_drug: str = "FOLFOX",
                          concentration_nM: float = 10.0) -> np.ndarray:
        """
        Simulate 128-dim cell image features for virtual patients.
        
        Pritamab effect:
          - ↑ cell circularity (apoptotic morphology)
          - ↓ cell density (fewer surviving cells)
          - ↑ nuclear fragmentation proxy
          - ↓ PrPc intensity proxy (antibody neutralisation)
        """
        rng = self.rng
        X = rng.standard_normal((n_samples, self.FEATURE_DIM)).astype(np.float32)

        # Base modifiers by KRAS allele
        kras_aggression = {"G12D": 1.20, "G12V": 1.15, "G12C": 1.10,
                           "G13D": 1.05, "WT":   0.90}.get(kras_allele, 1.0)
        prpc_mod = 1.3 if prpc_high else 0.9

        # Cell density [0:4] — higher KRAS → denser tumour cells
        X[:, 0:4] += kras_aggression * prpc_mod * 0.5

        # Pritamab effect
        if pritamab_treated:
            # ↓ density
            X[:, 0:4]   -= 1.8 * prpc_mod
            # ↑ circularity (apoptotic rounding)
            X[:, 32:48] += 0.6
            # ↑ nuclear fragmentation
            X[:, 96:112] += 1.4 * prpc_mod
            # ↓ PrPc proxy [80:96]
            X[:, 80:96] -= 1.2 * prpc_mod
            # ↓ cell-cell contact [112:128]
            X[:, 112:128] -= 0.8

        # Chemo effect
        chemo_intensity = {"FOLFOX": 1.1, "FOLFIRI": 1.05,
                           "FOLFOXIRI": 1.25, "TAS-102": 0.9}.get(chemo_drug, 1.0)
        conc_factor = np.log1p(concentration_nM / 10.0)
        X[:, 0:4]    -= chemo_intensity * conc_factor * 0.4
        X[:, 96:112] += chemo_intensity * conc_factor * 0.5

        # Clip to reasonable range
        X = np.clip(X, -4.0, 4.0)
        return X


if __name__ == "__main__":
    ext = CellposeFeatureExtractor()
    feats = ext.simulate_features(10, pritamab_treated=True, prpc_high=True)
    print(f"Feature shape: {feats.shape}, mean: {feats.mean():.4f}")
