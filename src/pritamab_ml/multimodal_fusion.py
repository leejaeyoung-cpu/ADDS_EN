"""
Modal 4: CT Tumor Feature Extractor  +  Multimodal Fusion MLP
--------------------------------------------------------------
CT extractor: 64-dim tumor features (volume, density, RECIST, radiomics proxy)
Fusion MLP:   480-dim (128+256+32+64) → PFS / OS / Synergy predictions
"""

import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Modal 4: CT Feature Extractor ────────────────────────────────────────────

class CTTumorFeatureExtractor:
    """
    64-dim CT tumor feature vector.
    [0:16]   Volume & RECIST metrics
    [16:32]  Density distribution (Hounsfield units histogram)
    [32:48]  Radiomics texture proxies (GLCM, GLRLM surrogates)
    [48:64]  Temporal change features (% change from baseline)
    """
    FEATURE_DIM = 64

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        logger.info("CTTumorFeatureExtractor initialized (feature_dim=64)")

    def simulate_features(self, n_samples: int,
                          pritamab_treated: bool = False,
                          prpc_high: bool = True,
                          kras_allele: str = "G12D",
                          followup_months: float = 3.0) -> np.ndarray:
        """
        Simulate CT radiomics features for n_samples.
        Pritamab effect: ↓ tumor volume, ↓ density heterogeneity, ↑ necrosis
        """
        rng = self.rng
        feats = np.zeros((n_samples, self.FEATURE_DIM), dtype=np.float32)

        # Baseline tumor size (cm) — log-normal
        baseline_diam = rng.lognormal(mean=1.5, sigma=0.5, size=n_samples)
        baseline_vol  = (4/3) * np.pi * (baseline_diam / 2) ** 3

        kras_growth = {"G12D": 1.25, "G12V": 1.18, "G12C": 1.10,
                       "G13D": 1.05, "WT": 0.90}.get(kras_allele, 1.0)
        prpc_growth = 1.20 if prpc_high else 0.95

        # Simulate post-treatment tumor size
        if pritamab_treated:
            shrink = rng.uniform(0.25, 0.65, n_samples)  # 25-65% shrinkage
            if prpc_high:
                shrink += 0.10
            post_diam = baseline_diam * (1 - shrink * prpc_growth)
        else:
            growth   = rng.uniform(-0.10, 0.40, n_samples) * kras_growth
            post_diam = baseline_diam * (1 + growth * followup_months / 3.0)

        post_diam = np.clip(post_diam, 0.1, 20.0)
        post_vol  = (4/3) * np.pi * (post_diam / 2) ** 3

        # [0:16] Volume/RECIST
        feats[:, 0]  = baseline_diam
        feats[:, 1]  = post_diam
        feats[:, 2]  = baseline_vol
        feats[:, 3]  = post_vol
        feats[:, 4]  = (post_diam - baseline_diam) / (baseline_diam + 1e-6)
        feats[:, 5]  = (post_vol  - baseline_vol)  / (baseline_vol  + 1e-6)
        feats[:, 6]  = (feats[:, 4] < -0.30).astype(np.float32)  # PR flag
        feats[:, 7]  = (feats[:, 4] >  0.20).astype(np.float32)  # PD flag
        feats[:, 8]  = followup_months
        feats[:, 9]  = baseline_diam * 2   # Simulated longest axis
        feats[:, 10] = post_diam * 2
        feats[:, 11:16] = rng.standard_normal((n_samples, 5)).astype(np.float32) * 0.2

        # [16:32] HU density distribution (higher HU = more viable tumor)
        baseline_HU = rng.normal(45, 15, n_samples)
        post_HU     = baseline_HU * (post_vol / (baseline_vol + 1e-6)) ** 0.3
        if pritamab_treated:
            post_HU -= rng.uniform(5, 20, n_samples)  # Necrosis ↓ HU
        feats[:, 16] = baseline_HU
        feats[:, 17] = post_HU
        feats[:, 18] = post_HU - baseline_HU
        feats[:, 19] = rng.uniform(0, 1, n_samples).astype(np.float32)  # heterogeneity
        if pritamab_treated:
            feats[:, 19] *= 0.6  # Less heterogeneity after response
        feats[:, 20:32] = rng.standard_normal((n_samples, 12)).astype(np.float32) * 0.3

        # [32:48] Radiomics texture proxies
        feats[:, 32] = rng.uniform(0, 1, n_samples).astype(np.float32)    # GLCM energy
        feats[:, 33] = rng.uniform(0, 5, n_samples).astype(np.float32)    # GLCM entropy
        feats[:, 34:48] = rng.standard_normal((n_samples, 14)).astype(np.float32) * 0.4
        if pritamab_treated:
            feats[:, 33] -= 0.5  # Less entropy (more homogeneous after treatment)

        # [48:64] Temporal change
        feats[:, 48] = feats[:, 4]                    # Δ diameter (%)
        feats[:, 49] = feats[:, 5]                    # Δ volume (%)
        feats[:, 50] = feats[:, 18] / 10              # Δ HU (normalized)
        feats[:, 51] = (feats[:, 4] < 0).astype(np.float32)   # Any shrinkage
        feats[:, 52:64] = rng.standard_normal((n_samples, 12)).astype(np.float32) * 0.2

        return feats.astype(np.float32)


# ── Fusion MLP (NumPy-based, no PyTorch required) ────────────────────────────

class LayerNorm:
    def __init__(self, dim: int, eps=1e-6):
        self.gamma = np.ones(dim,  dtype=np.float32)
        self.beta  = np.zeros(dim, dtype=np.float32)
        self.eps   = eps
    def __call__(self, x):
        mu  = x.mean(-1, keepdims=True)
        std = x.std(-1,  keepdims=True) + self.eps
        return self.gamma * (x - mu) / std + self.beta


class DenseLayer:
    def __init__(self, in_dim: int, out_dim: int, rng):
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
    def __call__(self, x):
        return x @ self.W + self.b


def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))


class PritamamFusionModel:
    """
    Multimodal Fusion MLP (NumPy, CPU-only, no PyTorch dependency).
    
    Architecture:
      Input: 480-dim (128 cell + 256 rna + 32 pkpd + 64 ct)
      Encoder: 480 → 256 → 128 → 64
      Heads:
        PFS:     64 → 32 → 1  (regression, months)
        OS:      64 → 32 → 1  (regression, months)
        Synergy: 64 → 32 → 1  (sigmoid probability)
    """
    INPUT_DIM = 480

    def __init__(self, seed: int = 2026):
        rng = np.random.default_rng(seed)
        # Encoder
        self.enc1  = DenseLayer(480, 256, rng)
        self.ln1   = LayerNorm(256)
        self.enc2  = DenseLayer(256, 128, rng)
        self.ln2   = LayerNorm(128)
        self.enc3  = DenseLayer(128, 64,  rng)
        self.ln3   = LayerNorm(64)
        # PFS head
        self.pfs1  = DenseLayer(64, 32, rng)
        self.pfs2  = DenseLayer(32, 1,  rng)
        # OS head
        self.os1   = DenseLayer(64, 32, rng)
        self.os2   = DenseLayer(32, 1,  rng)
        # Synergy head
        self.syn1  = DenseLayer(64, 32, rng)
        self.syn2  = DenseLayer(32, 1,  rng)
        # Calibration (set after fitting)
        self.pfs_scale = 8.25   # target mean PFS (Pritamab+FOLFOX)
        self.pfs_bias  = 2.0
        self.os_scale  = 17.5
        self.os_bias   = 4.0
        logger.info("PritamamFusionModel initialized (pure NumPy)")

    def _forward_encoder(self, x: np.ndarray) -> np.ndarray:
        h = relu(self.ln1(self.enc1(x)))
        h = relu(self.ln2(self.enc2(h)))
        h = relu(self.ln3(self.enc3(h)))
        return h

    def forward(self, x: np.ndarray) -> dict:
        """
        x: (n, 480)
        Returns: {'pfs': (n,), 'os': (n,), 'synergy_prob': (n,), 'embedding': (n,64)}
        """
        h = self._forward_encoder(x)
        # PFS (softplus activation → always positive)
        pfs_raw = self.pfs2(relu(self.pfs1(h))).squeeze(-1)
        pfs     = np.log1p(np.exp(pfs_raw)) * self.pfs_scale + self.pfs_bias

        # OS
        os_raw  = self.os2(relu(self.os1(h))).squeeze(-1)
        os_     = np.log1p(np.exp(os_raw))  * self.os_scale  + self.os_bias

        # Synergy probability
        syn_raw = self.syn2(relu(self.syn1(h))).squeeze(-1)
        synprob = sigmoid(syn_raw)

        return {"pfs": pfs, "os": os_, "synergy_prob": synprob, "embedding": h}

    def calibrate(self, target_pfs_mean: float, target_os_mean: float,
                  target_synergy_rate: float,
                  x_cal: np.ndarray):
        """Calibrate output scales against known targets."""
        out = self.forward(x_cal)
        raw_pfs_mean = np.log1p(np.exp(out["pfs"] / self.pfs_scale)).mean()
        raw_os_mean  = np.log1p(np.exp(out["os"]  / self.os_scale)).mean()
        if raw_pfs_mean > 0:
            self.pfs_scale = target_pfs_mean / max(raw_pfs_mean, 0.1)
        if raw_os_mean > 0:
            self.os_scale  = target_os_mean  / max(raw_os_mean, 0.1)
        logger.info(f"Calibrated: pfs_scale={self.pfs_scale:.2f}, "
                    f"os_scale={self.os_scale:.2f}")


if __name__ == "__main__":
    ct = CTTumorFeatureExtractor()
    feats = ct.simulate_features(10, pritamab_treated=True, prpc_high=True)
    print(f"CT feats: {feats.shape}")

    model = PritamamFusionModel()
    X = np.random.randn(10, 480).astype(np.float32)
    out = model.forward(X)
    print(f"PFS: {out['pfs'].round(2)}")
    print(f"OS:  {out['os'].round(2)}")
    print(f"Syn: {out['synergy_prob'].round(3)}")
