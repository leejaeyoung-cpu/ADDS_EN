"""
DeepSynergy Inference Module
Loads trained model and provides prediction API for drug synergy.
"""

import sys, json, logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import features from training script
from scripts.train_deep_synergy import (
    DeepSynergyModel, DRUG_SMILES, CELL_LINE_FEATURES, DRUG_CLASSES,
    get_drug_features, get_cell_features, KNOWN_SYNERGY_RULES,
)


class DeepSynergyPredictor:
    """Production inference for DeepSynergy drug combo predictions."""

    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = model_path or Path("F:/ADDS/models/deep_synergy/deep_synergy_v1.pt")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        input_dim = ckpt['input_dim']
        hidden_dims = ckpt.get('hidden_dims', [2048, 1024, 512, 128])

        self.model = DeepSynergyModel(input_dim=input_dim, hidden_dims=hidden_dims)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()

        self.test_results = ckpt.get('test_results', {})
        self.crc_results = ckpt.get('crc_finetune_results', ckpt.get('crc_results', {}))
        self.train_size = ckpt.get('train_size', 0)
        self.timestamp = ckpt.get('timestamp', 'unknown')

        # Cache drug features
        self._drug_cache = {}
        logger.info(f"DeepSynergy loaded: {input_dim}-dim, "
                    f"test_r={self.test_results.get('pearson_r', 'N/A')}, "
                    f"crc_r={self.crc_results.get('pearson_r', 'N/A')}")

    def _get_drug_feat(self, drug_name):
        if drug_name not in self._drug_cache:
            self._drug_cache[drug_name] = get_drug_features(drug_name)
        return self._drug_cache[drug_name]

    def predict_synergy(self, drug_a, drug_b, cell_line="HCT116"):
        """
        Predict synergy score for a drug combination.

        Returns dict with synergy_score, classification, confidence, etc.
        """
        feat_a = self._get_drug_feat(drug_a)
        feat_b = self._get_drug_feat(drug_b)
        feat_cell = get_cell_features(cell_line)
        x = np.concatenate([feat_a, feat_b, feat_cell]).astype(np.float32)

        with torch.no_grad():
            t = torch.from_numpy(x).unsqueeze(0).to(self.device)
            score = self.model(t).item()

        # Competitive binding correction (from energy framework)
        class_a = DRUG_CLASSES.get(drug_a, "unknown")
        class_b = DRUG_CLASSES.get(drug_b, "unknown")
        competitive_penalty = 0
        if class_a == class_b and class_a in ("egfr_inhib", "immune_ckpt", "antimetabolite"):
            competitive_penalty = -5.0
            score = min(score, score + competitive_penalty)

        # Classification
        if score > 5:
            classification = "synergistic"
        elif score < -5:
            classification = "antagonistic"
        else:
            classification = "additive"

        # Confidence based on distance from thresholds
        if classification == "synergistic":
            confidence = min(0.95, 0.5 + (score - 5) * 0.05)
        elif classification == "antagonistic":
            confidence = min(0.95, 0.5 + (-5 - score) * 0.05)
        else:
            dist = min(abs(score - 5), abs(score + 5))
            confidence = min(0.95, 0.3 + dist * 0.05)

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "cell_line": cell_line,
            "synergy_score": round(score, 2),
            "classification": classification,
            "confidence": round(confidence, 3),
            "competitive_binding_penalty": competitive_penalty,
            "drug_a_class": class_a,
            "drug_b_class": class_b,
            "drug_a_has_smiles": DRUG_SMILES.get(drug_a) is not None,
            "drug_b_has_smiles": DRUG_SMILES.get(drug_b) is not None,
            "model_version": "DeepSynergy_v1",
            "model_metrics": {
                "training_data": self.train_size,
                "test_pearson_r": self.test_results.get('pearson_r'),
                "crc_pearson_r": self.crc_results.get('pearson_r'),
            },
        }

    def predict_batch(self, drug_pairs, cell_line="HCT116"):
        """Predict synergy for multiple drug pairs."""
        results = []
        for drug_a, drug_b in drug_pairs:
            results.append(self.predict_synergy(drug_a, drug_b, cell_line))
        return results

    def get_available_drugs(self):
        """Return available drugs grouped by class."""
        by_class = {}
        for drug, cls in DRUG_CLASSES.items():
            by_class.setdefault(cls, []).append({
                "name": drug,
                "has_smiles": DRUG_SMILES.get(drug) is not None,
            })
        return by_class

    def get_available_cell_lines(self):
        """Return available cell lines."""
        return list(CELL_LINE_FEATURES.keys())


# Singleton
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DeepSynergyPredictor()
    return _predictor


if __name__ == "__main__":
    pred = get_predictor()

    # Test predictions
    tests = [
        ("5-Fluorouracil", "Oxaliplatin", "HCT116"),   # FOLFOX — synergistic
        ("Cetuximab", "Panitumumab", "HCT116"),          # Same target — antagonistic
        ("Vemurafenib", "Trametinib", "HT29"),           # BRAF+MEK — synergistic
        ("Pembrolizumab", "Nivolumab", "DLD-1"),         # Same target — antagonistic
        ("Irinotecan", "Cetuximab", "HCT116"),           # FOLFIRI-like — synergistic
    ]

    print("=" * 70)
    print("DeepSynergy Predictions")
    print("=" * 70)
    for a, b, cell in tests:
        r = pred.predict_synergy(a, b, cell)
        print(f"  {a:20s} + {b:20s} [{cell}]: "
              f"score={r['synergy_score']:+6.1f}  "
              f"{r['classification']:13s} "
              f"(conf={r['confidence']:.2f})"
              f"{'  COMPETITIVE' if r['competitive_binding_penalty'] else ''}")
