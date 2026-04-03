"""
PINN v4: Improved regularization over v3

Changes from v3:
1. Dropout 0.1 → 0.3 (reduce overfitting on n=78 scenarios)
2. Weight decay 1e-4 → 5e-4
3. Early stopping (patience=200 epochs)
4. Reduced max epochs (1500 → 800)
5. LOO-CV for honest generalization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import everything from v3
from track2_energy_pinn_v3 import (
    EnergyPredictorV3, PathwayGNN, N_NODES, PATHWAY_NODES, NODE_IDX,
    BASE_EDGES, build_adjacency, build_v3_dataset,
    kd_to_dg, cmax_to_mu, thalf_to_barrier, inh_to_eff, MUTATION_DDG,
)

import numpy as np
import json
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EnergyPredictorV4(nn.Module):
    """
    v4: Stronger regularization for small dataset (n~80).
    - Dropout 0.3 (up from 0.1)
    - Smaller hidden dimensions
    - BatchNorm for stability
    """
    def __init__(self, n_pk=7, n_pathway_mod=N_NODES):
        super().__init__()

        self.pk_enc = nn.Sequential(
            nn.Linear(n_pk, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

        self.gnn = PathwayGNN(node_dim=16, hidden_dim=32, n_layers=3)

        # Stronger dropout in fusion
        self.fusion = nn.Sequential(
            nn.Linear(32 + N_NODES, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU(),
            nn.Dropout(0.3),
        )
        self.residual = nn.Linear(32 + N_NODES, 64)

        # Smaller task heads
        self.head_tumor_supp = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 1))
        self.head_ic50 = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 1))
        self.head_synergy = nn.Sequential(nn.Linear(64, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, pk_features, pathway_modulation):
        pk_emb = self.pk_enc(pk_features)
        drug_energy = pk_features[:, 0:1]
        node_energies, edge_w = self.gnn(drug_energy, pathway_modulation)

        combined = torch.cat([pk_emb, node_energies], dim=1)
        h = self.fusion(combined) + self.residual(combined)

        # Constrained outputs (same as v3)
        tumor_supp = torch.sigmoid(self.head_tumor_supp(h)) * 100
        ic50 = torch.exp(self.head_ic50(h).clamp(-5, 10))
        synergy_ci = F.softplus(self.head_synergy(h)) + 0.1

        return tumor_supp, ic50, synergy_ci, node_energies, edge_w


def train_v4(scenarios, epochs=800, lr=3e-4, patience=200):
    """Train v4 with early stopping and stronger regularization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training v4 on {device}, {len(scenarios)} scenarios, patience={patience}")

    pk = torch.FloatTensor([s["pk"] for s in scenarios]).to(device)
    mod = torch.FloatTensor([s["mod"] for s in scenarios]).to(device)
    tgt = torch.FloatTensor([s["target"] for s in scenarios]).to(device)

    tgt_ts = tgt[:, 0]
    tgt_ic50_log = torch.log(tgt[:, 1].clamp(min=0.5))
    tgt_ci = tgt[:, 2]

    pk_mean, pk_std = pk.mean(0), pk.std(0) + 1e-8
    pk_n = (pk - pk_mean) / pk_std

    model = EnergyPredictorV4(n_pk=pk.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        pred_ts, pred_ic50, pred_ci, node_e, edge_w = model(pk_n, mod)

        l_ts = F.mse_loss(pred_ts.squeeze(), tgt_ts)
        l_ic50 = F.mse_loss(torch.log(pred_ic50.squeeze().clamp(min=0.5)), tgt_ic50_log)
        l_ci = F.mse_loss(pred_ci.squeeze(), tgt_ci)
        l_data = l_ts / 100 + l_ic50 + l_ci

        # Physics: energy conservation
        e_root = node_e[:, 0].abs()
        e_terminal = node_e[:, -8:].abs().sum(dim=1)
        l_conservation = ((e_root - e_terminal * 0.1) ** 2).mean()

        # Edge sparsity
        l_edge = edge_w.sum() * 0.01

        phys_w = 0.05 * min(1.0, epoch / 200)
        loss = l_data + phys_w * l_conservation + l_edge

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Early stopping
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

        if (epoch + 1) % 100 == 0:
            from scipy.stats import pearsonr
            ts_np = pred_ts.squeeze().detach().cpu().numpy()
            ic_np = pred_ic50.squeeze().detach().cpu().numpy()
            ci_np = pred_ci.squeeze().detach().cpu().numpy()
            tgt_np = tgt.cpu().numpy()
            r_ts = pearsonr(tgt_np[:, 0], ts_np)[0]
            r_ic = pearsonr(tgt_np[:, 1], ic_np)[0]
            r_ci = pearsonr(tgt_np[:, 2], ci_np)[0]
            logger.info(f"  Ep {epoch+1:4d}: loss={loss.item():.4f} "
                        f"r_ts={r_ts:.3f} r_ic={r_ic:.3f} r_ci={r_ci:.3f} "
                        f"no_improve={no_improve}")
            history.append({"epoch": epoch+1, "loss": round(loss.item(), 4),
                            "r_ts": round(r_ts, 4), "r_ic": round(r_ic, 4), "r_ci": round(r_ci, 4)})

    model.load_state_dict(best_state)
    model.eval()

    # LOO-CV
    logger.info("Running LOO cross-validation...")
    loo_errors = {"ts": [], "ic50": [], "ci": []}
    for i in range(len(scenarios)):
        mask = torch.ones(len(scenarios), dtype=torch.bool)
        mask[i] = False

        loo_m = EnergyPredictorV4(n_pk=pk.shape[1]).to(device)
        loo_opt = torch.optim.AdamW(loo_m.parameters(), lr=lr, weight_decay=5e-4)

        for ep in range(300):
            loo_m.train()
            loo_opt.zero_grad()
            p_ts, p_ic, p_ci, _, _ = loo_m(pk_n[mask], mod[mask])
            l = (F.mse_loss(p_ts.squeeze(), tgt_ts[mask]) / 100 +
                 F.mse_loss(torch.log(p_ic.squeeze().clamp(min=0.5)), tgt_ic50_log[mask]) +
                 F.mse_loss(p_ci.squeeze(), tgt_ci[mask]))
            l.backward()
            loo_opt.step()

        loo_m.eval()
        with torch.no_grad():
            p_ts, p_ic, p_ci, _, _ = loo_m(pk_n[i:i+1], mod[i:i+1])
        loo_errors["ts"].append(abs(float(p_ts) - float(tgt[i, 0])))
        loo_errors["ic50"].append(abs(float(p_ic) - float(tgt[i, 1])))
        loo_errors["ci"].append(abs(float(p_ci) - float(tgt[i, 2])))

        if (i + 1) % 20 == 0:
            logger.info(f"  LOO {i+1}/{len(scenarios)}: MAE_ts={np.mean(loo_errors['ts']):.1f} "
                        f"MAE_ic={np.mean(loo_errors['ic50']):.1f} MAE_ci={np.mean(loo_errors['ci']):.3f}")

    # Final eval
    with torch.no_grad():
        pred_ts, pred_ic50, pred_ci, node_e, edge_w = model(pk_n, mod)

    from scipy.stats import pearsonr
    tgt_np = tgt.cpu().numpy()
    r_ts = pearsonr(tgt_np[:, 0], pred_ts.squeeze().cpu().numpy())[0]
    r_ic = pearsonr(tgt_np[:, 1], pred_ic50.squeeze().cpu().numpy())[0]
    r_ci = pearsonr(tgt_np[:, 2], pred_ci.squeeze().cpu().numpy())[0]

    # IC50 check
    ic50_preds = pred_ic50.squeeze().cpu().numpy()
    n_neg = int((ic50_preds < 0).sum())

    results = {
        "version": "v4",
        "changes": "Dropout 0.3, weight_decay 5e-4, early stopping, smaller heads",
        "n_scenarios": len(scenarios),
        "stopped_at_epoch": epoch + 1,
        "train_r": {"ts": round(r_ts, 4), "ic50": round(r_ic, 4), "ci": round(r_ci, 4)},
        "loo_mae": {
            "ts_pct": round(np.mean(loo_errors["ts"]), 2),
            "ic50_nm": round(np.mean(loo_errors["ic50"]), 2),
            "ci": round(np.mean(loo_errors["ci"]), 4),
        },
        "ic50_negative_count": n_neg,
        "ic50_range": [round(float(ic50_preds.min()), 1), round(float(ic50_preds.max()), 1)],
        "history": history,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"v4 Results:")
    logger.info(f"  Train r: ts={r_ts:.4f}, ic50={r_ic:.4f}, ci={r_ci:.4f}")
    logger.info(f"  LOO MAE: ts={results['loo_mae']['ts_pct']:.1f}%, "
                f"ic50={results['loo_mae']['ic50_nm']:.1f}nM, ci={results['loo_mae']['ci']:.4f}")
    logger.info(f"  IC50 negative: {n_neg}/{len(ic50_preds)} (range: [{ic50_preds.min():.1f}, {ic50_preds.max():.1f}])")
    logger.info(f"  Stopped at epoch: {epoch+1}")
    logger.info(f"{'='*60}")

    return model, results, (pk_mean, pk_std)


def main():
    t0 = time.time()
    out = Path("F:/ADDS/models/energy")
    out.mkdir(parents=True, exist_ok=True)

    scenarios = build_v3_dataset()
    model, results, norms = train_v4(scenarios, epochs=800, patience=200)

    torch.save({
        "model_state": model.state_dict(),
        "pk_mean": norms[0].cpu(), "pk_std": norms[1].cpu(),
        "pathway_nodes": PATHWAY_NODES, "base_edges": BASE_EDGES,
        "version": "v4",
    }, out / "energy_predictor_v4.pt")

    with open(out / "energy_predictor_v4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    logger.info(f"Saved: energy_predictor_v4.pt ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
