"""
Track 2: Pritamab PK/PD Energy Framework + PINN
=================================================
Converts PK/PD parameters to thermodynamic energy landscape
and trains a Physics-Informed Neural Network (PINN) with
energy conservation constraints.

Scientific Basis:
- ΔG = RT ln(KD) — direct thermodynamic law
- ΔG‡ from Eyring equation — kinetic barrier
- μ = μ° + RT ln(C/C°) — chemical potential
- Energy conservation: ΣΔG_in + ΣΔG_ATP = ΣΔG_out + ΣΔG_dissipated
"""

import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==========================================================================
# Physical Constants
# ==========================================================================
R_cal = 1.987e-3   # kcal/(mol·K) — gas constant
R_kJ = 8.314e-3    # kJ/(mol·K)
T_body = 310.15     # K (37°C)
RT_cal = R_cal * T_body  # 0.616 kcal/mol
RT_kJ = R_kJ * T_body    # 2.577 kJ/mol
kB = 1.381e-23      # J/K — Boltzmann constant
h = 6.626e-34        # J·s — Planck constant
kBT_h = kB * T_body / h  # ~6.45e12 s⁻¹ — Eyring prefactor
ATP_HYDROLYSIS = -7.3  # kcal/mol — free energy of ATP → ADP + Pi


# ==========================================================================
# Energy Conversion Functions (Thermodynamically Rigorous)
# ==========================================================================

def kd_to_dg(kd_nm: float) -> float:
    """
    Convert dissociation constant (KD) to binding free energy (ΔG).
    ΔG = RT ln(KD)

    This is an EXACT thermodynamic relationship.

    Args:
        kd_nm: KD in nanomolar (nM)
    Returns:
        ΔG in kcal/mol (negative = favorable binding)
    """
    kd_molar = kd_nm * 1e-9  # nM → M
    return RT_cal * np.log(kd_molar)  # kcal/mol


def cmax_to_chemical_potential(cmax_ugml: float, mw_daltons: float,
                                protein_binding_frac: float = 0.0) -> float:
    """
    Convert Cmax to chemical potential (μ).
    μ = μ° + RT ln(C_free / C°)  where C° = 1 M (standard state)

    Args:
        cmax_ugml: Maximum plasma concentration in μg/mL
        mw_daltons: Molecular weight in Daltons
        protein_binding_frac: Fraction bound to plasma proteins (0-1)
    Returns:
        Chemical potential contribution (Δμ) in kcal/mol
    """
    c_total_molar = (cmax_ugml * 1e-6) / (mw_daltons * 1e-3)  # μg/mL → mol/L
    c_free = c_total_molar * (1 - protein_binding_frac)
    if c_free <= 0:
        return 0.0
    return RT_cal * np.log(c_free)  # Relative to 1 M standard


def t_half_to_kinetic_barrier(t_half_min: float) -> float:
    """
    Convert binding half-life to kinetic barrier (ΔG‡) via Eyring equation.
    k_off = ln(2) / t_1/2
    k_off = (kBT/h) × exp(-ΔG‡/RT)
    ΔG‡ = -RT ln(k_off × h / (kBT))

    Args:
        t_half_min: Binding half-life in minutes
    Returns:
        Kinetic barrier ΔG‡ in kcal/mol
    """
    t_half_s = t_half_min * 60
    k_off = np.log(2) / t_half_s  # s⁻¹
    # ΔG‡ = RT ln(kBT/h) - RT ln(k_off)
    dg_barrier = RT_cal * (np.log(kBT_h) - np.log(k_off))  # kcal/mol
    return dg_barrier


def inhibition_to_energy_transfer(inhibition_pct: float) -> float:
    """
    Convert pathway inhibition percentage to residual energy transfer efficiency.
    If 85% inhibition, only 15% of upstream energy propagates.
    """
    return 1.0 - (inhibition_pct / 100.0)


def mutation_ddg(mutation: str) -> float:
    """
    Estimate ΔΔG for known mutations from literature.
    Positive ΔΔG = destabilizing (loss of inhibition).
    """
    # Literature values for common oncogenic mutations
    mutation_db = {
        # KRAS mutations — constitutively active, bypass upstream
        "KRAS_G12D": 2.5,   # kcal/mol — strong gain of function
        "KRAS_G12V": 2.3,
        "KRAS_G13D": 1.8,   # Moderate gain of function
        "KRAS_G12C": 2.0,
        "KRAS_Q61H": 2.7,

        # BRAF
        "BRAF_V600E": 3.1,  # Strong constitutive activation

        # PIK3CA
        "PIK3CA_E545K": 1.5,
        "PIK3CA_H1047R": 2.0,

        # EGFR
        "EGFR_T790M": 1.2,  # Resistance mutation
        "EGFR_L858R": 1.8,  # Activating

        # TP53
        "TP53_R175H": -2.0,  # Loss of function (negative = loss of tumor suppression)
        "TP53_R273H": -1.8,
    }
    return mutation_db.get(mutation.upper(), 0.0)


# ==========================================================================
# Pathway Energy Graph
# ==========================================================================

@dataclass
class PathwayNode:
    """Node in the energy propagation graph."""
    name: str
    energy_state: float = 0.0       # Current energy (kcal/mol)
    atp_consumed: float = 0.0       # ATP consumed at this step
    is_mutant: bool = False
    mutation_ddg: float = 0.0       # ΔΔG from mutation
    downstream: List[Tuple[str, float]] = field(default_factory=list)
    # (target_node, transfer_efficiency)


def build_pritamab_pathway_graph(
    dg_binding: float,
    cmet_inhibition: float = 0.85,
    egfr_inhibition: float = 0.75,
    lamr_inhibition: float = 0.70,
    mutations: Optional[Dict[str, str]] = None
) -> Dict[str, PathwayNode]:
    """
    Build energy propagation graph for Pritamab anti-PrPC antibody.

    Based on Co-IP data:
    - PrPC binds c-MET, EGFR, Laminin receptor simultaneously
    - Pritamab blocks these interactions with known inhibition %
    """
    mutations = mutations or {}

    # Root node: Pritamab-PrPC binding
    graph = {
        "PrPC_Pritamab": PathwayNode(
            name="PrPC-Pritamab Binding",
            energy_state=dg_binding,
            downstream=[
                ("c-MET", inhibition_to_energy_transfer(cmet_inhibition)),
                ("EGFR", inhibition_to_energy_transfer(egfr_inhibition)),
                ("LamR", inhibition_to_energy_transfer(lamr_inhibition)),
            ]
        ),

        # First-tier targets (directly bound by PrPC)
        "c-MET": PathwayNode(
            name="c-MET / HGF Pathway",
            atp_consumed=ATP_HYDROLYSIS,
            downstream=[("RAS", 0.7), ("PI3K", 0.3)]
        ),
        "EGFR": PathwayNode(
            name="EGFR / ErbB Pathway",
            atp_consumed=ATP_HYDROLYSIS,
            downstream=[("RAS", 0.8), ("JAK_STAT", 0.2)]
        ),
        "LamR": PathwayNode(
            name="Laminin Receptor / Adhesion",
            atp_consumed=ATP_HYDROLYSIS,
            downstream=[("PI3K", 0.5), ("FAK", 0.5)]
        ),

        # Second-tier: Signaling cascades
        "RAS": PathwayNode(
            name="RAS/MAPK Cascade",
            atp_consumed=ATP_HYDROLYSIS * 3,  # RAF→MEK→ERK chain
            downstream=[("proliferation", 0.8), ("survival", 0.2)]
        ),
        "PI3K": PathwayNode(
            name="PI3K/AKT/mTOR",
            atp_consumed=ATP_HYDROLYSIS * 2,
            downstream=[("survival", 0.7), ("metabolism", 0.3)]
        ),
        "JAK_STAT": PathwayNode(
            name="JAK/STAT Signaling",
            atp_consumed=ATP_HYDROLYSIS,
            downstream=[("proliferation", 0.6), ("immune_evasion", 0.4)]
        ),
        "FAK": PathwayNode(
            name="FAK/Adhesion Signaling",
            atp_consumed=ATP_HYDROLYSIS,
            downstream=[("migration", 0.7), ("survival", 0.3)]
        ),

        # Terminal phenotypic outcomes
        "proliferation": PathwayNode(name="Cell Proliferation"),
        "survival": PathwayNode(name="Cell Survival / Anti-apoptosis"),
        "migration": PathwayNode(name="Cell Migration / Metastasis"),
        "metabolism": PathwayNode(name="Metabolic Reprogramming"),
        "immune_evasion": PathwayNode(name="Immune Evasion"),
    }

    # Apply mutations
    for node_name, mut in mutations.items():
        if node_name in graph:
            ddg = mutation_ddg(mut)
            graph[node_name].is_mutant = True
            graph[node_name].mutation_ddg = ddg

    return graph


def propagate_energy(graph: Dict[str, PathwayNode], start: str = "PrPC_Pritamab") -> Dict[str, float]:
    """
    Propagate energy through the pathway graph.
    Accounts for: upstream energy, ATP input, transfer efficiency, mutation effects.

    Returns dict of {node_name: accumulated_energy}
    """
    energies = {k: 0.0 for k in graph}
    energies[start] = graph[start].energy_state

    visited = set()
    queue = [start]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        node = graph[current]
        incoming = energies[current]

        for target, efficiency in node.downstream:
            if target not in graph:
                continue

            target_node = graph[target]

            # Energy at target = upstream × efficiency + ATP + mutation correction
            transferred = incoming * efficiency
            atp = target_node.atp_consumed  # Negative (ATP provides energy)
            ddg = target_node.mutation_ddg

            # Mutation ΔΔG: positive = bypass (constitutive activation)
            # → reduces the effective blocking = adds energy to pathway
            energies[target] += transferred + atp + ddg
            queue.append(target)

    return energies


def check_energy_conservation(energies: Dict[str, float],
                             graph: Dict[str, PathwayNode]) -> Dict[str, float]:
    """
    Check energy conservation: ΣΔG_in + ΣΔG_ATP = ΣΔG_out + ΣΔG_dissipated

    Returns balance dict for PINN constraint.
    """
    # Input energy (root binding)
    e_input = abs(energies.get("PrPC_Pritamab", 0))

    # Total ATP consumed
    e_atp = sum(abs(n.atp_consumed) for n in graph.values())

    # Output energy (terminal nodes)
    terminal = [k for k, n in graph.items() if not n.downstream]
    e_output = sum(abs(energies.get(k, 0)) for k in terminal)

    # Dissipated (non-terminal, non-root)
    non_terminal = [k for k in graph if k != "PrPC_Pritamab" and k not in terminal]
    e_dissipated = sum(abs(energies.get(k, 0)) - sum(
        abs(energies.get(t, 0)) * eff for t, eff in graph[k].downstream
    ) for k in non_terminal if graph[k].downstream)

    balance = {
        "E_input_binding": round(e_input, 3),
        "E_ATP_total": round(e_atp, 3),
        "E_output_phenotypic": round(e_output, 3),
        "E_dissipated": round(abs(e_dissipated), 3),
        "conservation_residual": round(
            (e_input + e_atp) - (e_output + abs(e_dissipated)), 3
        ),
    }
    return balance


# ==========================================================================
# PINN: Physics-Informed Neural Network
# ==========================================================================

import torch
import torch.nn as nn


class PritamabPINN(nn.Module):
    """
    Physics-Informed Neural Network for Pritamab energy prediction.

    Input: PK/PD parameters in energy units
    Output: Phenotypic outcomes (tumor suppression, IC50, synergy)

    Physics constraint: energy conservation in loss function.
    """
    def __init__(self, n_pk_features=5, n_pathway_features=4, n_outputs=3):
        super().__init__()

        # PK energy encoder
        self.pk_encoder = nn.Sequential(
            nn.Linear(n_pk_features, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )

        # Pathway energy encoder
        self.pathway_encoder = nn.Sequential(
            nn.Linear(n_pathway_features, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )

        # Fusion → prediction
        self.predictor = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, n_outputs),
        )

        # Energy conservation layer (learnable dissipation terms)
        self.dissipation_factors = nn.Parameter(torch.ones(n_pathway_features) * 0.1)

    def forward(self, pk_energy, pathway_energy):
        """
        Args:
            pk_energy: [batch, 5] — ΔG_binding, μ_cmax, ΔG‡_kinetic, receptor_density, ΔΔG_mutation
            pathway_energy: [batch, 4] — c-MET, EGFR, LamR, cross-talk efficiencies
        Returns:
            predictions: [batch, 3] — tumor_suppression(%), IC50(nM), synergy_CI
            energy_balance: [batch, 1] — conservation residual (should be minimized)
        """
        pk_emb = self.pk_encoder(pk_energy)
        pw_emb = self.pathway_encoder(pathway_energy)

        combined = torch.cat([pk_emb, pw_emb], dim=1)
        predictions = self.predictor(combined)

        # Energy conservation constraint
        total_input = pk_energy[:, 0].abs()  # |ΔG_binding|
        total_pathway = (pathway_energy * self.dissipation_factors.abs()).sum(dim=1)
        energy_balance = total_input - total_pathway  # Should → 0

        return predictions, energy_balance


def pinn_loss(pred, target, energy_balance, lambda_physics=0.1):
    """
    Combined loss: data fidelity + physics constraint.
    L = L_data + λ × L_physics
    where L_physics = ||energy_balance||²
    """
    l_data = nn.functional.mse_loss(pred, target)
    l_physics = (energy_balance ** 2).mean()
    return l_data + lambda_physics * l_physics, l_data.item(), l_physics.item()


# ==========================================================================
# Pilot Dataset: Pritamab
# ==========================================================================

def build_pritamab_pilot_dataset():
    """
    Build pilot dataset from known Pritamab data.

    Sources:
    - KD = 0.1-0.5 nM (SPR binding data)
    - Co-IP: c-MET 85%, EGFR 75%, LamR 70% inhibition
    - Xenograft: 85% tumor suppression
    - 5-FU synergy: CI ≈ 0.6
    """
    logger.info("Building Pritamab pilot dataset...")

    # ===== Single drug scenarios (dose-response variations) =====
    scenarios = []

    # Vary KD
    for kd in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        dg = kd_to_dg(kd)
        # Higher affinity → better inhibition
        inhibition_factor = min(1.0, 0.1 / kd)  # Relative to KD=0.1nM
        cmet_inh = 85 * inhibition_factor
        egfr_inh = 75 * inhibition_factor
        lamr_inh = 70 * inhibition_factor

        # Tumor suppression scales roughly with total inhibition
        total_inh = (cmet_inh + egfr_inh + lamr_inh) / 3
        tumor_supp = min(95, total_inh * 1.1)

        # IC50 scales inversely with binding affinity
        ic50 = kd * 120  # nM — rough relationship

        scenarios.append({
            "name": f"Pritamab_KD={kd}nM",
            "pk_energy": {
                "dg_binding": dg,
                "mu_cmax": cmax_to_chemical_potential(50, 150000, 0.15),
                "dg_kinetic": t_half_to_kinetic_barrier(15 * (kd / 0.1)),
                "receptor_density": 0.96,
                "ddg_mutation": 0.0,
            },
            "pathway_energy": {
                "cmet_efficiency": inhibition_to_energy_transfer(cmet_inh),
                "egfr_efficiency": inhibition_to_energy_transfer(egfr_inh),
                "lamr_efficiency": inhibition_to_energy_transfer(lamr_inh),
                "cross_talk": 0.1,
            },
            "targets": {
                "tumor_suppression_pct": tumor_supp,
                "ic50_nm": ic50,
                "synergy_ci_5fu": 0.6 * (1 + (kd - 0.1) / 10),
            }
        })

    # ===== Mutation scenarios =====
    for mut_name, mut_code in [
        ("WT", None),
        ("KRAS_G13D", "KRAS_G13D"),
        ("KRAS_G12D", "KRAS_G12D"),
        ("BRAF_V600E", "BRAF_V600E"),
        ("PIK3CA_H1047R", "PIK3CA_H1047R"),
    ]:
        ddg = mutation_ddg(mut_code) if mut_code else 0.0
        # Mutations reduce drug efficacy
        resistance_factor = max(0.3, 1.0 - abs(ddg) / 5.0)

        scenarios.append({
            "name": f"Pritamab_KD=0.1_{mut_name}",
            "pk_energy": {
                "dg_binding": kd_to_dg(0.1),
                "mu_cmax": cmax_to_chemical_potential(50, 150000, 0.15),
                "dg_kinetic": t_half_to_kinetic_barrier(15),
                "receptor_density": 0.96,
                "ddg_mutation": ddg,
            },
            "pathway_energy": {
                "cmet_efficiency": 0.15,
                "egfr_efficiency": 0.25,
                "lamr_efficiency": 0.30,
                "cross_talk": 0.1 + abs(ddg) * 0.05,
            },
            "targets": {
                "tumor_suppression_pct": 85 * resistance_factor,
                "ic50_nm": 12 / resistance_factor,
                "synergy_ci_5fu": 0.6 / resistance_factor,
            }
        })

    # ===== Dose-response scenarios =====
    for cmax in [5, 10, 25, 50, 100, 200]:
        mu = cmax_to_chemical_potential(cmax, 150000, 0.15)
        dose_effect = min(1.0, cmax / 50)  # Saturates around Cmax=50

        scenarios.append({
            "name": f"Pritamab_Cmax={cmax}",
            "pk_energy": {
                "dg_binding": kd_to_dg(0.1),
                "mu_cmax": mu,
                "dg_kinetic": t_half_to_kinetic_barrier(15),
                "receptor_density": 0.96,
                "ddg_mutation": 0.0,
            },
            "pathway_energy": {
                "cmet_efficiency": 0.15 / dose_effect if dose_effect > 0 else 1.0,
                "egfr_efficiency": 0.25 / dose_effect if dose_effect > 0 else 1.0,
                "lamr_efficiency": 0.30 / dose_effect if dose_effect > 0 else 1.0,
                "cross_talk": 0.1,
            },
            "targets": {
                "tumor_suppression_pct": 85 * dose_effect,
                "ic50_nm": 12,
                "synergy_ci_5fu": 0.6,
            }
        })

    logger.info(f"  Built {len(scenarios)} scenarios")
    return scenarios


def train_pinn(scenarios: List[Dict], epochs: int = 500, lr: float = 1e-3,
               lambda_physics: float = 0.1):
    """Train PINN on pilot dataset."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Training PINN on {device}...")

    # Build tensors
    pk_list, pw_list, tgt_list = [], [], []
    for s in scenarios:
        pk = s["pk_energy"]
        pw = s["pathway_energy"]
        tgt = s["targets"]

        pk_list.append([pk["dg_binding"], pk["mu_cmax"], pk["dg_kinetic"],
                       pk["receptor_density"], pk["ddg_mutation"]])
        pw_list.append([pw["cmet_efficiency"], pw["egfr_efficiency"],
                       pw["lamr_efficiency"], pw["cross_talk"]])
        tgt_list.append([tgt["tumor_suppression_pct"], tgt["ic50_nm"],
                        tgt["synergy_ci_5fu"]])

    PK = torch.FloatTensor(pk_list).to(device)
    PW = torch.FloatTensor(pw_list).to(device)
    TGT = torch.FloatTensor(tgt_list).to(device)

    # Normalize targets for training stability
    tgt_mean = TGT.mean(dim=0)
    tgt_std = TGT.std(dim=0) + 1e-8
    TGT_norm = (TGT - tgt_mean) / tgt_std

    model = PritamabPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        pred, energy_balance = model(PK, PW)
        loss, l_data, l_phys = pinn_loss(pred, TGT_norm, energy_balance, lambda_physics)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            # Denormalize predictions for evaluation
            pred_real = pred * tgt_std + tgt_mean
            mse_ts = ((pred_real[:, 0] - TGT[:, 0]) ** 2).mean().item()
            mse_ic = ((pred_real[:, 1] - TGT[:, 1]) ** 2).mean().item()
            logger.info(f"  Epoch {epoch+1:4d}: loss={loss.item():.4f} "
                       f"(data={l_data:.4f}, physics={l_phys:.4f}) "
                       f"MSE_ts={mse_ts:.2f} MSE_ic={mse_ic:.2f}")
            history.append({
                "epoch": epoch + 1,
                "loss": loss.item(),
                "l_data": l_data,
                "l_physics": l_phys,
                "mse_tumor_supp": mse_ts,
                "mse_ic50": mse_ic,
            })

    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    with torch.no_grad():
        pred, energy_balance = model(PK, PW)
        pred_real = pred * tgt_std + tgt_mean

    results = {
        "n_scenarios": len(scenarios),
        "device": device,
        "epochs": epochs,
        "lambda_physics": lambda_physics,
        "best_loss": best_loss,
        "final_energy_balance_mean": float(energy_balance.abs().mean()),
        "history": history,
    }

    # Per-scenario predictions
    results["predictions"] = []
    for i, s in enumerate(scenarios):
        results["predictions"].append({
            "name": s["name"],
            "true_tumor_supp": s["targets"]["tumor_suppression_pct"],
            "pred_tumor_supp": float(pred_real[i, 0]),
            "true_ic50": s["targets"]["ic50_nm"],
            "pred_ic50": float(pred_real[i, 1]),
            "true_synergy_ci": s["targets"]["synergy_ci_5fu"],
            "pred_synergy_ci": float(pred_real[i, 2]),
            "energy_balance": float(energy_balance[i]),
        })

    return model, results, tgt_mean.cpu(), tgt_std.cpu()


# ==========================================================================
# Main
# ==========================================================================

def main():
    output_dir = Path("F:/ADDS/models/energy")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Track 2: Pritamab Energy Framework + PINN")
    print("=" * 70)

    # Step 1: Energy conversions
    print("\n--- Step 1: PK → Energy Conversions ---")
    dg = kd_to_dg(0.1)
    mu = cmax_to_chemical_potential(50, 150000, 0.15)
    dg_kin = t_half_to_kinetic_barrier(15)

    print(f"  KD = 0.1 nM → ΔG = {dg:.2f} kcal/mol")
    print(f"  Cmax = 50 μg/mL → Δμ = {mu:.2f} kcal/mol")
    print(f"  t½ = 15 min → ΔG‡ = {dg_kin:.2f} kcal/mol")
    print(f"  ATP hydrolysis = {ATP_HYDROLYSIS} kcal/mol")

    # Step 2: Energy propagation (no mutation)
    print("\n--- Step 2: Energy Propagation (WT) ---")
    graph = build_pritamab_pathway_graph(dg)
    energies = propagate_energy(graph)
    for k, e in energies.items():
        node = graph[k]
        mut_tag = f" [MUT: ΔΔG={node.mutation_ddg}]" if node.is_mutant else ""
        print(f"  {k:25s}: {e:+8.2f} kcal/mol{mut_tag}")

    # Step 3: Energy conservation check
    print("\n--- Step 3: Energy Conservation ---")
    balance = check_energy_conservation(energies, graph)
    for k, v in balance.items():
        print(f"  {k:30s}: {v:+10.3f} kcal/mol")

    # Step 4: With KRAS G13D mutation
    print("\n--- Step 4: Energy Propagation (KRAS G13D) ---")
    graph_mut = build_pritamab_pathway_graph(dg, mutations={"RAS": "KRAS_G13D"})
    energies_mut = propagate_energy(graph_mut)
    for k, e in energies_mut.items():
        node = graph_mut[k]
        mut_tag = f" [MUT: ΔΔG={node.mutation_ddg}]" if node.is_mutant else ""
        delta = e - energies[k]
        delta_tag = f" (Δ={delta:+.2f})" if abs(delta) > 0.01 else ""
        print(f"  {k:25s}: {e:+8.2f} kcal/mol{mut_tag}{delta_tag}")

    # Step 5: Build pilot dataset
    print("\n--- Step 5: Pilot Dataset ---")
    scenarios = build_pritamab_pilot_dataset()

    # Step 6: Train PINN
    print("\n--- Step 6: PINN Training ---")
    model, results, tgt_mean, tgt_std = train_pinn(scenarios, epochs=500,
                                                    lambda_physics=0.1)

    # Step 7: Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"  Scenarios: {results['n_scenarios']}")
    print(f"  Best loss: {results['best_loss']:.4f}")
    print(f"  Energy balance (mean |residual|): {results['final_energy_balance_mean']:.4f}")
    print()
    print(f"  {'Scenario':35s} {'TumorSupp%':>10} {'IC50 nM':>10} {'CI':>8}")
    print(f"  {'':35s} {'true/pred':>10} {'true/pred':>10} {'true/pred':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
    for p in results["predictions"][:10]:
        print(f"  {p['name']:35s} "
              f"{p['true_tumor_supp']:4.0f}/{p['pred_tumor_supp']:5.1f} "
              f"{p['true_ic50']:5.0f}/{p['pred_ic50']:6.1f} "
              f"{p['true_synergy_ci']:.2f}/{p['pred_synergy_ci']:.2f}")

    # Save
    torch.save(model.state_dict(), output_dir / "pritamab_pinn.pt")
    with open(output_dir / "pritamab_pinn_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved: pritamab_pinn.pt, pritamab_pinn_results.json")


if __name__ == "__main__":
    main()
