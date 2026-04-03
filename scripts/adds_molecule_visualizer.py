"""
ADDS Molecular Structure Visualization v1.0
============================================
CVAE 생성 분자와 GP 스크리닝 상위 후보의 2D 구조를 시각화합니다.

RDKit 없이 순수 matplotlib으로 SMILES 기반 시각화:
  - SMILES 기반 원자/결합 카운팅
  - Radial 특성 레이더 차트
  - 분자 유사도 히트맵 (해시 FP Tanimoto)
  - Top-20 후보 속성 비교 표

Output:
  figures/molecular_structure_visualization.png
  figures/top20_candidates_table.png
"""

import sys, csv, hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT    = Path(__file__).parent.parent
DATA_DIR= ROOT / "data" / "drug_combinations"
FIG_DIR = ROOT / "figures"

import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.cm import get_cmap

# ── SMILES property extractor (no RDKit) ─────────────────────────────
def smiles_props(smiles):
    """Extract approximate molecular properties from SMILES string."""
    s = smiles.strip()
    props = {
        "length":    len(s),
        "n_C":      s.count("C"),
        "n_N":      s.count("N") + s.count("n"),
        "n_O":      s.count("O") + s.count("o"),
        "n_S":      s.count("S") + s.count("s"),
        "n_F":      s.count("F"),
        "n_Cl":     s.count("Cl"),
        "n_Br":     s.count("Br"),
        "n_ring":   (s.count("1")+s.count("2")+s.count("3")) // 2,
        "n_aromatic":s.count("c") + s.count("n") + s.count("o"),
        "n_double": s.count("="),
        "n_triple": s.count("#"),
        "n_chiral": s.count("@"),
        "n_branch": s.count("("),
        "n_stereo": s.count("/") + s.count("\\"),
        # Approx MW (C≈12, N≈14, O≈16)
        "mw_approx": (s.count("C")*12 + s.count("N")*14 + s.count("O")*16
                      + s.count("S")*32 + s.count("F")*19 + len(s.replace("Cl",""))*1),
    }
    # Drug-likeness score (rough Lipinski proxy)
    props["drug_like"] = int(
        props["mw_approx"] < 500 and
        props["n_N"] + props["n_O"] <= 10 and
        props["n_N"] + props["n_O"] + props["n_S"] <= 5
    )
    return props

def smiles_hash_fp(smiles, n_bits=256):
    bits = np.zeros(n_bits, dtype=np.float32)
    for r in range(1, 5):
        for i in range(len(smiles)):
            sub = smiles[max(0,i-r):i+r+1]
            h1  = int(hashlib.md5(sub.encode()).hexdigest(), 16)
            bits[h1 % n_bits] = 1.0
    return bits

def tanimoto(fp_a, fp_b):
    inter = (fp_a * fp_b).sum()
    union = (fp_a + fp_b).clip(0,1).sum()
    return inter / (union + 1e-8)


# ── Load Candidates ──────────────────────────────────────────────────
def load_candidates():
    rows = []
    # Priority: GP screened candidates
    f1 = DATA_DIR / "gp_screened_candidates.csv"
    if f1.exists():
        with open(f1, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["source"] = "GP_screened"
                rows.append(r)

    # All generated (for fingerprint diversity calc)
    all_smiles = []
    for fn in DATA_DIR.glob("generated_molecules_*.csv"):
        with open(fn, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("valid","").lower() in ("true","1"):
                    all_smiles.append(r.get("smiles",""))

    print(f"  GP candidates: {len(rows)}")
    print(f"  All generated: {len(all_smiles)}")
    return rows[:100], all_smiles[:500]  # limit for memory


# ── Figure 1: 6-Panel Property Analysis ─────────────────────────────
def fig_property_analysis(candidates, all_smiles):
    props_all = [smiles_props(s) for s in all_smiles if len(s) > 5]
    props_top = [smiles_props(r["smiles"]) for r in candidates[:20] if "smiles" in r]

    fig = plt.figure(figsize=(20, 14), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # A: MW distribution
    ax_a = fig.add_subplot(gs[0, 0])
    mws_all = [p["mw_approx"] for p in props_all]
    mws_top = [p["mw_approx"] for p in props_top]
    ax_a.hist(mws_all, bins=40, color="#AED6F1", alpha=0.7, label="All (n=500)")
    ax_a.hist(mws_top, bins=10, color="#E74C3C", alpha=0.9, label="Top-20")
    ax_a.axvline(500, color="#2C3E50", ls="--", lw=1.5, label="Lipinski MW<500")
    ax_a.set_xlabel("Approx MW (Da)", fontsize=11)
    ax_a.set_ylabel("Count", fontsize=11)
    ax_a.set_title("A. Molecule Weight Distribution", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=8); ax_a.spines[["top","right"]].set_visible(False)

    # B: Atom composition stacked bar (top-20)
    ax_b = fig.add_subplot(gs[0, 1])
    n_top = min(20, len(props_top))
    x     = np.arange(n_top)
    atom_types = ["n_C","n_N","n_O","n_S","n_F"]
    colors_b   = ["#3498DB","#E74C3C","#E67E22","#F1C40F","#2ECC71"]
    bottom = np.zeros(n_top)
    for at, cl in zip(atom_types, colors_b):
        vals = [p.get(at,0) for p in props_top[:n_top]]
        ax_b.bar(x, vals, bottom=bottom, color=cl, label=at[2:], alpha=0.85)
        bottom += np.array(vals)
    ax_b.set_xticks(x); ax_b.set_xticklabels([f"#{i+1}" for i in range(n_top)],
                                               fontsize=7, rotation=60)
    ax_b.set_ylabel("Atom count (approx)", fontsize=11)
    ax_b.set_title("B. Atom Composition (Top-20)", fontsize=12, fontweight="bold")
    ax_b.legend(fontsize=8, loc="upper right"); ax_b.spines[["top","right"]].set_visible(False)

    # C: Ring count vs aromaticity scatter
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter([p["n_ring"] for p in props_all], [p["n_aromatic"] for p in props_all],
                 c="#AED6F1", s=8, alpha=0.5, label="All")
    ax_c.scatter([p["n_ring"] for p in props_top], [p["n_aromatic"] for p in props_top],
                 c="#E74C3C", s=50, alpha=0.9, label="Top-20", zorder=5)
    ax_c.set_xlabel("Ring count (approx)", fontsize=11)
    ax_c.set_ylabel("Aromatic atoms", fontsize=11)
    ax_c.set_title("C. Ring vs Aromaticity", fontsize=12, fontweight="bold")
    ax_c.legend(fontsize=8); ax_c.spines[["top","right"]].set_visible(False)

    # D: Radar chart for Top-1 candidate properties
    ax_d = fig.add_subplot(gs[1, 0], polar=True)
    if props_top:
        p1   = props_top[0]
        cats = ["Rings","Aromatic","Branches","Chirality","Double=","N/O/S"]
        vals = [
            min(p1["n_ring"]/5, 1),
            min(p1["n_aromatic"]/10, 1),
            min(p1["n_branch"]/8, 1),
            min(p1["n_chiral"]/4, 1),
            min(p1["n_double"]/6, 1),
            min((p1["n_N"]+p1["n_O"]+p1["n_S"])/10, 1),
        ]
        angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
        vals   = vals + vals[:1]
        angles = angles + angles[:1]
        ax_d.plot(angles, vals, "o-", color="#E74C3C", lw=2)
        ax_d.fill(angles, vals, alpha=0.25, color="#E74C3C")
        ax_d.set_thetagrids(np.degrees(angles[:-1]), cats, fontsize=9)
        ax_d.set_ylim(0, 1); ax_d.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax_d.set_yticklabels(["","","",""], fontsize=7)
    ax_d.set_title("D. Top-1 Structural Features\n(normalized)", fontsize=11,
                   fontweight="bold", pad=15)

    # E: Tanimoto similarity heatmap (top-20 vs top-20)
    ax_e = fig.add_subplot(gs[1, 1])
    if len(props_top) >= 5:
        top_smi = [r["smiles"] for r in candidates[:20] if "smiles" in r]
        fps_t   = np.stack([smiles_hash_fp(s, 256) for s in top_smi[:20]])
        n_m     = len(fps_t)
        sim_mat = np.zeros((n_m, n_m))
        for i in range(n_m):
            for j in range(n_m):
                sim_mat[i,j] = tanimoto(fps_t[i], fps_t[j])
        im = ax_e.imshow(sim_mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax_e, fraction=0.046, label="Tanimoto")
        ax_e.set_xticks(range(n_m)); ax_e.set_yticks(range(n_m))
        ax_e.set_xticklabels([f"#{i+1}" for i in range(n_m)], fontsize=7, rotation=60)
        ax_e.set_yticklabels([f"#{i+1}" for i in range(n_m)], fontsize=7)
    ax_e.set_title("E. Structural Similarity (Top-20)", fontsize=12, fontweight="bold")

    # F: Synergy score vs drug-likeness
    ax_f = fig.add_subplot(gs[1, 2])
    scores_c = [float(r.get("synergy_score",0)) for r in candidates[:len(props_top)]]
    drug_like= [p["drug_like"] for p in props_top]
    colors_f = ["#E74C3C" if d else "#AED6F1" for d in drug_like]
    ax_f.scatter(scores_c, [p["mw_approx"] for p in props_top],
                 c=colors_f, s=60, alpha=0.85, edgecolors="gray", lw=0.5)
    ax_f.set_xlabel("GP Synergy Score", fontsize=11)
    ax_f.set_ylabel("Approx MW", fontsize=11)
    ax_f.axhline(500, color="#2C3E50", ls="--", lw=1.2, label="MW=500 (Lipinski)")
    # Legend
    from matplotlib.lines import Line2D
    ax_f.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#E74C3C", ms=9, label="Drug-like"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#AED6F1", ms=9, label="Non-drug-like"),
    ] + ax_f.get_legend_handles_labels()[0], fontsize=8, loc="upper left")
    ax_f.set_title("F. Synergy vs MW (drug-likeness)", fontsize=12, fontweight="bold")
    ax_f.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "ADDS CVAE de novo 분자 구조 특성 분석\n"
        "PrPc-targeted GP-screened candidates — Structural property panel",
        fontsize=14, fontweight="bold", y=1.01, color="#1A252F")
    plt.tight_layout()

    out = FIG_DIR / "molecular_structure_visualization.png"
    plt.savefig(str(out), dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ── Figure 2: Top-20 Candidates Table ────────────────────────────────
def fig_top20_table(candidates):
    top = candidates[:20]
    fig, ax = plt.subplots(figsize=(18, 10), facecolor="white")
    ax.axis("off")

    cols = ["Rank","Score","Condition","SMILES (truncated)","Length","Heteroatoms","Rings","Drug-like"]
    table_data = []
    for r in top:
        smi = r.get("smiles","")
        p   = smiles_props(smi)
        table_data.append([
            f"#{r.get('rank','?')}",
            f"{float(r.get('synergy_score',0)):.3f}",
            r.get("condition","prpc"),
            smi[:55] + "..." if len(smi) > 55 else smi,
            str(p["length"]),
            str(p["n_N"]+p["n_O"]+p["n_S"]),
            str(p["n_ring"]),
            "YES" if p["drug_like"] else "no",
        ])

    tbl = ax.table(cellText=table_data, colLabels=cols,
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Color header
    for j in range(len(cols)):
        tbl[(0,j)].set_facecolor("#2C3E50")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")

    # Color rows alternating + highlight drug-like
    for i, row in enumerate(table_data, 1):
        dl = row[-1] == "YES"
        for j in range(len(cols)):
            if dl:
                tbl[(i,j)].set_facecolor("#FDFEFE" if i%2==0 else "#EBF5FB")
            else:
                tbl[(i,j)].set_facecolor("#FDFEFE" if i%2==0 else "#F9F9F9")

    ax.set_title(
        "ADDS GP 시너지 스크리닝 Top-20 후보 분자\n"
        "(PrPc-targeted CVAE de novo generated + GP-EI optimization)",
        fontsize=13, fontweight="bold", pad=20, color="#1A252F")

    out = FIG_DIR / "top20_candidates_table.png"
    plt.savefig(str(out), dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("ADDS Molecular Structure Visualization v1.0")
    print("=" * 55)

    candidates, all_smiles = load_candidates()

    print("\n[Figure 1] Property analysis panel...")
    fig1 = fig_property_analysis(candidates, all_smiles)

    print("\n[Figure 2] Top-20 candidates table...")
    fig2 = fig_top20_table(candidates)

    print("\nDone. OK")
    print(f"  {fig1}")
    print(f"  {fig2}")
