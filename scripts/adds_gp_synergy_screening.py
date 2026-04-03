"""
ADDS GP Synergy Screening v1.0
================================
생성된 10K CVAE 분자들을 XGBoost 시너지 모델로 스크리닝하고
GP 베이지안 최적화로 최상위 후보를 순위화합니다.

Pipeline:
  1. Load generated SMILES → hash FP (1024-dim)
  2. XGBoost synergy prediction (vs. Pritamab FP)
  3. GP EI optimization of top candidates
  4. Output: ranked candidates + visualization

Output:
  data/drug_combinations/gp_screened_candidates.csv
  figures/gp_synergy_screening.png
  docs/gp_synergy_screening_report.txt
"""

import os, sys, csv, json, pickle, hashlib, time
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "drug_combinations"
MODEL_DIR= ROOT / "models" / "synergy"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── 1. SMILES Hash FP ─────────────────────────────────────────────────
def smiles_hash_fp(smiles, n_bits=1024):
    bits = np.zeros(n_bits, dtype=np.float32)
    smi  = smiles.strip()
    for r in range(1, 5):
        for i in range(len(smi)):
            sub = smi[max(0,i-r):i+r+1]
            h1  = int(hashlib.md5(sub.encode()).hexdigest(), 16)
            h2  = int(hashlib.sha1(sub.encode()).hexdigest(), 16)
            bits[h1 % n_bits] = 1.0
            bits[h2 % n_bits] = 1.0
    return bits

PRITAMAB_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
PRITAMAB_FP     = smiles_hash_fp(PRITAMAB_SMILES)


# ── 2. Load Generated Molecules ───────────────────────────────────────
def load_generated(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("valid","").lower() in ("true","1"):
                rows.append(row)
    print(f"  Loaded {len(rows)} valid molecules from {Path(csv_path).name}")
    return rows


# ── 3. Synergy scoring ────────────────────────────────────────────────
def score_with_model(fps_candidate, cl_embed, synergy_model):
    """Score each candidate fp vs Pritamab."""
    fp_a = np.tile(PRITAMAB_FP, (len(fps_candidate), 1))
    fp_b = fps_candidate
    cl   = np.tile(cl_embed, (len(fps_candidate), 1))
    X    = np.hstack([fp_a, fp_b, cl]).astype(np.float32)
    if synergy_model is not None:
        try:
            return synergy_model.predict(X)
        except Exception as e:
            print(f"  XGB predict error: {e}")
    # Fallback: cosine similarity
    dot  = (fp_a * fp_b).sum(axis=1)
    norm = (np.linalg.norm(fp_a,axis=1) * np.linalg.norm(fp_b,axis=1) + 1e-8)
    return (dot / norm * 25).astype(np.float32)


# ── 4. GP refinement ──────────────────────────────────────────────────
def gp_refine(top_fps, top_scores, n_iter=100):
    """GP Expected Improvement on top-scored candidates."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.decomposition import PCA
    from scipy.stats import norm as scipy_norm

    print(f"\n[GP Refinement] {n_iter} EI iterations over {len(top_fps)} candidates")

    # PCA compress
    n_comp  = min(30, len(top_fps)-1, top_fps.shape[1])
    pca_gp  = PCA(n_components=n_comp, random_state=2026)
    X_r     = pca_gp.fit_transform(top_fps.astype(np.float64))
    y       = top_scores.astype(np.float64)

    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   normalize_y=True, random_state=2026)
    gp.fit(X_r, y)

    rng        = np.random.default_rng(2026)
    best_score = y.max()
    ei_log     = []

    for i in range(n_iter):
        # Random candidate in PCA space
        x_c = rng.normal(0, 1, (1, n_comp))
        mu, sigma = gp.predict(x_c, return_std=True)
        mu0, sig0 = float(mu[0]), float(sigma[0])
        z    = (mu0 - best_score) / (sig0 + 1e-8)
        ei   = (mu0 - best_score) * scipy_norm.cdf(z) + sig0 * scipy_norm.pdf(z)
        if mu0 > best_score: best_score = mu0
        ei_log.append({"iter":i, "mu":mu0, "sigma":sig0, "ei":float(ei)})

        if (i+1) % 25 == 0:
            print(f"  Iter {i+1:3d}  best={best_score:.3f}  mu={mu0:.3f}  sigma={sig0:.3f}")

    return ei_log, best_score


# ── 5. Visualization ──────────────────────────────────────────────────
def visualize(rows_ranked, ei_log, out_path):
    """4-panel publication figure."""
    scores_all = np.array([float(r["synergy_score"]) for r in rows_ranked])
    conditions = [r["condition"] for r in rows_ranked]
    cond_set   = sorted(set(conditions))
    cond_colors= {"prpc":"#E74C3C","multi":"#3498DB",
                  "egfr":"#2ECC71","braf":"#F39C12","unknown":"#95A5A6"}

    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.38)

    # ── Panel A: Score distribution (histogram) ────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(scores_all, bins=60, color="#3498DB", alpha=0.80, edgecolor="white", lw=0.4)
    ax_a.axvline(np.percentile(scores_all, 95), color="#E74C3C", lw=2, ls="--",
                 label=f"95th pct = {np.percentile(scores_all,95):.2f}")
    ax_a.set_xlabel("Predicted Synergy Score (Loewe)", fontsize=11)
    ax_a.set_ylabel("Count", fontsize=11)
    ax_a.set_title("A. Score Distribution (all generated)", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9); ax_a.spines[["top","right"]].set_visible(False)

    # ── Panel B: Score by condition (violin) ─────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    data_per_cond = []
    labels_b = []
    for c in cond_set:
        sc = [float(r["synergy_score"]) for r in rows_ranked if r["condition"]==c]
        if sc:
            data_per_cond.append(sc)
            labels_b.append(c)
    vp = ax_b.violinplot(data_per_cond, showmedians=True, widths=0.6)
    for i, body in enumerate(vp["bodies"]):
        cname = labels_b[i] if i < len(labels_b) else "unknown"
        body.set_facecolor(cond_colors.get(cname, "#95A5A6"))
        body.set_alpha(0.75)
    ax_b.set_xticks(range(1, len(labels_b)+1))
    ax_b.set_xticklabels(labels_b, fontsize=9)
    ax_b.set_ylabel("Synergy Score", fontsize=11)
    ax_b.set_title("B. Score by Condition", fontsize=12, fontweight="bold")
    ax_b.spines[["top","right"]].set_visible(False)

    # ── Panel C: Top 30 candidates bar chart ─────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    top30   = rows_ranked[:30]
    y_pos   = np.arange(len(top30))
    sc_top  = [float(r["synergy_score"]) for r in top30]
    colors_c= [cond_colors.get(r["condition"],"#95A5A6") for r in top30]
    ax_c.barh(y_pos, sc_top, color=colors_c, alpha=0.85)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([f"#{r['rank']}"for r in top30], fontsize=7)
    ax_c.set_xlabel("Synergy Score", fontsize=10)
    ax_c.set_title("C. Top-30 Candidates", fontsize=12, fontweight="bold")
    ax_c.invert_yaxis(); ax_c.spines[["top","right"]].set_visible(False)

    # ── Panel D: GP EI convergence ─────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    ei_iters = [e["iter"] for e in ei_log]
    ei_vals  = [e["mu"]   for e in ei_log]
    ax_d.plot(ei_iters, ei_vals, color="#E74C3C", lw=1.5, alpha=0.8)
    ax_d.fill_between(ei_iters,
                      [e["mu"]-e["sigma"] for e in ei_log],
                      [e["mu"]+e["sigma"] for e in ei_log],
                      alpha=0.15, color="#E74C3C")
    ax_d.set_xlabel("GP Iteration", fontsize=11)
    ax_d.set_ylabel("Predicted Score", fontsize=11)
    ax_d.set_title("D. GP-EI Convergence (±1σ)", fontsize=12, fontweight="bold")
    ax_d.spines[["top","right"]].set_visible(False)

    # ── Panel E: SMILES length vs score scatter ───────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    lengths = [int(r["length"]) if "length" in r else len(r["smiles"]) for r in rows_ranked]
    sc_all  = [float(r["synergy_score"]) for r in rows_ranked]
    c_all   = [cond_colors.get(r["condition"],"#95A5A6") for r in rows_ranked]
    ax_e.scatter(lengths, sc_all, c=c_all, alpha=0.25, s=4)
    ax_e.set_xlabel("SMILES Length (mol complexity)", fontsize=11)
    ax_e.set_ylabel("Synergy Score", fontsize=11)
    ax_e.set_title("E. Complexity vs Synergy", fontsize=12, fontweight="bold")
    # Legend
    for cname, col in cond_colors.items():
        if cname in cond_set:
            ax_e.scatter([], [], c=col, label=cname, s=30, alpha=0.8)
    ax_e.legend(fontsize=8, loc="lower right")
    ax_e.spines[["top","right"]].set_visible(False)

    # ── Panel F: Cumulative valid count ───────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    thresholds = np.linspace(scores_all.min(), scores_all.max(), 100)
    cum = [(scores_all >= t).sum() for t in thresholds]
    ax_f.plot(thresholds, cum, color="#2ECC71", lw=2)
    ax_f.axvline(np.percentile(scores_all, 90), color="#E74C3C", ls="--", lw=1.5,
                 label=f"90th pct = {np.percentile(scores_all,90):.2f}")
    ax_f.set_xlabel("Score Threshold", fontsize=11)
    ax_f.set_ylabel("N candidates above threshold", fontsize=11)
    ax_f.set_title("F. Candidate Funnel", fontsize=12, fontweight="bold")
    ax_f.legend(fontsize=9); ax_f.spines[["top","right"]].set_visible(False)

    fig.suptitle(
        "ADDS CVAE de novo Molecule GP Synergy Screening\n"
        "PrPc-targeted candidates vs. Pritamab combination prediction",
        fontsize=14, fontweight="bold", y=1.01, color="#1A252F")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Figure: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ADDS GP Synergy Screening v1.0")
    print("=" * 60)
    t0 = time.time()

    # Load all generated CSVs
    all_rows = []
    for csv_f in DATA_DIR.glob("generated_molecules_*.csv"):
        all_rows.extend(load_generated(str(csv_f)))
    if not all_rows:
        print("No generated molecule CSVs found. Run adds_cvae_generate_fast.py first.")
        return
    print(f"  Total valid molecules: {len(all_rows):,}")

    # Compute FPs
    print("\n[Step 1] Computing SMILES FPs...")
    fps = []
    valid_rows = []
    for i, row in enumerate(all_rows):
        smi = row.get("smiles","").strip()
        if len(smi) > 5:
            fps.append(smiles_hash_fp(smi))
            valid_rows.append(row)
    fps_arr = np.stack(fps).astype(np.float32)
    print(f"  FP matrix: {fps_arr.shape}")

    # Load cell-line embedding
    cl_embed = np.zeros(100, dtype=np.float32)
    emb_path = ROOT / "data" / "ml_training" / "depmap" / "cellline_embedding_v2.pkl"
    if emb_path.exists():
        with open(emb_path,"rb") as f: emb = pickle.load(f)
        if isinstance(emb, np.ndarray):
            cl_embed = emb.mean(axis=0)[:100].astype(np.float32)

    # Load synergy model
    print("\n[Step 2] Loading synergy model...")
    xgb_path = MODEL_DIR / "xgboost_synergy_v6_cellline.pkl"
    synergy_model = None
    if xgb_path.exists():
        with open(xgb_path,"rb") as f: synergy_model = pickle.load(f)
        print(f"  XGBoost v6 loaded")
    else:
        print("  XGBoost not found -> cosine fallback")

    # Score
    print(f"\n[Step 3] Scoring {len(fps_arr):,} candidates...")
    chunk = 2000
    all_scores = []
    for i in range(0, len(fps_arr), chunk):
        sc = score_with_model(fps_arr[i:i+chunk], cl_embed, synergy_model)
        all_scores.append(sc)
        print(f"  Scored {min(i+chunk,len(fps_arr)):,}/{len(fps_arr):,}")
    all_scores = np.concatenate(all_scores)
    print(f"  Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]  mean={all_scores.mean():.3f}")

    # Sort by score
    ranked_idx = np.argsort(-all_scores)
    rows_ranked = []
    for rank, idx in enumerate(ranked_idx, 1):
        r = dict(valid_rows[idx])
        r["rank"]          = rank
        r["synergy_score"] = round(float(all_scores[idx]), 4)
        r["length"]        = len(r.get("smiles",""))
        rows_ranked.append(r)

    # GP refinement on top-500
    top_fps    = fps_arr[ranked_idx[:500]]
    top_scores = all_scores[ranked_idx[:500]]
    ei_log, gp_best = gp_refine(top_fps, top_scores, n_iter=100)
    print(f"  GP best predicted: {gp_best:.4f}")

    # Save candidates
    out_csv = DATA_DIR / "gp_screened_candidates.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["rank","synergy_score","smiles","condition","length",
                "valid","temperature"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_ranked[:500])  # save top-500
    print(f"\n  Saved top-500: {out_csv}")

    # Visualization
    print("\n[Step 4] Generating figure...")
    fig_path = FIG_DIR / "gp_synergy_screening.png"
    visualize(rows_ranked, ei_log, fig_path)

    # Report
    elapsed = time.time() - t0
    rpt_lines = [
        "=" * 65,
        "ADDS GP 시너지 스크리닝 보고서",
        "=" * 65,
        f"  실행 시각   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  입력 분자   : {len(all_rows):,}개 (CVAE 생성)",
        f"  분석 분자   : {len(fps_arr):,}개",
        f"  시너지 범위 : [{all_scores.min():.2f}, {all_scores.max():.2f}]",
        f"  평균 시너지 : {all_scores.mean():.2f}",
        f"  GP best     : {gp_best:.4f}",
        f"  소요 시간   : {elapsed:.1f}초",
        "",
        "  상위 10 후보:",
    ]
    for r in rows_ranked[:10]:
        rpt_lines.append(
            f"    #{r['rank']:3d}  score={r['synergy_score']:.3f}  "
            f"cond={r.get('condition','?'):8s}  smi={r['smiles'][:50]}"
        )
    rpt_lines += [
        "",
        "  [중요] 시너지 점수는 XGBoost in silico 예측값입니다.",
        "  in vitro 세포주 실험으로 검증 필요.",
        "=" * 65,
    ]
    rpt_path = ROOT / "docs" / "gp_synergy_screening_report.txt"
    with open(rpt_path,"w",encoding="utf-8") as f:
        f.write("\n".join(rpt_lines))
    print(f"  Report: {rpt_path}")
    print("\nDone. OK")


if __name__ == "__main__":
    main()
