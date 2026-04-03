"""
Paper 3: Thermodynamic Basis for Toxicity Reduction in KRAS-Mutant Cancers
==========================================================================
Activation Energy Modulation by Anti-PrPC Antibody (Pritamab)
Enables Low-Dose Combination Therapy

Computational Framework:
1. TCGA PRNP expression × KRAS mutation status correlation
2. KRAS pathway energy landscape simulation (with/without PrPC)
3. Dose-response modeling: EC50 shift with Pritamab
4. Combination analysis: FOLFOX + Pritamab CI/Bliss
5. Publication-quality figure generation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("F:/ADDS/outputs/paper3_pritamab_kras")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ============================================================================
# Constants
# ============================================================================
R = 1.987e-3      # kcal/(mol·K)
T = 310            # K (37°C body temp)
RT = R * T         # 0.616 kcal/mol

# KRAS mutation prevalence by cancer type (from literature)
KRAS_MUTATION_PREVALENCE = {
    'PAAD': 0.90,   # Pancreatic: ~90%
    'COAD': 0.45,   # Colorectal: ~45%
    'READ': 0.45,   # Rectal: ~45%
    'LUAD': 0.32,   # Lung adeno: ~32%
    'STAD': 0.10,   # Gastric: ~10%
    'BRCA': 0.05,   # Breast: ~5%
}

# Drug pharmacokinetic data (from BindingDB / literature)
DRUG_DATA = {
    '5-FU': {'name': '5-Fluorouracil', 'IC50_nM': 8000, 'Ki_nM': 100, 
             'EC50_nM': 12000, 'hill_n': 1.2, 'MTD_mg_m2': 425,
             'target': 'Thymidylate Synthase'},
    'Oxaliplatin': {'name': 'Oxaliplatin', 'IC50_nM': 2500, 'Ki_nM': None,
                    'EC50_nM': 3750, 'hill_n': 1.0, 'MTD_mg_m2': 85,
                    'target': 'DNA'},
    'Irinotecan': {'name': 'Irinotecan', 'IC50_nM': 5000, 'Ki_nM': 50,
                   'EC50_nM': 7500, 'hill_n': 1.3, 'MTD_mg_m2': 180,
                   'target': 'Topoisomerase I'},
    'Sotorasib': {'name': 'Sotorasib (KRAS G12C)', 'IC50_nM': 50, 'Ki_nM': 10,
                  'EC50_nM': 75, 'hill_n': 1.5, 'MTD_mg_m2': 960,
                  'target': 'KRAS G12C'},
}

# KRAS pathway steps and their activation energies (literature-estimated)
# PrPC contributions are conservative estimates based on scaffold/co-activator role
PATHWAY_STEPS = [
    {'name': 'KRAS-GTP activation', 'dG_normal': 3.0, 'dG_mutant': 0.8,
     'prpc_contribution': 0.5},   # PrPC as membrane scaffold lowers GTP loading barrier
    {'name': 'RAF recruitment', 'dG_normal': 2.5, 'dG_mutant': 1.5,
     'prpc_contribution': 0.25},  # Lipid raft co-localization assists RAF
    {'name': 'MEK phosphorylation', 'dG_normal': 2.0, 'dG_mutant': 1.8,
     'prpc_contribution': 0.10},  # Minimal direct PrPC effect downstream
    {'name': 'ERK activation', 'dG_normal': 1.5, 'dG_mutant': 1.3,
     'prpc_contribution': 0.05},  # Indirect only
    {'name': 'Nuclear translocation', 'dG_normal': 1.0, 'dG_mutant': 0.9,
     'prpc_contribution': 0.02},  # Negligible
]

# Coupling factor: fraction of ΔΔG‡ that translates to EC50 shift
# α < 1 reflects partial thermodynamic coupling between pathway barrier 
# and drug cytotoxicity (drug targets are downstream of KRAS signaling)
ALPHA_COUPLING = 0.35


# ============================================================================
# Part 1: TCGA PRNP Expression × KRAS Mutation Analysis
# ============================================================================
def analyze_tcga_prnp_kras():
    """Analyze PRNP expression stratified by KRAS mutation prevalence."""
    log.info("=" * 60)
    log.info("Part 1: TCGA PRNP-KRAS Correlation Analysis")
    log.info("=" * 60)
    
    tcga_file = Path("F:/ADDS/data/analysis/prpc_validation/open_data/real/tcga_all_cancers_prnp_real.csv")
    if not tcga_file.exists():
        log.warning("TCGA data not found: %s", tcga_file)
        return None
    
    df = pd.read_csv(tcga_file)
    log.info("Loaded %d samples, %d cancer types", len(df), df['cancer_type'].nunique())
    
    # Add KRAS mutation prevalence
    df['kras_prevalence'] = df['cancer_type'].map(KRAS_MUTATION_PREVALENCE)
    df = df.dropna(subset=['kras_prevalence'])
    
    # Tumor samples only
    df_tumor = df[df['sample_type'] == 'Tumor'].copy()
    log.info("Tumor samples: %d", len(df_tumor))
    
    # Per-cancer-type statistics
    results = {}
    for ct in df_tumor['cancer_type'].unique():
        sub = df_tumor[df_tumor['cancer_type'] == ct]
        results[ct] = {
            'n': len(sub),
            'prnp_mean': float(sub['PRNP_log2'].mean()),
            'prnp_median': float(sub['PRNP_log2'].median()),
            'prnp_std': float(sub['PRNP_log2'].std()),
            'kras_prevalence': KRAS_MUTATION_PREVALENCE.get(ct, 0),
        }
        log.info("  %s (n=%d): PRNP=%.2f±%.2f, KRAS_prev=%.0f%%",
                ct, results[ct]['n'], results[ct]['prnp_mean'], 
                results[ct]['prnp_std'], results[ct]['kras_prevalence']*100)
    
    # Correlation: PRNP expression vs KRAS prevalence
    ct_list = sorted(results.keys())
    prnp_means = [results[ct]['prnp_mean'] for ct in ct_list]
    kras_prevs = [results[ct]['kras_prevalence'] for ct in ct_list]
    
    r_val, p_val = stats.pearsonr(prnp_means, kras_prevs)
    rho, rho_p = stats.spearmanr(prnp_means, kras_prevs)
    log.info("\nCorrelation (PRNP mean vs KRAS prevalence):")
    log.info("  Pearson r=%.3f, p=%.4f", r_val, p_val)
    log.info("  Spearman rho=%.3f, p=%.4f", rho, rho_p)
    
    return {
        'df': df_tumor,
        'results_by_ct': results,
        'correlation': {'pearson_r': r_val, 'pearson_p': p_val,
                       'spearman_rho': rho, 'spearman_p': rho_p},
    }


# ============================================================================
# Part 2: Energy Landscape Simulation
# ============================================================================
def simulate_energy_landscape():
    """Model KRAS pathway energy barriers with/without PrPC."""
    log.info("\n" + "=" * 60)
    log.info("Part 2: KRAS Pathway Energy Landscape")
    log.info("=" * 60)
    
    scenarios = {
        'Normal (WT KRAS)': [],
        'KRAS Mutant + PrPC high': [],
        'KRAS Mutant + Pritamab': [],  # PrPC neutralized
    }
    
    for step in PATHWAY_STEPS:
        # Normal: standard activation energy
        scenarios['Normal (WT KRAS)'].append(step['dG_normal'])
        
        # KRAS mutant with high PrPC: lowest barrier (worst case)
        mutant_with_prpc = step['dG_mutant'] - step['prpc_contribution']
        scenarios['KRAS Mutant + PrPC high'].append(max(0.1, mutant_with_prpc))
        
        # KRAS mutant + Pritamab: PrPC contribution removed
        scenarios['KRAS Mutant + Pritamab'].append(step['dG_mutant'])
    
    step_names = [s['name'] for s in PATHWAY_STEPS]
    
    # Calculate total pathway energy
    for scenario, energies in scenarios.items():
        total = sum(energies)
        log.info("  %s: total ΔG‡ = %.2f kcal/mol", scenario, total)
        for name, e in zip(step_names, energies):
            log.info("    %s: %.2f kcal/mol", name, e)
    
    # ΔΔG‡ from Pritamab — use RATE-LIMITING STEP (largest contribution)
    # not cumulative sum, since pathway flux is limited by the slowest step
    step_ddGs = [scenarios['KRAS Mutant + Pritamab'][i] - scenarios['KRAS Mutant + PrPC high'][i]
                 for i in range(len(step_names))]
    ddG_rls = max(step_ddGs)  # Rate-limiting step contribution
    ddG_total = sum(step_ddGs)  # Total (for reporting)
    rls_idx = step_ddGs.index(ddG_rls)
    
    log.info("\n  Per-step ΔΔG‡ contributions:")
    for name, d in zip(step_names, step_ddGs):
        log.info("    %s: +%.3f kcal/mol", name, d)
    log.info("  Rate-limiting step: %s (ΔΔG‡ = +%.2f kcal/mol)", 
            step_names[rls_idx], ddG_rls)
    log.info("  Total cumulative ΔΔG‡ = +%.2f kcal/mol", ddG_total)
    
    # Effective ΔΔG‡ for EC50 shift: rate-limiting step × coupling factor
    ddG_effective = ddG_rls
    log.info("  Effective ΔΔG‡ (rate-limiting): +%.2f kcal/mol", ddG_effective)
    log.info("  Coupling factor α = %.2f", ALPHA_COUPLING)
    ddG_for_ec50 = ddG_effective * ALPHA_COUPLING
    log.info("  ΔΔG‡ for EC50 shift (α × ΔΔG‡_RLS): +%.3f kcal/mol", ddG_for_ec50)
    
    # Kinetic consequence at the rate-limiting step
    rate_ratio_rls = np.exp(-ddG_rls / RT)
    log.info("  Rate-limiting step rate ratio: %.3f (%.1f%% reduction)",
            rate_ratio_rls, (1-rate_ratio_rls)*100)
    
    return {
        'scenarios': scenarios,
        'step_names': step_names,
        'ddG_pritamab': ddG_effective,  # Rate-limiting step ΔΔG‡
        'ddG_for_ec50': ddG_for_ec50,   # After coupling factor
        'ddG_total': ddG_total,
        'rate_reduction': 1 - rate_ratio_rls,
        'step_ddGs': step_ddGs,
    }


# ============================================================================
# Part 3: Dose-Response Modeling
# ============================================================================
def hill_equation(conc, EC50, n=1.0):
    """Hill equation: f = C^n / (EC50^n + C^n)"""
    return (conc**n) / (EC50**n + conc**n)


def simulate_dose_response(ddG_pritamab):
    """Model dose-response curves with/without Pritamab."""
    log.info("\n" + "=" * 60)
    log.info("Part 3: Dose-Response Modeling (EC50 Shift)")
    log.info("=" * 60)
    
    results = {}
    
    for drug_key, drug in DRUG_DATA.items():
        EC50_alone = drug['EC50_nM']
        n = drug['hill_n']
        
        # EC50 shift with Pritamab
        # When activation energy barrier increases by ΔΔG‡,
        # less drug is needed → effective EC50 decreases
        # Use α-coupled ΔΔG‡ for biologically realistic shift
        ddG_coupled = ddG_pritamab * ALPHA_COUPLING
        EC50_shift_factor = np.exp(-ddG_coupled / RT)
        EC50_with_pritamab = EC50_alone * EC50_shift_factor
        
        dose_reduction = 1 - (EC50_with_pritamab / EC50_alone)
        
        # Concentration range for dose-response curve
        conc_range = np.logspace(np.log10(EC50_alone * 0.01), 
                                 np.log10(EC50_alone * 100), 200)
        
        f_alone = hill_equation(conc_range, EC50_alone, n)
        f_pritamab = hill_equation(conc_range, EC50_with_pritamab, n)
        
        results[drug_key] = {
            'EC50_alone': EC50_alone,
            'EC50_pritamab': EC50_with_pritamab,
            'dose_reduction': dose_reduction,
            'conc_range': conc_range,
            'f_alone': f_alone,
            'f_pritamab': f_pritamab,
            'hill_n': n,
        }
        
        log.info("  %s:", drug['name'])
        log.info("    EC50 alone:     %.0f nM", EC50_alone)
        log.info("    EC50 + Pritamab: %.0f nM", EC50_with_pritamab)
        log.info("    Dose reduction: %.1f%%", dose_reduction * 100)
        
        # At 50% inhibition, how much less drug needed?
        conc_50_alone = EC50_alone
        conc_50_pritamab = EC50_with_pritamab
        log.info("    For 50%% inhibition: %.0f → %.0f nM (%.1f%% less)", 
                conc_50_alone, conc_50_pritamab, dose_reduction * 100)
    
    return results


# ============================================================================
# Part 4: Combination Analysis (FOLFOX + Pritamab)
# ============================================================================
def combination_analysis(dose_response_results, ddG_pritamab):
    """Bliss independence + Combination Index for FOLFOX + Pritamab."""
    log.info("\n" + "=" * 60)
    log.info("Part 4: Combination Analysis (FOLFOX + Pritamab)")
    log.info("=" * 60)
    
    # FOLFOX = 5-FU + Oxaliplatin
    fu = DRUG_DATA['5-FU']
    oxa = DRUG_DATA['Oxaliplatin']
    
    EC50_shift = np.exp(-ddG_pritamab * ALPHA_COUPLING / RT)
    
    # Dose ranges (fraction of standard dose: 0.1x to 2x)
    fu_doses = np.linspace(0, fu['EC50_nM'] * 3, 50)
    oxa_doses = np.linspace(0, oxa['EC50_nM'] * 3, 50)
    
    # CI matrix: with and without Pritamab
    CI_alone = np.zeros((len(fu_doses), len(oxa_doses)))
    CI_pritamab = np.zeros((len(fu_doses), len(oxa_doses)))
    f_combined_alone = np.zeros((len(fu_doses), len(oxa_doses)))
    f_combined_pritamab = np.zeros((len(fu_doses), len(oxa_doses)))
    
    for i, fd in enumerate(fu_doses):
        for j, od in enumerate(oxa_doses):
            if fd == 0 and od == 0:
                continue
                
            # Without Pritamab
            f_fu = hill_equation(fd, fu['EC50_nM'], fu['hill_n'])
            f_oxa = hill_equation(od, oxa['EC50_nM'], oxa['hill_n'])
            f_comb = 1 - (1 - f_fu) * (1 - f_oxa)  # Bliss
            f_combined_alone[i, j] = f_comb
            if fd > 0 and od > 0:
                CI_alone[i, j] = fd/fu['IC50_nM'] + od/oxa['IC50_nM']
            
            # With Pritamab (shifted EC50)
            f_fu_p = hill_equation(fd, fu['EC50_nM'] * EC50_shift, fu['hill_n'])
            f_oxa_p = hill_equation(od, oxa['EC50_nM'] * EC50_shift, oxa['hill_n'])
            f_comb_p = 1 - (1 - f_fu_p) * (1 - f_oxa_p)  # Bliss
            f_combined_pritamab[i, j] = f_comb_p
            if fd > 0 and od > 0:
                # With Pritamab, effective IC50 is reduced
                CI_pritamab[i, j] = fd/(fu['IC50_nM']*EC50_shift) + od/(oxa['IC50_nM']*EC50_shift)
    
    # Therapeutic index improvement
    # Standard FOLFOX: 5-FU 400mg/m² + Oxa 85mg/m²
    std_fu_conc = fu['EC50_nM'] * 0.8  # ~80% of EC50
    std_oxa_conc = oxa['EC50_nM'] * 0.8
    
    f_std = 1 - (1-hill_equation(std_fu_conc, fu['EC50_nM'], fu['hill_n'])) * \
                (1-hill_equation(std_oxa_conc, oxa['EC50_nM'], oxa['hill_n']))
    
    # With Pritamab, what dose achieves same efficacy?
    for frac in np.arange(0.1, 1.0, 0.01):
        red_fu = std_fu_conc * frac
        red_oxa = std_oxa_conc * frac
        f_red = 1 - (1-hill_equation(red_fu, fu['EC50_nM']*EC50_shift, fu['hill_n'])) * \
                    (1-hill_equation(red_oxa, oxa['EC50_nM']*EC50_shift, oxa['hill_n']))
        if f_red >= f_std:
            log.info("  Standard FOLFOX inhibition: %.1f%%", f_std*100)
            log.info("  Same inhibition with Pritamab at %.0f%% dose", frac*100)
            log.info("  → %.0f%% dose reduction possible", (1-frac)*100)
            break
    
    return {
        'fu_doses': fu_doses,
        'oxa_doses': oxa_doses,
        'CI_alone': CI_alone,
        'CI_pritamab': CI_pritamab,
        'f_combined_alone': f_combined_alone,
        'f_combined_pritamab': f_combined_pritamab,
        'dose_reduction_frac': frac,
    }


# ============================================================================
# Part 5: Figure Generation
# ============================================================================
def generate_figures(tcga_data, energy_data, dose_data, combo_data):
    """Generate publication-quality figures."""
    log.info("\n" + "=" * 60)
    log.info("Part 5: Figure Generation")
    log.info("=" * 60)
    
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
    })
    
    # ================================================================
    # Figure 1: Energy Landscape
    # ================================================================
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    step_names = energy_data['step_names']
    scenarios = energy_data['scenarios']
    colors = {'Normal (WT KRAS)': '#2ecc71', 
              'KRAS Mutant + PrPC high': '#e74c3c',
              'KRAS Mutant + Pritamab': '#3498db'}
    
    x_pos = np.arange(len(step_names))
    width = 0.25
    
    for idx, (label, energies) in enumerate(scenarios.items()):
        bars = ax.bar(x_pos + idx*width - width, energies, width, 
                     label=label, color=colors[label], alpha=0.85,
                     edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, energies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Activation Energy ΔG‡ (kcal/mol)')
    ax.set_xlabel('KRAS Signaling Pathway Step')
    ax.set_title('Fig 1. KRAS Pathway Activation Energy Landscape\n'
                'Effect of PrPC Neutralization by Pritamab')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace(' ', '\n') for s in step_names], fontsize=9)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation for ΔΔG‡
    ax.annotate(f'ΔΔG‡(RLS) = +{energy_data["ddG_pritamab"]:.2f} kcal/mol\n'
               f'α-coupled EC₅₀ shift: {energy_data["ddG_for_ec50"]:.3f} kcal/mol\n'
               f'→ {energy_data["rate_reduction"]*100:.0f}% RLS rate reduction',
               xy=(0.5, 3.5), fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                         edgecolor='orange', alpha=0.9),
               ha='center')
    
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / 'fig1_energy_landscape.png')
    fig1.savefig(FIG_DIR / 'fig1_energy_landscape.pdf')
    log.info("  Saved Fig 1: Energy Landscape")
    plt.close(fig1)
    
    # ================================================================
    # Figure 2: TCGA PRNP by Cancer Type × KRAS Prevalence
    # ================================================================
    if tcga_data is not None:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        df = tcga_data['df']
        res = tcga_data['results_by_ct']
        
        # Panel A: Violin plot of PRNP expression by cancer type
        ct_order = sorted(res.keys(), key=lambda x: res[x]['kras_prevalence'], reverse=True)
        data_list = [df[df['cancer_type'] == ct]['PRNP_log2'].values for ct in ct_order]
        
        parts = ax1.violinplot(data_list, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            kp = KRAS_MUTATION_PREVALENCE.get(ct_order[i], 0)
            color = plt.cm.Reds(kp)
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax1.set_xticks(range(1, len(ct_order)+1))
        labels = [f"{ct}\n(KRAS:{KRAS_MUTATION_PREVALENCE[ct]*100:.0f}%)" for ct in ct_order]
        ax1.set_xticklabels(labels, fontsize=9)
        ax1.set_ylabel('PRNP Expression (log₂ RSEM)')
        ax1.set_title('A. PRNP Expression by Cancer Type\n(Ordered by KRAS Mutation Prevalence)')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Panel B: Scatter plot — PRNP mean vs KRAS prevalence
        prnp_means = [res[ct]['prnp_mean'] for ct in ct_order]
        kras_prevs = [res[ct]['kras_prevalence'] for ct in ct_order]
        sizes = [res[ct]['n'] * 0.5 for ct in ct_order]
        
        ax2.scatter(kras_prevs, prnp_means, s=sizes, c=kras_prevs,
                   cmap='Reds', edgecolors='black', linewidths=0.8, alpha=0.9)
        
        for ct, kp, pm in zip(ct_order, kras_prevs, prnp_means):
            ax2.annotate(ct, (kp, pm), textcoords="offset points", 
                        xytext=(10, 5), fontsize=10, fontweight='bold')
        
        # Trend line
        slope, intercept, r_val, p_val, se = stats.linregress(kras_prevs, prnp_means)
        x_fit = np.linspace(0, 1, 100)
        ax2.plot(x_fit, slope*x_fit + intercept, 'r--', alpha=0.7, linewidth=1.5,
                label=f'r={tcga_data["correlation"]["pearson_r"]:.3f}, '
                      f'p={tcga_data["correlation"]["pearson_p"]:.3f}')
        
        ax2.set_xlabel('KRAS Mutation Prevalence')
        ax2.set_ylabel('Mean PRNP Expression (log₂)')
        ax2.set_title('B. PRNP Expression vs KRAS Mutation Prevalence')
        ax2.legend(fontsize=10, loc='upper left')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        fig2.tight_layout()
        fig2.savefig(FIG_DIR / 'fig2_tcga_prnp_kras.png')
        fig2.savefig(FIG_DIR / 'fig2_tcga_prnp_kras.pdf')
        log.info("  Saved Fig 2: TCGA PRNP-KRAS")
        plt.close(fig2)
    
    # ================================================================
    # Figure 3: Dose-Response Curves (Drug Alone vs Drug + Pritamab)
    # ================================================================
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    drug_colors = {'5-FU': '#e74c3c', 'Oxaliplatin': '#3498db', 
                   'Irinotecan': '#2ecc71', 'Sotorasib': '#9b59b6'}
    
    for idx, (drug_key, dr) in enumerate(dose_data.items()):
        ax = axes[idx//2, idx%2]
        color = drug_colors[drug_key]
        
        conc = dr['conc_range']
        ax.semilogx(conc, dr['f_alone']*100, '-', color=color, linewidth=2.5,
                    label=f'{drug_key} alone')
        ax.semilogx(conc, dr['f_pritamab']*100, '--', color=color, linewidth=2.5,
                    alpha=0.7, label=f'{drug_key} + Pritamab')
        
        # Mark EC50 positions
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(dr['EC50_alone'], color=color, linestyle=':', alpha=0.4)
        ax.axvline(dr['EC50_pritamab'], color=color, linestyle=':', alpha=0.4)
        
        # Arrow showing EC50 shift
        ax.annotate('', xy=(dr['EC50_pritamab'], 50), 
                   xytext=(dr['EC50_alone'], 50),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(np.sqrt(dr['EC50_alone'] * dr['EC50_pritamab']), 55, 
               f'{dr["dose_reduction"]*100:.0f}% ↓', ha='center', fontsize=10,
               fontweight='bold', color='darkred')
        
        ax.set_xlabel('Drug Concentration (nM)')
        ax.set_ylabel('Inhibition (%)')
        ax.set_title(f'{DRUG_DATA[drug_key]["name"]}')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(-5, 105)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig3.suptitle('Fig 3. Dose-Response Curves: EC₅₀ Shift with Pritamab', 
                 fontsize=14, fontweight='bold', y=1.02)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / 'fig3_dose_response.png')
    fig3.savefig(FIG_DIR / 'fig3_dose_response.pdf')
    log.info("  Saved Fig 3: Dose-Response")
    plt.close(fig3)
    
    # ================================================================
    # Figure 4: Combination Heatmap (FOLFOX ± Pritamab)
    # ================================================================
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fu_doses = combo_data['fu_doses'] / DRUG_DATA['5-FU']['EC50_nM']
    oxa_doses = combo_data['oxa_doses'] / DRUG_DATA['Oxaliplatin']['EC50_nM']
    
    im1 = ax1.contourf(oxa_doses, fu_doses, combo_data['f_combined_alone']*100,
                       levels=np.arange(0, 105, 10), cmap='YlOrRd')
    ax1.contour(oxa_doses, fu_doses, combo_data['f_combined_alone']*100,
               levels=[50], colors='black', linewidths=2)
    plt.colorbar(im1, ax=ax1, label='Inhibition (%)')
    ax1.set_xlabel('Oxaliplatin (× EC₅₀)')
    ax1.set_ylabel('5-FU (× EC₅₀)')
    ax1.set_title('A. FOLFOX Alone')
    
    im2 = ax2.contourf(oxa_doses, fu_doses, combo_data['f_combined_pritamab']*100,
                       levels=np.arange(0, 105, 10), cmap='YlOrRd')
    ax2.contour(oxa_doses, fu_doses, combo_data['f_combined_pritamab']*100,
               levels=[50], colors='black', linewidths=2)
    plt.colorbar(im2, ax=ax2, label='Inhibition (%)')
    ax2.set_xlabel('Oxaliplatin (× EC₅₀)')
    ax2.set_ylabel('5-FU (× EC₅₀)')
    ax2.set_title('B. FOLFOX + Pritamab')
    
    fig4.suptitle(f'Fig 4. FOLFOX Combination: {(1-combo_data["dose_reduction_frac"])*100:.0f}% '
                 f'Dose Reduction with Pritamab', fontsize=14, fontweight='bold', y=1.02)
    fig4.tight_layout()
    fig4.savefig(FIG_DIR / 'fig4_combination_heatmap.png')
    fig4.savefig(FIG_DIR / 'fig4_combination_heatmap.pdf')
    log.info("  Saved Fig 4: Combination Heatmap")
    plt.close(fig4)
    
    # ================================================================
    # Figure 5: Therapeutic Index (Normal vs Tumor)
    # ================================================================
    fig5, ax = plt.subplots(figsize=(10, 6))
    
    # Concept: tumor has high PrPC → Pritamab effective
    # Normal tissue: low PrPC → Pritamab minimal effect
    conc = np.logspace(1, 5, 200)
    
    fu_EC50 = DRUG_DATA['5-FU']['EC50_nM']
    EC50_shift = np.exp(-energy_data['ddG_for_ec50'] / RT)
    
    # Tumor cell kill
    f_tumor_alone = hill_equation(conc, fu_EC50, 1.2)
    f_tumor_pritamab = hill_equation(conc, fu_EC50 * EC50_shift, 1.2)
    
    # Normal cell toxicity (no PrPC → no Pritamab effect)
    f_normal = hill_equation(conc, fu_EC50 * 1.5, 1.0)  # Normal cells more resistant
    
    ax.semilogx(conc, f_tumor_alone*100, 'r-', linewidth=2.5, label='Tumor (5-FU alone)')
    ax.semilogx(conc, f_tumor_pritamab*100, 'r--', linewidth=2.5, 
               label='Tumor (5-FU + Pritamab)')
    ax.semilogx(conc, f_normal*100, 'b-', linewidth=2.5, label='Normal tissue')
    
    # Shade therapeutic window
    # Standard dose point
    std_dose = fu_EC50 * 0.8
    ax.axvline(std_dose, color='gray', linestyle=':', alpha=0.5, label='Standard dose')
    
    # Reduced dose with Pritamab
    reduced_dose = std_dose * combo_data['dose_reduction_frac']
    ax.axvline(reduced_dose, color='green', linestyle=':', alpha=0.7, label='Reduced dose')
    
    # Annotation: therapeutic window
    f_tumor_std = hill_equation(std_dose, fu_EC50, 1.2) * 100
    f_normal_std = hill_equation(std_dose, fu_EC50*1.5, 1.0) * 100
    f_tumor_red = hill_equation(reduced_dose, fu_EC50*EC50_shift, 1.2) * 100
    f_normal_red = hill_equation(reduced_dose, fu_EC50*1.5, 1.0) * 100
    
    ax.annotate(f'Standard: TI = {f_tumor_std/max(f_normal_std,0.1):.1f}',
               xy=(std_dose, f_tumor_std), xytext=(std_dose*3, f_tumor_std-10),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=10, color='red')
    
    ti_reduced = f_tumor_red / max(f_normal_red, 0.1)
    ax.annotate(f'Reduced+Pritamab: TI = {ti_reduced:.1f}',
               xy=(reduced_dose, f_tumor_red), xytext=(reduced_dose*3, f_tumor_red+10),
               arrowprops=dict(arrowstyle='->', color='green'),
               fontsize=10, color='green')
    
    ax.set_xlabel('5-FU Concentration (nM)')
    ax.set_ylabel('Cell Kill / Toxicity (%)')
    ax.set_title('Fig 5. Therapeutic Index Improvement:\n'
                'Tumor-Selective PrPC Expression Enables Dose Reduction')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(-5, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig5.tight_layout()
    fig5.savefig(FIG_DIR / 'fig5_therapeutic_index.png')
    fig5.savefig(FIG_DIR / 'fig5_therapeutic_index.pdf')
    log.info("  Saved Fig 5: Therapeutic Index")
    plt.close(fig5)


# ============================================================================
# Main
# ============================================================================
def main():
    log.info("=" * 60)
    log.info("Paper 3: Pritamab + KRAS Activation Energy Framework")
    log.info("=" * 60)
    
    # Part 1: TCGA Analysis
    tcga_data = analyze_tcga_prnp_kras()
    
    # Part 2: Energy Landscape
    energy_data = simulate_energy_landscape()
    
    # Part 3: Dose-Response Modeling
    dose_data = simulate_dose_response(energy_data['ddG_pritamab'])
    
    # Part 4: Combination Analysis
    combo_data = combination_analysis(dose_data, energy_data['ddG_pritamab'])
    
    # Part 5: Figures
    generate_figures(tcga_data, energy_data, dose_data, combo_data)
    
    # Save results JSON
    results = {
        'energy_landscape': {
            'ddG_rls_kcal_mol': energy_data['ddG_pritamab'],
            'ddG_for_ec50_kcal_mol': energy_data['ddG_for_ec50'],
            'alpha_coupling': ALPHA_COUPLING,
            'rate_reduction_pct': energy_data['rate_reduction'] * 100,
            'scenarios': {k: v for k, v in energy_data['scenarios'].items()},
        },
        'dose_response': {
            drug: {
                'EC50_alone_nM': float(dr['EC50_alone']),
                'EC50_pritamab_nM': float(dr['EC50_pritamab']),
                'dose_reduction_pct': float(dr['dose_reduction'] * 100),
            } for drug, dr in dose_data.items()
        },
        'combination': {
            'folfox_dose_reduction_pct': float((1 - combo_data['dose_reduction_frac']) * 100),
        },
    }
    
    if tcga_data is not None:
        results['tcga_correlation'] = tcga_data['correlation']
    
    with open(OUT_DIR / 'paper3_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("  ΔΔG‡ from Pritamab: +%.2f kcal/mol", energy_data['ddG_pritamab'])
    log.info("  Signaling reduction: %.1f%%", energy_data['rate_reduction']*100)
    log.info("  FOLFOX dose reduction: %.0f%%", (1-combo_data['dose_reduction_frac'])*100)
    log.info("  All figures saved to: %s", FIG_DIR)
    log.info("  Results saved to: %s", OUT_DIR / 'paper3_results.json')


if __name__ == "__main__":
    main()
