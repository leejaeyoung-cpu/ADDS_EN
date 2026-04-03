"""
verify_energy_pipeline.py
Full integration test: Step 1 → Step 2 → Step 3
"""
import sys, os, warnings
sys.path.insert(0, r'f:\ADDS')
warnings.filterwarnings('ignore')

PASS, FAIL = [], []

def ck(name, cond, d=''):
    if cond:
        PASS.append(name)
        print(f'  PASS | {name}')
    else:
        FAIL.append(name)
        print(f'  FAIL | {name} | {str(d)[:70]}')

# ──────────────────────────────────────────────────────────────────────────────
print('=== STEP 1: imaging_to_energy ===')
from src.pritamab_ml.imaging_to_energy import ImagingToEnergyMapper, PatientEnergyProfile
mapper = ImagingToEnergyMapper()

sA = [{'total_cells':520,'mean_area_um2':310,'mean_circularity':0.50,
        'irregular_count':290,'normal_count':120}] * 3
ctA = {'tumor_volume_cc':45,'mean_hu':-8,'std_hu':50,'necrosis_ratio':0.20}
pA = mapper.compute_profile(sA, ctA, 'G12D')

ck('ProfileA type', isinstance(pA, PatientEnergyProfile))
kras_ddg = pA.ddg_per_pathway['KRAS_ERK']
min_ddg   = min(pA.ddg_per_pathway.values())
ck('G12D: KRAS has lowest ddg', abs(kras_ddg - min_ddg) < 1e-6, pA.ddg_per_pathway)
ck('G12D: sensitivity_rank[0]=KRAS_ERK', pA.sensitivity_rank[0] == 'KRAS_ERK', pA.sensitivity_rank)
ck('ddg_effective in (0.1, 1.5)', 0.1 < pA.ddg_effective < 1.5, pA.ddg_effective)
ck('all 4 pathways present', len(pA.ddg_per_pathway) == 4, pA.ddg_per_pathway.keys())

sB = [{'total_cells':80,'mean_area_um2':520,'mean_circularity':0.75,
        'irregular_count':10,'normal_count':65}] * 3
ctB = {'tumor_volume_cc':12,'mean_hu':-35,'std_hu':70,'necrosis_ratio':0.45}
pB = mapper.compute_profile(sB, ctB, 'WT')

hif_rank = pB.sensitivity_rank.index('HIF_VEGF')
ck('Hypoxia case: HIF in top-2', hif_rank < 2, pB.sensitivity_rank)
ck('Low-prolif: ddg_eff(B) > ddg_eff(A)',
   pB.ddg_effective > pA.ddg_effective,
   f'B={pB.ddg_effective:.3f} A={pA.ddg_effective:.3f}')

pC = mapper.compute_profile(sA[:1], ct_data=None, kras_allele='G12V')
ck('No-CT profile OK', isinstance(pC, PatientEnergyProfile))

# ──────────────────────────────────────────────────────────────────────────────
print()
print('=== STEP 2: pkpd_feature_module ===')
from src.pritamab_ml.pkpd_feature_module import PKPDFeatureModule
mod = PKPDFeatureModule()

f_static = mod.compute_features(5, kras_allele='G12D')
ck('Static shape (5,32)', f_static.shape == (5, 32), f_static.shape)
ck('Static: no NaN', not bool((f_static != f_static).any()))

f_dyn = mod.compute_features(5, kras_allele='G12D', patient_profile=pA)
ck('Dynamic shape (5,32)', f_dyn.shape == (5, 32))
expected_f16 = pA.ddg_per_pathway['KRAS_ERK'] / 2.0
ck('Dynamic feature[16]=KRAS ddg/2',
   abs(f_dyn[0, 16] - expected_f16) < 1e-5,
   f'{f_dyn[0,16]:.4f} vs {expected_f16:.4f}')
expected_f20 = pA.k_per_pathway['KRAS_ERK']
ck('Dynamic feature[20]=k_KRAS',
   abs(f_dyn[0, 20] - expected_f20) < 1e-5,
   f'{f_dyn[0,20]:.4f} vs {expected_f20:.4f}')
ck('Dynamic != Static in [16:24]',
   not (f_dyn[:, 16:24] == f_static[:, 16:24]).all())

# ──────────────────────────────────────────────────────────────────────────────
print()
print('=== STEP 3: pathway_drug_optimizer ===')
from src.pritamab_ml.pathway_drug_optimizer import PathwayDrugOptimizer, DrugCocktailRecommendation
opt = PathwayDrugOptimizer(max_drugs=4)

recA = opt.optimize(pA, kras_allele='G12D')
ck('RecA type', isinstance(recA, DrugCocktailRecommendation))
ck('RecA: Pritamab included', 'Pritamab' in recA.recommended_drugs, recA.recommended_drugs)
kras_drugs = {'Sotorasib', 'Trametinib', 'FOLFOX', 'FOLFIRI'}
ck('RecA: G12D -> KRAS-targeting drug present',
   bool(kras_drugs & set(recA.recommended_drugs)), recA.recommended_drugs)
ck('RecA: doses in (0.1, 10)',
   all(0.1 < v < 10 for v in recA.doses_relative.values()), recA.doses_relative)
ck('RecA: synergy matrix (4,4)',
   recA.synergy_matrix.shape == (4, 4), recA.synergy_matrix.shape)
ck('RecA: risk_score in [0,1]', 0 <= recA.risk_score <= 1, recA.risk_score)
ck('RecA: narrative non-empty', len(recA.narrative) > 20)

recB = opt.optimize(pB, kras_allele='WT')
hif_drugs = {'Bevacizumab', 'Lenvatinib', 'Regorafenib'}
ck('RecB: HIF-targeting drug (저산소)',
   bool(hif_drugs & set(recB.recommended_drugs)), recB.recommended_drugs)
ck('RecB: Pritamab included', 'Pritamab' in recB.recommended_drugs)

# ──────────────────────────────────────────────────────────────────────────────
print()
print(f'=== FINAL: PASS {len(PASS)} / FAIL {len(FAIL)} / TOTAL {len(PASS)+len(FAIL)} ===')
if FAIL:
    print('FAILED items:')
    for f in FAIL:
        print(f'  - {f}')
else:
    print('All checks passed!')

print()
print('--- Case A (G12D) Recommendation ---')
print(f'  Pathway rank : {recA.pathway_priority}')
print(f'  Cocktail     : {recA.recommended_drugs}')
print(f'  Doses        : {recA.doses_relative}')
print(f'  Coverage     : {recA.coverage_scores}')
print(f'  Risk         : {recA.risk_score:.1%}')

print()
print('--- Case B (WT, Hypoxia) Recommendation ---')
print(f'  Pathway rank : {recB.pathway_priority}')
print(f'  Cocktail     : {recB.recommended_drugs}')
