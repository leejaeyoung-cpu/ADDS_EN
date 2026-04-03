import sys, warnings
sys.path.insert(0, r'f:\ADDS')
warnings.filterwarnings('ignore')

from src.pritamab_ml.imaging_to_energy import ImagingToEnergyMapper, BASELINE_DDG, MAX_DELTA_DDG, RT_KCAL
from src.pritamab_ml.pathway_drug_optimizer import PathwayDrugOptimizer
import numpy as np

mapper = ImagingToEnergyMapper()
optimizer = PathwayDrugOptimizer(max_drugs=4)

cases = [
    {
        'name': 'Case A: KRAS G12D 고증식형',
        'cellpose': [{'total_cells':520,'mean_area_um2':310,'mean_circularity':0.50,'irregular_count':290,'normal_count':120}]*3,
        'ct': {'tumor_volume_cc':45,'mean_hu':-8,'std_hu':50,'necrosis_ratio':0.20},
        'kras': 'G12D'
    },
    {
        'name': 'Case B: KRAS WT 저산소 주도형',
        'cellpose': [{'total_cells':80,'mean_area_um2':520,'mean_circularity':0.75,'irregular_count':10,'normal_count':65}]*3,
        'ct': {'tumor_volume_cc':12,'mean_hu':-35,'std_hu':70,'necrosis_ratio':0.45},
        'kras': 'WT'
    },
    {
        'name': 'Case C: G12V 고침윤형',
        'cellpose': [{'total_cells':200,'mean_area_um2':280,'mean_circularity':0.38,'irregular_count':150,'normal_count':30}]*5,
        'ct': {'tumor_volume_cc':28,'mean_hu':15,'std_hu':35,'necrosis_ratio':0.05},
        'kras': 'G12V'
    },
    {
        'name': 'Case D: G12C 초기형 (CT 없음)',
        'cellpose': [{'total_cells':60,'mean_area_um2':400,'mean_circularity':0.70,'irregular_count':8,'normal_count':48}],
        'ct': None,
        'kras': 'G12C'
    },
]

print("=" * 70)
print("CASE RESULTS")
print("=" * 70)
for c in cases:
    p = mapper.compute_profile(c['cellpose'], c['ct'], c['kras'])
    rec = optimizer.optimize(p, kras_allele=c['kras'])
    print(f"\n{c['name']}")
    print(f"  indices : {p.imaging_indices}")
    print(f"  ddg     : {p.ddg_per_pathway}")
    print(f"  k       : {p.k_per_pathway}")
    print(f"  rank    : {p.sensitivity_rank}")
    print(f"  ddg_eff : {p.ddg_effective}")
    print(f"  cocktail: {rec.recommended_drugs}")
    print(f"  doses   : {rec.doses_relative}")

print("\n" + "=" * 70)
print("SENSITIVITY RANGE (baseline → max activation)")
print("=" * 70)
for pw, base in BASELINE_DDG.items():
    min_ddg = base * 0.25   # 75% max reduction
    k_base = np.exp(-base / RT_KCAL)
    k_max  = np.exp(-min_ddg / RT_KCAL)
    ratio  = k_max / k_base
    print(f"  {pw:12s}: ddg [{min_ddg:.3f},{base:.3f}] kcal/mol | k_ratio={ratio:.1f}x")
