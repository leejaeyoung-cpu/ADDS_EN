import json, numpy as np

with open(r'f:\ADDS\models\pritamab_fusion_v4_report.json') as f:
    r = json.load(f)

print('=== v4 Final Report ===')
print('Dataset:', r['dataset_size'], 'samples')
print('5-CV r_syn:', round(r['cv_5fold']['mean'],4), '+/-', round(r['cv_5fold']['std'],4))
print('Best val r_syn:', r['full_train']['best_r_syn'], '@ ep', r['full_train']['best_ep'])
print('Drug-rank rho:', r['drug_rank']['rho'], ' Top-2:', r['drug_rank']['top2'])
print()

probs = r['drug_rank']['drug_probs']
gt_bliss = {'Pritamab+Oxali':22.5,'Oxaliplatin':21.7,'FOLFOX':20.5,
            'FOLFIRI':18.8,'5-FU':18.4,'FOLFOXIRI':18.1,'Irinotecan':17.3}
drugs = list(gt_bliss.keys())
pred_v = np.array([probs[d] for d in drugs])
gt_v   = np.array([gt_bliss[d] for d in drugs])

print('Drug predictions (sorted by GT):')
gt_sorted = sorted(gt_bliss.items(), key=lambda x: -x[1])
for rank, (d, bliss) in enumerate(gt_sorted, 1):
    pred_rank = sorted(probs.items(), key=lambda x: -x[1])
    pred_rank_num = [i+1 for i,(k,_) in enumerate(pred_rank) if k==d][0]
    ok = 'OK' if pred_rank_num == rank else ('+-1' if abs(pred_rank_num-rank)<=1 else 'MISS')
    print(f'  GT#{rank} {d:22s}: pred#{pred_rank_num}  syn_prob={probs[d]:.4f}  [{ok}]')

print()
print('Key finding: FOLFOXIRI pred#1 (actually GT#6)')
print('FOLFOXIRI has Oxaliplatin+Irinotecan -> both potent -> inflated score')
print()
print('=== VERSION HISTORY ===')
print('v1: r_syn=0.631, drug-rank rho=0.371')
print('v2: r_syn=0.571, drug-rank rho=0.657')
print('v3b:r_syn=0.631, drug-rank rho=0.500')
print('v4: r_syn=0.937, drug-rank rho=0.393')
print()
print('r_syn DRAMATICALLY improved (0.63->0.937)')
print('drug-rank rho dropped because FOLFOXIRI feature vector')
print('gets boosted by Oxaliplatin+Irinotecan dual high-potency')
