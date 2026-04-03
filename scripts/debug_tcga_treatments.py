"""Investigate TCGA case treatment structure"""
import json

with open("F:/ADDS/data/ml_training/tcga/tcga_coad_cases.json") as f:
    cases = json.load(f)

# Investigate treatment structure
for i, case in enumerate(cases[:10]):
    case_id = case.get("submitter_id", "?")
    diags = case.get("diagnoses", [])
    if diags:
        d = diags[0]
        treatments = d.get("treatments", [])
        stage = d.get("ajcc_pathologic_stage", "?")
        print(f"Case {i}: {case_id}")
        print(f"  Stage: {stage}")
        print(f"  Treatment count: {len(treatments)}")
        for t in treatments:
            print(f"    Type: {t.get('treatment_type', '?')}")
            print(f"    Agent: {t.get('therapeutic_agents', '?')}")
        print(f"  Diag keys: {list(d.keys())}")
        print()

# Count treatment types
all_types = {}
n_with_treatment = 0
for case in cases:
    diags = case.get("diagnoses", [])
    if diags:
        d = diags[0]
        ts = d.get("treatments", [])
        if ts:
            n_with_treatment += 1
            for t in ts:
                tt = t.get("treatment_type", "unknown")
                all_types[tt] = all_types.get(tt, 0) + 1

print(f"\nCases with treatments: {n_with_treatment}/{len(cases)}")
print(f"Treatment types:")
for tt, n in sorted(all_types.items(), key=lambda x: -x[1]):
    print(f"  {tt}: {n}")
