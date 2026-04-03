"""API integration test for Energy Framework endpoints."""
import sys, os, json
from pathlib import Path

sys.path.insert(0, 'F:/ADDS')
sys.path.insert(0, 'F:/ADDS/scripts')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

passed, failed = 0, 0

def test(name, fn):
    global passed, failed
    print(f"\n=== {name} ===")
    try:
        fn()
        passed += 1
        print("  PASS")
    except Exception as e:
        failed += 1
        print(f"  FAIL: {e}")

# Test 1: Model loading
def t1():
    from track2_energy_pinn_v3 import EnergyPredictorV3
    import torch
    for fname in ['energy_predictor_v3_calibrated_v2.pt',
                  'energy_predictor_v3_calibrated.pt',
                  'energy_predictor_v3.pt']:
        p = Path(f'F:/ADDS/models/energy/{fname}')
        if p.exists():
            ckpt = torch.load(p, map_location='cpu', weights_only=False)
            model = EnergyPredictorV3(n_pk=7)
            model.load_state_dict(ckpt['model_state'])
            model.eval()
            r = ckpt.get('pearson_r', 'N/A')
            print(f"  Loaded: {fname}, r={r}")
            return
    raise FileNotFoundError("No model checkpoint found")

test("Model Loading", t1)

# Test 2: Schema validation
def t2():
    from backend.schemas.energy_schemas import (
        EnergyPredictRequest, EnergyPredictResponse,
        DrugCombValidationResponse, PathwayGraphResponse,
        CalibrationStatusResponse
    )
    req = EnergyPredictRequest(drug_name='Cetuximab', kd_nm=0.5)
    print(f"  EnergyPredictRequest: drug={req.drug_name}, kd={req.kd_nm}")
    resp = EnergyPredictResponse(
        drug_name='test', binding_energy_kcal=-10.5,
        predicted_tumor_suppression_pct=75.0,
        predicted_ic50_nm=15.0)
    print(f"  EnergyPredictResponse: ts={resp.predicted_tumor_suppression_pct}%")

test("Schema Validation", t2)

# Test 3: Saved results files
def t3():
    for fname, label in [
        ('drugcomb_validation_v2.json', 'v2 (improved)'),
        ('drugcomb_validation.json', 'v1'),
    ]:
        p = Path(f'F:/ADDS/models/energy/{fname}')
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            n = r.get('n_datapoints', '?')
            cells = r.get('n_cell_lines', 1)
            pr = r.get('pearson_r', '?')
            sa = r.get('synergy_accuracy', '?')
            aa = r.get('antagonist_accuracy', '?')
            print(f"  {label}: n={n}, cells={cells}, r={pr}, syn_acc={sa}, ant_acc={aa}")

test("Validation Results", t3)

# Test 4: PPI data
def t4():
    p = Path('F:/ADDS/data/real_ppi/ppi_summary.json')
    assert p.exists(), "ppi_summary.json not found"
    with open(p) as f:
        s = json.load(f)
    print(f"  STRING edges: {s.get('string_edges', 0)}")
    print(f"  Literature: {s.get('literature_edges', 0)}")
    print(f"  Merged: {s.get('merged_edges', 0)}")
    print(f"  Co-IP evidence: {s.get('edges_with_coip', 0)}")

test("PPI Data", t4)

# Test 5: Competitive binding detection
def t5():
    sys.path.insert(0, 'F:/ADDS/scripts')
    from validate_drugcomb_v2 import detect_competitive_binding
    # Same target antibodies should be detected as competitive
    assert detect_competitive_binding("Cetuximab", "Panitumumab") == 1.0
    print("  Cetuximab + Panitumumab: penalty=1.0 (same EGFR mAb)")
    # Same mechanism prodrugs
    assert detect_competitive_binding("5-Fluorouracil", "Capecitabine") == 1.0
    print("  5-FU + Capecitabine: penalty=1.0 (same TS inhibitor)")
    # Different mechanisms should NOT be competitive
    assert detect_competitive_binding("Cetuximab", "Bevacizumab") == 0.0
    print("  Cetuximab + Bevacizumab: penalty=0.0 (different targets)")
    # Partial overlap
    p = detect_competitive_binding("5-Fluorouracil", "Oxaliplatin")
    assert 0 < p < 1
    print(f"  5-FU + Oxaliplatin: penalty={p} (partial overlap)")

test("Competitive Binding Detection", t5)

# Test 6: Energy API endpoint import
def t6():
    from backend.api.energy_api import router, _load_model
    routes = [r.path for r in router.routes]
    print(f"  Routes: {routes}")
    assert '/predict' in routes
    assert '/validate' in routes
    assert '/pathway-graph' in routes
    assert '/calibration-status' in routes

test("API Router", t6)

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'='*50}")
