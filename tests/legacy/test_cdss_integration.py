# CDSS Integration Engine - Validation Test
# Mock data test without Unicode characters

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from medical_imaging.cdss.integration_engine import (
    CDSSIntegrationEngine,
    CellposeResults,
    CTDetectionResults,
    ClinicalData
)

def main():
    print("="*60)
    print("ADDS CDSS Integration Engine - Validation Test")
    print("="*60)
    
    # Create mock data
    print("\n[Step 1] Creating mock data...")
    
    cellpose = CellposeResults(
        cell_count=2450,
        mean_area_um2=185.0,
        mean_circularity=0.78,
        morphology_score=9.1,
        ki67_index=0.45
    )
    print(f"[OK] Cellpose: {cellpose.cell_count} cells, Ki-67: {cellpose.ki67_index*100:.0f}%")
    
    ct = CTDetectionResults(
        tumor_detected=True,
        total_candidates=33,
        high_conf_candidates=7,
        max_confidence=0.963,
        tumor_size_mm=15.2,
        tumor_location="Sigmoid colon",
        tnm_stage="T2N1M0"
    )
    print(f"[OK] CT: Tumor detected, TNM: {ct.tnm_stage}, Confidence: {ct.max_confidence*100:.1f}%")
    
    clinical = ClinicalData(
        patient_id="P12345",
        age=58,
        gender="M",
        kras_status="Wild-type",
        tp53_status="Wild-type",
        msi_status="MSS",
        liver_function="Normal",
        kidney_function="Normal",
        ecog_performance=0,
        comorbidities=[]
    )
    print(f"[OK] Clinical: {clinical.age}y {clinical.gender}, ECOG: {clinical.ecog_performance}")
    
    # Initialize engine
    print("\n[Step 2] Initializing Integration Engine...")
    engine = CDSSIntegrationEngine(openai_client=None)
    print("[OK] Engine initialized")
    
    # Integrate data
    print("\n[Step 3] Integrating patient data...")
    profile = engine.integrate_patient_data(cellpose, ct, clinical)
    print("[OK] Integration complete")
    
    # Verify results
    print("\n[Step 4] Verifying results...")
    print("-"*60)
    
    tests_passed = []
    
    # Test 1: Cancer stage
    expected_stage = "IIB"
    actual_stage = profile.cancer_stage
    test_pass = actual_stage == expected_stage
    tests_passed.append(test_pass)
    status = "[PASS]" if test_pass else "[FAIL]"
    print(f"{status} Cancer Stage: {actual_stage} (Expected: {expected_stage})")
    
    # Test 2: Risk level
    expected_risk = "Medium-High"
    actual_risk = profile.risk_level
    test_pass = actual_risk == expected_risk
    tests_passed.append(test_pass)
    status = "[PASS]" if test_pass else "[FAIL]"
    print(f"{status} Risk Level: {actual_risk} (Expected: {expected_risk})")
    
    # Test 3: Prognosis
    actual_prognosis = profile.prognosis_5yr_survival
    test_pass = 0.60 <= actual_prognosis <= 0.75
    tests_passed.append(test_pass)
    status = "[PASS]" if test_pass else "[FAIL]"
    print(f"{status} 5-Year Survival: {actual_prognosis*100:.1f}% (Expected: 60-75%)")
    
    # Test 4: Therapy count
    therapy_count = len(profile.recommended_therapies)
    test_pass = therapy_count >= 2
    tests_passed.append(test_pass)
    status = "[PASS]" if test_pass else "[FAIL]"
    print(f"{status} Therapy Options: {therapy_count} (Expected: >= 2)")
    
    # Test 5: Primary therapy
    if profile.recommended_therapies:
        first_therapy = profile.recommended_therapies[0]
        test_pass = "FOLFOX" in first_therapy.therapy_name
        tests_passed.append(test_pass)
        status = "[PASS]" if test_pass else "[FAIL]"
        print(f"{status}   - Primary: {first_therapy.therapy_name}")
        print(f"      - Confidence: {first_therapy.confidence*100:.0f}%")
        print(f"      - Efficacy: {first_therapy.predicted_efficacy*100:.0f}%")
        print(f"      - Drugs: {', '.join(first_therapy.drug_combination[:2])}...")
    
    # Test 6: JSON serialization
    profile_dict = profile.to_dict()
    test_pass = isinstance(profile_dict, dict) and 'patient_id' in profile_dict
    tests_passed.append(test_pass)
    status = "[PASS]" if test_pass else "[FAIL]"
    print(f"{status} JSON Serialization")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(tests_passed)
    total = len(tests_passed)
    
    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("[SUCCESS] CDSS Integration Engine is working correctly!")
        print("[SUCCESS] Ready for Streamlit UI testing")
        return 0
    else:
        print("\n[WARNING] Some tests failed")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
