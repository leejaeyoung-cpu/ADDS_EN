# OpenAI Medical Interpretation Test
# Test the OpenAI integration for CDSS

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from medical_imaging.cdss.integration_engine import (
    CDSSIntegrationEngine,
    CellposeResults,
    CTDetectionResults,
    ClinicalData
)

from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()

def test_openai_interpretation():
    """Test OpenAI medical interpretation generation"""
    
    print("="*60)
    print("OpenAI Medical Interpretation Test")
    print("="*60)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found in .env")
        return False
    
    print(f"\n[OK] API Key found: {api_key[:20]}...")
    
    # Create mock data
    print("\n[Step 1] Creating patient profile...")
    
    cellpose = CellposeResults(
        cell_count=2450,
        mean_area_um2=185.0,
        mean_circularity=0.78,
        morphology_score=9.1,
        ki67_index=0.45
    )
    
    ct = CTDetectionResults(
        tumor_detected=True,
        total_candidates=33,
        high_conf_candidates=7,
        max_confidence=0.963,
        tumor_size_mm=15.2,
        tumor_location="Sigmoid colon",
        tnm_stage="T2N1M0"
    )
    
    clinical = ClinicalData(
        patient_id="P12345",
        age=58,
        gender="M",
        kras_status="Wild-type",
        msi_status="MSS",
        liver_function="Normal",
        kidney_function="Normal",
        ecog_performance=0
    )
    
    print("[OK] Patient data created")
    
    # Initialize engine with OpenAI
    print("\n[Step 2] Initializing Integration Engine with OpenAI...")
    
    try:
        openai_client = OpenAI()
        engine = CDSSIntegrationEngine(openai_client=openai_client)
        print("[OK] Engine initialized with OpenAI client")
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI: {e}")
        return False
    
    # Integrate data
    print("\n[Step 3] Running integration...")
    
    try:
        profile = engine.integrate_patient_data(cellpose, ct, clinical)
        print("[OK] Integration complete")
        print(f"  - Cancer Stage: {profile.cancer_stage}")
        print(f"  - Risk Level: {profile.risk_level}")
        print(f"  - 5-Year Survival: {profile.prognosis_5yr_survival*100:.1f}%")
    except Exception as e:
        print(f"[ERROR] Integration failed: {e}")
        return False
    
    # Test doctor interpretation
    print("\n[Step 4] Generating doctor interpretation...")
    print("-"*60)
    
    if profile.doctor_interpretation:
        print("[SUCCESS] Doctor interpretation generated:")
        print("\n" + "="*60)
        print(profile.doctor_interpretation)
        print("="*60)
    else:
        print("[WARNING] No doctor interpretation generated")
    
    # Test patient interpretation
    print("\n[Step 5] Generating patient interpretation...")
    print("-"*60)
    
    if profile.patient_interpretation:
        print("[SUCCESS] Patient interpretation generated:")
        print("\n" + "="*60)
        print(profile.patient_interpretation)
        print("="*60)
    else:
        print("[WARNING] No patient interpretation generated")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    has_doctor = profile.doctor_interpretation is not None
    has_patient = profile.patient_interpretation is not None
    
    print(f"Doctor Interpretation: {'[OK]' if has_doctor else '[FAIL]'}")
    print(f"Patient Interpretation: {'[OK]' if has_patient else '[FAIL]'}")
    
    if has_doctor and has_patient:
        print("\n[SUCCESS] OpenAI integration working correctly!")
        print("[SUCCESS] Medical interpretations generated successfully!")
        return True
    else:
        print("\n[WARNING] Some interpretations missing")
        return False

if __name__ == "__main__":
    try:
        success = test_openai_interpretation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
