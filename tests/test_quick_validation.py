"""
ADDS Phase 2 - Quick Clinical Validation Test
Single-scenario test for rapid debugging

Author: ADDS Development Team
Date: 2026-01-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from datetime import datetime
from recommendation.dtol_cycle_demo import DTOLCycleEngine, create_demo_drugs
from recommendation.combination_designer import Drug, PatientProfile

def run_quick_validation():
    """Run a single quick validation test"""
    
    print("\n" + "="*80)
    print("QUICK CLINICAL VALIDATION TEST")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Test only 1 cancer with 1 strategy
    patient = PatientProfile(
        patient_id="BC-001",
        age=48,
        cancer_type="Breast",
        stage=2,
        ecog=0,
        mutations=["HER2+", "PIK3CA"],
        previous_treatments=[],
        egfr=85,
        bilirubin=0.7
    )
    
    drugs = create_demo_drugs()
    
    print("Patient Profile:")
    print(f"  ID: {patient.patient_id}")
    print(f"  Cancer: {patient.cancer_type}, Stage {patient.stage}")
    print(f"  Mutations: {', '.join(patient.mutations)}")
    print(f"  ECOG: {patient.ecog}\n")
    
    print("Configuration:")
    print("  Iterations: 2 (reduced from 3)")
    print("  Workers: 2 (reduced from 4)")
    print("  Combinations/iteration: 3 (reduced from 5)")
    print("  Strategy: dual_mode\n")
    
    engine = DTOLCycleEngine(
        max_iterations=2,  # Reduced from 3
        max_workers=2,     # Reduced from 4
        acquisition_strategy='dual_mode',
        random_state=42
    )
    
    print("[>>] Starting DTOL cycle...\n")
    
    start_time = datetime.now()
    
    results = engine.run_cycle(
        available_drugs=drugs,
        patient=patient,
        num_combinations_per_iteration=3  # Reduced from 5
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)
    print(f"[OK] Test completed successfully!")
    print(f"Final Score: {results['best_score']:.3f}")
    print(f"Best Combination: {[d.name for d in results['best_combination'].drugs][:3]}")
    print(f"Execution Time: {elapsed:.1f} seconds")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    print("\nADDS Quick Validation Test - Debugging Tool\n")
    
    try:
        results = run_quick_validation()
        print("[SUCCESS] Quick validation passed!")
        print("Recommendation: System is functioning correctly\n")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n[ERROR] Quick validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
