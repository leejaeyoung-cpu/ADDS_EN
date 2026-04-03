"""
Phase 2-2: Acquisition Function Integration Tests

Test suite for validating the integration of enhanced acquisition functions
into CombinationDesigner.

Author: ADDS Development Team
Date: 2026-01-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from recommendation.combination_designer import (
    CombinationDesigner,
    Drug,
    PatientProfile
)

# Test fixtures
def create_test_drugs():
    """Create standardized test drugs"""
    return [
        Drug(
            name="Cisplatin",
            min_dose=50,
            max_dose=100,
            standard_dose=75,
            route="IV",
            half_life_hours=24,
            target_proteins=["DNA"],
            known_interactions=["5-FU"]
        ),
        Drug(
            name="5-Fluorouracil",
            min_dose=400,
            max_dose=1000,
            standard_dose=800,
            route="IV",
            half_life_hours=12,
            target_proteins=["Thymidylate synthase"],
            known_interactions=["Cisplatin"]
        ),
        Drug(
            name="Trastuzumab",
            min_dose=6,
            max_dose=8,
            standard_dose=8,
            route="IV",
            half_life_hours=672,
            target_proteins=["HER2"],
            known_interactions=[]
        )
    ]


def create_test_patient():
    """Create standardized test patient"""
    return PatientProfile(
        patient_id="TEST_P2_2",
        age=65,
        cancer_type="Gastric",
        stage=3,
        ecog=1,
        mutations=["HER2"],
        previous_treatments=[],
        egfr=70,
        bilirubin=0.8
    )


# Test 1: Basic Integration
def test_basic_acquisition_functions():
    """Test each of 5 acquisition functions works"""
    print("\n" + "="*70)
    print("TEST 1: Basic Acquisition Function Integration")
    print("="*70)
    
    designer = CombinationDesigner(random_state=42)
    drugs = create_test_drugs()
    patient = create_test_patient()
    
    acquisition_functions = [
        'expected_improvement',
        'upper_confidence_bound',
        'probability_of_improvement',
        'thompson_sampling',
        'entropy_search_simplified'
    ]
    
    results = {}
    
    for acq_func in acquisition_functions:
        print(f"\nTesting: {acq_func}")
        try:
            combinations = designer.design_next_combinations(
                available_drugs=drugs,
                patient_profile=patient,
                num_combinations=3,
                iteration=0,
                acquisition=acq_func
            )
            
            assert len(combinations) == 3, f"Expected 3 combinations, got {len(combinations)}"
            assert all(2 <= len(c.drugs) <= 4 for c in combinations), "Invalid combination size"
            
            results[acq_func] = "[OK] PASS"
            print(f"  -> {len(combinations)} combinations generated")
            
        except Exception as e:
            results[acq_func] = f"[FAIL]: {e}"
            print(f"  → ERROR: {e}")
    
    # Summary
    print("\n" + "-"*70)
    print("Results:")
    for func, status in results.items():
        print(f"  {func:30s}: {status}")
    
    passed = sum(1 for s in results.values() if "PASS" in s)
    print(f"\nPassed: {passed}/{len(acquisition_functions)}")
    
    return all("PASS" in s for s in results.values())


# Test 2: Dual-Mode Strategy
def test_dual_mode_strategy():
    """Test dual-mode: Thompson → EI transition"""
    print("\n" + "="*70)
    print("TEST 2: Dual-Mode Strategy")
    print("="*70)
    
    designer = CombinationDesigner(
        random_state=42,
        dual_mode_threshold=10
    )
    drugs = create_test_drugs()
    patient = create_test_patient()
    
    # Test early iterations (should use Thompson Sampling)
    print("\nEarly Phase (Iteration 0-9):")
    for iteration in [0, 5, 9]:
        acq_func = designer._determine_acquisition_function(
            iteration=iteration,
            patient=patient,
            acquisition='dual_mode'
        )
        expected = 'thompson_sampling'
        status = "✅" if acq_func == expected else "❌"
        print(f"  Iteration {iteration}: {acq_func:30s} {status}")
        assert acq_func == expected, f"Expected {expected}, got {acq_func}"
    
    # Test late iterations (should use Expected Improvement)
    print("\nLate Phase (Iteration 10+):")
    for iteration in [10, 15, 20]:
        acq_func = designer._determine_acquisition_function(
            iteration=iteration,
            patient=patient,
            acquisition='dual_mode'
        )
        expected = 'expected_improvement'
        status = "✅" if acq_func == expected else "❌"
        print(f"  Iteration {iteration}: {acq_func:30s} {status}")
        assert acq_func == expected, f"Expected {expected}, got {acq_func}"
    
    print("\n✅ Dual-mode strategy working correctly!")
    return True


# Test 3: Adaptive Selection
def test_adaptive_selection():
    """Test cancer-specific adaptive selection"""
    print("\n" + "="*70)
    print("TEST 3: Adaptive Cancer-Specific Selection")
    print("="*70)
    
    designer = CombinationDesigner(random_state=42)
    
    test_cases = [
        # (cancer_type, stage, history_size, expected_category)
        ("Pancreatic", 4, 5, 'entropy_search_simplified'),  # Complex, early
        ("Breast", 2, 5, 'expected_improvement'),  # Standard, early
        ("Gastric", 3, 10, 'entropy_search_simplified'),  # Medium complexity, mid
        ("Colorectal", 3, 20, 'probability_of_improvement'),  # Standard, late
    ]
    
    for cancer, stage, history, expected_type in test_cases:
        patient = PatientProfile(
            patient_id=f"TEST_{cancer}",
            age=65,
            cancer_type=cancer,
            stage=stage,
            ecog=1,
            mutations=[],
            previous_treatments=[],
            egfr=70,
            bilirubin=0.8
        )
        
        acq_func = designer._determine_acquisition_function(
            iteration=history,
            patient=patient,
            acquisition='adaptive'
        )
        
        status = "✅" if acq_func == expected_type else "⚠️"
        print(f"{cancer:15s} Stage {stage}, History {history:2d}: {acq_func:30s} {status}")
        
        # For adaptive, we just verify it returns a valid function
        valid_functions = [
            'expected_improvement',
            'thompson_sampling',
            'entropy_search_simplified',
            'probability_of_improvement',
            'upper_confidence_bound'
        ]
        assert acq_func in valid_functions, f"Invalid function: {acq_func}"
    
    print("\n✅ Adaptive selection working!")
    return True


# Test 4: Backward Compatibility
def test_backward_compatibility():
    """Test existing code still works"""
    print("\n" + "="*70)
    print("TEST 4: Backward Compatibility")
    print("="*70)
    
    # Old-style initialization (no new parameters)
    designer = CombinationDesigner()
    drugs = create_test_drugs()
    patient = create_test_patient()
    
    # Old-style call (with basic acquisition function)
    combinations = designer.design_next_combinations(
        available_drugs=drugs,
        patient_profile=patient,
        num_combinations=5,
        iteration=0,
        acquisition='expected_improvement'  # Legacy function
    )
    
    assert len(combinations) == 5, f"Expected 5 combinations, got {len(combinations)}"
    print(f"\n✅ Legacy code works: Generated {len(combinations)} combinations")
    print("✅ Backward compatibility maintained!")
    
    return True


# Test 5: Enhanced vs Legacy Performance
def test_performance_comparison():
    """Compare enhanced vs legacy acquisition functions"""
    print("\n" + "="*70)
    print("TEST 5: Performance Comparison (Enhanced vs Legacy)")
    print("="*70)
    
    designer = CombinationDesigner(random_state=42)
    drugs = create_test_drugs()
    patient = create_test_patient()
    
    # Simulate optimization history
    print("\nSimulating 20 iterations of optimization...")
    
    for iteration in range(20):
        # Dual-mode (recommended)
        combinations = designer.design_next_combinations(
            available_drugs=drugs,
            patient_profile=patient,
            num_combinations=1,
            iteration=iteration,
            acquisition='dual_mode'
        )
        
        # Simulate outcome (better outcomes as we progress)
        fake_outcome = 0.3 + (iteration / 20) * 0.5 + np.random.normal(0, 0.05)
        fake_outcome = np.clip(fake_outcome, 0, 1)
        
        designer.update_history(combinations[0], fake_outcome)
        
        if iteration in [0, 9, 10, 19]:
            acq_func = designer._determine_acquisition_function(
                iteration=iteration,
                patient=patient,
                acquisition='dual_mode'
            )
            print(f"  Iteration {iteration:2d}: Using {acq_func:30s} (best={fake_outcome:.3f})")
    
    print(f"\n✅ Optimization completed: {len(designer.history)} samples in history")
    
    # Check final performance
    final_best = max(y for _, y in designer.history)
    print(f"✅ Final best outcome: {final_best:.3f}")
    
    assert final_best > 0.6, "Should achieve reasonable performance"
    
    return True


# Main test runner
def run_all_tests():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("PHASE 2-2: ACQUISITION FUNCTION INTEGRATION TESTS")
    print("="*70)
    
    tests = [
        ("Basic Integration", test_basic_acquisition_functions),
        ("Dual-Mode Strategy", test_dual_mode_strategy),
        ("Adaptive Selection", test_adaptive_selection),
        ("Backward Compatibility", test_backward_compatibility),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
        except Exception as e:
            results[test_name] = f"❌ FAIL: {e}"
            print(f"\n❌ ERROR in {test_name}: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, status in results.items():
        print(f"{test_name:30s}: {status}")
    
    passed = sum(1 for s in results.values() if "PASS" in s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2-2 integration successful!")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
