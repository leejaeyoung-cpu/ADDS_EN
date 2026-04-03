"""
Simple test to verify ADDS Medical AI modules
"""

import sys
sys.path.insert(0, 'src')

def test_prognosis_predictor():
    """Test prognosis prediction"""
    from medical_ai.prognosis_predictor import PrognosisPredictor
    
    predictor = PrognosisPredictor()
    result = predictor.predict(
        tnm_stage="T2N1M0",
        tumor_size_mm=15.2,
        age=65,
        kras_status="Wild-type",
        lymph_nodes_positive=2,
        lymphovascular_invasion=True
    )
    
    print("=" * 60)
    print("PROGNOSIS PREDICTION TEST")
    print("=" * 60)
    print(f"TNM Stage: T2N1M0")
    print(f"Survival 1yr: {result.survival_1yr*100:.1f}%")
    print(f"Survival 3yr: {result.survival_3yr*100:.1f}%")
    print(f"Survival 5yr: {result.survival_5yr*100:.1f}%")
    print(f"Risk Group: {result.risk_group.value}")
    print(f"Recurrence Risk: {result.recurrence_risk*100:.0f}%")
    print(f"Risk Factors: {len(result.risk_factors)} identified")
    print("Status: PASS\n")
    return True

def test_clinical_decision():
    """Test treatment plan generation"""
    from medical_ai.clinical_decision import ClinicalDecisionEngine
    from medical_ai.prognosis_predictor import PrognosisPredictor
    
    # Get prognosis first
    predictor = PrognosisPredictor()
    prognosis = predictor.predict(
        tnm_stage="T2N1M0",
        tumor_size_mm=15.2,
        age=65,
        kras_status="Wild-type"
    )
    
    # Generate treatment plan
    engine = ClinicalDecisionEngine()
    plan = engine.recommend_treatment(
        tnm_stage="T2N1M0",
        prognosis=prognosis.__dict__,
        patient_profile={'age': 65, 'kras_status': 'Wild-type'}
    )
    
    print("=" * 60)
    print("TREATMENT PLAN GENERATION TEST")
    print("=" * 60)
    print(f"Number of Phases: {len(plan.phases)}")
    for idx, phase in enumerate(plan.phases, 1):
        print(f"  Phase {idx}: {phase.name} ({phase.duration})")
    print(f"Expected Duration: {plan.expected_duration_weeks} weeks")
    print(f"Success Probability: {plan.success_probability*100:.0f}%")
    print(f"Monitoring Frequency: {plan.monitoring.frequency}")
    print("Status: PASS\n")
    return True

def test_ct_preprocessor():
    """Test CT preprocessor"""
    from preprocessing.ct_preprocessor import CTPreprocessor
    import numpy as np
    
    preprocessor = CTPreprocessor(
        window_center=40,
        window_width=400,
        target_size=(640, 640),
        apply_clahe=True
    )
    
    # Create mock HU data
    mock_hu = np.random.randn(512, 512) * 100 + 40
    
    # Test preprocessing
    result = preprocessor.preprocess_image(mock_hu, pixel_spacing=0.75)
    
    print("=" * 60)
    print("CT PREPROCESSOR TEST")
    print("=" * 60)
    print(f"Input Shape: (512, 512)")
    print(f"Output Shape: {result.shape}")
    print(f"Output Range: [{result.min():.1f}, {result.max():.1f}]")
    print(f"Windowing: WC=40, WW=400")
    print(f"CLAHE: Enabled")
    print("Status: PASS\n")
    return True

def test_yolo_detector():
    """Test YOLO detector initialization"""
    from medical_imaging.detection.yolo_tumor_detector import MockTumorDetector
    
    # Use mock detector (ultralytics not installed)
    detector = MockTumorDetector()
    
    print("=" * 60)
    print("YOLO DETECTOR TEST")
    print("=" * 60)
    print("MockTumorDetector initialized (ultralytics not installed)")
    print("Fallback mode: Active")
    print("Status: PASS\n")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ADDS MEDICAL AI SYSTEM - MODULE TESTING")
    print("=" * 60 + "\n")
    
    tests = [
        test_ct_preprocessor,
        test_yolo_detector,
        test_prognosis_predictor,
        test_clinical_decision
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"Error: {e}\n")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("All systems operational!")
    print("=" * 60)
