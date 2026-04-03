"""
Simple Integration Test for CT Pipeline (without PyRadiomics)
Tests core components without requiring difficult-to-install dependencies
"""

import sys
from pathlib import Path
import logging
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

try:
    # Test imports
    from src.medical_imaging.tumor_classifier import TumorClassifier, BiomarkerPredictor
    from src.medical_imaging.adds_integrator import ADDSIntegrator
    print("[OK] Successfully imported pipeline modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_tumor_classification():
    """Test tumor classification with mock radiomics data"""
    print("\n" + "="*80)
    print("TEST 1: Tumor Classification & Staging")
    print("="*80)
    
    # Mock radiomics features
    radiomics = {
        'original_shape_Sphericity': 0.55,
        'original_shape_VoxelVolume': 6500,
        'original_shape_SurfaceVolumeRatio': 0.32,
        'original_firstorder_Entropy': 4.5,
        'original_firstorder_Mean': 45.3,
        'original_glcm_Contrast': 165,
        'original_glcm_Correlation': 0.78
    }
    
    try:
        classifier = TumorClassifier()
        biomarker = BiomarkerPredictor()
        
        # Classify tumor
        tnm_result = classifier.predict_tnm(radiomics)
        
        print(f"✓ Classification: {tnm_result['classification']}")
        print(f"  Malignancy probability: {tnm_result.get('malignancy_probability', 0):.2f}")
        
        if tnm_result['classification'] == 'Malignant':
            tnm = tnm_result['tnm_stage']
            print(f"  TNM Stage: T{tnm['T']}, N{tnm['N']}, M{tnm['M']}")
            print(f"  Overall Stage: {tnm_result['overall_stage']}")
        
        # Biomarkers
        msi = biomarker.predict_msi_status(radiomics)
        print(f"\n✓ MSI Status: {msi['status']} (probability: {msi['probability']:.2f})")
        
        classification = {
            **tnm_result,
            'msi_status': msi
        }
        
        return classification
    
    except Exception as e:
        print(f"Classification failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_adds_integration(classification):
    """Test ADDS integration"""
    print("\n" + "="*80)
    print("TEST 2: ADDS System Integration")
    print("="*80)
    
    if classification is None:
        print("Skipping (no classification results)")
        return None
    
    # Mock radiomics for ADDS test
    radiomics = {
        'original_shape_Sphericity': 0.55,
        'original_shape_SurfaceVolumeRatio': 0.32,
        'original_firstorder_Entropy': 4.5,
        'original_glcm_Contrast': 165
    }
    
    try:
        integrator = ADDSIntegrator()
        
        tumor_analysis = {
            'volume_mm3': 6500,
            'centroid': [150, 200, 220],
            'confidence': 0.92
        }
        
        adds_input = integrator.prepare_adds_input(
            patient_id='PT-TEST-001',
            volume=None,
            tumor_analysis=tumor_analysis,
            radiomics=radiomics,
            classification=classification
        )
        
        print(f"✓ ADDS input prepared")
        print(f"  Tumor Stage: {adds_input['tumor_info']['stage']}")
        print(f"  Tumor Volume: {adds_input['tumor_characteristics']['volume_cm3']:.2f} cm³")
        print(f"  Location: {adds_input['tumor_info']['location']}")
        
        # Drug sensitivities
        print("\n✓ Drug Sensitivity Predictions:")
        sensitivities = adds_input['predicted_drug_sensitivity']
        for drug, pred in sensitivities.items():
            response_rate = pred['predicted_response_rate']
            print(f"  {drug}: {response_rate:.2%}")
            print(f"    → {pred['rationale']}")
        
        # Biomarkers
        print("\n✓ Imaging Biomarkers:")
        biomarkers = adds_input['imaging_biomarkers']
        msi = biomarkers['msi_status_predicted']
        print(f"  MSI Status: {msi['status']} (confidence: {msi['confidence']:.2f})")
        
        # Treatment plan
        print("\n✓ Generating treatment plan...")
        treatment_plan = integrator._generate_fallback_plan(adds_input)
        
        if treatment_plan['recommended_regimen']['primary_drugs']:
            primary = treatment_plan['recommended_regimen']['primary_drugs'][0]
            print(f"  Recommended primary: {primary['name']}")
            print(f"  Expected response: {primary['predicted_response_rate']:.2%}")
            print(f"  Rationale: {primary['rationale']}")
        
        # Save result
        output_dir = Path("outputs/integration_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / "adds_integration_test.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'adds_input': adds_input,
                'treatment_plan': treatment_plan
            }, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n📝 Result saved to: {result_file}")
        
        return {'adds_input': adds_input, 'treatment_plan': treatment_plan}
    
    except Exception as e:
        print(f"ADDS integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_complete_pipeline():
    """Test complete integration"""
    print("\n" + "="*80)
    print("CT-BASED CRC DETECTION PIPELINE - INTEGRATION TEST")
    print("="*80)
    print("\nNOTE: Running simplified test without PyRadiomics")
    print("      Using mock radiomics features for demonstration\n")
    
    results = {}
    
    # Test 1: Classification
    classification = test_tumor_classification()
    results['classification_test'] = 'PASS' if classification else 'FAIL'
    
    # Test 2: ADDS Integration
    adds_result = test_adds_integration(classification)
    results['adds_integration_test'] = 'PASS' if adds_result else 'FAIL'
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"{icon} {test_name}: {status}")
    
    all_passed = all(status == 'PASS' for status in results.values())
    
    if all_passed:
        print("\n🎉 All tests PASSED!")
        print("\n✅ Core Pipeline Components Verified:")
        print("   - Stage 4: Radiomics Extraction (mocked)")
        print("   - Stage 5: Classification & TNM Staging ✓")
        print("   - Stage 6: ADDS Integration ✓")
        print("\n📌 Next Steps:")
        print("   1. Install PyRadiomics for full radiomics extraction")
        print("   2. Integrate nnU-Net for automatic segmentation")
        print("   3. Test with real CT DICOM data")
    else:
        print("\n⚠️  Some tests failed. Check error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)
