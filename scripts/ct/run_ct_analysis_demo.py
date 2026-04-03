"""
Run CT Analysis on Real Patient Data
Demonstrates complete 6-stage pipeline execution
"""

from ct_crc_detection_pipeline import IntegratedCRCDetectionPipeline
from pathlib import Path
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def analyze_ct_data():
    """Run complete CT analysis"""
    
    print("\n" + "="*80)
    print("CT CRC DETECTION PIPELINE - REAL DATA ANALYSIS")
    print("="*80)
    
    # Configuration
    config = {
        'target_spacing': [1.0, 1.0, 1.0],
        'radiomics_bin_width': 25,
        'models_dir': 'models',
        'output_dir': 'outputs/crc_detection',
        'save_intermediate': True,
        'use_gpu': True
    }
    
    # Initialize pipeline
    print("\n[INIT] Initializing integrated pipeline...")
    pipeline = IntegratedCRCDetectionPipeline(config)
    print("[OK] Pipeline initialized")
    
    # Patient data
    patient_id = "PT-INHA-DEMO-001"
    dicom_folder = Path("F:/ADDS/CTdata/CTdcm")
    
    print(f"\n[DATA]")
    print(f"  Patient ID: {patient_id}")
    print(f"  DICOM folder: {dicom_folder}")
    
    if dicom_folder.exists():
        num_files = len(list(dicom_folder.glob("*.dcm")))
        print(f"  DICOM files: {num_files}")
    else:
        print(f"  ERROR: Folder not found!")
        return
    
    # Run analysis
    print("\n" + "="*80)
    print("RUNNING ANALYSIS...")
    print("="*80)
    
    result = pipeline.process_patient(
        dicom_folder=dicom_folder,
        patient_id=patient_id
    )
    
    # Display results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nPatient ID: {result['patient_id']}")
    print(f"Status: {result['status'].upper()}")
    
    if result['status'] == 'success':
        print(f"Duration: {result['duration_seconds']:.1f} seconds")
        
        # Stages
        print("\n[PIPELINE STAGES]")
        for stage_name, stage_data in result.get('stages', {}).items():
            status = "✓" if stage_data.get('status') == 'success' else "○"
            print(f"  {status} {stage_name}")
            if 'shape' in stage_data:
                print(f"      Volume shape: {stage_data['shape']}")
        
        # Tumors
        tumors = result.get('tumors', [])
        print(f"\n[TUMORS] Detected: {len(tumors)}")
        
        for tumor in tumors[:3]:  # Show first 3
            print(f"\n  Tumor {tumor['tumor_id']}:")
            print(f"    Volume: {tumor['volume_mm3']:.2f} mm³ ({tumor['volume_mm3']/1000:.2f} cm³)")
            print(f"    Centroid: {[f'{c:.1f}' for c in tumor['centroid']]}")
            
            cls = tumor['classification']
            print(f"    Classification: {cls['classification']}")
            
            if cls['classification'] == 'Malignant':
                tnm = cls['tnm_stage']
                print(f"    TNM Stage: T{tnm['T']}, N{tnm['N']}, M{tnm['M']}")
                print(f"    Overall Stage: {cls['overall_stage']}")
                
                # Biomarkers
                msi = cls.get('msi_status', {})
                kras = cls.get('kras_mutation', {})
                
                if msi:
                    print(f"    MSI Status: {msi.get('status', 'Unknown')} ({msi.get('probability', 0):.1%})")
                if kras:
                    print(f"    KRAS: {kras.get('status', 'Unknown')} ({kras.get('probability', 0):.1%})")
        
        # ADDS Integration
        if 'adds_integration' in result:
            adds = result['adds_integration']
            print(f"\n[ADDS INTEGRATION]")
            print(f"  Status: {adds['status']}")
            
            if adds['status'] == 'success':
                tp = adds.get('treatment_plan', {})
                regimen = tp.get('recommended_regimen', {})
                
                # Primary drugs
                primary_drugs = regimen.get('primary_drugs', [])
                if primary_drugs:
                    print(f"\n  Primary Therapy:")
                    for drug in primary_drugs[:2]:
                        print(f"    • {drug.get('name', 'Unknown')}")
                        print(f"      Response Rate: {drug.get('predicted_response_rate', 0):.1%}")
                
                # Targeted therapy
                targeted = regimen.get('targeted_therapy', [])
                if targeted:
                    print(f"\n  Targeted Therapy:")
                    for drug in targeted[:2]:
                        print(f"    • {drug.get('name', 'Unknown')}")
                        print(f"      Benefit: {drug.get('predicted_benefit', 0):.1%}")
        
        # Output location
        output_dir = Path(f"outputs/crc_detection/{patient_id}")
        print(f"\n[OUTPUT]")
        print(f"  Results saved to: {output_dir}")
        
        # Save JSON
        json_path = output_dir / f"{patient_id}_summary.json"
        if not json_path.exists():
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  JSON report: {json_path}")
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE")
        print("="*80)
        
    else:
        print(f"\nERROR: {result.get('error', 'Unknown error')}")
        print("\n" + "="*80)
        print("✗ ANALYSIS FAILED")
        print("="*80)


if __name__ == "__main__":
    analyze_ct_data()
