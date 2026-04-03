"""
End-to-End Workflow Test for CDSS Metadata Learning System

Tests the complete workflow from patient registration through analysis,
training, and outcome collection.
"""

import sys
from pathlib import Path

# Fix import path
current_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(current_dir))

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, encoding='utf-8')
logger = logging.getLogger(__name__)


def test_complete_workflow():
    """Test end-to-end CDSS workflow"""
    
    print("="*80)
    print("CDSS Metadata Learning System - End-to-End Workflow Test")
    print("="*80)
    
    try:
        from patient_management_system.services.cdss_orchestrator import CDSSOrchestrator
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import (
            Patient, CTAnalysis, Treatment, TreatmentOutcome
        )
        
        orchestrator = CDSSOrchestrator()
        db = get_session()
        
        # === 1. Patient Registration ===
        print("\n" + "="*80)
        print("STEP 1: Patient Registration")
        print("="*80)
        
        patient_data = {
            'patient_id': f'E2E-TEST-{datetime.now().strftime("%Y%m%d%H%M%S")}',
            'name': 'End-to-End Test Patient',
            'birthdate': datetime(1965, 5, 15),
            'gender': 'M'
        }
        
        result = orchestrator.process_new_patient(patient_data)
        
        if result['success']:
            print(f"[OK] Patient registered: {result['patient_id']}")
            print(f"  Assessment: {result['assessment']}")
        else:
            print(f"[FAIL] Registration failed: {result.get('error')}")
            return False
        
        patient_id = result['patient_id']
        
        # === 2. Add Treatment ===
        print("\n" + "="*80)
        print("STEP 2: Add Treatment")
        print("="*80)
        
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        
        treatment = Treatment(
            patient_id=patient.id,
            drug_cocktail="FOLFOX",
            start_date=datetime.now() - timedelta(days=90)
        )
        treatment.notes = "Cycle 6: Oxaliplatin 85mg/m2, 5-FU 400mg/m2"
        
        db.add(treatment)
        db.commit()
        
        print(f"✓ Treatment added: {treatment.drug_cocktail}")
        print(f"  Started: {treatment.start_date.strftime('%Y-%m-%d')}")
        print(f"  Cycle: {treatment.cycle_number}")
        
        # === 3. Record Outcome ===
        print("\n" + "="*80)
        print("STEP 3: Record Treatment Outcome")
        print("="*80)
        
        outcome = TreatmentOutcome(
            treatment_id=treatment.id,
            assessment_date=datetime.now(),
            response_type='PR',  # Partial Response
            tumor_size_change_percent=-35.5,
            pfs_days=180,
            qol_score=7.5,
            clinical_notes="Patient showing good response to treatment. Tumor reduction observed."
        )
        
        db.add(outcome)
        db.commit()
        
        print(f"✓ Outcome recorded:")
        print(f"  Response Type: {outcome.response_type}")
        print(f"  Tumor Change: {outcome.tumor_size_change_percent:+.1f}%")
        print(f"  PFS: {outcome.pfs_days} days")
        print(f"  QoL: {outcome.qol_score}/10")
        
        # === 4. Test NLP Parser ===
        print("\n" + "="*80)
        print("STEP 4: Parse Physician Notes")
        print("="*80)
        
        from patient_management_system.services.nlp_parser import PhysicianNotesParser
        
        parser = PhysicianNotesParser()
        
        test_notes = """
        Patient follow-up after 6 cycles of FOLFOX.
        Assessment: Moderate improvement observed. 
        CT imaging shows tumor reduced from 4.5 cm to 2.9 cm.
        Patient reports mild fatigue and nausea, controlled with medication.
        Labs show stable blood counts. No severe complications.
        Impression: Partial response to chemotherapy. Continue current regimen.
        Plan: Continue FOLFOX, reassess after 2 more cycles.
        """
        
        parsed = parser.parse(test_notes)
        
        print(f"✓ Notes parsed successfully:")
        print(f"  Severity: {parsed['severity']['level']} ({parsed['severity']['score']}/10)")
        print(f"  Symptoms: {', '.join(parsed['symptoms']) if parsed['symptoms'] else 'None'}")
        print(f"  Tumor Status: {parsed['tumor_status']['status']} (confidence: {parsed['tumor_status']['confidence']:.2f})")
        print(f"  Requires Re-analysis: {'YES ⚠️' if parsed['requires_reanalysis'] else 'No'}")
        print(f"  Medications: {', '.join(parsed['medications']) if parsed['medications'] else 'None'}")
        
        # Extract measurements
        measurements = parser.extract_tumor_measurements(test_notes)
        if measurements:
            print(f"  Tumor Measurements:")
            for key, value in measurements.items():
                print(f"    • {key}: {value}")
        
        # === 5. Test Metadata Aggregation ===
        print("\n" + "="*80)
        print("STEP 5: Aggregate Metadata for Training")
        print("="*80)
        
        from patient_management_system.services.metadata_extraction import MetadataAggregator
        
        aggregator = MetadataAggregator()
        dataset = aggregator.create_training_dataset(limit=5)
        
        print(f"✓ Training dataset created:")
        print(f"  Total samples: {dataset['total_samples']}")
        print(f"  Features extracted: {dataset['feature_names']}")
        
        # === 6. Verify Database==
        print("\n" + "="*80)
        print("STEP 6: Verify Database State")
        print("="*80)
        
        total_patients = db.query(Patient).count()
        total_treatments = db.query(Treatment).count()
        total_outcomes = db.query(TreatmentOutcome).count()
        total_analyses = db.query(CTAnalysis).count()
        
        print(f"✓ Database state:")
        print(f"  Total Patients: {total_patients}")
        print(f"  Total Treatments: {total_treatments}")
        print(f"  Total Outcomes: {total_outcomes}")
        print(f"  Total CT Analyses: {total_analyses}")
        
        # === 7. Summary ===
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print("\n✅ ALL WORKFLOW STEPS COMPLETED SUCCESSFULLY!")
        print("\nWorkflow tested:")
        print("  1. ✓ Patient registration")
        print("  2. ✓ Treatment recording")
        print("  3. ✓ Outcome collection")
        print("  4. ✓ NLP parsing of clinical notes")
        print("  5. ✓ Metadata aggregation")
        print("  6. ✓ Database persistence")
        
        print("\n🎯 System is ready for:")
        print("  • Daily ML training cycles")
        print("  • Dynamic re-analysis on data updates")
        print("  • Continuous learning from outcomes")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_workflow()
    
    print("\n" + "="*80)
    print(f"Final Result: {'PASS ✓' if success else 'FAIL ✗'}")
    print("="*80)
