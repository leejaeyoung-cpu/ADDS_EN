"""
Integration test for precision oncology pipeline
Tests end-to-end workflow from patient registration to final report
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.clinical.clinical_database import ClinicalDatabase
from src.clinical.cohort_classifier import CohortClassifier
from src.pathology.spatial_analyzer import SpatialAnalyzer
from src.pathology.heterogeneity_metrics import HeterogeneityAnalyzer
from src.recommendation.drug_optimizer import DrugCombinationOptimizer
from src.recommendation.dosage_calculator import DosageCalculator
from src.recommendation.schedule_planner import SchedulePlanner
from src.reporting.clinical_report import ClinicalReportGenerator


class IntegrationTester:
    """통합 테스트 실행기"""
    
    def __init__(self):
        self.db = ClinicalDatabase()
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
    
    def run_all_tests(self):
        """전체 테스트 실행"""
        print("="*60)
        print("ADDS Precision Oncology Pipeline - Integration Test")
        print("="*60)
        
        # Load sample data
        samples = self._load_sample_data()
        
        if not samples:
            print("[ERROR] No sample data found!")
            return
        
        # Run tests for each sample
        for i, sample in enumerate(samples[:3], 1):  # Test first 3 samples
            print(f"\n[TEST {i}/{min(3, len(samples))}] Testing {sample['patient']['patient_id']} ({sample['scenario']})")
            print("-"*60)
            self._test_patient_workflow(sample)
        
        # Summary
        print("\n" + "="*60)
        print(f"TEST SUMMARY:")
        print(f"  [PASS] {self.passed_tests} tests passed")
        print(f"  [FAIL] {self.failed_tests} tests failed")
        if self.warnings:
            print(f"  [WARN] {len(self.warnings)} warnings")
        print("="*60)
        
        return self.failed_tests == 0
    
    def _load_sample_data(self):
        """샘플 데이터 로드"""
        sample_file = Path('data/samples/all_samples.json')
        
        if not sample_file.exists():
            print(f"[WARN] Sample file not found: {sample_file}")
            print("[INFO] Generating sample data...")
            
            # Generate samples
            from scripts.generate_sample_data import SampleDataGenerator
            generator = SampleDataGenerator()
            return generator.save_sample_dataset(num_patients=10)
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _test_patient_workflow(self, sample):
        """환자별 전체 워크플로우 테스트"""
        patient_data = sample['patient']
        quant_data = sample['quantitative_analysis']
        
        try:
            # Step 1: Patient Registration
            print("  [1/6] Patient registration...", end=" ")
            self.db.save_patient(patient_data)
            
            # Save genomic variants
            for variant in patient_data['genomic_variants']:
                self.db.add_genomic_variant(patient_data['patient_id'], variant)
            
            self._pass("OK")
            
            # Step 2: Cohort Classification
            print("  [2/6] Cohort classification...", end=" ")
            classifier = CohortClassifier()
            cohort = classifier.classify_patient(
                quantitative_data=quant_data,
                clinical_data=patient_data,
                genomic_data=patient_data['genomic_variants']
            )
            
            self._pass(f"OK - {cohort['cohort_name']}")
            
            # Save cohort
            cohort_id = self.db.save_cohort_classification(
                patient_data['patient_id'],
                cohort
            )
            
            # Step 3: Treatment Recommendation
            print("  [3/6] Treatment recommendation...", end=" ")
            optimizer = DrugCombinationOptimizer()
            recommendation = optimizer.recommend_regimen(
                cohort_classification=cohort,
                quantitative_results=quant_data,
                clinical_profile=patient_data,
                genomic_variants=patient_data['genomic_variants']
            )
            
            self._pass(f"OK - {recommendation['primary_regimen'].get('name', 'Custom')}")
            
            # Save recommendation
            rec_id = self.db.save_recommendation(
                patient_data['patient_id'],
                cohort_id,
                recommendation
            )
            
            # Step 4: Dosage Calculation
            print("  [4/6] Dosage calculation...", end=" ")
            dosage_calc = DosageCalculator()
            dosage_plan = dosage_calc.calculate_regimen_dosages(
                regimen=recommendation['primary_regimen'],
                patient_profile=patient_data
            )
            
            self._pass(f"OK - {len(dosage_plan['drugs'])} drugs")
            
            # Step 5: Schedule Generation
            print("  [5/6] Schedule generation...", end=" ")
            planner = SchedulePlanner()
            schedule = planner.generate_schedule(
                regimen=recommendation['primary_regimen'],
                dosage_plan=dosage_plan,
                start_date='2026-02-01',
                num_cycles=6
            )
            
            conflicts = planner.identify_conflicts(schedule)
            if conflicts:
                self.warnings.append(f"{patient_data['patient_id']}: {len(conflicts)} schedule conflicts")
            
            self._pass(f"OK - {len(schedule['cycles'])} cycles")
            
            # Step 6: Clinical Report
            print("  [6/6] Clinical report generation...", end=" ")
            report_gen = ClinicalReportGenerator()
            report = report_gen.generate_treatment_recommendation_report(
                patient_data=patient_data,
                quantitative_analysis=quant_data,
                clinical_metadata=patient_data,
                genomic_data=patient_data['genomic_variants'],
                cohort_classification=cohort,
                recommendation=recommendation,
                dosage_plan=dosage_plan,
                schedule=schedule
            )
            
            # Export report
            markdown_report = report_gen.export_to_markdown(report)
            report_path = Path(f"data/reports/{patient_data['patient_id']}_report.md")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(markdown_report)
            
            self._pass("OK")
            
            print(f"\n  [RESULT] Workflow completed successfully!")
            print(f"           Report saved: {report_path}")
            
        except Exception as e:
            self._fail(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _pass(self, message):
        """테스트 통과"""
        print(message)
        self.passed_tests += 1
    
    def _fail(self, message):
        """테스트 실패"""
        print(f"[FAIL] {message}")
        self.failed_tests += 1
    
    def test_individual_modules(self):
        """개별 모듈 테스트"""
        print("\n" + "="*60)
        print("INDIVIDUAL MODULE TESTS")
        print("="*60)
        
        # Test SpatialAnalyzer
        print("\n[TEST] SpatialAnalyzer...")
        try:
            import numpy as np
            analyzer = SpatialAnalyzer()
            centroids = np.random.rand(100, 2) * 1000
            results = analyzer.analyze_spatial_distribution(centroids)
            
            assert 'mean_nnd' in results
            assert 'clark_evans_index' in results
            print("  [PASS] SpatialAnalyzer OK")
            self.passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] {str(e)}")
            self.failed_tests += 1
        
        # Test HeterogeneityAnalyzer
        print("\n[TEST] HeterogeneityAnalyzer...")
        try:
            import pandas as pd
            het_analyzer = HeterogeneityAnalyzer()
            
            cell_features = pd.DataFrame({
                'area': np.random.rand(100) * 500 + 100,
                'circularity': np.random.rand(100),
                'eccentricity': np.random.rand(100),
                'solidity': np.random.rand(100) * 0.5 + 0.5
            })
            
            results = het_analyzer.calculate_morphological_heterogeneity(cell_features)
            
            assert 'overall_heterogeneity' in results
            assert 'heterogeneity_grade' in results
            print("  [PASS] HeterogeneityAnalyzer OK")
            self.passed_tests += 1
        except Exception as e:
            print(f"  [FAIL] {str(e)}")
            self.failed_tests += 1


if __name__ == "__main__":
    tester = IntegrationTester()
    
    # Run integration tests
    success = tester.run_all_tests()
    
    # Run module tests
    tester.test_individual_modules()
    
    # Exit code
    sys.exit(0 if success else 1)
