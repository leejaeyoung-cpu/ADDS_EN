"""
Unit tests for ADDS modules
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.synergy_calculator import SynergyCalculator
from data.data_integrator import DataIntegrator, DataValidator
from evaluation.data_validator import DataQualityValidator


class TestSynergyCalculator(unittest.TestCase):
    """Test synergy calculation methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = SynergyCalculator()
    
    def test_bliss_synergy(self):
        """Test Bliss independence calculation"""
        # Test synergistic case
        synergy = self.calculator.calculate_bliss(
            effect_a=0.4,
            effect_b=0.3,
            effect_combination=0.8
        )
        self.assertGreater(synergy, 0, "Should be synergistic")
        
        # Test antagonistic case
        synergy = self.calculator.calculate_bliss(
            effect_a=0.4,
            effect_b=0.3,
            effect_combination=0.5
        )
        self.assertLess(synergy, 0, "Should be antagonistic")
    
    def test_hsa_synergy(self):
        """Test HSA calculation"""
        synergy = self.calculator.calculate_hsa(
            effect_a=0.4,
            effect_b=0.3,
            effect_combination=0.6
        )
        self.assertGreater(synergy, 0, "Should be synergistic")
        self.assertAlmostEqual(synergy, 0.2, places=2)
    
    def test_loewe_synergy(self):
        """Test Loewe additivity"""
        synergy = self.calculator.calculate_loewe(
            dose_a=10.0,
            dose_b=15.0,
            ic50_a=20.0,
            ic50_b=30.0,
            effect_combination=0.5
        )
        # CI should be 0.5 + 0.5 = 1.0, so synergy = 0
        self.assertAlmostEqual(synergy, 0.0, places=1)
    
    def test_ic50_fitting(self):
        """Test IC50 curve fitting"""
        # Generate synthetic dose-response data
        doses = np.array([0.1, 1, 10, 100, 1000])
        true_ic50 = 50
        effects = 1 / (1 + (doses / true_ic50) ** -1)
        
        ic50, params = self.calculator.fit_ic50(doses, effects)
        
        self.assertIsNotNone(ic50)
        self.assertGreater(ic50, 0)
        # Should be close to true IC50
        self.assertAlmostEqual(ic50, true_ic50, delta=20)


class TestDataIntegrator(unittest.TestCase):
    """Test data integration functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.integrator = DataIntegrator()
    
    def test_create_experiment(self):
        """Test experiment record creation"""
        exp = self.integrator.create_experiment_record(
            experiment_name="Test Experiment",
            experiment_type="cell_viability",
            cell_line="MCF-7"
        )
        
        self.assertIn('experiment_id', exp)
        self.assertEqual(exp['experiment_name'], "Test Experiment")
        self.assertEqual(exp['experiment_type'], "cell_viability")
    
    def test_add_drug_combination(self):
        """Test drug combination creation"""
        exp = self.integrator.create_experiment_record(
            experiment_name="Test",
            experiment_type="test"
        )
        
        combo = self.integrator.add_drug_combination(
            experiment_id=exp['experiment_id'],
            compounds=['Drug A', 'Drug B'],
            concentrations=[10.0, 20.0],
            concentration_units='μM'
        )
        
        self.assertIn('combination_id', combo)
        self.assertEqual(len(combo['compounds']), 2)
        self.assertEqual(len(combo['concentrations']), 2)
    
    def test_data_validator(self):
        """Test data validation"""
        valid_data = {
            'experiment_name': 'Test',
            'experiment_type': 'cell_viability'
        }
        
        report = DataValidator.validate_experiment_data(valid_data)
        self.assertTrue(report['valid'])
        
        invalid_data = {}
        report = DataValidator.validate_experiment_data(invalid_data)
        self.assertFalse(report['valid'])
        self.assertGreater(len(report['issues']), 0)


class TestDataQualityValidator(unittest.TestCase):
    """Test data quality validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = DataQualityValidator()
    
    def test_missing_values_check(self):
        """Test missing value detection"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8],
            'C': [9, 10, 11, 12]
        })
        
        report = self.validator.check_missing_values(df)
        
        self.assertEqual(report['total_rows'], 4)
        self.assertEqual(report['columns']['A']['missing_count'], 1)
        self.assertEqual(report['columns']['B']['missing_count'], 1)
        self.assertEqual(report['columns']['C']['missing_count'], 0)
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        np.random.seed(42)
        
        # Create data with outliers
        data = np.random.normal(50, 10, 100)
        data[0] = 200  # Outlier
        
        df = pd.DataFrame({'values': data})
        
        report = self.validator.detect_outliers(df, method='iqr')
        
        self.assertIn('values', report)
        self.assertGreater(report['values']['num_outliers'], 0)
    
    def test_image_features_validation(self):
        """Test image feature validation"""
        # Valid features
        valid_df = pd.DataFrame({
            'area': [100, 150, 120],
            'circularity': [0.8, 0.9, 0.75],
            'eccentricity': [0.3, 0.4, 0.35]
        })
        
        report = self.validator.validate_image_features(valid_df)
        self.assertTrue(report['valid'])
        
        # Invalid features
        invalid_df = pd.DataFrame({
            'area': [-10, 150, 120],  # Negative area
            'circularity': [0.8, 1.5, 0.75],  # Out of range
            'eccentricity': [0.3, 0.4, 0.35]
        })
        
        report = self.validator.validate_image_features(invalid_df)
        self.assertFalse(report['valid'])
        self.assertGreater(len(report['issues']), 0)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestSynergyCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityValidator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
