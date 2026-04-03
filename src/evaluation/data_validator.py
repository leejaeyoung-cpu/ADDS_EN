"""
Data quality validation for ADDS
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json

from ..utils import get_logger

logger = get_logger(__name__)


class DataQualityValidator:
    """
    Validate data quality and consistency
    """
    
    def __init__(self):
        """Initialize validator"""
        self.validation_results = []
        logger.info("✓ Data quality validator initialized")
    
    def validate_experiment_data(
        self,
        data: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate experiment data structure
        
        Args:
            data: Experiment data dictionary
            required_fields: List of required fields
        
        Returns:
            Validation report
        """
        if required_fields is None:
            required_fields = ['experiment_name', 'experiment_type', 'date_performed']
        
        issues = []
        warnings = []
        
        # Check required fields
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(f"Missing required field: {field}")
            elif isinstance(data[field], str) and not data[field].strip():
                issues.append(f"Empty value for required field: {field}")
        
        # Check data types
        if 'date_performed' in data and data['date_performed']:
            try:
                pd.to_datetime(data['date_performed'])
            except:
                issues.append("Invalid date format in 'date_performed'")
        
        # Check metadata
        if 'metadata' in data and data['metadata']:
            if not isinstance(data['metadata'], dict):
                warnings.append("Metadata should be a dictionary")
        
        report = {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'data_id': data.get('experiment_id', 'unknown')
        }
        
        self.validation_results.append(report)
        return report
    
    def check_missing_values(
        self,
        df: pd.DataFrame,
        critical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check for missing values in DataFrame
        
        Args:
            df: Input DataFrame
            critical_columns: Columns that should not have missing values
        
        Returns:
            Report of missing values
        """
        missing_report = {}
        
        # Overall missing values
        total_missing = df.isnull().sum()
        missing_pct = (total_missing / len(df) * 100).round(2)
        
        missing_report['total_rows'] = len(df)
        missing_report['columns'] = {}
        
        for col in df.columns:
            missing_report['columns'][col] = {
                'missing_count': int(total_missing[col]),
                'missing_percentage': float(missing_pct[col])
            }
        
        # Check critical columns
        if critical_columns:
            critical_issues = []
            for col in critical_columns:
                if col in df.columns and df[col].isnull().any():
                    critical_issues.append(f"Critical column '{col}' has missing values")
            
            missing_report['critical_issues'] = critical_issues
        
        logger.info(f"Missing value check: {len(df.columns)} columns analyzed")
        return missing_report
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Detect outliers in numerical data
        
        Args:
            df: Input DataFrame
            columns: Columns to check (None for all numeric)
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            Outlier report
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_report = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
            
            outlier_report[col] = {
                'num_outliers': len(outliers),
                'percentage': (len(outliers) / len(df) * 100).round(2),
                'outlier_indices': outliers.index.tolist()[:10]  # First 10
            }
        
        logger.info(f"Outlier detection: {len(columns)} columns analyzed using {method}")
        return outlier_report
    
    def check_data_consistency(
        self,
        experiments: List[Dict],
        results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Check consistency across related datasets
        
        Args:
            experiments: List of experiment records
            results: List of result records
        
        Returns:
            Consistency report
        """
        issues = []
        
        # Check for orphaned results
        exp_ids = {exp['experiment_id'] for exp in experiments}
        
        orphaned_count = 0
        for result in results:
            if 'experiment_id' in result and result['experiment_id'] not in exp_ids:
                orphaned_count += 1
                if orphaned_count <= 5:  # Report first 5
                    issues.append(f"Result {result.get('result_id', 'unknown')} references non-existent experiment")
        
        if orphaned_count > 5:
            issues.append(f"... and {orphaned_count - 5} more orphaned results")
        
        # Check for experiments without results
        result_exp_ids = {r['experiment_id'] for r in results if 'experiment_id' in r}
        experiments_without_results = [exp for exp in experiments 
                                      if exp['experiment_id'] not in result_exp_ids]
        
        report = {
            'consistent': len(issues) == 0,
            'issues': issues,
            'num_experiments': len(experiments),
            'num_results': len(results),
            'orphaned_results': orphaned_count,
            'experiments_without_results': len(experiments_without_results)
        }
        
        logger.info(f"Consistency check: {len(experiments)} experiments, {len(results)} results")
        return report
    
    def validate_image_features(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate extracted image features
        
        Args:
            features_df: DataFrame with cell features
        
        Returns:
            Validation report
        """
        issues = []
        warnings = []
        
        # Check for negative values where they shouldn't exist
        positive_only = ['area', 'perimeter', 'major_axis_length', 'minor_axis_length']
        for col in positive_only:
            if col in features_df.columns:
                if (features_df[col] < 0).any():
                    issues.append(f"{col} contains negative values")
        
        # Check for values out of expected range
        if 'circularity' in features_df.columns:
            invalid_circ = features_df[(features_df['circularity'] < 0) | 
                                       (features_df['circularity'] > 1)]
            if len(invalid_circ) > 0:
                issues.append(f"Circularity out of range [0,1]: {len(invalid_circ)} cells")
        
        if 'eccentricity' in features_df.columns:
            invalid_ecc = features_df[(features_df['eccentricity'] < 0) | 
                                     (features_df['eccentricity'] > 1)]
            if len(invalid_ecc) > 0:
                issues.append(f"Eccentricity out of range [0,1]: {len(invalid_ecc)} cells")
        
        # Check for suspiciously small or large cells
        if 'area' in features_df.columns:
            median_area = features_df['area'].median()
            very_small = features_df[features_df['area'] < median_area * 0.01]
            very_large = features_df[features_df['area'] > median_area * 100]
            
            if len(very_small) > 0:
                warnings.append(f"{len(very_small)} cells with suspiciously small area")
            if len(very_large) > 0:
                warnings.append(f"{len(very_large)} cells with suspiciously large area")
        
        report = {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'num_cells': len(features_df)
        }
        
        logger.info(f"Image feature validation: {len(features_df)} cells checked")
        return report
    
    def generate_validation_report(
        self,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            output_path: Path to save report
        
        Returns:
            Summary report
        """
        summary = {
            'total_validations': len(self.validation_results),
            'passed': sum(1 for r in self.validation_results if r['valid']),
            'failed': sum(1 for r in self.validation_results if not r['valid']),
            'details': self.validation_results
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Validation report saved to {output_path}")
        
        return summary
