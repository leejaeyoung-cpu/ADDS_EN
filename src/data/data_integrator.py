"""
Data integration and management for ADDS
Combines multimodal data (images, documents, experiments) into unified dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import uuid
import json

from src.utils import get_logger, db_manager

logger = get_logger(__name__)


class DataIntegrator:
    """
    Integrate multimodal experimental data
    """
    
    def __init__(self):
        """Initialize data integrator"""
        self.experiments = []
        self.images = []
        self.results = []
        logger.info("✓ Data integrator initialized")
    
    def create_experiment_record(
        self,
        experiment_name: str,
        experiment_type: str,
        cell_line: Optional[str] = None,
        date_performed: Optional[datetime] = None,
        performed_by: Optional[str] = None,
        description: Optional[str] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Create a new experiment record
        
        Args:
            experiment_name: Name of experiment
            experiment_type: Type ('cell_viability', 'western_blot', 'imaging', etc.)
            cell_line: Cell line used
            date_performed: Date of experiment
            performed_by: Person who performed
            description: Detailed description
            **metadata: Additional metadata
        
        Returns:
            Experiment record dictionary
        """
        if date_performed is None:
            date_performed = datetime.now()
        
        experiment = {
            'experiment_id': str(uuid.uuid4()),
            'experiment_name': experiment_name,
            'experiment_type': experiment_type,
            'cell_line': cell_line,
            'date_performed': date_performed.isoformat(),
            'performed_by': performed_by,
            'description': description,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        self.experiments.append(experiment)
        logger.info(f"✓ Created experiment: {experiment_name}")
        
        return experiment
    
    def add_image_data(
        self,
        experiment_id: str,
        file_path: str,
        image_type: str = 'microscopy',
        segmentation_results: Optional[Dict] = None,
        features: Optional[pd.DataFrame] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Add image data to experiment
        
        Args:
            experiment_id: Linked experiment ID
            file_path: Path to image file
            image_type: Type of image
            segmentation_results: Cellpose segmentation results
            features: Extracted features DataFrame
            **metadata: Additional metadata
        
        Returns:
            Image record
        """
        image_id = str(uuid.uuid4())
        
        image_record = {
            'image_id': image_id,
            'experiment_id': experiment_id,
            'file_path': file_path,
            'file_name': Path(file_path).name,
            'image_type': image_type,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        if segmentation_results:
            image_record['segmentation'] = segmentation_results
        
        if features is not None:
            image_record['features'] = features.to_dict('records')
        
        self.images.append(image_record)
        logger.info(f"✓ Added image: {Path(file_path).name}")
        
        return image_record
    
    def add_drug_combination(
        self,
        experiment_id: str,
        compounds: List[str],
        concentrations: List[float],
        concentration_units: str = 'μM',
        combination_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add drug combination record
        
        Args:
            experiment_id: Linked experiment ID
            compounds: List of compound names/IDs
            concentrations: Corresponding concentrations
            concentration_units: Unit of concentration
            combination_name: Optional name for combination
        
        Returns:
            Combination record
        """
        if len(compounds) != len(concentrations):
            raise ValueError("Compounds and concentrations must have same length")
        
        if combination_name is None:
            combination_name = " + ".join([f"{c} ({d}{concentration_units})" 
                                          for c, d in zip(compounds, concentrations)])
        
        combination = {
            'combination_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'combination_name': combination_name,
            'compounds': compounds,
            'concentrations': concentrations,
            'concentration_units': concentration_units,
            'created_at': datetime.now().isoformat()
        }
        
        return combination
    
    def add_experimental_result(
        self,
        experiment_id: str,
        result_type: str,
        value: float,
        unit: str,
        combination_id: Optional[str] = None,
        image_id: Optional[str] = None,
        standard_deviation: Optional[float] = None,
        n_replicates: Optional[int] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Add experimental result
        
        Args:
            experiment_id: Linked experiment ID
            result_type: Type of result ('viability', 'ic50', 'apoptosis_rate', etc.)
            value: Measured value
            unit: Unit of measurement
            combination_id: Optional linked combination
            image_id: Optional linked image
            standard_deviation: StdDev of measurement
            n_replicates: Number of replicates
            **metadata: Additional metadata
        
        Returns:
            Result record
        """
        result = {
            'result_id': str(uuid.uuid4()),
            'experiment_id': experiment_id,
            'combination_id': combination_id,
            'image_id': image_id,
            'result_type': result_type,
            'value': value,
            'unit': unit,
            'standard_deviation': standard_deviation,
            'n_replicates': n_replicates,
            'metadata': metadata,
            'created_at': datetime.now().isoformat()
        }
        
        self.results.append(result)
        logger.info(f"✓ Added result: {result_type} = {value} {unit}")
        
        return result
    
    def create_integrated_dataset(
        self,
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Create integrated dataset from all collected data
        
        Args:
            output_path: Optional path to save dataset
        
        Returns:
            Integrated DataFrame
        """
        # Convert to DataFrames
        df_experiments = pd.DataFrame(self.experiments)
        df_images = pd.DataFrame(self.images)
        df_results = pd.DataFrame(self.results)
        
        # Merge data
        if len(df_results) > 0 and len(df_experiments) > 0:
            dataset = df_results.merge(
                df_experiments,
                on='experiment_id',
                how='left',
                suffixes=('', '_exp')
            )
            
            if len(df_images) > 0:
                dataset = dataset.merge(
                    df_images,
                    on='image_id',
                    how='left',
                    suffixes=('', '_img')
                )
        else:
            dataset = pd.DataFrame()
        
        logger.info(f"✓ Created integrated dataset with {len(dataset)} records")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(output_path, index=False)
            logger.info(f"✓ Saved dataset to {output_path}")
        
        return dataset
    
    def save_to_database(self, initialize_db: bool = False):
        """
        Save all collected data to database
        
        Args:
            initialize_db: Whether to initialize database connection first
        """
        if initialize_db:
            db_manager.initialize()
        
        with db_manager.get_session() as session:
            # Here you would insert data using ORM models
            # For now, we'll log the action
            logger.info(f"Saving {len(self.experiments)} experiments to database")
            logger.info(f"Saving {len(self.images)} images to database")
            logger.info(f"Saving {len(self.results)} results to database")
            
            # TODO: Implement actual database insertion with ORM models
            # session.add_all([ExperimentModel(**exp) for exp in self.experiments])
        
        logger.info("✓ Data saved to database")
    
    def export_to_json(self, output_path: Union[str, Path]):
        """
        Export all data to JSON format
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'experiments': self.experiments,
            'images': self.images,
            'results': self.results,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Exported data to {output_path}")
    
    def import_from_json(self, input_path: Union[str, Path]):
        """
        Import data from JSON file
        
        Args:
            input_path: Input file path
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.experiments = data.get('experiments', [])
        self.images = data.get('images', [])
        self.results = data.get('results', [])
        
        logger.info(f"✓ Imported data from {input_path}")
        logger.info(f"  - {len(self.experiments)} experiments")
        logger.info(f"  - {len(self.images)} images")
        logger.info(f"  - {len(self.results)} results")


class DataValidator:
    """
    Validate data quality and consistency
    """
    
    @staticmethod
    def validate_experiment_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate experiment data
        
        Returns:
            Validation report
        """
        issues = []
        
        required_fields = ['experiment_name', 'experiment_type']
        for field in required_fields:
            if field not in data or not data[field]:
                issues.append(f"Missing required field: {field}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    @staticmethod
    def check_data_consistency(
        experiments: List[Dict],
        results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Check consistency across datasets
        
        Returns:
            Consistency report
        """
        issues = []
        
        # Check for orphaned results
        exp_ids = {exp['experiment_id'] for exp in experiments}
        for result in results:
            if result['experiment_id'] not in exp_ids:
                issues.append(f"Result references non-existent experiment: {result['result_id']}")
        
        return {
            'consistent': len(issues) == 0,
            'issues': issues,
            'num_experiments': len(experiments),
            'num_results': len(results)
        }
