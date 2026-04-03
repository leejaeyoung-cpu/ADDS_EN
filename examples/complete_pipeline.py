"""
Example: Complete pipeline demonstration for ADDS
Demonstrates image processing, synergy calculation, and data integration
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Import ADDS modules
from src.preprocessing.image_processor import CellposeProcessor
from src.preprocessing.document_parser import DocumentParser
from src.utils.synergy_calculator import SynergyCalculator
from src.data.data_integrator import DataIntegrator
from src.utils import get_logger

logger = get_logger(__name__)


def example_1_image_processing():
    """
    Example 1: Process cell images with Cellpose
    """
    logger.info("=" * 50)
    logger.info("Example 1: Image Processing with Cellpose")
    logger.info("=" * 50)
    
    # Initialize processor
    processor = CellposeProcessor(
        model_type='cyto2',
        gpu=True,  # Set to False if no GPU
        batch_size=8
    )
    
    # Create sample images directory (if you have real images, use those)
    data_dir = Path('data/raw/images')
    output_dir = Path('data/processed/cellpose_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Looking for images in: {data_dir}")
    
    # Process all images in directory
    if data_dir.exists():
        image_files = list(data_dir.glob('*.tif')) + list(data_dir.glob('*.tiff'))
        
        if image_files:
            logger.info(f"Found {len(image_files)} images")
            
            for img_path in image_files[:3]:  # Process first 3
                logger.info(f"\nProcessing: {img_path.name}")
                
                results = processor.process_and_save(
                    image_path=img_path,
                    output_dir=output_dir,
                    save_masks=True,
                    save_features=True,
                    save_visualization=True
                )
                
                logger.info(f"  Cells detected: {results['metadata']['num_cells']}")
                logger.info(f"  Health score: {results['metrics']['estimated_health_score']:.2f}")
                logger.info(f"  Results saved to: {output_dir}")
        else:
            logger.warning("No .tif/.tiff images found. Place your images in data/raw/images/")
    else:
        logger.warning(f"Directory not found: {data_dir}")
        logger.info("Create the directory and add your cell images to proceed.")
    
    logger.info("\n✓ Example 1 completed\n")


def example_2_synergy_calculation():
    """
    Example 2: Calculate drug synergy scores
    """
    logger.info("=" * 50)
    logger.info("Example 2: Drug Synergy Calculation")
    logger.info("=" * 50)
    
    calculator = SynergyCalculator()
    
    # Example drug combination experiment
    experiments = [
        {
            'drug_a': 'Doxorubicin',
            'drug_b': 'Cisplatin',
            'dose_a': 10.0,
            'dose_b': 15.0,
            'effect_a': 0.35,
            'effect_b': 0.30,
            'effect_comb': 0.75,
            'ic50_a': 5.0,
            'ic50_b': 8.0
        },
        {
            'drug_a': 'Paclitaxel',
            'drug_b': 'Gemcitabine',
            'dose_a': 5.0,
            'dose_b': 20.0,
            'effect_a': 0.40,
            'effect_b': 0.25,
            'effect_comb': 0.70,
            'ic50_a': 3.0,
            'ic50_b': 12.0
        }
    ]
    
    results = []
    
    for exp in experiments:
        logger.info(f"\nAnalyzing: {exp['drug_a']} + {exp['drug_b']}")
        
        synergy = calculator.calculate_all_synergies(
            dose_a=exp['dose_a'],
            dose_b=exp['dose_b'],
            effect_a=exp['effect_a'],
            effect_b=exp['effect_b'],
            effect_combination=exp['effect_comb'],
            ic50_a=exp['ic50_a'],
            ic50_b=exp['ic50_b']
        )
        
        logger.info(f"  Bliss Score: {synergy['bliss']:.3f}")
        logger.info(f"  HSA Score: {synergy['hsa']:.3f}")
        logger.info(f"  Loewe Score: {synergy['loewe']:.3f}")
        logger.info(f"  ZIP Score: {synergy['zip']:.3f}")
        logger.info(f"  Synergistic: {synergy['is_synergistic']}")
        
        # Create report
        report = calculator.create_synergy_report(
            drug_a_name=exp['drug_a'],
            drug_b_name=exp['drug_b'],
            synergy_scores=synergy,
            dose_a=exp['dose_a'],
            dose_b=exp['dose_b']
        )
        
        results.append(report)
    
    # Combine and save results
    combined_results = pd.concat(results, ignore_index=True)
    output_path = Path('data/outputs/synergy_results.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_results.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Synergy results saved to: {output_path}")
    logger.info("\n✓ Example 2 completed\n")


def example_3_data_integration():
    """
    Example 3: Integrate multimodal data
    """
    logger.info("=" * 50)
    logger.info("Example 3: Data Integration")
    logger.info("=" * 50)
    
    integrator = DataIntegrator()
    
    # Create experiment record
    experiment = integrator.create_experiment_record(
        experiment_name="Doxorubicin + Cisplatin Combination Test",
        experiment_type="cell_viability",
        cell_line="MCF-7",
        performed_by="연구팀",
        description="항암제 칵테일 시너지 테스트"
    )
    
    logger.info(f"Created experiment: {experiment['experiment_id']}")
    
    # Add drug combination
    combination = integrator.add_drug_combination(
        experiment_id=experiment['experiment_id'],
        compounds=['Doxorubicin', 'Cisplatin'],
        concentrations=[10.0, 15.0],
        concentration_units='μM'
    )
    
    logger.info(f"Added combination: {combination['combination_name']}")
    
    # Add experimental results
    integrator.add_experimental_result(
        experiment_id=experiment['experiment_id'],
        combination_id=combination['combination_id'],
        result_type='viability',
        value=0.75,
        unit='fraction',
        standard_deviation=0.05,
        n_replicates=3
    )
    
    logger.info("Added experimental result")
    
    # Create integrated dataset
    dataset = integrator.create_integrated_dataset(
        output_path='data/outputs/integrated_dataset.csv'
    )
    
    logger.info(f"Integrated dataset shape: {dataset.shape}")
    
    # Export to JSON
    integrator.export_to_json('data/outputs/experiment_data.json')
    
    logger.info("\n✓ Example 3 completed\n")


def example_4_document_processing():
    """
    Example 4: Process PDF documents
    """
    logger.info("=" * 50)
    logger.info("Example 4: Document Processing")
    logger.info("=" * 50)
    
    parser = DocumentParser()
    
    # Check for PDF files
    pdf_dir = Path('data/raw/documents')
    
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob('*.pdf'))
        
        if pdf_files:
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for pdf_path in pdf_files[:2]:  # Process first 2
                logger.info(f"\nProcessing: {pdf_path.name}")
                
                result = parser.parse_experiment_report(pdf_path)
                
                logger.info(f"  Pages: {result.get('num_pages', 'N/A')}")
                logger.info(f"  Compounds found: {result.get('compounds', [])}")
                logger.info(f"  Tables extracted: {len(result.get('tables', []))}")
                
                # Save extracted data
                output_path = Path(f'data/processed/documents/{pdf_path.stem}_extracted.json')
                parser.save_structured_data(result, output_path, format='json')
                logger.info(f"  Saved to: {output_path}")
        else:
            logger.warning("No PDF files found. Place PDFs in data/raw/documents/")
    else:
        logger.warning(f"Directory not found: {pdf_dir}")
        logger.info("Create the directory and add PDF documents to proceed.")
    
    logger.info("\n✓ Example 4 completed\n")


def main():
    """
    Run all examples
    """
    logger.info("=" * 50)
    logger.info("ADDS - Complete Pipeline Examples")
    logger.info("=" * 50)
    logger.info("")
    
    # Run each example
    try:
        # Example 1: Image Processing
        example_1_image_processing()
        
        # Example 2: Synergy Calculation
        example_2_synergy_calculation()
        
        # Example 3: Data Integration
        example_3_data_integration()
        
        # Example 4: Document Processing
        example_4_document_processing()
        
        logger.info("=" * 50)
        logger.info("✓ All examples completed successfully!")
        logger.info("=" * 50)
        logger.info("")
        logger.info("Check the following directories for outputs:")
        logger.info("  - data/processed/cellpose_results/")
        logger.info("  - data/processed/documents/")
        logger.info("  - data/outputs/")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
