"""
CDSS Batch Processing Helper
Process multiple images for Cellpose and CT analysis
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.medical_imaging.cdss.integration_engine import CellposeResults, CTDetectionResults
from src.data.cdss_database import get_database

# Try to import analysis modules
try:
    from src.preprocessing.image_processor import CellposeProcessor
    from skimage import measure
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

try:
    import pydicom
    from src.medical_imaging.detection.candidate_detector import TumorDetector
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def process_cellpose_batch(
    uploaded_files,
    pixel_size_um: float = 0.5,
    patient_id: str = "UNKNOWN",
    progress_callback=None
) -> Tuple[List[Dict], Dict]:
    """
    Process multiple Cellpose images in batch
    
    Args:
        uploaded_files: List of uploaded image files
        pixel_size_um: Pixel size in micrometers
        patient_id: Patient identifier
        progress_callback: Streamlit progress bar callback
    
    Returns:
        results: List of analysis results for each image
        summary: Summary statistics
    """
    if not CELLPOSE_AVAILABLE:
        return [], {"error": "Cellpose not available"}
    
    results = []
    total_cells = 0
    total_images = len(uploaded_files)
    
    # Initialize Cellpose
    processor = CellposeProcessor(model_type='cyto2', gpu=True)
    
    for idx, image_file in enumerate(uploaded_files):
        try:
            # Update progress
            if progress_callback:
                progress_callback((idx + 1) / total_images, 
                                 f"Processing {image_file.name}... ({idx+1}/{total_images})")
            
            # Load image
            img = np.array(Image.open(image_file))
            
            # Segment
            masks, flows, metadata = processor.segment_image(
                img,
                diameter=None,
                channels=[0, 0]
            )
            
            num_cells = metadata['num_cells']
            
            if num_cells == 0:
                result = {
                    'image_name': image_file.name,
                    'cell_count': 0,
                    'mean_area_um2': 0.0,
                    'mean_circularity': 0.0,
                    'morphology_score': 0.0,
                    'ki67_index': 0.0,
                    'status': 'no_cells_detected'
                }
            else:
                # Extract features
                regions = measure.regionprops(masks)
                
                pixel_area_um2 = pixel_size_um ** 2
                areas = [r.area * pixel_area_um2 for r in regions]
                mean_area = np.mean(areas)
                
                # Circularities
                circularities = []
                for r in regions:
                    if r.perimeter > 0:
                        circ = 4 * np.pi * r.area / (r.perimeter ** 2)
                        circularities.append(min(circ, 1.0))
                mean_circularity = np.mean(circularities) if circularities else 0.5
                
                # Morphology score
                area_std = np.std(areas)
                area_cv = area_std / mean_area if mean_area > 0 else 0
                morphology_score = max(0, 10 - area_cv * 20)
                
                # Ki-67 estimate
                img_area_um2 = img.shape[0] * img.shape[1] * pixel_area_um2
                cell_density = num_cells / (img_area_um2 / 1000000)
                ki67_estimate = min(0.9, cell_density / 10000)
                
                result = {
                    'image_name': image_file.name,
                    'cell_count': num_cells,
                    'mean_area_um2': float(mean_area),
                    'mean_circularity': float(mean_circularity),
                    'morphology_score': float(morphology_score),
                    'ki67_index': float(ki67_estimate),
                    'status': 'success'
                }
            
            results.append(result)
            total_cells += num_cells
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            results.append({
                'image_name': image_file.name,
                'cell_count': 0,
                'status': f'error: {str(e)}'
            })
    
    # Summary statistics
    successful = [r for r in results if r.get('status') == 'success']
    summary = {
        'total_images': total_images,
        'successful': len(successful),
        'failed': total_images - len(successful),
        'total_cells': total_cells,
        'mean_cells_per_image': total_cells / len(successful) if successful else 0,
        'mean_ki67': np.mean([r['ki67_index'] for r in successful]) if successful else 0
    }
    
    return results, summary


def process_ct_batch(
    uploaded_files,
    patient_id: str = "UNKNOWN",
    progress_callback=None
) -> Tuple[List[Dict], Dict]:
    """
    Process multiple DICOM files in batch
    
    Args:
        uploaded_files: List of uploaded DICOM files
        patient_id: Patient identifier
        progress_callback: Streamlit progress bar callback
    
    Returns:
        results: List of CT analysis results
        summary: Summary statistics
    """
    if not PYDICOM_AVAILABLE:
        return [], {"error": "pydicom not available"}
    
    results = []
    total_candidates = 0
    total_high_conf = 0
    total_files = len(uploaded_files)
    
    detector = TumorDetector(min_area_mm2=10.0, max_area_mm2=5000.0)
    
    for idx, dicom_file in enumerate(uploaded_files):
        try:
            # Update progress
            if progress_callback:
                progress_callback((idx + 1) / total_files,
                                 f"Processing {dicom_file.name}... ({idx+1}/{total_files})")
            
            # Read DICOM
            dcm = pydicom.dcmread(dicom_file)
            
            # Get pixel array
            hu_image = dcm.pixel_array.astype(float)
            
            # Apply rescale
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                hu_image = hu_image * dcm.RescaleSlope + dcm.RescaleIntercept
            
            # Get pixel spacing
            if hasattr(dcm, 'PixelSpacing'):
                pixel_spacing = (float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]))
            else:
                pixel_spacing = (1.0, 1.0)
            
            # Detect candidates
            candidates = detector.detect_candidates_2d(
                hu_image,
                pixel_spacing,
                body_mask=None,
                slice_index=0,
                method='multi_threshold'
            )
            
            high_conf = [c for c in candidates if c.confidence_score > 0.7]
            tumor_detected = len(high_conf) > 0
            max_confidence = max([c.confidence_score for c in candidates], default=0.0)
            
            # Estimate TNM
            tnm_stage = None
            tumor_size_mm = None
            if high_conf:
                largest = max(high_conf, key=lambda x: x.area_mm2)
                tumor_size_mm = np.sqrt(largest.area_mm2)
                
                if tumor_size_mm < 10:
                    tnm_stage = "T1N0M0"
                elif tumor_size_mm < 20:
                    tnm_stage = "T2N0M0"
                elif tumor_size_mm < 50:
                    tnm_stage = "T2N1M0"
                else:
                    tnm_stage = "T3N1M0"
            
            result = {
                'dicom_name': dicom_file.name,
                'tumor_detected': tumor_detected,
                'total_candidates': len(candidates),
                'high_conf_candidates': len(high_conf),
                'max_confidence': float(max_confidence),
                'tumor_size_mm': float(tumor_size_mm) if tumor_size_mm else None,
                'tnm_stage': tnm_stage,
                'status': 'success'
            }
            
            results.append(result)
            total_candidates += len(candidates)
            total_high_conf += len(high_conf)
            
        except Exception as e:
            print(f"Error processing {dicom_file.name}: {e}")
            results.append({
                'dicom_name': dicom_file.name,
                'tumor_detected': False,
                'total_candidates': 0,
                'status': f'error: {str(e)}'
            })
    
    # Summary
    successful = [r for r in results if r.get('status') == 'success']
    tumors_found = sum(1 for r in successful if r.get('tumor_detected'))
    
    summary = {
        'total_files': total_files,
        'successful': len(successful),
        'failed': total_files - len(successful),
        'tumors_detected': tumors_found,
        'total_candidates': total_candidates,
        'total_high_conf': total_high_conf,
        'detection_rate': tumors_found / len(successful) if successful else 0
    }
    
    return results, summary


def save_batch_to_database(
    results: List[Dict],
    summary: Dict,
    patient_id: str,
    analysis_type: str,
    created_by: str = "System"
) -> int:
    """
    Save batch analysis results to database
    
    Args:
        results: List of analysis results
        summary: Summary statistics
        patient_id: Patient identifier
        analysis_type: 'cellpose_batch' or 'ct_batch'
        created_by: User name
    
    Returns:
        record_id: Database record ID
    """
    db = get_database()
    
    # Create main record
    record_id = db.create_analysis_record(
        patient_id=patient_id,
        analysis_type=analysis_type,
        total_items=summary.get('total_images', summary.get('total_files', 0)),
        summary=str(summary),
        created_by=created_by
    )
    
    # Save batch results
    if analysis_type == 'cellpose_batch':
        db.save_cellpose_batch(record_id, results)
    elif analysis_type == 'ct_batch':
        db.save_ct_batch(record_id, results)
    
    return record_id


def load_patient_history(patient_id: str) -> List[Dict]:
    """
    Load analysis history for a patient
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        List of analysis records
    """
    db = get_database()
    return db.get_patient_records(patient_id)


def export_results_to_csv(record_id: int, output_path: str):
    """
    Export analysis results to CSV
    
    Args:
        record_id: Database record ID
        output_path: Output file path
    """
    db = get_database()
    db.export_to_csv(output_path, record_id=record_id)
