import json
from pathlib import Path
import pydicom
import numpy as np
from typing import Dict, List, Optional
import sys
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.medical_imaging.detection.candidate_detector import TumorDetector, CTPreprocessor
from src.medical_imaging.cdss.integration_engine import CTDetectionResults, CellposeResults, ClinicalData

try:
    from src.preprocessing.image_processor import CellposeProcessor
    from skimage import measure
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("Warning: CellposeProcessor not available")


def load_sample_patient(sample_id: str = "PT-TEST-1000") -> Optional[Dict]:
    """Load patient sample data from JSON"""
    try:
        data_dir = Path(__file__).parent.parent.parent / "data" / "samples"
        sample_file = data_dir / f"{sample_id}.json"
        
        if sample_file.exists():
            with open(sample_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading sample: {e}")
        return None


def list_available_samples() -> List[str]:
    """List all available patient samples"""
    try:
        data_dir = Path(__file__).parent.parent.parent / "data" / "samples"
        samples = []
        for file in data_dir.glob("PT-*.json"):
            samples.append(file.stem)
        return sorted(samples)
    except:
        return []


def sample_to_cellpose_results(sample_data: Dict) -> CellposeResults:
    """Convert sample JSON to CellposeResults"""
    quant = sample_data.get('quantitative_analysis', {})
    patient = sample_data.get('patient', {})
    
    return CellposeResults(
        cell_count=quant.get('num_cells', 0),
        mean_area_um2=quant.get('mean_area', 0.0),
        mean_circularity=0.75,  # Default
        morphology_score=min(10.0, quant.get('overall_heterogeneity', 0.5) * 10),
        ki67_index=patient.get('ki67_index', 30) / 100.0  # Convert to 0-1
    )


def sample_to_clinical_data(sample_data: Dict) -> ClinicalData:
    """Convert sample JSON to ClinicalData"""
    patient = sample_data.get('patient', {})
    
    # Check for TP53 mutation
    tp53_mutant = any(v['gene_name'] == 'TP53' for v in patient.get('genomic_variants', []))
    
    return ClinicalData(
        patient_id=patient.get('patient_id', 'UNKNOWN'),
        age=patient.get('age', 60),
        gender="M" if patient.get('gender') == 'Male' else 'F',
        kras_status="Wild-type",
        tp53_status="Mutant" if tp53_mutant else "Wild-type",
        msi_status=patient.get('microsatellite_status', 'MSS'),
        liver_function=patient.get('hepatic_function', 'Normal'),
        kidney_function="Normal",
        ecog_performance=patient.get('ecog_score', 0)
    )


def analyze_dicom_slice(dicom_file) -> Optional[CTDetectionResults]:
    """Analyze a single DICOM slice for tumor detection"""
    try:
        # Read DICOM
        if isinstance(dicom_file, str):
            dcm = pydicom.dcmread(dicom_file)
        else:
            # Streamlit UploadedFile
            dcm = pydicom.dcmread(dicom_file)
        
        # Get pixel array
        hu_image = dcm.pixel_array.astype(float)
        
        # Apply rescale if available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            hu_image = hu_image * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Get pixel spacing
        if hasattr(dcm, 'PixelSpacing'):
            pixel_spacing = (float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1]))
        else:
            pixel_spacing = (1.0, 1.0)
        
        # Detect candidates
        detector = TumorDetector(min_area_mm2=10.0, max_area_mm2=5000.0)
        candidates = detector.detect_candidates_2d(
            hu_image,
            pixel_spacing,
            body_mask=None,
            slice_index=0,
            method='multi_threshold'
        )
        
        # Filter high confidence candidates
        high_conf = [c for c in candidates if c.confidence_score > 0.7]
        
        # Determine if tumor detected
        tumor_detected = len(high_conf) > 0
        
        # Get max confidence
        max_confidence = max([c.confidence_score for c in candidates], default=0.0)
        
        # Estimate TNM from size
        tnm_stage = "Unknown"
        tumor_size_mm = None
        if high_conf:
            largest = max(high_conf, key=lambda x: x.area_mm2)
            tumor_size_mm = np.sqrt(largest.area_mm2)  # Approximate diameter
            
            # Simple TNM estimation
            if tumor_size_mm < 10:
                tnm_stage = "T1N0M0"
            elif tumor_size_mm < 20:
                tnm_stage = "T2N0M0"
            elif tumor_size_mm < 50:
                tnm_stage = "T2N1M0"
            else:
                tnm_stage = "T3N1M0"
        
        return CTDetectionResults(
            tumor_detected=tumor_detected,
            total_candidates=len(candidates),
            high_conf_candidates=len(high_conf),
            max_confidence=max_confidence,
            tumor_size_mm=tumor_size_mm,
            tumor_location="CT Slice (location TBD)",
            tnm_stage=tnm_stage if tumor_detected else None
        )
        
    except Exception as e:
        print(f"Error analyzing DICOM: {e}")
        return None


def analyze_cellpose_image(image_file, pixel_size_um: float = 0.5) -> Optional[CellposeResults]:
    """
    Analyze a microscopy image with Cellpose
    
    Args:
        image_file: Uploaded image file (Streamlit UploadedFile or path)
        pixel_size_um: Pixel size in micrometers (for area calculation)
    
    Returns:
        CellposeResults with segmentation metrics
    """
    if not CELLPOSE_AVAILABLE:
        print("Cellpose not available")
        return None
    
    try:
        # Load image
        if isinstance(image_file, str):
            img = np.array(Image.open(image_file))
        else:
            # Streamlit UploadedFile
            img = np.array(Image.open(image_file))
        
        # Initialize Cellpose
        processor = CellposeProcessor(model_type='cyto2', gpu=True)
        
        # Segment image
        masks, flows, metadata = processor.segment_image(
            img,
            diameter=None,  # Auto-detect
            channels=[0, 0]  # Grayscale
        )
        
        num_cells = metadata['num_cells']
        
        if num_cells == 0:
            return CellposeResults(
                cell_count=0,
                mean_area_um2=0.0,
                mean_circularity=0.0,
                morphology_score=0.0,
                ki67_index=0.0
            )
        
        # Extract cell features
        regions = measure.regionprops(masks)
        
        # Calculate areas (convert pixels to um^2)
        pixel_area_um2 = pixel_size_um ** 2
        areas = [r.area * pixel_area_um2 for r in regions]
        mean_area = np.mean(areas)
        
        # Calculate circularities
        circularities = []
        for r in regions:
            if r.perimeter > 0:
                circ = 4 * np.pi * r.area / (r.perimeter ** 2)
                circularities.append(min(circ, 1.0))
        mean_circularity = np.mean(circularities) if circularities else 0.5
        
        # Calculate morphology score (based on size uniformity)
        area_std = np.std(areas)
        area_cv = area_std / mean_area if mean_area > 0 else 0
        # Lower CV = higher score (more uniform)
        morphology_score = max(0, 10 - area_cv * 20)
        
        # Estimate Ki-67 (placeholder - would need immunofluorescence in real application)
        # For demo: Use cell density as proxy (more cells = higher proliferation)
        img_area_um2 = img.shape[0] * img.shape[1] * pixel_area_um2
        cell_density = num_cells / (img_area_um2 / 1000000)  # cells per mm^2
        ki67_estimate = min(0.9, cell_density / 10000)  # Cap at 90%
        
        return CellposeResults(
            cell_count=num_cells,
            mean_area_um2=float(mean_area),
            mean_circularity=float(mean_circularity),
            morphology_score=float(morphology_score),
            ki67_index=float(ki67_estimate)
        )
        
    except Exception as e:
        print(f"Error analyzing Cellpose image: {e}")
        import traceback
        traceback.print_exc()
        return None
