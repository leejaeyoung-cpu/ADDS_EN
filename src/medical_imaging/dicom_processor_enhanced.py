"""
Enhanced DICOM Processor with 2026 Best Practices
- Automatic series detection by SeriesInstanceUID
- Multi-file volume reconstruction with HU conversion
- Robust metadata extraction and validation
- Clinical context enhancement
"""

import numpy as np
import pydicom
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class DICOMMetadata:
    """Clinical metadata extracted from DICOM"""
    # Patient Info
    patient_id: str
    patient_age: Optional[int]
    patient_sex: Optional[str]
    
    # Study Info
    study_uid: str
    study_date: Optional[str]
    study_description: str
    
    # Series Info
    series_uid: str
    series_number: int
    series_description: str
    modality: str
    
    # Image Info
    slice_count: int
    slice_thickness: float
    pixel_spacing: Tuple[float, float]
    image_orientation: Optional[List[float]]
    
    # Acquisition Parameters
    kvp: float
    exposure: float
    
    # Quality Indicators
    warnings: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'patient': {
                'id': self.patient_id,
                'age': self.patient_age,
                'sex': self.patient_sex
            },
            'study': {
                'uid': self.study_uid,
                'date': self.study_date,
                'description': self.study_description
            },
            'series': {
                'uid': self.series_uid,
                'number': self.series_number,
                'description': self.series_description,
                'modality': self.modality
            },
            'image': {
                'slice_count': self.slice_count,
                'slice_thickness_mm': self.slice_thickness,
                'pixel_spacing_mm': self.pixel_spacing,
                'orientation': self.image_orientation
            },
            'acquisition': {
                'kvp': self.kvp,
                'exposure': self.exposure
            },
            'quality': {
                'warnings': self.warnings
            }
        }


class EnhancedDICOMProcessor:
    """
    2026 Best Practices DICOM Processing
    
    Features:
    - Automatic series detection and grouping
    - Proper HU (Hounsfield Unit) conversion
    - Comprehensive metadata extraction
    - Image quality validation
    - Multi-phase CT support
    """
    
    def __init__(self, enforce_valid_dicom: bool = False):
        """
        Initialize processor
        
        Args:
            enforce_valid_dicom: If True, raise errors for non-standard DICOM
        """
        self.enforce_valid = enforce_valid_dicom
        if enforce_valid_dicom:
            pydicom.config.enforce_valid_values = True
    
    def detect_series(self, dicom_dir: Path) -> Dict[str, List[Dict]]:
        """
        Detect and group DICOM files by SeriesInstanceUID
        
        Args:
            dicom_dir: Directory containing DICOM files
            
        Returns:
            Dictionary mapping series_uid to list of file info dicts
            {
                'series_uid_1': [
                    {'path': Path, 'instance': int, 'position': float},
                    ...
                ],
                ...
            }
        """
        logger.info(f"Scanning directory: {dicom_dir}")
        
        series_map = defaultdict(list)
        invalid_files = []
        
        # Find all DICOM files (.dcm extension or no extension)
        dicom_files = list(dicom_dir.rglob("*.dcm"))
        if not dicom_files:
            # Try files without extension
            dicom_files = [f for f in dicom_dir.rglob("*") if f.is_file() and not f.suffix]
        
        logger.info(f"Found {len(dicom_files)} potential DICOM files")
        
        for dcm_file in dicom_files:
            try:
                # Read metadata only (faster)
                ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)
                
                # Verify it's actually DICOM
                if not hasattr(ds, 'SeriesInstanceUID'):
                    invalid_files.append(str(dcm_file))
                    continue
                
                series_uid = ds.SeriesInstanceUID
                instance_num = int(getattr(ds, 'InstanceNumber', 0))
                
                # Get slice position for ordering
                if hasattr(ds, 'ImagePositionPatient'):
                    # Use Z-coordinate (slice position)
                    slice_position = float(ds.ImagePositionPatient[2])
                else:
                    slice_position = instance_num
                
                series_map[series_uid].append({
                    'path': dcm_file,
                    'instance': instance_num,
                    'position': slice_position,
                    'dataset': ds  # Keep for metadata
                })
                
            except Exception as e:
                logger.warning(f"Failed to read {dcm_file}: {e}")
                invalid_files.append(str(dcm_file))
                continue
        
        if invalid_files:
            logger.warning(f"Skipped {len(invalid_files)} invalid files")
        
        # Sort each series by slice position
        for series_uid in series_map:
            series_map[series_uid].sort(key=lambda x: x['position'])
            logger.info(f"Series {series_uid[:16]}... : {len(series_map[series_uid])} slices")
        
        return dict(series_map)
    
    def select_ct_abdomen_series(self, series_map: Dict[str, List[Dict]]) -> Optional[List[Dict]]:
        """
        Heuristically select CT abdomen/pelvis series
        
        Priority:
        1. Series description contains 'abd', 'abdom', 'pelv', 'portal'
        2. CT modality with most slices
        3. Fallback to largest series
        
        Args:
            series_map: Output from detect_series()
            
        Returns:
            Selected series file list or None
        """
        if not series_map:
            return None
        
        # Filter CT only
        ct_series = {}
        for series_uid, files in series_map.items():
            ds = files[0]['dataset']
            if ds.Modality == 'CT':
                ct_series[series_uid] = files
        
        if not ct_series:
            logger.warning("No CT series found, using all series")
            ct_series = series_map
        
        # Try keyword matching
        abdomen_keywords = ['abd', 'abdom', 'pelv', 'portal', 'venous', 'colon']
        for series_uid, files in ct_series.items():
            ds = files[0]['dataset']
            desc = getattr(ds, 'SeriesDescription', '').lower()
            
            if any(kw in desc for kw in abdomen_keywords):
                logger.info(f"Selected series by keyword match: '{desc}'")
                return files
        
        # Fallback: largest series
        largest_series = max(ct_series.values(), key=len)
        logger.info(f"Selected series by size: {len(largest_series)} slices")
        return largest_series
    
    def reconstruct_volume(self, series_files: List[Dict]) -> Tuple[np.ndarray, DICOMMetadata]:
        """
        Reconstruct 3D volume from DICOM series with proper HU conversion
        
        Args:
            series_files: List of file info dicts from detect_series()
            
        Returns:
            (volume, metadata) where volume is in Hounsfield Units
        """
        logger.info(f"Reconstructing volume from {len(series_files)} slices")
        
        slices = []
        
        for file_info in series_files:
            ds = pydicom.dcmread(file_info['path'])
            
            # Get pixel array
            pixel_array = ds.pixel_array.astype(np.float32)
            
            # Apply Rescale Slope and Intercept for HU conversion
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
                logger.debug(f"Applied HU conversion: slope={slope}, intercept={intercept}")
            else:
                logger.warning("No RescaleSlope/Intercept found, HU conversion skipped")
            
            slices.append(pixel_array)
        
        # Stack into 3D volume (Z, Y, X)
        volume = np.stack(slices, axis=0)
        
        # Extract metadata
        metadata = self._extract_metadata(series_files)
        
        logger.info(f"Volume reconstructed: shape={volume.shape}, "
                   f"HU range=[{volume.min():.1f}, {volume.max():.1f}]")
        
        return volume, metadata
    
    def _extract_metadata(self, series_files: List[Dict]) -> DICOMMetadata:
        """Extract comprehensive metadata from DICOM series"""
        # Use first slice for most metadata
        ds = series_files[0]['dataset']
        
        # Parse age (format: '065Y' -> 65)
        age = None
        if hasattr(ds, 'PatientAge'):
            age_str = ds.PatientAge
            try:
                age = int(age_str.rstrip('YMD'))  # Remove Y/M/D suffix
            except:
                pass
        
        # Pixel spacing
        pixel_spacing = (1.0, 1.0)  # Default
        if hasattr(ds, 'PixelSpacing'):
            pixel_spacing = tuple(float(x) for x in ds.PixelSpacing)
        
        # Image orientation
        orientation = None
        if hasattr(ds, 'ImageOrientationPatient'):
            orientation = [float(x) for x in ds.ImageOrientationPatient]
        
        # Quality warnings
        warnings = []
        
        # Check slice thickness
        slice_thickness = float(getattr(ds, 'SliceThickness', 0))
        if slice_thickness > 5.0:
            warnings.append(f"Thick slices ({slice_thickness:.1f}mm) may reduce sensitivity")
        
        # Check slice count
        if len(series_files) < 20:
            warnings.append(f"Few slices ({len(series_files)}) may indicate partial scan")
        
        # Check acquisition parameters
        kvp = float(getattr(ds, 'KVP', 0))
        if kvp > 0 and kvp < 100:
            warnings.append(f"Low kVp ({kvp}) may affect image quality")
        
        metadata = DICOMMetadata(
            patient_id=getattr(ds, 'PatientID', 'UNKNOWN'),
            patient_age=age,
            patient_sex=getattr(ds, 'PatientSex', None),
            study_uid=ds.StudyInstanceUID,
            study_date=getattr(ds, 'StudyDate', None),
            study_description=getattr(ds, 'StudyDescription', ''),
            series_uid=ds.SeriesInstanceUID,
            series_number=int(getattr(ds, 'SeriesNumber', 0)),
            series_description=getattr(ds, 'SeriesDescription', ''),
            modality=ds.Modality,
            slice_count=len(series_files),
            slice_thickness=slice_thickness,
            pixel_spacing=pixel_spacing,
            image_orientation=orientation,
            kvp=kvp,
            exposure=float(getattr(ds, 'Exposure', 0)),
            warnings=warnings
        )
        
        return metadata
    
    def validate_series(self, series_files: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate series consistency and quality
        
        Returns:
            (is_valid, warnings)
        """
        warnings = []
        
        # Check minimum slices
        if len(series_files) < 10:
            warnings.append(f"Only {len(series_files)} slices (recommend >20)")
        
        # Check spacing consistency
        spacings = []
        for file_info in series_files[:min(5, len(series_files))]:
            ds = file_info['dataset']
            if hasattr(ds, 'SliceThickness'):
                spacings.append(float(ds.SliceThickness))
        
        if spacings:
            if max(spacings) - min(spacings) > 0.1:
                warnings.append(f"Inconsistent slice thickness: {spacings}")
        
        # Check for gaps in slice positions
        positions = [f['position'] for f in series_files]
        diffs = np.diff(positions)
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            if np.any(np.abs(diffs - mean_diff) > mean_diff * 0.5):
                warnings.append("Potential gaps in slice positions detected")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def process_dicom_directory(self, dicom_dir: Union[str, Path], 
                                select_series: bool = True) -> Tuple[np.ndarray, DICOMMetadata]:
        """
        Complete pipeline: Directory -> Volume + Metadata
        
        Args:
            dicom_dir: Directory containing DICOM files
            select_series: If True, auto-select abdomen series
            
        Returns:
            (ct_volume, metadata)
        """
        dicom_dir = Path(dicom_dir)
        
        # Step 1: Detect series
        series_map = self.detect_series(dicom_dir)
        
        if not series_map:
            raise ValueError(f"No valid DICOM series found in {dicom_dir}")
        
        # Step 2: Select series
        if select_series:
            target_series = self.select_ct_abdomen_series(series_map)
        else:
            # Use first series
            target_series = list(series_map.values())[0]
        
        if target_series is None:
            raise ValueError("Failed to select appropriate series")
        
        # Step 3: Validate
        is_valid, validation_warnings = self.validate_series(target_series)
        if not is_valid:
            logger.warning("Series validation issues: " + "; ".join(validation_warnings))
        
        # Step 4: Reconstruct
        volume, metadata = self.reconstruct_volume(target_series)
        
        # Add validation warnings to metadata
        metadata.warnings.extend(validation_warnings)
        
        return volume, metadata
    
    def save_metadata(self, metadata: DICOMMetadata, output_path: Union[str, Path]):
        """Save metadata to JSON file"""
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.info(f"Metadata saved to {output_path}")


def demo_usage():
    """Example usage"""
    processor = EnhancedDICOMProcessor()
    
    # Process DICOM directory
    dicom_dir = Path("data/patient_data/case_001")
    
    try:
        volume, metadata = processor.process_dicom_directory(dicom_dir)
        
        print(f"✅ Successfully processed DICOM series")
        print(f"   Patient ID: {metadata.patient_id}")
        print(f"   Series: {metadata.series_description}")
        print(f"   Volume shape: {volume.shape}")
        print(f"   HU range: [{volume.min():.1f}, {volume.max():.1f}]")
        
        if metadata.warnings:
            print(f"⚠️  Warnings:")
            for warning in metadata.warnings:
                print(f"     - {warning}")
        
        # Save metadata
        processor.save_metadata(metadata, "output_metadata.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_usage()
