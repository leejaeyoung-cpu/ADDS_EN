"""
CT Analyzer UI Backend
Handles DICOM processing, inference, and results for UI
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import time

logger = logging.getLogger(__name__)


class CTAnalyzerUI:
    """
    Backend logic for CT analysis UI
    Bridges SOTA pipeline with Streamlit interface
    """
    
    def __init__(self):
        self.predictor = None
        self.ensemble = None
        self.current_volume = None
        self.current_metadata = None
        self.prediction = None
        self.uncertainty = None
    
    def process_dicom_series(
        self,
        files: List,
        progress_callback=None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process uploaded files (DICOM or regular images)
        
        Args:
            files: List of file objects (from st.file_uploader)
            progress_callback: Optional callback for progress updates
        
        Returns:
            processed_volume: Preprocessed 3D volume
            metadata: File metadata
        """
        logger.info(f"Processing {len(files)} files...")
        
        if progress_callback:
            progress_callback(0.1, "Loading files...")
        
        # Detect file type
        first_file = files[0]
        file_ext = first_file.name.split('.')[-1].lower()
        
        is_dicom = file_ext in ['dcm', 'dicom']
        
        if is_dicom:
            # DICOM processing
            from ..data.preprocessing_pipeline import DICOMSeriesLoader, MedicalPreprocessor
            
            loader = DICOMSeriesLoader()
            volume, metadata = loader.load_series(files)
            
            logger.info(f"Loaded DICOM volume shape: {volume.shape}")
            logger.info(f"Spacing: {metadata.get('spacing')}")
        else:
            # Regular image processing
            logger.info("Processing as regular images...")
            volume, metadata = self._process_image_stack(files, progress_callback)
        
        if progress_callback:
            progress_callback(0.3, "Preprocessing volume...")
        
        # Preprocess
        from ..data.preprocessing_pipeline import MedicalPreprocessor
        
        preprocessor = MedicalPreprocessor(target_spacing=(1.0, 1.0, 1.0))
        processed = preprocessor.preprocess_volume(
            volume,
            metadata,
            apply_windowing=True if is_dicom else False,
            apply_resampling=True,
            apply_normalization=True
        )
        
        if progress_callback:
            progress_callback(0.5, "Preprocessing complete!")
        
        # Store for later use
        self.current_volume = processed
        self.current_metadata = metadata
        
        logger.info(f"Processed volume shape: {processed.shape}")
        
        return processed, metadata
    
    def _process_image_stack(
        self,
        image_files: List,
        progress_callback=None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process regular images as a 3D stack
        
        Args:
            image_files: List of image file objects
            progress_callback: Optional progress callback
        
        Returns:
            volume: 3D volume (D, H, W)
            metadata: Basic metadata
        """
        from PIL import Image
        import io
        
        logger.info(f"Processing {len(image_files)} images as stack...")
        
        slices = []
        
        for i, img_file in enumerate(image_files):
            # Load image
            image = Image.open(io.BytesIO(img_file.read()))
            
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # To numpy
            img_array = np.array(image, dtype=np.float32)
            slices.append(img_array)
            
            if progress_callback and i % 10 == 0:
                progress = 0.1 + (i / len(image_files)) * 0.2
                progress_callback(progress, f"Loading images... {i+1}/{len(image_files)}")
        
        # Stack to 3D
        volume = np.stack(slices, axis=0)
        
        # Create basic metadata
        metadata = {
            'spacing': [1.0, 1.0, 1.0],  # Assume isotropic
            'source': 'image_stack',
            'num_slices': len(slices),
            'image_shape': volume.shape
        }
        
        logger.info(f"Created volume from images: {volume.shape}")
        
        return volume, metadata
    
    def run_inference(
        self,
        volume: np.ndarray,
        settings: Dict,
        progress_callback=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run inference on preprocessed volume
        
        Args:
            volume: Preprocessed volume (D, H, W)
            settings: Inference settings dict
                - use_ensemble: bool
                - use_tta: bool
                - apply_postprocessing: bool
                - checkpoint_paths: list (for ensemble)
                - checkpoint_path: str (for single model)
            progress_callback: Progress callback
        
        Returns:
            prediction: Binary segmentation (D, H, W)
            uncertainty: Uncertainty map (optional, D, H, W)
        """
        from ..inference.predictor import SOTAPredictor
        from ..inference.ensemble import EnsemblePredictor
        from ..inference.postprocess import postprocess_segmentation
        
        logger.info("Starting inference...")
        
        if progress_callback:
            progress_callback(0.5, "Initializing model...")
        
        # Create predictor
        if settings.get('use_ensemble', False):
            checkpoint_paths = settings.get('checkpoint_paths', [])
            
            if not checkpoint_paths:
                # Use default paths (if trained)
                checkpoint_paths = [
                    "models/sota/fold_0/best_model.pth",
                    "models/sota/fold_1/best_model.pth",
                    "models/sota/fold_2/best_model.pth",
                    "models/sota/fold_3/best_model.pth",
                    "models/sota/fold_4/best_model.pth"
                ]
            
            # Check if checkpoints exist
            existing_checkpoints = [p for p in checkpoint_paths if Path(p).exists()]
            
            if not existing_checkpoints:
                logger.warning("No trained checkpoints found. Using mock prediction.")
                return self._mock_prediction(volume)
            
            predictor = EnsemblePredictor(
                checkpoint_paths=existing_checkpoints,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                patch_size=(96, 96, 96),
                overlap=0.5,
                ensemble_mode='average'
            )
            
            if progress_callback:
                progress_callback(0.6, f"Running ensemble ({len(existing_checkpoints)} models)...")
            
            # Predict with ensemble
            prediction, uncertainty = predictor.predict(
                volume,
                return_uncertainty=True
            )
            
        else:
            checkpoint_path = settings.get('checkpoint_path', 'models/sota/fold_0/best_model.pth')
            
            if not Path(checkpoint_path).exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}. Using mock prediction.")
                return self._mock_prediction(volume)
            
            predictor = SOTAPredictor(
                checkpoint_path=checkpoint_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                patch_size=(96, 96, 96),
                overlap=0.5,
                use_tta=settings.get('use_tta', False),
                apply_postprocessing=False  # We'll apply separately
            )
            
            if progress_callback:
                progress_callback(0.6, "Running inference...")
            
            # Predict
            prediction, probs = predictor.predict(volume, return_probabilities=True)
            uncertainty = None
        
        if progress_callback:
            progress_callback(0.8, "Inference complete!")
        
        # Apply post-processing if requested
        if settings.get('apply_postprocessing', True):
            if progress_callback:
                progress_callback(0.85, "Applying post-processing...")
            
            # Use less aggressive filtering for demo/mock predictions
            min_size = 500  # Reduced from 1000 for better demo visibility
            
            prediction, stats = postprocess_segmentation(
                prediction,
                min_size=min_size,
                max_size=50000,
                fill_holes_size=500,
                apply_morphology=True
            )
            
            logger.info(f"Post-processing stats: {stats}")
        
        if progress_callback:
            progress_callback(1.0, "Analysis complete!")
        
        # Store results
        self.prediction = prediction
        self.uncertainty = uncertainty
        
        return prediction, uncertainty
    
    def _mock_prediction(
        self,
        volume: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate mock prediction for demonstration
        (Used when trained model is not available)
        """
        logger.warning("Using mock prediction (no trained model)")
        
        # Create a simple mock tumor in center
        D, H, W = volume.shape
        
        prediction = np.zeros((D, H, W), dtype=np.uint8)
        
        # Handle single-slice or very shallow volumes
        if D <= 3:
            logger.info(f"Shallow volume detected (D={D}), creating 2D circular tumor")
            center_y, center_x = H // 2, W // 2
            radius_2d = min(H, W) // 6
            
            for y in range(H):
                for x in range(W):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist < radius_2d:
                        prediction[:, y, x] = 1  # Apply to all slices
            
            tumor_voxels = prediction.sum()
            logger.info(f"2D Mock tumor created: {tumor_voxels} voxels, radius={radius_2d}")
        else:
            # 3D spherical tumor for multi-slice volumes
            center_z, center_y, center_x = D // 2, H // 2, W // 2
            radius = max(min(D, H, W) // 6, 5)  # Ensure minimum radius of 5
            
            logger.info(f"Creating 3D mock tumor: center=({center_z}, {center_y}, {center_x}), radius={radius}")
            
            for z in range(D):
                for y in range(H):
                    for x in range(W):
                        dist = np.sqrt(
                            (z - center_z)**2 +
                            (y - center_y)**2 +
                            (x - center_x)**2
                        )
                        if dist < radius:
                            prediction[z, y, x] = 1
            
            # Add a second smaller tumor for variety (only if volume is large enough)
            if D > 20 and H > 100 and W > 100:
                tumor2_z = center_z + D // 4
                tumor2_y = center_y - H // 6
                tumor2_x = center_x + W // 6
                radius2 = radius // 2
                
                for z in range(D):
                    for y in range(H):
                        for x in range(W):
                            dist = np.sqrt(
                                (z - tumor2_z)**2 +
                                (y - tumor2_y)**2 +
                                (x - tumor2_x)**2
                            )
                            if dist < radius2:
                                prediction[z, y, x] = 1
            
            tumor_voxels = prediction.sum()
            logger.info(f"3D Mock prediction created: {tumor_voxels} tumor voxels")
        
        # No uncertainty available in mock mode — return zeros instead of random noise
        # to avoid misleading users with fake confidence intervals
        logger.warning("Mock prediction: uncertainty map is zeroed (no model available)")
        uncertainty = np.zeros((D, H, W), dtype=np.float32)
        
        return prediction, uncertainty
    
    def calculate_metrics(
        self,
        prediction: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """
        Calculate metrics from prediction
        
        Args:
            prediction: Binary segmentation
            metadata: DICOM metadata
        
        Returns:
            metrics: Dictionary of metrics
        """
        spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
        
        # Calculate volume
        num_voxels = prediction.sum()
        voxel_volume = np.prod(spacing)  # mm³
        tumor_volume_mm3 = num_voxels * voxel_volume
        tumor_volume_cm3 = tumor_volume_mm3 / 1000
        
        # Calculate bounding box
        if num_voxels > 0:
            coords = np.argwhere(prediction > 0)
            bbox_min = coords.min(axis=0)
            bbox_max = coords.max(axis=0)
            bbox_size = bbox_max - bbox_min + 1
            bbox_size_mm = (bbox_size * np.array(spacing)).tolist()
        else:
            bbox_size_mm = [0, 0, 0]
        
        metrics = {
            'tumor_volume_cm3': float(tumor_volume_cm3),
            'tumor_volume_mm3': float(tumor_volume_mm3),
            'num_voxels': int(num_voxels),
            'bbox_size_mm': bbox_size_mm,
            'spacing': spacing
        }
        
        return metrics
    
    def get_slice_view(
        self,
        volume: np.ndarray,
        segmentation: np.ndarray,
        axis: int,
        slice_idx: int,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Get 2D slice with overlay
        
        Args:
            volume: 3D volume
            segmentation: 3D segmentation
            axis: 0 (axial), 1 (sagittal), 2 (coronal)
            slice_idx: Slice index
            alpha: Overlay transparency
        
        Returns:
            rgb_slice: RGB image with overlay
        """
        # Extract slice
        if axis == 0:  # Axial
            img_slice = volume[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        elif axis == 1:  # Sagittal
            img_slice = volume[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :]
        else:  # Coronal
            img_slice = volume[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]
        
        # Normalize image to 0-255
        img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8) * 255).astype(np.uint8)
        
        # Create RGB
        rgb_slice = np.stack([img_slice, img_slice, img_slice], axis=-1)
        
        # Add red overlay for segmentation
        red_overlay = np.zeros_like(rgb_slice)
        red_overlay[seg_slice > 0, 0] = 255  # Red channel
        
        # Blend
        rgb_slice = (rgb_slice * (1 - alpha) + red_overlay * alpha).astype(np.uint8)
        
        return rgb_slice


# Test
if __name__ == "__main__":
    print("Testing CTAnalyzerUI...")
    
    analyzer = CTAnalyzerUI()
    
    # Test mock prediction
    dummy_volume = np.random.randn(64, 128, 128).astype(np.float32)
    prediction, uncertainty = analyzer._mock_prediction(dummy_volume)
    
    print(f"Mock prediction shape: {prediction.shape}")
    print(f"Tumor voxels: {prediction.sum()}")
    
    # Test metrics
    metadata = {'spacing': [1.0, 1.0, 1.0]}
    metrics = analyzer.calculate_metrics(prediction, metadata)
    
    print(f"Metrics: {metrics}")
    print("[OK] CTAnalyzerUI test passed!")
