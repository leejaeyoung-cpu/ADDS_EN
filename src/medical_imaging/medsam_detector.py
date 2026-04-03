#!/usr/bin/env python3
"""
MedSAM CT Tumor Detection
Segment Anything Model for medical CT images
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pydicom

# SAM imports
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment-anything not installed")

class MedSAMDetector:
    """MedSAM-based CT tumor detection"""
    
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        """
        Initialize MedSAM detector
        
        Args:
            model_type: 'vit_h', 'vit_l', or 'vit_b'
            checkpoint_path: Path to SAM checkpoint
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything package required")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if checkpoint_path is None:
            # Default checkpoint path
            checkpoint_path = "models/sam_vit_h_4b8939.pth"
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"\n{'='*70}")
            print("SAM Checkpoint not found!")
            print("="*70)
            print(f"\nPlease download SAM weights:")
            print(f"\n1. Create models directory:")
            print(f"   mkdir models")
            print(f"\n2. Download ViT-H checkpoint:")
            print(f"   curl -L -o models/sam_vit_h_4b8939.pth \\")
            print(f"   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
            print(f"\n3. Or use smaller model (ViT-L):")
            print(f"   curl -L -o models/sam_vit_l_0b3195.pth \\")
            print(f"   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
            print(f"\n{'='*70}\n")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load SAM model
        print(f"Loading SAM model from {checkpoint_path}...")
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.sam.to(device=self.device)
        
        self.predictor = SamPredictor(self.sam)
        print("SAM model loaded successfully!")
    
    def load_ct_slice(self, dicom_path):
        """Load and preprocess CT DICOM slice"""
        dcm = pydicom.dcmread(dicom_path)
        image = dcm.pixel_array.astype(float)
        
        # Apply HU conversion
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Window to soft tissue
        window_level = 40
        window_width = 400
        img_min = window_level - window_width / 2
        img_max = window_level + window_width / 2
        
        image = np.clip(image, img_min, img_max)
        image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # Convert to 3-channel for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image_rgb, image
    
    def predict_with_points(self, image, points, labels=None):
        """
        Predict segmentation with point prompts
        
        Args:
            image: RGB image
            points: np.array of shape (N, 2) with (x, y) coordinates
            labels: np.array of shape (N,) with 1 for foreground, 0 for background
        
        Returns:
            masks, scores, logits
        """
        if labels is None:
            labels = np.ones(len(points))
        
        # Set image
        self.predictor.set_image(image)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        return masks, scores, logits
    
    def predict_with_box(self, image, box):
        """
        Predict segmentation with bounding box
        
        Args:
            image: RGB image
            box: np.array [x1, y1, x2, y2]
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False,
        )
        
        return masks, scores, logits

def test_medsam():
    """Test MedSAM on Inha CT data"""
    
    print("\n" + "="*70)
    print("MedSAM CT Tumor Detection Test")
    print("="*70)
    
    try:
        # Initialize detector
        detector = MedSAMDetector(
            model_type="vit_h",
            checkpoint_path="models/sam_vit_h_4b8939.pth"
        )
    except FileNotFoundError as e:
        print(f"\nSetup required: {e}")
        return
    
    # Load test slice
    test_slice = Path("CTdata/CTdcm/20050.dcm")
    
    if not test_slice.exists():
        print(f"Test file not found: {test_slice}")
        return
    
    print(f"\nLoading CT slice: {test_slice.name}")
    image_rgb, image_gray = detector.load_ct_slice(test_slice)
    
    print(f"Image shape: {image_rgb.shape}")
    print(f"Device: {detector.device}")
    
    # Test with center point (example)
    h, w = image_gray.shape
    center_point = np.array([[w//2, h//2]])
    
    print(f"\nTesting with center point: {center_point[0]}")
    print("Running inference...")
    
    masks, scores, logits = detector.predict_with_points(
        image_rgb,
        center_point,
        labels=np.array([1])
    )
    
    print(f"\nResults:")
    print(f"  Masks generated: {len(masks)}")
    print(f"  Scores: {scores}")
    print(f"  Best score: {scores.max():.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(image_gray, cmap='gray')
    axes[0].scatter(center_point[0,0], center_point[0,1], c='red', s=100, marker='*')
    axes[0].set_title('Original + Input Point')
    axes[0].axis('off')
    
    # Three masks
    for idx, (mask, score) in enumerate(zip(masks, scores)):
        ax = axes[idx + 1]
        ax.imshow(image_gray, cmap='gray')
        ax.imshow(mask, alpha=0.5, cmap='Reds')
        ax.set_title(f'Mask {idx+1} (Score: {score:.3f})')
        ax.axis('off')
    
    plt.tight_layout()
    
    output_file = "CTdata/visualizations/medsam_test_result.png"
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    
    plt.close()
    
    print("\n" + "="*70)
    print("MedSAM test complete!")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Review visualization")
    print("2. Create annotation interface for point selection")
    print("3. Annotate 5-10 tumor slices")
    print("4. Fine-tune on annotated data")

if __name__ == '__main__':
    test_medsam()
