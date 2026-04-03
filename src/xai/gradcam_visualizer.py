"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ADDS
Visualizes which image regions are important for CNN-based predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_init import *

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image


class GradCAM:
    """
    Grad-CAM visuali for Cellpose and other CNNs
    
    Shows heatmap of important image regions for segmentation/classification
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model (e.g., Cellpose network)
            target_layer: Name of layer to visualize (e.g., 'layer4', 'conv_final')
        """
        self.model = model
        self.model.eval()
        
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks(target_layer)
    
    def _register_hooks(self, layer_name: str):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.target_layer = module
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
        
        if self.target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model")
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = highest predicted class)
            
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (H, W, 3) uint8
            heatmap: Grad-CAM heatmap (H', W') float [0, 1]
            alpha: Transparency of heatmap
            colormap: OpenCV colormap
            
        Returns:
            Overlaid image (H, W, 3) uint8
        """
        # Resize heatmap to match image
        h, w = image.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to RGB
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Overlay
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid


class CellposeGradCAM:
    """
    Specialized Grad-CAM for Cellpose segmentation
    
    Shows which image regions influenced cell detection
    """
    
    def __init__(self, cellpose_model):
        """
        Initialize Cellpose Grad-CAM
        
        Args:
            cellpose_model: Cellpose model instance
        """
        self.cellpose_model = cellpose_model
    
    def visualize_segmentation_importance(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        save_path: Optional[Path] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize which image regions drove segmentation
        
        Args:
            image: Original image (H, W) or (H, W, 3)
            masks: Segmentation masks (H, W)
            save_path: Optional path to save figure
            
        Returns:
            (Figure, importance_map)
        """
        # Compute gradient-based importance
        # For Cellpose, we approximate using mask boundaries
        importance_map = self._compute_boundary_importance(masks)
        
        # Create visualization
        fig = self._create_visualization(image, masks, importance_map)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, importance_map
    
    def _compute_boundary_importance(self, masks: np.ndarray) -> np.ndarray:
        """
        Compute importance based on cell boundaries
        
        High importance = near cell boundaries (where decision was made)
        """
        # Detect edges using Sobel
        if masks.dtype != np.float32:
            masks_float = masks.astype(np.float32)
        else:
            masks_float = masks
        
        # Sobel gradients
        grad_x = cv2.Sobel(masks_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(masks_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        importance = (gradient_mag - gradient_mag.min()) / (gradient_mag.max() - gradient_mag.min() + 1e-8)
        
        # Apply Gaussian smoothing
        importance = cv2.GaussianBlur(importance, (15, 15), 0)
        
        return importance
    
    def _create_visualization(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        importance_map: np.ndarray
    ) -> plt.Figure:
        """Create 4-panel visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Segmentation masks
        axes[0, 1].imshow(masks, cmap='nipy_spectral')
        axes[0, 1].set_title('Segmentation Masks', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Importance heatmap
        im = axes[1, 0].imshow(importance_map, cmap='hot')
        axes[1, 0].set_title('Importance Heatmap', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # Overlay
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = (image * 255).astype(np.uint8)
        
        importance_uint8 = (importance_map * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(importance_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_colored, 0.4, 0)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay: Important Regions', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.suptitle('Grad-CAM: Segmentation Importance Analysis', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig


def demo_gradcam():
    """Demonstration with synthetic data"""
    
    # Create synthetic image with cells
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some circular "cells"
    centers = [(64, 64), (128, 128), (192, 64), (64, 192)]
    radii = [20, 25, 18, 22]
    
    for center, radius in zip(centers, radii):
        cv2.circle(image, center, radius, 200, -1)
    
    # Add noise
    noise = np.random.normal(0, 20, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Create synthetic masks
    masks = np.zeros_like(image)
    for i, (center, radius) in enumerate(zip(centers, radii), 1):
        cv2.circle(masks, center, radius, i, -1)
    
    # Initialize Cellpose Grad-CAM
    cellpose_gradcam = CellposeGradCAM(None)
    
    # Visualize
    fig, importance = cellpose_gradcam.visualize_segmentation_importance(
        image,
        masks,
        save_path=Path('gradcam_demo.png')
    )
    
    print("=" * 70)
    print("Grad-CAM Demo Complete")
    print("=" * 70)
    print(f"Generated importance map: {importance.shape}")
    print(f"Importance range: [{importance.min():.3f}, {importance.max():.3f}]")
    print(f"Saved visualization to: gradcam_demo.png")
    print("=" * 70)
    
    plt.show()


if __name__ == '__main__':
    demo_gradcam()
