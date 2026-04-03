"""
Advanced Image Enhancement Module
Provides upscaling and CLAHE preprocessing for cell images
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ImageEnhancer:
    """Image enhancement utilities for preprocessing"""
    
    def __init__(self):
        pass
    
    def upscale_image(
        self, 
        image: np.ndarray, 
        scale_factor: float = 2.0,
        interpolation: str = 'lanczos'
    ) -> np.ndarray:
        """
        Upscale image using various interpolation methods
        
        Args:
            image: Input image (numpy array)
            scale_factor: Scaling factor (1.5, 2.0, 2.5, 3.0)
            interpolation: 'cubic', 'lanczos', 'linear'
            
        Returns:
            Upscaled image
        """
        if scale_factor == 1.0:
            return image
        
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Select interpolation method
        interp_methods = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        interp = interp_methods.get(interpolation, cv2.INTER_LANCZOS4)
        
        # Resize image
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=interp)
        
        return upscaled
    
    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting (1.0-4.0)
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE-processed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # For color images, apply to each channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def denoise_image(
        self,
        image: np.ndarray,
        method: str = 'bilateral',
        strength: int = 10
    ) -> np.ndarray:
        """
        Apply denoising to image
        
        Args:
            image: Input image
            method: 'bilateral', 'gaussian', 'median'
            strength: Denoising strength
            
        Returns:
            Denoised image
        """
        if method == 'bilateral':
            denoised = cv2.bilateralFilter(image, strength, strength*2, strength/2)
        elif method == 'gaussian':
            kernel_size = strength if strength % 2 == 1 else strength + 1
            denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == 'median':
            kernel_size = strength if strength % 2 == 1 else strength + 1
            denoised = cv2.medianBlur(image, kernel_size)
        else:
            denoised = image
        
        return denoised
    
    def preprocess_pipeline(
        self,
        image: np.ndarray,
        upscale: bool = False,
        scale_factor: float = 2.0,
        clahe: bool = False,
        clip_limit: float = 2.0,
        denoise: bool = False,
        denoise_strength: int = 5
    ) -> Tuple[np.ndarray, dict]:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image
            upscale: Apply upscaling
            scale_factor: Upscaling factor
            clahe: Apply CLAHE
            clip_limit: CLAHE clip limit
            denoise: Apply denoising
            denoise_strength: Denoising strength
            
        Returns:
            Tuple of (processed image, metadata dict)
        """
        processed = image.copy()
        metadata = {
            'original_shape': image.shape,
            'steps_applied': []
        }
        
        # Step 1: Denoise (if enabled)
        if denoise:
            processed = self.denoise_image(processed, strength=denoise_strength)
            metadata['steps_applied'].append(f'Denoise (strength={denoise_strength})')
        
        # Step 2: Upscale (if enabled)
        if upscale and scale_factor > 1.0:
            processed = self.upscale_image(processed, scale_factor=scale_factor)
            metadata['steps_applied'].append(f'Upscale ({scale_factor}x)')
            metadata['upscaled_shape'] = processed.shape
        
        # Step 3: CLAHE (if enabled)
        if clahe:
            processed = self.apply_clahe(processed, clip_limit=clip_limit)
            metadata['steps_applied'].append(f'CLAHE (clip={clip_limit})')
        
        metadata['final_shape'] = processed.shape
        
        return processed, metadata
    
    def get_comparison_images(
        self,
        original: np.ndarray,
        processed: np.ndarray
    ) -> np.ndarray:
        """
        Create side-by-side comparison of original and processed images
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Concatenated comparison image
        """
        # Resize images to same size for comparison
        if original.shape != processed.shape:
            processed_resized = cv2.resize(processed, (original.shape[1], original.shape[0]))
        else:
            processed_resized = processed
        
        # Concatenate horizontally
        comparison = np.hstack([original, processed_resized])
        
        return comparison
