"""
Image Quality Enhancer
Advanced preprocessing for noise reduction, focus enhancement, and contrast optimization
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from scipy import ndimage
from skimage import filters, exposure


class ImageQualityEnhancer:
    """Advanced image quality enhancement utilities"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def denoise_image(
        self,
        image: np.ndarray,
        method: str = 'bilateral',
        **kwargs
    ) -> np.ndarray:
        """
        Remove noise from image
        
        Args:
            image: Input image (grayscale or RGB)
            method: Denoising method
                - 'bilateral': Edge-preserving filter (recommended)
                - 'nlm': Non-local means (slow but powerful)
                - 'gaussian': Fast but blurs edges
            **kwargs: Method-specific parameters
        
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # RGB image
            channels = cv2.split(image)
            denoised_channels = [
                self._denoise_single_channel(ch, method, **kwargs)
                for ch in channels
            ]
            return cv2.merge(denoised_channels)
        else:
            # Grayscale
            return self._denoise_single_channel(image, method, **kwargs)
    
    def _denoise_single_channel(
        self,
        channel: np.ndarray,
        method: str,
        **kwargs
    ) -> np.ndarray:
        """Denoise single channel"""
        if method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            return cv2.bilateralFilter(
                channel.astype(np.uint8), d, sigma_color, sigma_space
            )
        
        elif method == 'nlm':
            h = kwargs.get('h', 10)
            template_window_size = kwargs.get('template_window_size', 7)
            search_window_size = kwargs.get('search_window_size', 21)
            return cv2.fastNlMeansDenoising(
                channel.astype(np.uint8),
                None,
                h,
                template_window_size,
                search_window_size
            )
        
        elif method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            return ndimage.gaussian_filter(channel, sigma=sigma)
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def enhance_focus(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Enhance image focus using unsharp masking
        
        Args:
            image: Input image
            kernel_size: Gaussian kernel size (odd number)
            strength: Sharpening strength (0.5-2.0)
        
        Returns:
            Sharpened image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian blurred version
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Unsharp mask: original + strength * (original - blurred)
        sharpened = cv2.addWeighted(
            image, 1.0 + strength,
            blurred, -strength,
            0
        )
        
        # Clip values
        return np.clip(sharpened, 0, 255).astype(image.dtype)
    
    def optimize_contrast(
        self,
        image: np.ndarray,
        method: str = 'clahe',
        **kwargs
    ) -> np.ndarray:
        """
        Optimize image contrast
        
        Args:
            image: Input image
            method: Contrast enhancement method
                - 'clahe': Contrast Limited Adaptive Histogram Equalization
                - 'adaptive': Adaptive histogram equalization
                - 'gamma': Gamma correction
            **kwargs: Method-specific parameters
        
        Returns:
            Contrast-enhanced image
        """
        if method == 'clahe':
            clip_limit = kwargs.get('clip_limit', 2.0)
            tile_grid_size = kwargs.get('tile_grid_size', (8, 8))
            
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(
                    clipLimit=clip_limit,
                    tileGridSize=tile_grid_size
                )
                l_clahe = clahe.apply(l)
                
                # Merge back
                lab_clahe = cv2.merge([l_clahe, a, b])
                return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(
                    clipLimit=clip_limit,
                    tileGridSize=tile_grid_size
                )
                return clahe.apply(image.astype(np.uint8))
        
        elif method == 'adaptive':
            return exposure.equalize_adapthist(
                image,
                kernel_size=kwargs.get('kernel_size', None),
                clip_limit=kwargs.get('clip_limit', 0.03)
            )
        
        elif method == 'gamma':
            gamma = kwargs.get('gamma', 1.0)
            return exposure.adjust_gamma(image, gamma)
        
        else:
            raise ValueError(f"Unknown contrast method: {method}")
    
    def compute_quality_metrics(
        self,
        image: np.ndarray,
        original: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive quality metrics
        
        Args:
            image: Image to analyze
            original: Optional original image for comparison
        
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Focus score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['focus_score'] = laplacian.var()
        
        # Estimate SNR
        metrics['snr'] = self._estimate_snr(gray)
        
        # Brightness
        metrics['brightness'] = np.mean(gray)
        
        # RMS contrast
        metrics['contrast'] = np.std(gray)
        
        # Sharpness (gradient magnitude)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        metrics['sharpness'] = np.mean(np.sqrt(gx**2 + gy**2))
        
        # Illumination uniformity (std of local means)
        kernel_size = min(gray.shape[0], gray.shape[1]) // 10
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        local_means = cv2.blur(gray, (kernel_size, kernel_size))
        metrics['uniformity'] = 1.0 / (1.0 + np.std(local_means))
        
        # If original provided, compute improvement
        if original is not None:
            orig_metrics = self.compute_quality_metrics(original)
            metrics['snr_improvement'] = (
                (metrics['snr'] - orig_metrics['snr']) / orig_metrics['snr'] * 100
            )
            metrics['focus_improvement'] = (
                (metrics['focus_score'] - orig_metrics['focus_score']) /
                orig_metrics['focus_score'] * 100
            )
        
        self.quality_metrics = metrics
        return metrics
    
    def _estimate_snr(
        self,
        image: np.ndarray,
        roi_size: int = 50
    ) -> float:
        """
        Estimate signal-to-noise ratio
        
        Args:
            image: Grayscale image
            roi_size: Size of ROI for measurement
        
        Returns:
            SNR value
        """
        h, w = image.shape
        
        # Sample signal region (center)
        center_y, center_x = h // 2, w // 2
        signal_roi = image[
            center_y - roi_size//2:center_y + roi_size//2,
            center_x - roi_size//2:center_x + roi_size//2
        ]
        signal = np.mean(signal_roi)
        
        # Sample noise region (corners)
        corners = [
            image[0:roi_size, 0:roi_size],
            image[0:roi_size, -roi_size:],
            image[-roi_size:, 0:roi_size],
            image[-roi_size:, -roi_size:]
        ]
        
        # Use corner with lowest variance as background
        noise_stds = [np.std(corner) for corner in corners]
        noise = min(noise_stds)
        
        if noise == 0:
            return float('inf')
        
        return signal / noise if noise > 0 else 0.0
    
    def get_quality_grade(self, metric_value: float, metric_name: str) -> str:
        """
        Convert metric value to quality grade
        
        Args:
            metric_value: Metric value
            metric_name: Name of metric
        
        Returns:
            Grade: 'Excellent', 'Good', 'Acceptable', 'Poor'
        """
        thresholds = {
            'focus_score': {
                'Excellent': 1000,
                'Good': 500,
                'Acceptable': 100,
                'Poor': 0
            },
            'snr': {
                'Excellent': 20,
                'Good': 15,
                'Acceptable': 10,
                'Poor': 0
            },
            'sharpness': {
                'Excellent': 50,
                'Good': 30,
                'Acceptable': 15,
                'Poor': 0
            },
            'uniformity': {
                'Excellent': 0.9,
                'Good': 0.7,
                'Acceptable': 0.5,
                'Poor': 0
            }
        }
        
        if metric_name not in thresholds:
            return 'Unknown'
        
        thresh = thresholds[metric_name]
        if metric_value >= thresh['Excellent']:
            return 'Excellent'
        elif metric_value >= thresh['Good']:
            return 'Good'
        elif metric_value >= thresh['Acceptable']:
            return 'Acceptable'
        else:
            return 'Poor'
    
    def apply_full_enhancement(
        self,
        image: np.ndarray,
        denoise: bool = True,
        sharpen: bool = True,
        enhance_contrast: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply complete enhancement pipeline
        
        Args:
            image: Input image
            denoise: Apply denoising
            sharpen: Apply sharpening
            enhance_contrast: Apply contrast enhancement
            **kwargs: Parameters for each method
        
        Returns:
            (enhanced_image, metrics_dict)
        """
        result = image.copy()
        original_metrics = self.compute_quality_metrics(image)
        
        if denoise:
            result = self.denoise_image(
                result,
                method=kwargs.get('denoise_method', 'bilateral')
            )
        
        if enhance_contrast:
            result = self.optimize_contrast(
                result,
                method=kwargs.get('contrast_method', 'clahe')
            )
        
        if sharpen:
            result = self.enhance_focus(
                result,
                kernel_size=kwargs.get('sharpen_kernel', 5),
                strength=kwargs.get('sharpen_strength', 1.0)
            )
        
        enhanced_metrics = self.compute_quality_metrics(result, original=image)
        
        return result, {
            'original': original_metrics,
            'enhanced': enhanced_metrics,
            'improvements': {
                'snr': enhanced_metrics.get('snr_improvement', 0),
                'focus': enhanced_metrics.get('focus_improvement', 0)
            }
        }
