"""
Cellpose-based Image Processing Pipeline for ADDS
Provides comprehensive cell segmentation and analysis
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image
import cv2
from cellpose import models, core
from skimage import measure, morphology
from skimage.measure import regionprops
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils import get_logger, config

logger = get_logger(__name__)


class CellposeProcessor:
    """
    Image segmentation and feature extraction using Cellpose
    """
    
    def __init__(
        self,
        model_type: str = None,
        gpu: bool = None,
        batch_size: int = None
    ):
        """
        Initialize Cellpose processor
        
        Args:
            model_type: Cellpose model ('cyto', 'cyto2', 'nuclei', or path to custom model)
            gpu: Whether to use GPU
            batch_size: Batch size for processing
        """
        # Load from config if not specified
        if model_type is None:
            model_type = config.get('cellpose.model_type', 'cyto2')
        if gpu is None:
            # Get default from config (which may come from .env)
            config_gpu = config.get('cellpose.gpu', True)
            cuda_available = core.use_gpu()
            # Both must be True to use GPU
            gpu = config_gpu and cuda_available
            logger.info(f"GPU setting from config: {config_gpu}, CUDA available: {cuda_available}, Using GPU: {gpu}")
        else:
            # Explicitly provided by UI or caller
            logger.info(f"GPU setting explicitly provided: {gpu}")
            # Still check CUDA availability
            if gpu and not core.use_gpu():
                logger.warning("GPU requested but CUDA not available, falling back to CPU")
                gpu = False
        
        if batch_size is None:
            batch_size = config.get('cellpose.batch_size', 8)
        
        self.model_type = model_type
        self.gpu = gpu
        self.batch_size = batch_size
        
        # Initialize Cellpose model
        if gpu:
            import torch
            if torch.cuda.is_available():
                device_idx = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_idx)
                logger.info(f"Initializing Cellpose with GPU: {device_name} (cuda:{device_idx})")
            else:
                logger.warning("GPU requested but CUDA not available, falling back to CPU")
                gpu = False
        
        if not gpu:
            logger.info(f"Initializing Cellpose model: {model_type} (CPU mode)")
        
        self.model = models.CellposeModel(gpu=gpu, model_type=model_type)

        
        # Get processing parameters from config
        self.diameter = config.get('cellpose.diameter', None)
        self.flow_threshold = config.get('cellpose.flow_threshold', 0.4)
        self.cellprob_threshold = config.get('cellpose.cellprob_threshold', 0.0)
        self.channels = config.get('cellpose.channels', [0, 0])
        
        logger.info("[OK] Cellpose processor initialized")
    
    def segment_image(
        self,
        image: Union[np.ndarray, str, Path],
        diameter: Optional[float] = None,
        channels: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Segment cells in a single image
        
        Args:
            image: Input image (numpy array or file path)
            diameter: Expected cell diameter (None for auto)
            channels: [cytoplasm, nucleus] channels (0=grayscale, 1=red, 2=green, 3=blue)
            **kwargs: Additional Cellpose parameters
        
        Returns:
            masks: Segmentation masks (HxW array, each cell has unique integer)
            flows: Flow field from Cellpose
            metadata: Dictionary with segmentation info
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        
        # Use defaults if not specified
        if diameter is None:
            diameter = self.diameter
        if channels is None:
            channels = self.channels
        
        # Run Cellpose segmentation
        logger.debug(f"Segmenting image of shape {image.shape}")
        
        # Cellpose 4.0+ returns (masks, flows, styles) - no diams
        masks, flows, styles = self.model.eval(
            image,
            diameter=diameter,
            channels=channels,
            flow_threshold=kwargs.get('flow_threshold', self.flow_threshold),
            cellprob_threshold=kwargs.get('cellprob_threshold', self.cellprob_threshold),
            batch_size=self.batch_size
        )
        
        num_cells = masks.max()
        logger.info(f"[OK] Segmented {num_cells} cells")
        
        metadata = {
            'num_cells': int(num_cells),
            'estimated_diameter': float(diameter) if diameter else None,  # Use input diameter
            'image_shape': image.shape,
            'model_type': self.model_type
        }
        
        return masks, flows, metadata
    
    def segment_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        show_progress: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Segment multiple images in batch
        
        Args:
            images: List of images or image paths
            show_progress: Show progress bar
        
        Returns:
            List of (masks, flows, metadata) tuples
        """
        results = []
        
        iterator = tqdm(images, desc="Segmenting images") if show_progress else images
        
        for img in iterator:
            try:
                masks, flows, metadata = self.segment_image(img)
                results.append((masks, flows, metadata))
            except Exception as e:
                logger.error(f"Error segmenting image: {e}")
                results.append((None, None, {'error': str(e)}))
        
        return results
    
    def extract_morphological_features(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        include_intensity: bool = True
    ) -> pd.DataFrame:
        """
        Extract morphological features from segmented cells
        
        Args:
            image: Original image
            masks: Segmentation masks
            include_intensity: Whether to include intensity features
        
        Returns:
            DataFrame with features for each cell
        """
        features_list = []
        
        # Get region properties
        props = regionprops(masks, intensity_image=image if include_intensity else None)
        
        for prop in props:
            features = {
                'cell_id': prop.label,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'circularity': (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'orientation': prop.orientation,
                'centroid_x': prop.centroid[1],
                'centroid_y': prop.centroid[0],
            }
            
            if include_intensity and image is not None:
                # Handle multi-channel images - convert to scalar if needed
                mean_int = prop.mean_intensity
                max_int = prop.max_intensity
                min_int = prop.min_intensity
                
                # If values are arrays (multi-channel), take the mean
                if hasattr(mean_int, '__len__'):
                    mean_int = float(np.mean(mean_int))
                if hasattr(max_int, '__len__'):
                    max_int = float(np.mean(max_int))
                if hasattr(min_int, '__len__'):
                    min_int = float(np.mean(min_int))
                
                features.update({
                    'mean_intensity': mean_int,
                    'max_intensity': max_int,
                    'min_intensity': min_int,
                })
            
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        logger.info(f"[OK] Extracted features for {len(df)} cells")
        
        return df
    
    def calculate_cell_viability_metrics(
        self,
        features_df: pd.DataFrame,
        total_area: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate cell viability and health metrics
        
        Args:
            features_df: DataFrame with cell features
            total_area: Total image area (for confluency calculation)
        
        Returns:
            Dictionary of aggregate metrics
        """
        if len(features_df) == 0:
            return {'cell_count': 0}
        
        metrics = {
            'cell_count': len(features_df),
            'mean_area': features_df['area'].mean(),
            'std_area': features_df['area'].std(),
            'mean_circularity': features_df['circularity'].mean(),
            'mean_eccentricity': features_df['eccentricity'].mean(),
        }
        
        if 'mean_intensity' in features_df.columns:
            metrics.update({
                'mean_intensity': features_df['mean_intensity'].mean(),
                'std_intensity': features_df['mean_intensity'].std(),
            })
        
        if total_area:
            metrics['confluency'] = features_df['area'].sum() / total_area
        
        # Estimate health based on morphology
        # Healthy cells are typically more circular and uniform
        health_score = (
            features_df['circularity'].mean() * 0.5 +
            (1 - features_df['area'].std() / (features_df['area'].mean() + 1e-6)) * 0.5
        )
        metrics['estimated_health_score'] = np.clip(health_score, 0, 1)
        
        return metrics
    
    def process_and_save(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        save_masks: bool = True,
        save_features: bool = True,
        save_visualization: bool = True,
        deep_analysis: bool = True,  # 심층 분석 활성화
        save_metadata: bool = True,
        save_report: bool = True
    ) -> Dict:
        """
        Complete pipeline: segment, extract features, quality assessment, and save results
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            save_masks: Save segmentation masks
            save_features: Save feature CSV
            save_visualization: Save visualization overlay
            deep_analysis: Enable deep analysis (metadata, quality, report)
            save_metadata: Save metadata JSON
            save_report: Save PDF report
        
        Returns:
            Dictionary with results and file paths
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and segment
        image = self._load_image(image_path)
        masks, flows, seg_metadata = self.segment_image(image)
        
        # Extract features
        features_df = self.extract_morphological_features(image, masks)
        metrics = self.calculate_cell_viability_metrics(
            features_df,
            total_area=image.shape[0] * image.shape[1]
        )
        
        base_name = image_path.stem
        
        # Convert features_df to list of dicts for JSON serialization
        cell_data = features_df.to_dict('records')
        
        results = {
            'image_path': str(image_path),
            'segmentation_metadata': seg_metadata,
            'metrics': metrics,
            'masks': masks,  # Add masks for interactive viewer
            'image': image,  # Add image for interactive viewer
            'cell_data': cell_data  # Add cell data for interactive viewer
        }
        
        # Deep Analysis
        if deep_analysis:
            logger.info("Performing deep analysis...")
            
            # 메타데이터 추출
            from preprocessing.metadata_extractor import ImageMetadataExtractor
            metadata_extractor = ImageMetadataExtractor()
            
            full_metadata = metadata_extractor.create_metadata(
                image_path,
                image,
                analysis_results=metrics
            )
            results['full_metadata'] = full_metadata
            
            # 품질 평가
            from preprocessing.quality_assessor import ImageQualityAssessor
            quality_assessor = ImageQualityAssessor()
            
            quality_assessment = quality_assessor.assess_overall_quality(image)
            results['quality_assessment'] = quality_assessment
            
            logger.info(f"Quality: {quality_assessment['overall_quality']} "
                       f"(Score: {quality_assessment['overall_score']:.2f})")
            
            # 메타데이터 저장
            if save_metadata:
                metadata_path = output_dir / f"{base_name}_metadata.json"
                metadata_extractor.save_metadata(full_metadata, metadata_path, format='json')
                results['metadata_path'] = str(metadata_path)
            
            # AI Platform Comparison (Optional)
            comparison_results = None
            hyperparameter_recs = None
            
            # Check if AI comparison is requested (via environment variable or parameter)
            enable_ai_comparison = os.getenv("ENABLE_AI_COMPARISON", "false").lower() == "true"
            
            if enable_ai_comparison:
                try:
                    from utils.ai_platform_comparator import AIPlatformComparator
                    
                    comparator = AIPlatformComparator()
                    if comparator.is_available():
                        logger.info("Running AI platform comparison...")
                        
                        # Prepare Cellpose results for comparison
                        cellpose_data = {
                            'num_cells': seg_metadata.get('num_cells', 0),
                            'mean_area': metrics.get('mean_area', 0),
                            'mean_circularity': metrics.get('mean_circularity', 0)
                        }
                        
                        # GPT-4V analysis
                        gpt4v_results = comparator.analyze_with_gpt4v(
                            image_path,
                            cellpose_data
                        )
                        
                        # Compare results
                        comparison_results = comparator.compare_results(
                            cellpose_data,
                            gpt4v_results
                        )
                        
                        # Generate hyperparameter recommendations
                        hyperparameter_recs = comparator.generate_hyperparameter_recommendations(
                            comparison_results,
                            {
                                'diameter': self.diameter,
                                'flow_threshold': self.flow_threshold,
                                'cellprob_threshold': self.cellprob_threshold
                            }
                        )
                        
                        results['ai_comparison'] = comparison_results
                        results['hyperparameter_recommendations'] = hyperparameter_recs
                        logger.info(f"AI comparison complete: {comparison_results.get('agreement_level', 'N/A')} agreement")
                    else:
                        logger.warning("AI comparison skipped: OpenAI API not available")
                        
                except Exception as e:
                    logger.warning(f"AI comparison failed: {e}")
            
            # 원본 이미지 저장 (visualizations보다 먼저)
            original_path = output_dir / f"{base_name}_original.png"
            Image.fromarray(image).save(original_path)
            results['original_path'] = str(original_path)
            
            # Visualization 생성 (PDF보다 먼저!)
            if save_visualization:
                logger.info("Generating all visualization images...")
                try:
                    viz_paths = self._generate_pipeline_visualizations(
                        image, masks, base_name, output_dir
                    )
                    results.update(viz_paths)
                    
                    # 기본 overlay도 저장
                    overlay = self._create_overlay(image, masks)
                    overlay_path = output_dir / f"{base_name}_overlay.png"
                    Image.fromarray(overlay).save(overlay_path)
                    results['visualization_path'] = str(overlay_path)
                    results['overlay_path'] = str(overlay_path)
                    
                    logger.info(f"Generated {len(viz_paths) + 1} visualization images")
                except Exception as e:
                    logger.warning(f"Visualization generation failed: {e}")
                    # Fallback
                    vis_path = output_dir / f"{base_name}_overlay.png"
                    self._save_visualization(image, masks, vis_path)
                    results['visualization_path'] = str(vis_path)
            
            # PDF 리포트 생성 (이미지 생성 후!)
            if save_report:
                from preprocessing.report_generator import AnalysisReportGenerator
                report_generator = AnalysisReportGenerator()
                
                report_path = output_dir / f"{base_name}_report.pdf"
                
                # Collect visualization paths for the report
                viz_paths_for_report = {}
                if 'original_path' in results:
                    viz_paths_for_report['original_path'] = results['original_path']
                if 'preprocessed_path' in results:
                    viz_paths_for_report['preprocessed_path'] = results['preprocessed_path']
                if 'colored_mask_path' in results:
                    viz_paths_for_report['colored_mask_path'] = results['colored_mask_path']
                if 'overlay_path' in results:
                    viz_paths_for_report['overlay_path'] = results['overlay_path']
                if 'contour_path' in results:
                    viz_paths_for_report['contour_path'] = results['contour_path']
                if 'heatmap_path' in results:
                    viz_paths_for_report['heatmap_path'] = results['heatmap_path']
                
                # Pass comparison results to report if available
                report_kwargs = {
                    'visualization_paths': viz_paths_for_report
                }
                
                if comparison_results:
                    report_kwargs['comparison_results'] = comparison_results
                if hyperparameter_recs:
                    report_kwargs['hyperparameter_recommendations'] = hyperparameter_recs
                
                report_generator.generate_report(
                    report_path,
                    image_path,
                    full_metadata,
                    quality_assessment,
                    {
                        'num_cells': seg_metadata.get('num_cells', 0),
                        'cell_density': metrics.get('cell_count', 0) / (image.shape[0] * image.shape[1] / 1000000) if metrics.get('cell_count', 0) > 0 else 0,
                        'mean_cell_area': metrics.get('mean_cell_area', 0),
                        'viability_score': metrics.get('estimated_health_score', 0)
                    },
                    **report_kwargs
                )
                results['report_path'] = str(report_path)
                logger.info(f"Report generated: {report_path.name}")
        
        # 기존 저장 기능
        if save_masks:
            masks_path = output_dir / f"{base_name}_masks.npy"
            np.save(masks_path, masks)
            results['masks_path'] = str(masks_path)
        
        if save_features:
            features_path = output_dir / f"{base_name}_features.csv"
            features_df.to_csv(features_path, index=False)
            results['features_path'] = str(features_path)
        
        # Visualization 및 원본 이미지는 이미 위에서 save_report 전에 생성됨
        # (이 섹션은 제거됨 - 중복 방지)
        
        logger.info(f"[COMPLETE] Processed: {image_path.name}")
        return results
    
    def _generate_pipeline_visualizations(self, image, masks, base_name, output_dir):
        """Generate all pipeline visualization images"""
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        viz_paths = {}
        
        # 1. Preprocessed
        preprocessed = self._create_preprocessed_image(image)
        p_path = output_dir / f'{base_name}_preprocessed.png'
        Image.fromarray(preprocessed).save(p_path)
        viz_paths['preprocessed_path'] = str(p_path)
        
        # 2. Colored mask
        colored = self._create_colored_mask(masks)
        c_path = output_dir / f'{base_name}_colored_mask.png'
        Image.fromarray(colored).save(c_path)
        viz_paths['colored_mask_path'] = str(c_path)
        
        # 3. Contours
        contour = self._create_contour_image(image, masks)
        ct_path = output_dir / f'{base_name}_contours.png'
        Image.fromarray(contour).save(ct_path)
        viz_paths['contour_path'] = str(ct_path)
        
        # 4. Heatmap
        heatmap = self._create_size_heatmap(image, masks)
        h_path = output_dir / f'{base_name}_heatmap.png'
        Image.fromarray(heatmap).save(h_path)
        viz_paths['heatmap_path'] = str(h_path)
        
        return viz_paths
    
    def _create_preprocessed_image(self, image):
        """Apply CLAHE preprocessing"""
        import cv2
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    def _create_colored_mask(self, masks):
        """Create colorful mask visualization"""
        import matplotlib.pyplot as plt
        from matplotlib import cm
        if masks.max() == 0:
            return np.ones((*masks.shape, 3), dtype=np.uint8) * 255
        normalized = masks.astype(float) / masks.max()
        colormap = cm.get_cmap('nipy_spectral')
        colored = colormap(normalized)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        colored_rgb[masks == 0] = [255, 255, 255]
        return colored_rgb
    
    def _create_contour_image(self, image, masks):
        """Create image with contours"""
        import cv2
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        for cell_id in range(1, int(masks.max()) + 1):
            cell_mask = (masks == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result
    
    def _create_size_heatmap(self, image, masks):
        """Create size-based heatmap"""
        import cv2
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from skimage.measure import regionprops
        props = regionprops(masks)
        if not props:
            return image
        areas = {p.label: p.area for p in props}
        min_a, max_a = min(areas.values()), max(areas.values())
        rng = max_a - min_a if max_a > min_a else 1
        hm_arr = np.zeros(masks.shape, dtype=float)
        for cid, area in areas.items():
            hm_arr[masks == cid] = (area - min_a) / rng
        cmap = cm.get_cmap('hot')
        colored = cmap(hm_arr)
        hm_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape)==2 else image
        return cv2.addWeighted(img_rgb, 0.4, hm_rgb, 0.6, 0)

    def _create_overlay(self, image, masks):
        """Create overlay of original image with colored masks"""
        import cv2
        
        # 원본 이미지 RGB 변환
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image.copy()
        
        # 컬러 마스크 생성
        colored_mask = self._create_colored_mask(masks)
        
        # 블렌딩 (원본 60% + 마스크 40%)
        overlay = cv2.addWeighted(img_rgb, 0.6, colored_mask, 0.4, 0)
        
        return overlay
    
    def _save_visualization(self, image, masks, output_path):
        """Save visualization with masks overlay"""
        overlay = self._create_overlay(image, masks)
        Image.fromarray(overlay).save(output_path)

    
    def _load_image(self, path: Union[str, Path]) -> np.ndarray:
        """Load image from file"""
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not load image: {path}")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _save_visualization(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        output_path: Union[str, Path],
        alpha: float = 0.5
    ):
        """Save visualization of segmentation overlay"""
        from cellpose import plot
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot image with masks overlay
        img_show = plot.mask_overlay(image, masks)
        ax.imshow(img_show)
        ax.set_title(f"Segmented Cells: {masks.max()}")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# Empty __init__ files for packages
def create_package_init_files():
    """Create __init__.py files for all packages"""
    packages = ['data', 'models', 'preprocessing', 'evaluation', 'ui']
    src_dir = Path(__file__).parent.parent
    
    for pkg in packages:
        init_file = src_dir / pkg / '__init__.py'
        if not init_file.exists():
            init_file.write_text(f'"""{pkg.capitalize()} package"""')
    
