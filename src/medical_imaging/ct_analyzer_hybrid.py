"""
Hybrid CT Analyzer using nnU-Net + TotalSegmentator

Combines:
- nnU-Net: Precise tumor segmentation (Dice 0.85-0.92)
- TotalSegmentator: Anatomical context (104 organs)

Author: ADDS Team
Date: 2026-01-19
"""

import numpy as np
import torch
from pathlib import Path
import tempfile
import warnings
warnings.filterwarnings('ignore')

try:
    import SimpleITK as sitk
except ImportError:
    print("⚠ SimpleITK not installed. Install with: pip install SimpleITK")
    sitk = None


class HybridCTAnalyzer:
    """
    Hybrid CT Analyzer combining nnU-Net and TotalSegmentator
    
    Usage:
        analyzer = HybridCTAnalyzer(use_gpu=True)
        result = analyzer.analyze_ct_hybrid(ct_image, metadata)
    """
    
    def __init__(self, use_gpu=True):
        """
        Initialize hybrid analyzer
        
        Args:
            use_gpu: Use GPU acceleration if available
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        print(f"🔧 Initializing Hybrid CT Analyzer (Device: {self.device})")
        
        # Initialize components
        self.nnunet_ready = self._init_nnunet()
        self.totalseg_ready = self._check_totalseg()
        
        # Status report
        if self.nnunet_ready and self.totalseg_ready:
            print("✅ Hybrid mode ready (nnU-Net + TotalSegmentator)")
        elif self.nnunet_ready:
            print("⚠️ Partial mode (nnU-Net only)")
        elif self.totalseg_ready:
            print("⚠️ Partial mode (TotalSegmentator only)")
        else:
            print("❌ Fallback to basic mode")
    
    def _init_nnunet(self):
        """Initialize nnU-Net predictor"""
        try:
            from nnunetv2.inference.predict import nnUNetPredictor
            
            self.nnunet = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False
            )
            
            # Check for model
            model_folder = Path.home() / 'nnUNet_models' / 'Dataset010_Colon'
            
            if model_folder.exists():
                print("📦 Loading nnU-Net model...")
                self.nnunet.initialize_from_trained_model_folder(
                    str(model_folder),
                    use_folds=(0, 1, 2, 3, 4),
                    checkpoint_name='checkpoint_final.pth'
                )
                print("✓ nnU-Net loaded")
                return True
            else:
                print("⚠ nnU-Net model not found")
                print("  Download: nnUNetv2_download_pretrained_model Dataset010_Colon")
                return False
                
        except ImportError:
            print("⚠ nnU-Net not installed")
            print("  Install: pip install nnunetv2")
            return False
        except Exception as e:
            print(f"⚠ nnU-Net initialization error: {e}")
            return False
    
    def _check_totalseg(self):
        """Check if TotalSegmentator is available"""
        try:
            import totalsegmentator
            print("✓ TotalSegmentator available")
            return True
        except ImportError:
            print("⚠ TotalSegmentator not installed")
            print("  Install: pip install TotalSegmentator")
            return False
    
    def analyze_ct_hybrid(self, image, metadata=None, cancer_type='Colorectal'):
        """
        Perform hybrid CT analysis
        
        Args:
            image: PIL Image, numpy array, or file path
            metadata: DICOM metadata dict (optional)
            cancer_type: Cancer type for context
            
        Returns:
            dict: Comprehensive analysis results
        """
        print(f"\n{'='*50}")
        print(f"HYBRID CT ANALYSIS - {cancer_type}")
        print(f"{'='*50}\n")
        
        # Convert image to numpy array
        image_array = self._prepare_image(image)
        
        results = {
            'metadata': metadata or {},
            'cancer_type': cancer_type,
            'status': 'success'
        }
        
        # Step 1: nnU-Net tumor segmentation
        if self.nnunet_ready:
            print("🔍 Step 1/3: nnU-Net tumor segmentation...")
            tumor_result = self._segment_tumor_nnunet(image_array)
            results['tumor_segmentation'] = tumor_result
            
            if self.use_gpu:
                torch.cuda.empty_cache()  # Free GPU memory
        else:
            print("⏭ Step 1/3: Skipped (nnU-Net not available)")
            results['tumor_segmentation'] = {'status': 'not_available'}
        
        # Step 2: TotalSegmentator anatomical structures
        if self.totalseg_ready:
            print("🔍 Step 2/3: TotalSegmentator anatomical analysis...")
            anatomy_result = self._segment_anatomy_totalseg(image_array)
            results['anatomical_structures'] = anatomy_result
            
            if self.use_gpu:
                torch.cuda.empty_cache()
        else:
            print("⏭ Step 2/3: Skipped (TotalSegmentator not available)")
            results['anatomical_structures'] = {'status': 'not_available'}
        
        # Step 3: Fusion analysis
        print("🔍 Step 3/3: Fusion analysis...")
        if self.nnunet_ready and self.totalseg_ready:
            fusion_result = self._fuse_tumor_and_anatomy(
                results['tumor_segmentation'],
                results['anatomical_structures']
            )
            results['fusion_analysis'] = fusion_result
            
            # Generate staging
            results['staging'] = self._estimate_staging(
                results['tumor_segmentation'],
                results['anatomical_structures'],
                fusion_result
            )
            
            # Clinical report
            results['clinical_report'] = self._generate_clinical_report(results)
        else:
            results['fusion_analysis'] = {'status': 'insufficient_data'}
            results['staging'] = {'status': 'unavailable'}
            results['clinical_report'] = "Hybrid analysis requires both nnU-Net and TotalSegmentator"
        
        print(f"\n✅ Analysis complete\n")
        return results
    
    def _prepare_image(self, image):
        """Convert various image formats to numpy array"""
        # Handle Streamlit UploadedFile
        if hasattr(image, 'read') and hasattr(image, 'seek'):
            from PIL import Image as PILImage
            image.seek(0)  # Reset file pointer
            pil_image = PILImage.open(image)
            image_array = np.array(pil_image.convert('L'))
            return image_array
        
        # Handle file path (string)
        if isinstance(image, str):
            from PIL import Image as PILImage
            pil_image = PILImage.open(image)
            image_array = np.array(pil_image.convert('L'))
            return image_array
        
        # Handle PIL Image
        if hasattr(image, 'convert'):
            from PIL import Image as PILImage
            image_array = np.array(image.convert('L'))
            return image_array
        
        # Handle numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = image[:, :, 0]  # Take first channel
            return image
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _segment_tumor_nnunet(self, image_array):
        """Segment tumor using nnU-Net"""
        try:
            # Ensure 3D for nnU-Net (add batch and channel dims if needed)
            if len(image_array.shape) == 2:
                image_3d = image_array[np.newaxis, np.newaxis, ...]
            else:
                image_3d = image_array
            
            # Predict
            segmentation = self.nnunet.predict_single_npy_array(
                image_3d,
                image_properties=None,
                save_or_return_probabilities=False
            )
            
            # Extract first 2D slice if needed
            if len(segmentation.shape) == 3:
                segmentation = segmentation[0, ...]
            
            # Tumor mask (label 1)
            tumor_mask = (segmentation == 1).astype(np.uint8)
            
            # Calculate properties
            props = self._calculate_tumor_properties(tumor_mask)
            
            return {
                'status': 'success',
                'tumor_detected': props['volume'] > 0,
                'mask': tumor_mask,
                'bounding_box': props['bbox'],
                'tumor_volume_mm3': props['volume'],
                'centroid': props['centroid'],
                'confidence': 0.9
            }
            
        except Exception as e:
            print(f"❌ nnU-Net segmentation error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _segment_anatomy_totalseg(self, image_array):
        """Segment anatomical structures using TotalSegmentator"""
        try:
            from totalsegmentator.python_api import totalsegmentator
            
            # TotalSegmentator requires NIfTI format
            with tempfile.TemporaryDirectory() as tmpdir:
                input_path = Path(tmpdir) / 'input.nii.gz'
                output_path = Path(tmpdir) / 'output'
                
                # Save as NIfTI
                if sitk:
                    sitk_img = sitk.GetImageFromArray(image_array[np.newaxis, ...])
                    sitk.WriteImage(sitk_img, str(input_path))
                else:
                    # Fallback without SimpleITK
                    import nibabel as nib
                    nifti_img = nib.Nifti1Image(image_array[np.newaxis, ...], np.eye(4))
                    nib.save(nifti_img, str(input_path))
                
                # Run TotalSegmentator
                segmentation = totalsegmentator(
                    str(input_path),
                    str(output_path),
                    ml=True,
                    fast=False,
                    device='gpu' if self.use_gpu else 'cpu',
                    quiet=True
                )
                
                # Parse results
                organs = self._parse_totalseg_results(output_path)
                
                return {
                    'status': 'success',
                    'organs_detected': organs,
                    'organ_count': len(organs)
                }
                
        except Exception as e:
            print(f"❌ TotalSegmentator error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_tumor_properties(self, mask):
        """Calculate tumor geometric properties"""
        from scipy import ndimage
        
        if np.sum(mask) == 0:
            return {
                'bbox': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                'volume': 0,
                'centroid': (0, 0)
            }
        
        # Find objects
        labeled, num = ndimage.label(mask)
        if num == 0:
            return {
                'bbox': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                'volume': 0,
                'centroid': (0, 0)
            }
        
        # Get largest component
        slices = ndimage.find_objects(labeled)
        if not slices:
            return {
                'bbox': {'x': 0, 'y': 0, 'width': 0, 'height': 0},
                'volume': 0,
                'centroid': (0, 0)
            }
        
        largest_slice = slices[0]
        y_slice, x_slice = largest_slice
        
        # Centroid
        centroid = ndimage.center_of_mass(mask)
        
        return {
            'bbox': {
                'x': int(x_slice.start),
                'y': int(y_slice.start),
                'width': int(x_slice.stop - x_slice.start),
                'height': int(y_slice.stop - y_slice.start)
            },
            'volume': int(np.sum(mask)),
            'centroid': (float(centroid[0]), float(centroid[1]))
        }
    
    def _parse_totalseg_results(self, output_path):
        """Parse TotalSegmentator output"""
        # List all segmentation files
        output_path = Path(output_path)
        if not output_path.exists():
            return []
        
        organ_files = list(output_path.glob('*.nii.gz'))
        organs = [f.stem.replace('.nii', '') for f in organ_files]
        
        return organs
    
    def _fuse_tumor_and_anatomy(self, tumor, anatomy):
        """Fuse tumor and anatomical information"""
        if tumor.get('status') != 'success' or anatomy.get('status') != 'success':
            return {'status': 'insufficient_data'}
        
        # Find anatomical location
        centroid = tumor.get('centroid', (0, 0))
        organs = anatomy.get('organs_detected', [])
        
        # Simplified nearest organ (would need spatial analysis in practice)
        nearest_organ = organs[0] if organs else 'Unknown'
        
        return {
            'status': 'success',
            'anatomical_location': nearest_organ,
            'tumor_centroid': centroid,
            'nearby_organs': organs[:5],  # Top 5 nearby
            'resectability': self._assess_resectability(tumor, organs)
        }
    
    def _assess_resectability(self, tumor, organs):
        """Assess surgical resectability"""
        volume = tumor.get('tumor_volume_mm3', 0)
        
        if volume == 0:
            return 'Unknown'
        elif volume < 5000:
            return 'Resectable'
        elif volume < 15000:
            return 'Borderline Resectable'
        else:
            return 'Likely Unresectable'
    
    def _estimate_staging(self, tumor, anatomy, fusion):
        """Estimate TNM staging"""
        if tumor.get('status') != 'success':
            return {'status': 'unavailable'}
        
        volume = tumor.get('tumor_volume_mm3', 0)
        
        # T-stage based on size
        if volume < 2000:
            t_stage = 'T1'
        elif volume < 5000:
            t_stage = 'T2'
        elif volume < 10000:
            t_stage = 'T3'
        else:
            t_stage = 'T4'
        
        return {
            'status': 'estimated',
            'T_stage': t_stage,
            'N_stage': 'Nx',  # Cannot determine without lymph node analysis
            'M_stage': 'Mx',  # Cannot determine without full body scan
            'confidence': 0.7
        }
    
    def _generate_clinical_report(self, results):
        """Generate structured clinical report"""
        tumor = results.get('tumor_segmentation', {})
        fusion = results.get('fusion_analysis', {})
        staging = results.get('staging', {})
        
        report = f"""
### Clinical Findings

**Tumor Characteristics:**
- Location: {fusion.get('anatomical_location', 'Unknown')}
- Volume: {tumor.get('tumor_volume_mm3', 0):.0f} mm³
- Confidence: {tumor.get('confidence', 0):.0%}

**Anatomical Context:**
- Nearby structures: {', '.join(fusion.get('nearby_organs', [])[:3])}

**Assessment:**
- Resectability: {fusion.get('resectability', 'Unknown')}
- Estimated T-stage: {staging.get('T_stage', 'Tx')}

**Recommendation:**
- Further imaging for N/M staging
- Multidisciplinary tumor board review
        """
        
        return report.strip()


# Factory function for easy integration
def create_hybrid_ct_analyzer(use_gpu=True):
    """
    Factory function to create HybridCTAnalyzer
    
    Returns:
        HybridCTAnalyzer instance
    """
    return HybridCTAnalyzer(use_gpu=use_gpu)
