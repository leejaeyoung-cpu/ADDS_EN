"""
Enhanced nnU-Net Predictor with 3D DICOM support and 5-fold ensemble
"""

import os
import numpy as np
from pathlib import Path
import torch
import tempfile
import nibabel as nib
from typing import List, Union

class nnUNetPredictorEnhanced:
    """
    개선된 nnU-Net 예측기
    - 3D DICOM series 지원
    - 5-fold ensemble 지원
    """
    
    def __init__(self, model_path=None, use_gpu=True, use_ensemble=False, folds=None):
        """
        Args:
            model_path: Base model directory
            use_gpu: GPU 사용 여부
            use_ensemble: 5-fold ensemble 사용 여부
            folds: 사용할 fold 리스트 (None이면 사용 가능한 모든 fold)
        """
        if model_path is None:
            model_path = r"C:\nnUNet_data\nnUNet_results\Dataset010_Colon\nnUNetTrainer__nnUNetPlans__3d_fullres"
        
        self.model_base_path = Path(model_path)
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.use_ensemble = use_ensemble
        self.predictor = None
        
        # 사용 가능한 folds 확인
        if folds is None:
            self.folds = self._detect_available_folds()
        else:
            self.folds = folds
        
        print(f"Available folds: {self.folds}")
        
        # nnU-Net 환경 변수
        os.environ['nnUNet_raw'] = r'C:\nnUNet_data\nnUNet_raw'
        os.environ['nnUNet_preprocessed'] = r'C:\nnUNet_data\nnUNet_preprocessed'
        os.environ['nnUNet_results'] = r'C:\nnUNet_data\nnUNet_results'
    
    def _detect_available_folds(self):
        """사용 가능한 fold 감지"""
        available = []
        for fold in range(5):
            fold_dir = self.model_base_path / f"fold_{fold}"
            checkpoint = fold_dir / "checkpoint_best.pth"
            if checkpoint.exists():
                available.append(fold)
        return tuple(available) if available else (0,)
    
    def load_model(self):
        """nnU-Net 모델 로드"""
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor as BasePredictor
            
            self.predictor = BasePredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device(self.device),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            
            # Model 초기화 (ensemble 또는 단일 fold)
            folds_to_use = self.folds if self.use_ensemble else (self.folds[0],)
            
            self.predictor.initialize_from_trained_model_folder(
                self.model_base_path,
                use_folds=folds_to_use,
                checkpoint_name='checkpoint_best.pth'
            )
            
            print(f"Model loaded with folds: {folds_to_use}")
            print(f"Ensemble: {self.use_ensemble}")
            
            return True
        except Exception as e:
            print(f"Model loading failed: {e}")
            return False
    
    def predict_from_nifti(self, nifti_path: str):
        """
        NIfTI 파일에서 예측
        
        Args:
            nifti_path: NIfTI 파일 경로 (.nii.gz)
        
        Returns:
            dict: 예측 결과
        """
        if self.predictor is None:
            if not self.load_model():
                return {'status': 'error', 'error': 'Model load failed'}
        
        try:
            with tempfile.TemporaryDirectory() as output_dir:
                # Prediction
                self.predictor.predict_from_files(
                    [[nifti_path]],
                    [os.path.join(output_dir, 'prediction.nii.gz')],
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=2,
                    num_processes_segmentation_export=2,
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0
                )
                
                # 결과 로드
                pred_nifti = nib.load(os.path.join(output_dir, 'prediction.nii.gz'))
                pred_mask = pred_nifti.get_fdata()
            
            # 결과 분석
            tumor_volume = np.sum(pred_mask > 0)
            tumor_detected = tumor_volume > 0
            
            return {
                'status': 'success',
                'tumor_detected': tumor_detected,
                'segmentation_mask': pred_mask,
                'tumor_volume': int(tumor_volume),
                'bounding_box': self._get_bounding_box_3d(pred_mask) if tumor_detected else None,
                'method': f"nnU-Net ({'Ensemble' if self.use_ensemble else f'Fold {self.folds[0]}'})"
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def predict_from_dicom_series(self, dicom_files: List):
        """
        DICOM series에서 직접 예측
        
        Args:
            dicom_files: DICOM 파일 리스트
        
        Returns:
            dict: 예측 결과
        """
        from .dicom_processor import DICOMVolumeProcessor
        
        try:
            # DICOM → NIfTI 변환
            processor = DICOMVolumeProcessor()
            nifti_path = processor.process_dicom_to_nifti(dicom_files)
            
            # Prediction
            result = self.predict_from_nifti(nifti_path)
            
            # Cleanup
            processor.cleanup()
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _get_bounding_box_3d(self, mask):
        """3D mask에서 bounding box 추출"""
        if np.sum(mask) == 0:
            return None
        
        z_indices, y_indices, x_indices = np.where(mask > 0)
        
        return {
            'x': int(x_indices.min()),
            'y': int(y_indices.min()),
            'z': int(z_indices.min()),
            'width': int(x_indices.max() - x_indices.min()),
            'height': int(y_indices.max() - y_indices.min()),
            'depth': int(z_indices.max() - z_indices.min())
        }
