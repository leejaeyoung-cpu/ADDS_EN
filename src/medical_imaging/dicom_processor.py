"""
3D DICOM Series Handler for nnU-Net
Multiple DICOM slices → 3D volume → nnU-Net inference
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import pydicom
from typing import List, Union
import os

class DICOMVolumeProcessor:
    """DICOM series를 3D volume으로 변환하고 nnU-Net 준비"""
    
    def __init__(self):
        self.temp_dir = None
    
    def load_dicom_series(self, dicom_files: List[Union[str, bytes]]):
        """
        DICOM series를 로드하고 정렬
        
        Args:
            dicom_files: DICOM 파일 경로 리스트 또는 bytes 리스트
        
        Returns:
            tuple: (슬라이스 배열, 메타데이터)
        """
        slices = []
        
        for dicom_file in dicom_files:
            try:
                # Streamlit UploadedFile 또는 경로 처리
                if hasattr(dicom_file, 'read'):
                    # UploadedFile
                    dicom_data = pydicom.dcmread(dicom_file)
                else:
                    # File path
                    dicom_data = pydicom.dcmread(dicom_file)
                
                slices.append(dicom_data)
            except Exception as e:
                print(f"Failed to read DICOM: {e}")
                continue
        
        if len(slices) == 0:
            raise ValueError("No valid DICOM files found")
        
        # Instance Number로 정렬
        slices.sort(key=lambda x: int(x.InstanceNumber) if hasattr(x, 'InstanceNumber') else 0)
        
        return slices
    
    def create_3d_volume(self, dicom_slices):
        """
        DICOM slices를 3D numpy array로 변환
        
        Args:
            dicom_slices: pydicom dataset 리스트
        
        Returns:
            np.ndarray: 3D volume (z, y, x)
        """
        # 첫 번째 슬라이스에서 크기 가져오기
        img_shape = dicom_slices[0].pixel_array.shape
        
        # 3D 배열 초기화
        volume = np.zeros((len(dicom_slices), img_shape[0], img_shape[1]), dtype=np.float32)
        
        # 각 슬라이스를 배열에 추가
        for i, dcm in enumerate(dicom_slices):
            volume[i, :, :] = dcm.pixel_array.astype(np.float32)
        
        return volume
    
    def save_as_nifti(self, volume: np.ndarray, output_path: str = None):
        """
        3D volume을 NIfTI 형식으로 저장
        
        Args:
            volume: 3D numpy array
            output_path: 저장 경로 (None이면 임시 파일)
        
        Returns:
            str: NIfTI 파일 경로
        """
        if output_path is None:
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(self.temp_dir, 'volume.nii.gz')
        
        # NIfTI 이미지 생성 (단순 identity affine)
        nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))
        
        # 저장
        nib.save(nifti_img, output_path)
        
        return output_path
    
    def process_dicom_to_nifti(self, dicom_files: List):
        """
        전체 파이프라인: DICOM series → NIfTI
        
        Args:
            dicom_files: DICOM 파일 리스트
        
        Returns:
            str: NIfTI 파일 경로
        """
        # 1. DICOM 로드 및 정렬
        slices = self.load_dicom_series(dicom_files)
        print(f"Loaded {len(slices)} DICOM slices")
        
        # 2. 3D volume 생성
        volume = self.create_3d_volume(slices)
        print(f"Created 3D volume: {volume.shape}")
        
        # 3. NIfTI 저장
        nifti_path = self.save_as_nifti(volume)
        print(f"Saved to NIfTI: {nifti_path}")
        
        return nifti_path
    
    def cleanup(self):
        """임시 파일 정리"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
