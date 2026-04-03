"""
3D CT Volume Reconstruction Service
DICOM 슬라이스들을 3D 볼륨으로 재구성

기능:
1. DICOM 시리즈 로딩 및 정렬
2. 슬라이스 간격 보정
3. 등방성 복셀 리샘플링
4. NIfTI 형식 저장
"""

import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VolumeMetadata:
    """3D 볼륨 메타데이터"""
    spacing: Tuple[float, float, float]  # (x, y, z) mm
    origin: Tuple[float, float, float]   # (x, y, z) mm
    direction: Tuple[float, ...]         # 9개 값
    dimensions: Tuple[int, int, int]     # (x, y, z)
    patient_id: str
    study_date: str
    modality: str


class CT3DReconstructor:
    """CT DICOM 슬라이스를 3D 볼륨으로 재구성"""
    
    def __init__(self):
        self.volume = None
        self.metadata = None
        self.sitk_image = None
    
    # ========================================================================
    # 1. DICOM 시리즈 로딩
    # ========================================================================
    
    def load_dicom_series(self, dicom_directory: str) -> np.ndarray:
        """
        DICOM 디렉토리에서 시리즈를 읽어 3D 볼륨 생성
        
        Args:
            dicom_directory: DICOM 파일들이 있는 디렉토리
            
        Returns:
            3D numpy array [z, y, x] (슬라이스, 높이, 너비)
        """
        logger.info(f"Loading DICOM series from: {dicom_directory}")
        
        # SimpleITK ImageSeriesReader 사용
        reader = sitk.ImageSeriesReader()
        
        series_ids = reader.GetGDCMSeriesIDs(dicom_directory)
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_directory}")
        
        # 첫 번째 시리즈 사용 (여러 시리즈가 있을 수 있음)
        series_id = series_ids[0]
        logger.info(f"Found {len(series_ids)} series, using series: {series_id}")
        
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_directory, series_id)
        logger.info(f"Found {len(dicom_files)} DICOM files")
        
        reader.SetFileNames(dicom_files)
        
        # 메타데이터 로딩 활성화
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        # 이미지 읽기
        self.sitk_image = reader.Execute()
        
        # NumPy 배열로 변환
        self.volume = sitk.GetArrayFromImage(self.sitk_image)
        
        # 메타데이터 추출
        self._extract_metadata(dicom_files[0])
        
        logger.info(f"Volume shape: {self.volume.shape}")
        logger.info(f"Spacing: {self.metadata.spacing}")
        logger.info(f"Origin: {self.metadata.origin}")
        
        return self.volume
    
    def _extract_metadata(self, sample_dicom_file: str):
        """DICOM 파일에서 메타데이터 추출"""
        dcm = pydicom.dcmread(sample_dicom_file)
        
        spacing = self.sitk_image.GetSpacing()
        origin = self.sitk_image.GetOrigin()
        direction = self.sitk_image.GetDirection()
        dimensions = self.sitk_image.GetSize()
        
        self.metadata = VolumeMetadata(
            spacing=spacing,
            origin=origin,
            direction=direction,
            dimensions=dimensions,
            patient_id=str(dcm.PatientID) if hasattr(dcm, 'PatientID') else 'Unknown',
            study_date=str(dcm.StudyDate) if hasattr(dcm, 'StudyDate') else 'Unknown',
            modality=str(dcm.Modality) if hasattr(dcm, 'Modality') else 'CT'
        )
    
    # ========================================================================
    # 2. 슬라이스 정렬 및 간격 보정
    # ========================================================================
    
    def resample_to_isotropic(
        self,
        target_spacing: float = 1.0,
        interpolator: str = 'linear'
    ) -> np.ndarray:
        """
        등방성 복셀로 리샘플링 (모든 축에서 동일한 spacing)
        
        Args:
            target_spacing: 목표 spacing (mm)
            interpolator: 'linear', 'nearest', 'bspline'
            
        Returns:
            리샘플링된 3D 배열
        """
        if self.sitk_image is None:
            raise ValueError("Must load DICOM series first")
        
        logger.info(f"Resampling to isotropic spacing: {target_spacing} mm")
        
        # 현재 spacing과 크기
        original_spacing = self.sitk_image.GetSpacing()
        original_size = self.sitk_image.GetSize()
        
        # 새로운 크기 계산
        new_size = [
            int(round(osz * ospc / target_spacing))
            for osz, ospc in zip(original_size, original_spacing)
        ]
        
        # Interpolator 선택
        interpolator_map = {
            'linear': sitk.sitkLinear,
            'nearest': sitk.sitkNearestNeighbor,
            'bspline': sitk.sitkBSpline
        }
        interp_method = interpolator_map.get(interpolator, sitk.sitkLinear)
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing([target_spacing] * 3)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(self.sitk_image.GetDirection())
        resampler.SetOutputOrigin(self.sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(self.sitk_image.GetPixelIDValue())
        resampler.SetInterpolator(interp_method)
        
        self.sitk_image = resampler.Execute(self.sitk_image)
        self.volume = sitk.GetArrayFromImage(self.sitk_image)
        
        # 메타데이터 업데이트
        self.metadata.spacing = self.sitk_image.GetSpacing()
        self.metadata.dimensions = self.sitk_image.GetSize()
        
        logger.info(f"Resampled shape: {self.volume.shape}")
        logger.info(f"New spacing: {self.metadata.spacing}")
        
        return self.volume
    
    # ========================================================================
    # 3. HU 값 정규화
    # ========================================================================
    
    def normalize_hu_values(self) -> np.ndarray:
        """
        HU 값을 0-255 범위로 정규화 (시각화용)
        
        Returns:
            정규화된 uint8 배열
        """
        # Windowing (일반적인 CT abdomen window)
        window_center = 40  # HU
        window_width = 400  # HU
        
        min_hu = window_center - window_width // 2
        max_hu = window_center + window_width // 2
        
        # Clipping
        volume_normalized = np.clip(self.volume, min_hu, max_hu)
        
        # 0-255로 스케일링
        volume_normalized = ((volume_normalized - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
        
        return volume_normalized
    
    # ========================================================================
    # 4. 피부 분할 (간단한 HU thresholding)
    # ========================================================================
    
    def segment_skin(self, threshold: int = -200) -> np.ndarray:
        """
        간단한 HU thresholding으로 피부/신체 윤곽 분할
        
        Args:
            threshold: HU threshold (공기 제거용)
            
        Returns:
            Binary mask [z, y, x]
        """
        logger.info(f"Segmenting skin with threshold: {threshold} HU")
        
        # 공기보다 높은 HU 값을 가진 영역 (신체 내부)
        body_mask = self.volume > threshold
        
        # Morphological operations으로 정리
        body_mask_sitk = sitk.GetImageFromArray(body_mask.astype(np.uint8))
        body_mask_sitk.CopyInformation(self.sitk_image)
        
        # Closing (구멍 메우기)
        body_mask_sitk = sitk.BinaryMorphologicalClosing(
            body_mask_sitk,
            kernelRadius=[3, 3, 3]
        )
        
        # Largest connected component (신체만 남기기)
        cc_filter = sitk.ConnectedComponentImageFilter()
        cc_image = cc_filter.Execute(body_mask_sitk)
        
        # 가장 큰 컴포넌트 선택
        label_stats = sitk.LabelShapeStatisticsImageFilter()
        label_stats.Execute(cc_image)
        
        if label_stats.GetNumberOfLabels() > 0:
            largest_label = max(
                range(1, label_stats.GetNumberOfLabels() + 1),
                key=lambda l: label_stats.GetNumberOfPixels(l)
            )
            body_mask_sitk = cc_image == largest_label
        
        body_mask = sitk.GetArrayFromImage(body_mask_sitk).astype(bool)
        
        logger.info(f"Body volume: {np.sum(body_mask) * np.prod(self.metadata.spacing) / 1000:.1f} liters")
        
        return body_mask
    
    # ========================================================================
    # 5. 종양 3D 좌표 계산
    # ========================================================================
    
    def calculate_tumor_coordinates(
        self,
        tumor_mask: np.ndarray
    ) -> Dict:
        """
        종양 마스크로부터 3D 좌표 계산
        
        Args:
            tumor_mask: Binary mask [z, y, x]
            
        Returns:
            {
                'centroid_mm': (x, y, z),
                'bbox_min_mm': (x, y, z),
                'bbox_max_mm': (x, y, z),
                'volume_cm3': float,
                'longest_diameter_mm': float
            }
        """
        if np.sum(tumor_mask) == 0:
            raise ValueError("Empty tumor mask")
        
        # Binary 마스크를 SimpleITK 이미지로 변환
        tumor_sitk = sitk.GetImageFromArray(tumor_mask.astype(np.uint8))
        tumor_sitk.CopyInformation(self.sitk_image)
        
        # Label statistics 계산
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(tumor_sitk)
        
        if stats.GetNumberOfLabels() == 0:
            raise ValueError("No labels found in tumor mask")
        
        # Label 1 사용 (binary mask이므로)
        label = 1
        
        # Centroid (물리 좌표계, mm)
        centroid = stats.GetCentroid(label)
        
        # Bounding box
        bbox = stats.GetBoundingBox(label)  # [x_start, y_start, z_start, x_size, y_size, z_size]
        
        # Index를 physical coordinates로 변환
        bbox_min_idx = [bbox[0], bbox[1], bbox[2]]
        bbox_max_idx = [bbox[0] + bbox[3] - 1, bbox[1] + bbox[4] - 1, bbox[2] + bbox[5] - 1]
        
        bbox_min = self.sitk_image.TransformIndexToPhysicalPoint(bbox_min_idx)
        bbox_max = self.sitk_image.TransformIndexToPhysicalPoint(bbox_max_idx)
        
        # Volume (mm³ → cm³)
        volume_mm3 = stats.GetPhysicalSize(label)
        volume_cm3 = volume_mm3 / 1000
        
        # Longest diameter (Feret diameter)
        feret_diameter = stats.GetFeretDiameter(label)
        
        coordinates = {
            'centroid_mm': centroid,
            'bbox_min_mm': bbox_min,
            'bbox_max_mm': bbox_max,
            'volume_cm3': volume_cm3,
            'longest_diameter_mm': feret_diameter,
            'num_voxels': stats.GetNumberOfPixels(label)
        }
        
        logger.info(f"Tumor centroid: {centroid}")
        logger.info(f"Tumor volume: {volume_cm3:.2f} cm³")
        logger.info(f"Longest diameter: {feret_diameter:.1f} mm")
        
        return coordinates
    
    # ========================================================================
    # 6. 파일 저장
    # ========================================================================
    
    def save_nifti(self, output_path: str):
        """NIfTI 형식으로 저장"""
        if self.sitk_image is None:
            raise ValueError("No image to save")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sitk.WriteImage(self.sitk_image, str(output_path))
        logger.info(f"Saved NIfTI: {output_path}")
    
    def save_mask_nifti(self, mask: np.ndarray, output_path: str):
        """마스크를 NIfTI로 저장"""
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_sitk.CopyInformation(self.sitk_image)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sitk.WriteImage(mask_sitk, str(output_path))
        logger.info(f"Saved mask NIfTI: {output_path}")
    
    # ========================================================================
    # 7. 유틸리티
    # ========================================================================
    
    def get_slice(self, axis: str, index: int) -> np.ndarray:
        """
        특정 축에서 슬라이스 추출
        
        Args:
            axis: 'axial' (z), 'sagittal' (x), 'coronal' (y)
            index: 슬라이스 인덱스
        """
        if axis == 'axial':
            return self.volume[index, :, :]
        elif axis == 'sagittal':
            return self.volume[:, :, index]
        elif axis == 'coronal':
            return self.volume[:, index, :]
        else:
            raise ValueError(f"Invalid axis: {axis}")


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ct_3d_reconstruction.py <dicom_directory>")
        sys.exit(1)
    
    dicom_dir = sys.argv[1]
    
    print("=" * 80)
    print("3D CT Volume Reconstruction")
    print("=" * 80)
    
    # Reconstructor 생성
    reconstructor = CT3DReconstructor()
    
    # DICOM 시리즈 로딩
    volume = reconstructor.load_dicom_series(dicom_dir)
    
    print(f"\n[1] Loaded volume: {volume.shape}")
    print(f"    Spacing: {reconstructor.metadata.spacing}")
    print(f"    Patient ID: {reconstructor.metadata.patient_id}")
    
    # 등방성 리샘플링 (1mm³ voxels)
    volume_iso = reconstructor.resample_to_isotropic(target_spacing=1.0)
    print(f"\n[2] Isotropic resampling: {volume_iso.shape}")
    
    # 피부 분할
    skin_mask = reconstructor.segment_skin(threshold=-200)
    print(f"\n[3] Skin segmentation: {np.sum(skin_mask)} voxels")
    
    # 저장
    output_dir = Path("output/3d_reconstruction")
    reconstructor.save_nifti(output_dir / "ct_volume.nii.gz")
    reconstructor.save_mask_nifti(skin_mask, output_dir / "skin_mask.nii.gz")
    
    print(f"\n[4] Saved to {output_dir}")
    print("=" * 80)
