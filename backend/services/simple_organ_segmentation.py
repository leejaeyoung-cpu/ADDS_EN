"""
Simple HU-based Organ Segmentation
TotalSegmentator의 대안으로 HU threshold 기반 장기 분할

주요 장기:
- Bone: HU > 300
- Liver: 40 < HU < 70
- Lung: -500 < HU < -300
- Fat: -120 < HU < -80
- Muscle: 10 < HU < 60
- Air/Lung: HU < -500
"""

import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
import json
from typing import Dict
from scipy import ndimage

logger = logging.getLogger(__name__)


# HU 기반 장기 레이블 (조정된 범위)
HU_ORGAN_LABELS = {
    1: {"name": "air_lung", "display": "공기/폐", "hu_min": -1000, "hu_max": -400, "color": "#000000"},
    2: {"name": "fat", "display": "지방", "hu_min": -150, "hu_max": -50, "color": "#FFD700"},
    3: {"name": "lung_tissue", "display": "폐 조직", "hu_min": -500, "hu_max": -200, "color": "#87CEEB"},
    4: {"name": "muscle", "display": "근육", "hu_min": 0, "hu_max": 70, "color": "#CD5C5C"},
    5: {"name": "liver", "display": "간", "hu_min": 30, "hu_max": 80, "color": "#8B4513"},
    6: {"name": "soft_tissue", "display": "연조직", "hu_min": 10, "hu_max": 50, "color": "#FFB6C1"},
    7: {"name": "bone", "display": "뼈", "hu_min": 200, "hu_max": 3000, "color": "#FFFFFF"},
}


class SimpleOrganSegmentation:
    """HU 기반 간단한 장기 분할"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def segment_organs_hu(
        self,
        ct_volume_path: str,
        output_dir: str
    ) -> str:
        """
        HU threshold 기반 장기 분할
        
        Returns:
            Multi-label organ file path
        """
        self.logger.info(f"Segmenting organs using HU thresholds from {ct_volume_path}")
        
        # CT 로딩
        ct_nii = nib.load(ct_volume_path)
        ct_volume = ct_nii.get_fdata()
        
        self.logger.info(f"CT volume shape: {ct_volume.shape}")
        self.logger.info(f"HU range: {ct_volume.min():.1f} to {ct_volume.max():.1f}")
        
        # Multi-label 볼륨 초기화
        ml_volume = np.zeros_like(ct_volume, dtype=np.uint8)
        
        # 각 장기별로 HU threshold 적용
        for label_id, organ_info in HU_ORGAN_LABELS.items():
            organ_name = organ_info['name']
            hu_min = organ_info['hu_min']
            hu_max = organ_info['hu_max']
            
            # HU 범위로 마스크 생성
            mask = (ct_volume >= hu_min) & (ct_volume <= hu_max)
            
            # 작은 노이즈 제거
            if organ_name not in ['bone', 'air_lung']:
                mask = ndimage.binary_opening(mask, iterations=1)
            
            # Multi-label에 추가 (우선순위: 나중 것이 덮어씀)
            ml_volume[mask] = label_id
            
            volume_voxels = np.sum(mask)
            spacing = ct_nii.header.get_zooms()
            volume_mm3 = volume_voxels * np.prod(spacing)
            volume_cm3 = volume_mm3 / 1000
            
            self.logger.info(f"  {organ_info['display']:10s} ({organ_name:15s}): "
                           f"HU [{hu_min:5.0f}, {hu_max:5.0f}] → {volume_cm3:8.1f} cm³")
        
        # 저장
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        ml_file = output_path / "organs_multilabel_hu.nii.gz"
        ml_nii = nib.Nifti1Image(ml_volume, ct_nii.affine, ct_nii.header)
        nib.save(ml_nii, str(ml_file))
        
        self.logger.info(f"Saved multi-label organ file: {ml_file}")
        
        return str(ml_file)
    
    def refine_organ_segmentation(
        self,
        ml_volume_path: str,
        output_path: str = None
    ) -> str:
        """
        장기 분할 정제 (연결성 기반)
        
        가장 큰 connected component만 유지
        """
        self.logger.info("Refining organ segmentation...")
        
        ml_nii = nib.load(ml_volume_path)
        ml_volume = ml_nii.get_fdata()
        
        refined = np.zeros_like(ml_volume)
        
        for label_id, organ_info in HU_ORGAN_LABELS.items():
            mask = (ml_volume == label_id)
            
            if np.sum(mask) == 0:
                continue
            
            # Connected components
            labeled, num_features = ndimage.label(mask)
            
            if num_features == 0:
                continue
            
            # 각 component의 크기 계산
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            
            # 가장 큰 것들만 유지 (top 3)
            top_n = min(3, num_features)
            largest_labels = np.argsort(sizes)[-top_n:] + 1
            
            for large_label in largest_labels:
                component = (labeled == large_label)
                refined[component] = label_id
            
            self.logger.info(f"  {organ_info['display']:10s}: kept {top_n}/{num_features} components")
        
        # 저장
        if output_path is None:
            input_path = Path(ml_volume_path)
            output_path = input_path.parent / f"{input_path.stem}_refined.nii.gz"
        
        refined_nii = nib.Nifti1Image(refined.astype(np.uint8), ml_nii.affine, ml_nii.header)
        nib.save(refined_nii, str(output_path))
        
        self.logger.info(f"Saved refined segmentation: {output_path}")
        
        return str(output_path)
    
    def get_organ_statistics(self, ml_volume_path: str) -> Dict:
        """장기 통계 계산"""
        self.logger.info("Calculating organ statistics...")
        
        ml_nii = nib.load(ml_volume_path)
        ml_volume = ml_nii.get_fdata()
        spacing = ml_nii.header.get_zooms()
        
        stats = {}
        
        for label_id, organ_info in HU_ORGAN_LABELS.items():
            organ_name = organ_info['name']
            mask = (ml_volume == label_id)
            
            if np.sum(mask) == 0:
                continue
            
            # 부피
            volume_mm3 = np.sum(mask) * np.prod(spacing)
            volume_cm3 = volume_mm3 / 1000
            
            # 중심점
            centroid_voxel = ndimage.center_of_mass(mask)
            centroid_mm = [centroid_voxel[i] * spacing[i] for i in range(3)]
            
            # 바운딩 박스
            coords = np.where(mask)
            bbox_min_voxel = [np.min(coords[i]) for i in range(3)]
            bbox_max_voxel = [np.max(coords[i]) for i in range(3)]
            
            bbox_min_mm = [bbox_min_voxel[i] * spacing[i] for i in range(3)]
            bbox_max_mm = [bbox_max_voxel[i] * spacing[i] for i in range(3)]
            
            stats[organ_name] = {
                'label_id': int(label_id),
                'display_name': organ_info['display'],
                'color': organ_info['color'],
                'hu_range': [organ_info['hu_min'], organ_info['hu_max']],
                'volume_cm3': float(volume_cm3),
                'centroid_mm': [float(x) for x in centroid_mm],
                'bbox_min_mm': [float(x) for x in bbox_min_mm],
                'bbox_max_mm': [float(x) for x in bbox_max_mm]
            }
        
        return stats
    
    def save_organ_info(self, stats: Dict, output_file: str):
        """장기 정보 JSON 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved organ info to {output_file}")


def main():
    """실행"""


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    service = SimpleOrganSegmentation()
    
    ct_volume = BASE_DIR / "output/protocol_test/ct_volume.nii.gz"
    output_dir = BASE_DIR / "output/organs_simple"
    
    print("=" * 80)
    print("Simple HU-based Organ Segmentation")
    print("=" * 80)
    print()
    
    # Step 1: HU 기반 분할
    print("[1] Segmenting organs using HU thresholds...")
    ml_file = service.segment_organs_hu(
        ct_volume_path=ct_volume,
        output_dir=output_dir
    )
    print(f"    Complete: {ml_file}")
    
    # Step 2: 정제
    print("\n[2] Refining segmentation (keeping largest components)...")
    refined_file = service.refine_organ_segmentation(ml_file)
    print(f"    Complete: {refined_file}")
    
    # Step 3: 통계
    print("\n[3] Calculating statistics...")
    stats = service.get_organ_statistics(refined_file)
    
    stats_file = f"{output_dir}/organ_statistics.json"
    service.save_organ_info(stats, stats_file)
    print(f"    Complete: {stats_file}")
    
    # 요약
    print("\n" + "=" * 80)
    print("Organ Segmentation Complete")
    print("=" * 80)
    print(f"Output: {refined_file}")
    print(f"Organs found: {len(stats)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
