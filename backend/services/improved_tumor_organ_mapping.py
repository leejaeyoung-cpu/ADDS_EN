"""
개선된 종양-장기 매핑 서비스
1. 공간적 오버랩 기반 매핑 (바운딩 박스 교차)
2. 조정된 HU 범위
"""

import os as _os
from pathlib import Path as _Path
BASE_DIR = _Path(_os.environ.get("ADDS_BASE_DIR", str(_Path(__file__).resolve().parent.parent.parent)))

import numpy as np
import nibabel as nib
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
from scipy import ndimage
from dataclasses import dataclass, asdict


@dataclass
class Tumor3D:
    """3D 종양 정보"""
    tumor_id: int
    volume_cm3: float
    centroid_mm: List[float]
    bbox_min_mm: List[float]
    bbox_max_mm: List[float]
    organ_name: str
    organ_label_id: int
    num_voxels: int
    overlap_percentage: float  # 장기와의 오버랩 비율


logger = logging.getLogger(__name__)


class ImprovedTumorOrganMapping:
    """개선된 종양-장기 매핑"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def map_tumors_spatial_overlap(
        self,
        labeled_tumor_path: str,
        organs_ml_path: str,
        organ_labels: Dict[int, Dict]
    ) -> List[Tumor3D]:
        """
        공간적 오버랩 기반 종양-장기 매핑
        
        각 종양이 어느 장기와 가장 많이 겹치는지 계산
        """
        self.logger.info("Mapping tumors using spatial overlap...")
        
        # 로딩
        tumor_nii = nib.load(labeled_tumor_path)
        tumors_volume = tumor_nii.get_fdata()
        spacing = tumor_nii.header.get_zooms()
        
        organs_nii = nib.load(organs_ml_path)
        organs_volume = organs_nii.get_fdata()
        
        # Get actual unique tumor IDs (not sequential!)
        unique_tumor_ids = np.unique(tumors_volume)
        unique_tumor_ids = unique_tumor_ids[unique_tumor_ids > 0]  # Remove background
        
        self.logger.info(f"Processing {len(unique_tumor_ids)} tumors...")
        
        tumor_list = []
        
        for tumor_id in unique_tumor_ids:
            tumor_mask = (tumors_volume == tumor_id)
            num_voxels = np.sum(tumor_mask)
            
            if num_voxels == 0:
                continue
            
            # 종양이 차지하는 복셀들에서 장기 레이블 확인
            tumor_voxels = tumor_mask > 0
            organ_labels_in_tumor = organs_volume[tumor_voxels]
            
            # 가장 많이 겹치는 장기 찾기
            unique_organs, counts = np.unique(organ_labels_in_tumor, return_counts=True)
            
            # 0 (배경) 제외
            non_zero_mask = unique_organs > 0
            if not np.any(non_zero_mask):
                # 모든 복셀이 배경 (장기 밖)
                organ_label = 0
                overlap_pct = 0.0
            else:
                unique_organs = unique_organs[non_zero_mask]
                counts = counts[non_zero_mask]
                
                # 가장 많이 겹치는 장기
                max_idx = np.argmax(counts)
                organ_label = int(unique_organs[max_idx])
                overlap_pct = (counts[max_idx] / num_voxels) * 100
            
            # 부피 계산
            volume_mm3 = num_voxels * np.prod(spacing)
            volume_cm3 = volume_mm3 / 1000
            
            # 중심점
            centroid_voxel = ndimage.center_of_mass(tumor_mask)
            centroid_mm = [centroid_voxel[i] * spacing[i] for i in range(3)]
            
            # 바운딩 박스
            coords = np.where(tumor_mask)
            bbox_min_voxel = [np.min(coords[i]) for i in range(3)]
            bbox_max_voxel = [np.max(coords[i]) for i in range(3)]
            
            bbox_min_mm = [bbox_min_voxel[i] * spacing[i] for i in range(3)]
            bbox_max_mm = [bbox_max_voxel[i] * spacing[i] for i in range(3)]
            
            # 장기 이름
            organ_name = "unknown"
            if organ_label in organ_labels:
                organ_name = organ_labels[organ_label]['name']
            
            tumor = Tumor3D(
                tumor_id=int(tumor_id),
                volume_cm3=float(volume_cm3),
                centroid_mm=[float(x) for x in centroid_mm],
                bbox_min_mm=[float(x) for x in bbox_min_mm],
                bbox_max_mm=[float(x) for x in bbox_max_mm],
                organ_name=organ_name,
                organ_label_id=int(organ_label),
                num_voxels=int(num_voxels),
                overlap_percentage=float(overlap_pct)
            )
            
            tumor_list.append(tumor)
        
        self.logger.info(f"Mapped {len(tumor_list)} tumors")
        
        # 매핑 품질 리포트
        mapped_count = sum(1 for t in tumor_list if t.organ_name != "unknown")
        self.logger.info(f"  Successfully mapped to organs: {mapped_count}/{len(tumor_list)}")
        
        return tumor_list
    
    def group_tumors_by_organ(self, tumors: List[Tumor3D]) -> Dict[str, List[Tumor3D]]:
        """장기별 그룹핑"""
        grouped = {}
        
        for tumor in tumors:
            organ = tumor.organ_name
            if organ not in grouped:
                grouped[organ] = []
            grouped[organ].append(tumor)
        
        # 통계
        self.logger.info("\nTumors by organ:")
        for organ, organ_tumors in sorted(grouped.items()):
            total_volume = sum(t.volume_cm3 for t in organ_tumors)
            avg_overlap = np.mean([t.overlap_percentage for t in organ_tumors])
            
            self.logger.info(
                f"  {organ:20s}: {len(organ_tumors):3d} tumors, "
                f"{total_volume:8.2f} cm³ total, "
                f"{avg_overlap:5.1f}% avg overlap"
            )
        
        return grouped
    
    def save_results(self, tumors: List[Tumor3D], output_file: str):
        """결과 저장"""


        tumors_dict = [asdict(t) for t in tumors]
        grouped = self.group_tumors_by_organ(tumors)
        
        summary = {
            'total_tumors': len(tumors),
            'organs_with_tumors': len(grouped),
            'tumors_by_organ': {
                organ: {
                    'count': len(organ_tumors),
                    'total_volume_cm3': sum(t.volume_cm3 for t in organ_tumors),
                    'avg_overlap_pct': float(np.mean([t.overlap_percentage for t in organ_tumors])),
                    'tumor_ids': [t.tumor_id for t in organ_tumors]
                }
                for organ, organ_tumors in grouped.items()
            }
        }
        
        output = {
            'summary': summary,
            'tumors': tumors_dict
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved results to {output_file}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 장기 레이블
    ORGAN_LABELS = {
        1: {"name": "air_lung"},
        2: {"name": "fat"},
        3: {"name": "lung_tissue"},
        4: {"name": "muscle"},
        5: {"name": "liver"},
        6: {"name": "soft_tissue"},
        7: {"name": "bone"},
    }
    
    service = ImprovedTumorOrganMapping()
    
    # 입력
    tumor_file = BASE_DIR / "output/tumor_organ_mapping/tumors_unique_3d.nii.gz"
    organs_file = BASE_DIR / "output/organs_simple/organs_multilabel_hu.nii_refined.nii.gz"
    output_file = BASE_DIR / "output/tumor_organ_mapping/tumor_organ_mapping_improved.json"
    
    print("=" * 80)
    print("Improved Tumor-Organ Mapping (Spatial Overlap)")
    print("=" * 80)
    print()
    
    # 매핑 실행
    print("[1] Mapping tumors using spatial overlap...")
    tumors = service.map_tumors_spatial_overlap(
        labeled_tumor_path=tumor_file,
        organs_ml_path=organs_file,
        organ_labels=ORGAN_LABELS
    )
    
    print(f"\n[2] Grouping by organ...")
    service.group_tumors_by_organ(tumors)
    
    print(f"\n[3] Saving results...")
    service.save_results(tumors, output_file)
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
