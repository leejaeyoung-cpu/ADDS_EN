"""
ADDS Tumor-Organ Mapping Service
3D Connected Components로 중복 종양 제거 및 장기별 매핑

주요 기능:
1. 3D Connected Components로 같은 종양 통합
2. 각 종양이 어느 장기에 속하는지 매핑
3. 장기별 종양 그룹핑
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


logger = logging.getLogger(__name__)


class TumorOrganMappingService:
    """종양-장기 매핑 서비스"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def remove_duplicate_tumors_3d(
        self,
        tumor_mask_path: str,
        output_path: str = None
    ) -> Tuple[str, int]:
        """
        3D Connected Components로 중복 종양 제거
        
        Args:
            tumor_mask_path: 종양 마스크 NIfTI 파일
            output_path: 출력 파일 경로 (None이면 자동 생성)
        
        Returns:
            (labeled_tumor_path, num_unique_tumors)
        """
        self.logger.info("Removing duplicate tumors using 3D connected components...")
        
        # 로딩
        tumor_nii = nib.load(tumor_mask_path)
        tumor_mask = tumor_nii.get_fdata() > 0
        
        # 3D Connected Components
        # 각 연결된 영역이 하나의 고유 종양
        labeled_tumors, num_tumors = ndimage.label(tumor_mask)
        
        self.logger.info(f"Found {num_tumors} unique 3D tumors (removed duplicates)")
        
        # 저장
        if output_path is None:
            input_path = Path(tumor_mask_path)
            output_path = input_path.parent / f"{input_path.stem}_unique_3d.nii.gz"
        
        labeled_nii = nib.Nifti1Image(
            labeled_tumors.astype(np.uint16),
            tumor_nii.affine,
            tumor_nii.header
        )
        nib.save(labeled_nii, str(output_path))
        
        self.logger.info(f"Saved unique 3D tumors to {output_path}")
        
        return str(output_path), num_tumors
    
    def map_tumors_to_organs(
        self,
        labeled_tumor_path: str,
        organs_ml_path: str,
        organ_labels: Dict[int, Dict]
    ) -> List[Tumor3D]:
        """
        각 종양이 어느 장기에 속하는지 매핑
        
        Args:
            labeled_tumor_path: Labeled tumor NIfTI (각 종양이 고유 ID)
            organs_ml_path: Multi-label organ NIfTI
            organ_labels: 장기 레이블 정의
        
        Returns:
            List of Tumor3D objects
        """
        self.logger.info("Mapping tumors to organs...")
        
        # 로딩
        tumor_nii = nib.load(labeled_tumor_path)
        tumors_volume = tumor_nii.get_fdata()
        spacing = tumor_nii.header.get_zooms()
        
        organs_nii = nib.load(organs_ml_path)
        organs_volume = organs_nii.get_fdata()
        
        num_tumors = int(tumors_volume.max())
        self.logger.info(f"Processing {num_tumors} tumors...")
        
        tumor_list = []
        
        for tumor_id in range(1, num_tumors + 1):
            tumor_mask = (tumors_volume == tumor_id)
            num_voxels = np.sum(tumor_mask)
            
            if num_voxels == 0:
                continue
            
            # 부피 계산
            volume_mm3 = num_voxels * np.prod(spacing)
            volume_cm3 = volume_mm3 / 1000
            
            # 중심점 (voxel 좌표)
            centroid_voxel = ndimage.center_of_mass(tumor_mask)
            centroid_voxel_int = tuple(int(round(x)) for x in centroid_voxel)
            
            # 중심점 (mm 좌표)
            centroid_mm = [
                centroid_voxel[i] * spacing[i]
                for i in range(3)
            ]
            
            # 바운딩 박스
            coords = np.where(tumor_mask)
            bbox_min_voxel = [np.min(coords[i]) for i in range(3)]
            bbox_max_voxel = [np.max(coords[i]) for i in range(3)]
            
            bbox_min_mm = [bbox_min_voxel[i] * spacing[i] for i in range(3)]
            bbox_max_mm = [bbox_max_voxel[i] * spacing[i] for i in range(3)]
            
            # 어느 장기에 속하는지 확인 (중심점 기준)
            organ_label = int(organs_volume[centroid_voxel_int])
            
            # 장기 이름 찾기
            organ_name = "unknown"
            if organ_label in organ_labels:
                organ_name = organ_labels[organ_label]['name']
            
            # Tumor3D 객체 생성
            tumor = Tumor3D(
                tumor_id=tumor_id,
                volume_cm3=float(volume_cm3),
                centroid_mm=[float(x) for x in centroid_mm],
                bbox_min_mm=[float(x) for x in bbox_min_mm],
                bbox_max_mm=[float(x) for x in bbox_max_mm],
                organ_name=organ_name,
                organ_label_id=int(organ_label),
                num_voxels=int(num_voxels)
            )
            
            tumor_list.append(tumor)
        
        self.logger.info(f"Mapped {len(tumor_list)} tumors to organs")
        
        return tumor_list
    
    def group_tumors_by_organ(
        self,
        tumors: List[Tumor3D]
    ) -> Dict[str, List[Tumor3D]]:
        """장기별로 종양 그룹핑"""
        
        grouped = {}
        
        for tumor in tumors:
            organ = tumor.organ_name
            
            if organ not in grouped:
                grouped[organ] = []
            
            grouped[organ].append(tumor)
        
        # 각 장기별 통계
        self.logger.info("\nTumors by organ:")
        for organ, organ_tumors in sorted(grouped.items()):
            total_volume = sum(t.volume_cm3 for t in organ_tumors)
            self.logger.info(f"  {organ:20s}: {len(organ_tumors):3d} tumors, "
                           f"{total_volume:8.2f} cm³ total")
        
        return grouped
    
    def save_tumor_organ_mapping(
        self,
        tumors: List[Tumor3D],
        output_file: str
    ):
        """종양-장기 매핑 결과를 JSON으로 저장"""
        
        # Tumor3D를 dict로 변환
        tumors_dict = [asdict(t) for t in tumors]
        
        # 장기별 그룹핑 정보 추가
        grouped = self.group_tumors_by_organ(tumors)
        
        summary = {
            'total_tumors': len(tumors),
            'organs_with_tumors': len(grouped),
            'tumors_by_organ': {
                organ: {
                    'count': len(organ_tumors),
                    'total_volume_cm3': sum(t.volume_cm3 for t in organ_tumors),
                    'tumor_ids': [t.tumor_id for t in organ_tumors]
                }
                for organ, organ_tumors in grouped.items()
            }
        }
        
        # 저장
        output = {
            'summary': summary,
            'tumors': tumors_dict
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved tumor-organ mapping to {output_file}")
    
    def create_organ_specific_tumor_masks(
        self,
        labeled_tumor_path: str,
        tumors: List[Tumor3D],
        output_dir: str
    ):
        """장기별 종양 마스크 생성"""
        
        self.logger.info("Creating organ-specific tumor masks...")
        
        # 로딩
        tumor_nii = nib.load(labeled_tumor_path)
        tumors_volume = tumor_nii.get_fdata()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 장기별로 그룹핑
        grouped = self.group_tumors_by_organ(tumors)
        
        for organ, organ_tumors in grouped.items():
            if organ == "unknown":
                continue
            
            # 해당 장기의 종양만 포함하는 마스크
            organ_tumor_mask = np.zeros_like(tumors_volume, dtype=np.uint8)
            
            for tumor in organ_tumors:
                tumor_voxels = (tumors_volume == tumor.tumor_id)
                organ_tumor_mask[tumor_voxels] = 1
            
            # 저장
            mask_file = output_path / f"tumors_{organ}.nii.gz"
            mask_nii = nib.Nifti1Image(
                organ_tumor_mask,
                tumor_nii.affine,
                tumor_nii.header
            )
            nib.save(mask_nii, str(mask_file))
            
            num_voxels = np.sum(organ_tumor_mask)
            self.logger.info(f"  {organ:20s}: {mask_file} ({num_voxels:,} voxels)")
        
        self.logger.info("Organ-specific tumor masks created")


def main():
    """테스트 실행"""


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # HU 기반 장기 레이블
    ORGAN_LABELS = {
        1: {"name": "air_lung"},
        2: {"name": "fat"},
        3: {"name": "lung_tissue"},
        4: {"name": "muscle"},
        5: {"name": "liver"},
        6: {"name": "soft_tissue"},
        7: {"name": "bone"},
    }
    
    service = TumorOrganMappingService()
    
    # 입력 파일
    tumor_mask = BASE_DIR / "output/protocol_test/tumor_mask.nii.gz"
    organs_ml = BASE_DIR / "output/organs_simple/organs_multilabel_hu.nii_refined.nii.gz"
    output_dir = BASE_DIR / "output/tumor_organ_mapping"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ADDS Tumor-Organ Mapping")
    print("=" * 80)
    print()
    
    # Step 1: 중복 제거 (3D connected components)
    print("[1] Removing duplicate tumors (3D connected components)...")
    labeled_tumor_path, num_unique = service.remove_duplicate_tumors_3d(
        tumor_mask_path=tumor_mask,
        output_path=f"{output_dir}/tumors_unique_3d.nii.gz"
    )
    print(f"    Original: 273 detections")
    print(f"    Unique 3D tumors: {num_unique}")
    print(f"    Reduction: {273 - num_unique} duplicates removed")
    
    # Step 2: 종양-장기 매핑
    print("\n[2] Mapping tumors to organs...")
    if not Path(organs_ml).exists():
        print(f"    Error: Organ segmentation not found at {organs_ml}")
        print("    Run organ_segmentation_service.py first!")
        return
    
    tumors = service.map_tumors_to_organs(
        labeled_tumor_path=labeled_tumor_path,
        organs_ml_path=organs_ml,
        organ_labels=ORGAN_LABELS
    )
    
    # Step 3: 장기별 그룹핑
    print("\n[3] Grouping tumors by organ...")
    grouped = service.group_tumors_by_organ(tumors)
    
    # Step 4: 결과 저장
    print("\n[4] Saving results...")
    mapping_file = f"{output_dir}/tumor_organ_mapping.json"
    service.save_tumor_organ_mapping(tumors, mapping_file)
    print(f"    ✓ Saved to {mapping_file}")
    
    # Step 5: 장기별 종양 마스크 생성
    print("\n[5] Creating organ-specific tumor masks...")
    service.create_organ_specific_tumor_masks(
        labeled_tumor_path=labeled_tumor_path,
        tumors=tumors,
        output_dir=f"{output_dir}/masks"
    )
    
    # 요약
    print("\n" + "=" * 80)
    print("Tumor-Organ Mapping Complete")
    print("=" * 80)
    print(f"Total unique 3D tumors: {len(tumors)}")
    print(f"Organs with tumors: {len(grouped)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
