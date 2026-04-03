"""
ADDS Organ Segmentation Service
TotalSegmentator를 사용한 자동 장기 분할

- 104개 장기 자동 인식
- 장기별 독립 마스크 생성
- 3D 메시 생성
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

logger = logging.getLogger(__name__)


# TotalSegmentator 장기 레이블 (주요 장기만)
ORGAN_LABELS = {
    1: {"name": "spleen", "color": "#8B4513", "display": "비장"},
    2: {"name": "kidney_right", "color": "#DC143C", "display": "우측 신장"},
    3: {"name": "kidney_left", "color": "#DC143C", "display": "좌측 신장"},
    4: {"name": "gallbladder", "color": "#32CD32", "display": "담낭"},
    5: {"name": "liver", "color": "#8B4513", "display": "간"},
    6: {"name": "stomach", "color": "#FFB6C1", "display": "위"},
    7: {"name": "pancreas", "color": "#FFA500", "display": "췌장"},
    55: {"name": "colon", "color": "#FF69B4", "display": "대장"},
    # 폐
    10: {"name": "lung_upper_lobe_left", "color": "#87CEEB", "display": "좌측 폐 상엽"},
    11: {"name": "lung_lower_lobe_left", "color": "#87CEEB", "display": "좌측 폐 하엽"},
    12: {"name": "lung_upper_lobe_right", "color": "#87CEEB", "display": "우측 폐 상엽"},
    13: {"name": "lung_middle_lobe_right", "color": "#87CEEB", "display": "우측 폐 중엽"},
    14: {"name": "lung_lower_lobe_right", "color": "#87CEEB", "display": "우측 폐 하엽"},
}


class OrganSegmentationService:
    """장기 분할 및 메시 생성 서비스"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def segment_organs_totalsegmentator(
        self, 
        ct_volume_path: str,
        output_dir: str,
        fast: bool = False,
        use_gpu: bool = True
    ) -> str:
        """
        TotalSegmentator로 장기 분할
        
        Args:
            ct_volume_path: CT NIfTI 파일 경로
            output_dir: 출력 디렉토리
            fast: 빠른 모드 (정확도 낮음)
            use_gpu: GPU 사용 여부
        
        Returns:
            분할된 장기 마스크 파일 경로
        """
        try:
            from totalsegmentator.python_api import totalsegmentator
            
            self.logger.info(f"Starting TotalSegmentator on {ct_volume_path}")
            self.logger.info(f"Fast mode: {fast}, GPU: {use_gpu}")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # TotalSegmentator 실행
            totalsegmentator(
                input=ct_volume_path,
                output=str(output_path),
                ml=True,  # Multi-label (하나의 파일에 모든 장기)
                fast=fast,
                device="gpu" if use_gpu else "cpu"
            )
            
            # Multi-label 파일 경로
            ml_file = output_path / "segmentations.nii.gz"
            
            if not ml_file.exists():
                # TotalSegmentator가 개별 파일로 저장한 경우
                self.logger.info("Merging individual organ masks into multi-label volume")
                ml_file = self._merge_organ_masks(output_path)
            
            self.logger.info(f"Organ segmentation complete: {ml_file}")
            return str(ml_file)
            
        except ImportError:
            self.logger.error("TotalSegmentator not installed. Install with: pip install TotalSegmentator")
            raise
        except Exception as e:
            self.logger.error(f"Organ segmentation failed: {e}", exc_info=True)
            raise
    
    def _merge_organ_masks(self, output_dir: Path) -> Path:
        """개별 장기 마스크를 하나의 multi-label 볼륨으로 병합"""
        
        # 첫 번째 파일로부터 shape 가져오기
        organ_files = list(output_dir.glob("*.nii.gz"))
        if not organ_files:
            raise FileNotFoundError(f"No organ masks found in {output_dir}")
        
        reference = nib.load(str(organ_files[0]))
        shape = reference.shape
        
        # Multi-label 볼륨 초기화
        ml_volume = np.zeros(shape, dtype=np.uint8)
        
        # 각 장기 마스크를 통합
        for organ_file in organ_files:
            organ_name = organ_file.stem.replace('.nii', '')
            
            # 이름으로 label ID 찾기
            label_id = None
            for oid, oinfo in ORGAN_LABELS.items():
                if oinfo['name'] == organ_name:
                    label_id = oid
                    break
            
            if label_id is None:
                continue
            
            # 마스크 로딩
            mask_nii = nib.load(str(organ_file))
            mask = mask_nii.get_fdata() > 0
            
            # Multi-label 볼륨에 추가
            ml_volume[mask] = label_id
        
        # 저장
        ml_file = output_dir / "organs_multilabel.nii.gz"
        ml_nii = nib.Nifti1Image(ml_volume, reference.affine, reference.header)
        nib.save(ml_nii, str(ml_file))
        
        self.logger.info(f"Merged {len(organ_files)} organ masks into {ml_file}")
        return ml_file
    
    def extract_organ_masks(
        self, 
        ml_volume_path: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Multi-label 볼륨에서 개별 장기 마스크 추출
        
        Args:
            ml_volume_path: Multi-label NIfTI 파일
            output_dir: 출력 디렉토리
        
        Returns:
            {organ_name: mask_file_path}
        """
        self.logger.info(f"Extracting organ masks from {ml_volume_path}")
        
        # 로딩
        ml_nii = nib.load(ml_volume_path)
        ml_volume = ml_nii.get_fdata()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        organ_masks = {}
        
        for label_id, organ_info in ORGAN_LABELS.items():
            organ_name = organ_info['name']
            
            # 해당 장기의 마스크 추출
            mask = (ml_volume == label_id).astype(np.uint8)
            
            if np.sum(mask) == 0:
                self.logger.warning(f"Organ {organ_name} (label {label_id}) not found in volume")
                continue
            
            # 저장
            mask_file = output_path / f"{organ_name}.nii.gz"
            mask_nii = nib.Nifti1Image(mask, ml_nii.affine, ml_nii.header)
            nib.save(mask_nii, str(mask_file))
            
            organ_masks[organ_name] = str(mask_file)
            
            volume_voxels = np.sum(mask)
            spacing = ml_nii.header.get_zooms()
            volume_mm3 = volume_voxels * np.prod(spacing)
            volume_cm3 = volume_mm3 / 1000
            
            self.logger.info(f"  {organ_info['display']:15s} ({organ_name:25s}): {volume_cm3:8.1f} cm³")
        
        self.logger.info(f"Extracted {len(organ_masks)} organ masks")
        return organ_masks
    
    def get_organ_statistics(self, ml_volume_path: str) -> Dict:
        """
        장기 통계 계산
        
        Returns:
            {
                organ_name: {
                    volume_cm3, centroid, bbox, label_id
                }
            }
        """
        from scipy import ndimage
        
        self.logger.info("Calculating organ statistics...")
        
        ml_nii = nib.load(ml_volume_path)
        ml_volume = ml_nii.get_fdata()
        spacing = ml_nii.header.get_zooms()
        
        stats = {}
        
        for label_id, organ_info in ORGAN_LABELS.items():
            organ_name = organ_info['name']
            mask = (ml_volume == label_id)
            
            if np.sum(mask) == 0:
                continue
            
            # 부피
            volume_mm3 = np.sum(mask) * np.prod(spacing)
            volume_cm3 = volume_mm3 / 1000
            
            # 중심점 (voxel 좌표)
            centroid_voxel = ndimage.center_of_mass(mask)
            
            # 중심점 (mm 좌표)
            centroid_mm = [
                centroid_voxel[i] * spacing[i]
                for i in range(3)
            ]
            
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
                'volume_cm3': float(volume_cm3),
                'centroid_mm': [float(x) for x in centroid_mm],
                'bbox_min_mm': [float(x) for x in bbox_min_mm],
                'bbox_max_mm': [float(x) for x in bbox_max_mm]
            }
        
        return stats
    
    def save_organ_info(self, stats: Dict, output_file: str):
        """장기 정보를 JSON으로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved organ info to {output_file}")


def main():
    """테스트 실행"""


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    service = OrganSegmentationService()
    
    # 입력 파일
    ct_volume = BASE_DIR / "output/protocol_test/ct_volume.nii.gz"
    output_dir = BASE_DIR / "output/organs"
    
    print("=" * 80)
    print("ADDS Organ Segmentation")
    print("=" * 80)
    print()
    
    # Step 1: TotalSegmentator 실행
    print("[1] Running TotalSegmentator (this may take 5-10 minutes)...")
    try:
        ml_file = service.segment_organs_totalsegmentator(
            ct_volume_path=ct_volume,
            output_dir=output_dir,
            fast=False,  # High quality
            use_gpu=True
        )
        print(f"    ✓ Complete: {ml_file}")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        print("\nNote: Install TotalSegmentator with:")
        print("  pip install TotalSegmentator")
        return
    
    # Step 2: 개별 장기 마스크 추출
    print("\n[2] Extracting individual organ masks...")
    organ_masks = service.extract_organ_masks(
        ml_volume_path=ml_file,
        output_dir=f"{output_dir}/individual_masks"
    )
    print(f"    ✓ Extracted {len(organ_masks)} organs")
    
    # Step 3: 통계 계산
    print("\n[3] Calculating organ statistics...")
    stats = service.get_organ_statistics(ml_file)
    
    stats_file = f"{output_dir}/organ_statistics.json"
    service.save_organ_info(stats, stats_file)
    print(f"    ✓ Saved to {stats_file}")
    
    # 요약
    print("\n" + "=" * 80)
    print("Organ Segmentation Complete")
    print("=" * 80)
    print(f"Multi-label file: {ml_file}")
    print(f"Individual masks: {output_dir}/individual_masks/")
    print(f"Statistics: {stats_file}")
    print(f"Total organs found: {len(stats)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
