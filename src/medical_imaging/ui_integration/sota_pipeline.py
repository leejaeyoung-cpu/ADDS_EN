# -*- coding: utf-8 -*-
"""
SOTA 종양 검출 파이프라인 (UI 통합용)
Swin-UNETR 기반 3D CT 종양 검출
"""
import sys
import io

# Windows 인코딩 설정 (최우선)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
from typing import List, Tuple, Dict, Any

# MONAI imports
try:
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import SwinUNETR
    from monai.transforms import Activations, AsDiscrete
except ImportError:
    print("Warning: MONAI not installed. Install with: pip install monai")


class SOTADetector:
    """SOTA 종양 검출 엔진 (Swin-UNETR)"""
    
    def __init__(self, checkpoint_path: str = "models/sota/fold_0/best_model.pth"):
        """
        초기화
        
        Args:
            checkpoint_path: 모델 체크포인트 경로
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # 모델 파라미터 (학습 시와 동일해야 함)
        self.img_size = (96, 96, 96)
        self.in_channels = 1
        self.out_channels = 2  # Background + Tumor
        self.feature_size = 48
    
    def load_model(self):
        """Swin-UNETR 모델 로드"""
        print(f"\n🔄 SOTA 모델 로딩...")
        print(f"   경로: {self.checkpoint_path}")
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {self.checkpoint_path}")
        
        # 모델 초기화 (학습 코드와 동일한 파라미터)
        self.model = SwinUNETR(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
            use_checkpoint=True,  # 메모리 절약
            spatial_dims=3  # 3D volumes
        )
        
        # 체크포인트 로드
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # State dict 키 확인
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 모델 로드 완료")
        print(f"   Device: {self.device}")
        
        # 체크포인트 정보
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            loss_val = checkpoint['loss']
            if isinstance(loss_val, (int, float)):
                print(f"   Loss: {loss_val:.4f}")
    
    def predict(self, volume: np.ndarray, roi_size: Tuple[int, int, int] = (96, 96, 96),
                sw_batch_size: int = 4, overlap: float = 0.5) -> np.ndarray:
        """
        3D Volume 추론
        
        Args:
            volume: (Z, Y, X) numpy array, normalized
            roi_size: Sliding window patch size
            sw_batch_size: Concurrent patches
            overlap: Overlap ratio (0-1)
        
        Returns:
            pred_mask: (Z, Y, X) binary mask (0: background, 1: tumor)
        """
        if self.model is None:
            self.load_model()
        
        # Numpy → Torch (1, 1, Z, Y, X)
        volume_tensor = torch.from_numpy(volume).float()
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        volume_tensor = volume_tensor.to(self.device)
        
        print(f"\n🔬 SOTA 추론 시작...")
        print(f"   Input shape: {volume_tensor.shape}")
        print(f"   Patch size: {roi_size}")
        print(f"   Overlap: {overlap}")
        
        # Sliding Window Inference
        with torch.no_grad():
            outputs = sliding_window_inference(
                inputs=volume_tensor,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=self.model,
                overlap=overlap,
                mode='gaussian',  # Gaussian blending
                device=self.device
            )
        
        # Softmax → Argmax
        outputs = torch.softmax(outputs, dim=1)  # (1, 2, Z, Y, X)
        pred_mask = torch.argmax(outputs, dim=1)  # (1, Z, Y, X)
        
        # Torch → Numpy
        pred_mask = pred_mask.squeeze(0).cpu().numpy()
        
        tumor_voxels = pred_mask.sum()
        total_voxels = pred_mask.size
        tumor_ratio = tumor_voxels / total_voxels * 100
        
        print(f"✅ 추론 완료")
        print(f"   Output shape: {pred_mask.shape}")
        print(f"   Tumor voxels: {tumor_voxels} / {total_voxels} ({tumor_ratio:.2f}%)")
        
        return pred_mask


def load_ct_volume(data_dir: str) -> Tuple[np.ndarray, int]:
    """
    CT 슬라이스들을 3D volume으로 로드
    
    Args:
        data_dir: CT 이미지 디렉토리
    
    Returns:
        volume: (Z, Y, X) numpy array
        num_slices: 슬라이스 개수
    """
    data_path = Path(data_dir)
    image_files = sorted(list(data_path.glob("*.jpg")))
    
    if not image_files:
        raise ValueError(f"이미지를 찾을 수 없습니다: {data_dir}")
    
    slices = []
    for img_path in image_files:
        img = Image.open(img_path).convert('L')
        slices.append(np.array(img))
    
    # Stack to 3D
    volume = np.stack(slices, axis=0)
    
    return volume, len(slices)


def preprocess_volume(volume: np.ndarray, 
                     normalize_method: str = 'minmax') -> np.ndarray:
    """
    CT Volume 전처리
    
    Args:
        volume: (Z, Y, X) numpy array
        normalize_method: 'minmax' or 'zscore'
    
    Returns:
        preprocessed: (Z, Y, X) normalized array
    """
    volume = volume.astype(np.float32)
    
    if normalize_method == 'minmax':
        # Min-Max to [-1, 1]
        v_min = volume.min()
        v_max = volume.max()
        volume = (volume - v_min) / (v_max - v_min + 1e-8)
        volume = volume * 2 - 1
    
    elif normalize_method == 'zscore':
        # Z-score (mean=0, std=1)
        mean = volume.mean()
        std = volume.std()
        volume = (volume - mean) / (std + 1e-8)
    
    return volume


def postprocess_predictions(pred_mask: np.ndarray, 
                           min_size_voxels: int = 50,
                           spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> List[Dict[str, Any]]:
    """
    예측 마스크 후처리
    
    Args:
        pred_mask: (Z, Y, X) binary mask
        min_size_voxels: 최소 종양 크기
        spacing: (z_spacing, y_spacing, x_spacing) in mm
    
    Returns:
        detections: List of tumor regions
    """
    from skimage import measure
    
    # Connected Component Labeling
    labeled_mask = measure.label(pred_mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    print(f"\n🔍 후처리 중...")
    print(f"   Components found: {len(regions)}")
    
    detections = []
    
    for region in regions:
        volume_voxels = region.area
        
        # Size filtering
        if volume_voxels < min_size_voxels:
            continue
        
        # 3D Bounding Box
        bbox = region.bbox  # (z_min, y_min, x_min, z_max, y_max, x_max)
        
        # Centroid
        centroid = region.centroid  # (z, y, x)
        
        # Volume in mm³ and mL
        volume_mm3 = volume_voxels * np.prod(spacing)
        volume_ml = volume_mm3 / 1000
        
        # Additional properties
        solidity = region.solidity
        extent = region.extent
        
        detection = {
            'centroid': tuple(float(c) for c in centroid),
            'bounding_box': tuple(int(b) for b in bbox),
            'volume_voxels': int(volume_voxels),
            'volume_mm3': float(volume_mm3),
            'volume_ml': float(volume_ml),
            'solidity': float(solidity),
            'extent': float(extent),
            # Convert for quality evaluator
            'slice_num': int(centroid[0]),
            'center': (int(centroid[2]), int(centroid[1])),  # (X, Y)
            'area_pixels': int(volume_voxels)  # Approximate
        }
        
        detections.append(detection)
    
    print(f"✅ 후처리 완료: {len(detections)} 개 영역")
    
    return detections


class SOTAPipeline:
    """완전한 SOTA 검출 파이프라인"""
    
    def __init__(self, checkpoint_path: str = "models/sota/fold_0/best_model.pth"):
        self.detector = SOTADetector(checkpoint_path)
    
    def run(self, data_dir: str, output_dir: str = "sota_results",
            spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[Dict, Dict]:
        """
        전체 파이프라인 실행
        
        Args:
            data_dir: CT 이미지 디렉토리
            output_dir: 결과 저장 디렉토리
            spacing: (slice_thickness, pixel_y, pixel_x) in mm
        
        Returns:
            combined_results: 전체 결과
            quality_report: 품질 평가 리포트
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("🔬 SOTA 종양 검출 파이프라인")
        print("=" * 70)
        
        # Step 1: Load
        print("\n[1/5] 데이터 로딩...")
        volume, num_slices = load_ct_volume(data_dir)
        print(f"   Volume shape: {volume.shape}")
        print(f"   Slices: {num_slices}")
        
        # Step 2: Preprocess
        print("\n[2/5] 전처리...")
        volume = preprocess_volume(volume, normalize_method='minmax')
        print(f"   Normalized range: [{volume.min():.3f}, {volume.max():.3f}]")
        
        # Step 3: Inference
        print("\n[3/5] SOTA 모델 추론...")
        pred_mask = self.detector.predict(volume)
        
        # Step 4: Postprocess
        print("\n[4/5] 후처리...")
        detections = postprocess_predictions(pred_mask, min_size_voxels=50, spacing=spacing)
        
        # Step 5: Quality Evaluation
        print("\n[5/5] 품질 평가...")
        quality_report, results = self._evaluate_quality(detections, num_slices)
        
        # Save results
        result_file = output_path / "sota_detection_report.json"
        combined_results = {
            'detections_3d': detections,
            'quality_report': quality_report,
            'summary': results['summary'],
            'total_slices': num_slices,
            'slices_with_tumors': results['slices_with_tumors'],
            'tumor_detections': results['tumor_detections']
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 70)
        print("✅ SOTA 검출 완료!")
        print(f"   검출된 영역: {len(detections)} 개")
        print(f"   총 부피: {sum(d['volume_ml'] for d in detections):.2f} mL")
        print(f"   품질 점수: {quality_report['score']}/100 {quality_report['quality']}")
        print(f"   결과 저장: {result_file}")
        print("=" * 70)
        
        return combined_results, quality_report
    
    def _evaluate_quality(self, detections: List[Dict], total_slices: int) -> Tuple[Dict, Dict]:
        """품질 평가 (기존 시스템 활용)"""
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from medical_imaging.validation.quality_metrics import DetectionQualityMetrics
            
            # Convert to quality evaluator format
            results = {
                'total_slices': total_slices,
                'slices_with_tumors': list(set([d['slice_num'] for d in detections])),
                'tumor_detections': detections,
                'summary': {
                    'total_tumor_regions': len(detections),
                    'affected_slices': len(set([d['slice_num'] for d in detections])),
                    'total_volume_ml': sum(d['volume_ml'] for d in detections)
                }
            }
            
            evaluator = DetectionQualityMetrics()
            quality_report = evaluator.assess_results(results)
            
            return quality_report, results
        
        except Exception as e:
            print(f"⚠️ 품질 평가 오류: {e}")
            # 기본 리포트 반환
            return {
                'score': 0,
                'quality': 'N/A',
                'status': 'error',
                'warnings': [str(e)]
            }, {
                'total_slices': total_slices,
                'slices_with_tumors': [],
                'tumor_detections': detections,
                'summary': {}
            }


if __name__ == "__main__":
    # 테스트 실행
    import argparse
    
    parser = argparse.ArgumentParser(description='SOTA 종양 검출')
    parser.add_argument('--data_dir', type=str, default='CTdata_cleaned',
                       help='CT 이미지 디렉토리')
    parser.add_argument('--output_dir', type=str, default='sota_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--checkpoint', type=str, default='models/sota/fold_0/best_model.pth',
                       help='모델 체크포인트 경로')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = SOTAPipeline(checkpoint_path=args.checkpoint)
    results, quality = pipeline.run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        spacing=(1.0, 1.0, 1.0)
    )
