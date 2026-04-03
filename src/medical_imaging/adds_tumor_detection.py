"""
ADDS CT Tumor Detection Integration Module
Mock 검출 시스템을 ADDS 메인 시스템에 통합
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json
from datetime import datetime

# ADDS 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze_tumor_location import TumorAnalyzer

class ADDSTumorDetection:
    """ADDS 시스템용 종양 검출 통합 클래스"""
    
    def __init__(self, mode='mock'):
        """
        초기화
        
        Args:
            mode: 'mock' 또는 'sota' (현재는 mock만 지원)
        """
        self.mode = mode
        self.results = None
        
    def analyze_ct_data(self, data_dir, output_dir=None, 
                       slice_thickness=1.0, pixel_spacing=(1.0, 1.0)):
        """
        CT 데이터 분석 및 종양 검출
        
        Args:
            data_dir: CT 이미지 디렉토리
            output_dir: 결과 저장 디렉토리
            slice_thickness: 슬라이스 간격 (mm)
            pixel_spacing: 픽셀 간격 (mm, mm)
            
        Returns:
            분석 결과 딕셔너리
        """
        print(f"\n🔍 ADDS CT 종양 검출 시작")
        print(f"   모드: {self.mode.upper()}")
        print(f"   데이터: {data_dir}")
        
        # TumorAnalyzer 초기화
        analyzer = TumorAnalyzer(
            data_dir=data_dir,
            output_dir=output_dir
        )
        
        # 분석 실행
        analyzer.analyze_volume(
            slice_thickness=slice_thickness,
            pixel_spacing=pixel_spacing
        )
        
        # 결과 저장
        self.results = analyzer.analysis_results
        
        return self.get_summary()
    
    def get_summary(self):
        """분석 결과 요약"""
        if not self.results:
            return None
        
        summary = self.results.get('summary', {})
        
        return {
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'total_slices': self.results['total_slices'],
            'affected_slices': summary.get('affected_slices', 0),
            'total_tumor_regions': summary.get('total_tumor_regions', 0),
            'total_volume_ml': summary.get('total_volume_ml', 0.0),
            'bounding_box_3d': summary.get('bounding_box_3d'),
            'slice_range': summary.get('slice_range')
        }
    
    def export_results(self, output_path):
        """결과를 JSON으로 저장"""
        if not self.results:
            print("⚠️  분석 결과가 없습니다.")
            return
        
        summary = self.get_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 결과 저장: {output_path}")
        
        return summary

def main():
    """테스트 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ADDS CT 종양 검출')
    parser.add_argument('--data', type=str, default='CTdata_cleaned',
                        help='CT 데이터 디렉토리')
    parser.add_argument('--output', type=str, default='adds_tumor_results',
                        help='결과 저장 디렉토리')
    parser.add_argument('--mode', type=str, default='mock',
                        choices=['mock', 'sota'],
                        help='검출 모드 (mock: 임계값 기반, sota: 딥러닝)')
    
    args = parser.parse_args()
    
    # ADDS 종양 검출 실행
    detector = ADDSTumorDetection(mode=args.mode)
    
    results = detector.analyze_ct_data(
        data_dir=args.data,
        output_dir=args.output
    )
    
    # 결과 저장
    detector.export_results('adds_tumor_detection_results.json')
    
    print("\n" + "="*70)
    print("ADDS CT 종양 검출 완료")
    print("="*70)
    print(f"\n📊 결과 요약:")
    print(f"   검출 모드: {results['mode'].upper()}")
    print(f"   총 슬라이스: {results['total_slices']}")
    print(f"   종양 검출 슬라이스: {results['affected_slices']}")
    print(f"   검출된 영역: {results['total_tumor_regions']}")
    print(f"   추정 부피: {results['total_volume_ml']:.2f} mL")
    
    print(f"\n✅ UI에서 확인:")
    print(f"   streamlit run ui_ct_tumor_analysis.py")

if __name__ == "__main__":
    main()
