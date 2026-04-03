#!/usr/bin/env python3
"""
AI-Hub 대장암 데이터 로더
데이터 유형을 자동 감지하고 적절한 로더 선택
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class AIHubDataLoader:
    """AI-Hub 데이터 자동 로더"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.data_type = None
        self.file_format = None
        
    def analyze_structure(self) -> Dict:
        """데이터 구조 분석"""
        print("Analyzing data structure...")
        
        # 파일 확장자 통계
        extensions = {}
        total_size = 0
        file_count = 0
        
        for file_path in self.data_root.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                total_size += file_path.stat().st_size
                file_count += 1
        
        # 데이터 유형 추정
        if '.dcm' in extensions or '.dicom' in extensions:
            self.data_type = 'CT/MRI'
            self.file_format = 'DICOM'
        elif '.tif' in extensions or '.tiff' in extensions:
            self.data_type = 'Pathology'
            self.file_format = 'TIFF'
        elif '.png' in extensions or '.jpg' in extensions:
            self.data_type = 'Endoscopy/Pathology'
            self.file_format = 'Image'
        
        result = {
            'total_files': file_count,
            'total_size_gb': total_size / (1024**3),
            'extensions': extensions,
            'estimated_type': self.data_type,
            'file_format': self.file_format,
        }
        
        return result
    
    def find_labels(self) -> List[Path]:
        """라벨 파일 찾기"""
        label_patterns = ['*.json', '*.xml', '*.txt', '*_mask.png', '*_label.png']
        label_files = []
        
        for pattern in label_patterns:
            label_files.extend(self.data_root.rglob(pattern))
        
        return label_files
    
    def load_sample(self, num_samples: int = 5) -> List[Dict]:
        """샘플 데이터 로드"""
        samples = []
        
        if self.file_format == 'DICOM':
            samples = self._load_dicom_samples(num_samples)
        elif self.file_format == 'TIFF':
            samples = self._load_tiff_samples(num_samples)
        elif self.file_format == 'Image':
            samples = self._load_image_samples(num_samples)
        
        return samples
    
    def _load_dicom_samples(self, n: int) -> List[Dict]:
        """DICOM 샘플 로드"""
        try:
            import pydicom
            
            dcm_files = list(self.data_root.rglob('*.dcm'))[:n]
            samples = []
            
            for dcm_path in dcm_files:
                try:
                    ds = pydicom.dcmread(dcm_path)
                    samples.append({
                        'path': str(dcm_path),
                        'shape': (ds.Rows, ds.Columns),
                        'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                        'modality': getattr(ds, 'Modality', 'Unknown'),
                    })
                except Exception as e:
                    print(f"Error loading {dcm_path}: {e}")
            
            return samples
        except ImportError:
            print("pydicom not installed. Install with: pip install pydicom")
            return []
    
    def _load_tiff_samples(self, n: int) -> List[Dict]:
        """TIFF 샘플 로드"""
        from PIL import Image
        
        tiff_files = list(self.data_root.rglob('*.tif*'))[:n]
        samples = []
        
        for tiff_path in tiff_files:
            try:
                img = Image.open(tiff_path)
                samples.append({
                    'path': str(tiff_path),
                    'shape': img.size,
                    'mode': img.mode,
                    'size_mb': tiff_path.stat().st_size / (1024**2),
                })
            except Exception as e:
                print(f"Error loading {tiff_path}: {e}")
        
        return samples
    
    def _load_image_samples(self, n: int) -> List[Dict]:
        """일반 이미지 샘플 로드"""
        from PIL import Image
        
        img_files = []
        for ext in ['.png', '.jpg', '.jpeg']:
            img_files.extend(list(self.data_root.rglob(f'*{ext}'))[:n])
        
        samples = []
        for img_path in img_files[:n]:
            try:
                img = Image.open(img_path)
                samples.append({
                    'path': str(img_path),
                    'shape': img.size,
                    'mode': img.mode,
                })
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        return samples
    
    def generate_report(self, output_path: str):
        """분석 리포트 생성"""
        structure = self.analyze_structure()
        labels = self.find_labels()
        samples = self.load_sample()
        
        report = {
            'structure': structure,
            'label_files': [str(p) for p in labels[:10]],
            'sample_data': samples,
        }
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 콘솔 출력
        print("\n" + "="*60)
        print("AI-Hub Data Analysis Report")
        print("="*60)
        print(f"Total Files: {structure['total_files']}")
        print(f"Total Size: {structure['total_size_gb']:.2f} GB")
        print(f"Estimated Type: {structure['estimated_type']}")
        print(f"File Format: {structure['file_format']}")
        print(f"\nFile Extensions:")
        for ext, count in structure['extensions'].items():
            print(f"  {ext}: {count} files")
        print(f"\nLabel Files Found: {len(labels)}")
        print(f"Sample Data Loaded: {len(samples)}")
        print("="*60)
        print(f"\nFull report saved to: {output_path}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_directory>")
        print("Example: python data_loader.py C:/Users/brook/Desktop/ADDS/data/aihub_colorectal/raw")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_file = Path(data_dir).parent / 'metadata' / 'analysis_report.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    loader = AIHubDataLoader(data_dir)
    loader.generate_report(str(output_file))
