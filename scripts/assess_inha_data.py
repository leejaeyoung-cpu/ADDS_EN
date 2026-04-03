#!/usr/bin/env python3
"""
인하대병원 데이터 Assessment 스크립트
CT 데이터 현황 파악
"""

import os
from pathlib import Path
from typing import Dict, List
import json

class InhaDataAssessment:
    """인하대병원 데이터 분석"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.report = {}
    
    def assess_ct_data(self) -> Dict:
        """CT 데이터 평가"""
        print("\n=== Assessing CT Data ===\n")
        
        # DICOM 파일 찾기
        dicom_files = list(self.data_root.rglob('*.dcm'))
        
        ct_report = {
            'total_files': len(dicom_files),
            'file_locations': [str(f.parent) for f in dicom_files[:5]],
        }
        
        # 샘플 DICOM 로드
        if dicom_files:
            try:
                import pydicom
                
                sample_dcm = pydicom.dcmread(dicom_files[0])
                ct_report['sample_metadata'] = {
                    'patient_id': getattr(sample_dcm, 'PatientID', 'Unknown'),
                    'modality': getattr(sample_dcm, 'Modality', 'Unknown'),
                    'shape': (sample_dcm.Rows, sample_dcm.Columns),
                    'scanner': getattr(sample_dcm, 'Manufacturer', 'Unknown'),
                }
                
                # 환자 수 추정
                patient_ids = set()
                for dcm_path in dicom_files[:100]:  # 샘플링
                    try:
                        ds = pydicom.dcmread(dcm_path)
                        patient_ids.add(getattr(ds, 'PatientID', 'Unknown'))
                    except:
                        pass
                
                ct_report['estimated_patients'] = len(patient_ids)
                
            except Exception as e:
                ct_report['error'] = str(e)
        
        self.report['ct'] = ct_report
        return ct_report
    
    def assess_pathology_data(self) -> Dict:
        """병리 데이터 평가"""
        print("\n=== Assessing Pathology Data ===\n")
        
        # 이미지 파일 찾기
        image_files = []
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.svs']:
            image_files.extend(list(self.data_root.rglob(f'*{ext}')))
        
        path_report = {
            'total_files': len(image_files),
            'formats': {},
        }
        
        # 형식별 분류
        for img in image_files:
            ext = img.suffix.lower()
            path_report['formats'][ext] = path_report['formats'].get(ext, 0) + 1
        
        # 샘플 이미지 로드
        if image_files:
            try:
                from PIL import Image
                
                sample_img = Image.open(image_files[0])
                path_report['sample_metadata'] = {
                    'size': sample_img.size,
                    'mode': sample_img.mode,
                    'format': sample_img.format,
                }
            except Exception as e:
                path_report['error'] = str(e)
        
        self.report['pathology'] = path_report
        return path_report
    
    def assess_clinical_data(self) -> Dict:
        """임상 데이터 평가"""
        print("\n=== Assessing Clinical Data ===\n")
        
        # CSV, Excel 파일 찾기
        clinical_files = []
        for ext in ['.csv', '.xlsx', '.xls']:
            clinical_files.extend(list(self.data_root.rglob(f'*{ext}')))
        
        clin_report = {
            'total_files': len(clinical_files),
            'file_list': [str(f.name) for f in clinical_files],
        }
        
        self.report['clinical'] = clin_report
        return clin_report
    
    def generate_report(self, output_path: str):
        """종합 리포트 생성"""
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        # 콘솔 출력
        print("\n" + "="*70)
        print("인하대병원 데이터 Assessment Report")
        print("="*70)
        
        if 'ct' in self.report:
            print(f"\n📊 CT Data:")
            print(f"  Total DICOM files: {self.report['ct']['total_files']}")
            if 'estimated_patients' in self.report['ct']:
                print(f"  Estimated patients: {self.report['ct']['estimated_patients']}")
        
        if 'pathology' in self.report:
            print(f"\n🔬 Pathology Data:")
            print(f"  Total image files: {self.report['pathology']['total_files']}")
            print(f"  Formats: {self.report['pathology']['formats']}")
        
        if 'clinical' in self.report:
            print(f"\n📋 Clinical Data:")
            print(f"  Files: {self.report['clinical']['total_files']}")
        
        print(f"\n{'='*70}")
        print(f"Full report: {output_path}")
        print("="*70 + "\n")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python assess_inha_data.py <data_directory>")
        print("\n인하대병원 데이터 디렉토리를 지정하세요:")
        print("  예: python assess_inha_data.py C:/Users/brook/Desktop/ADDS/data/inha")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_file = Path(data_dir).parent / 'inha_data_assessment.json'
    
    assessor = InhaDataAssessment(data_dir)
    
    # 데이터 평가
    assessor.assess_ct_data()
    assessor.assess_pathology_data()
    assessor.assess_clinical_data()
    
    # 리포트 생성
    assessor.generate_report(str(output_file))
