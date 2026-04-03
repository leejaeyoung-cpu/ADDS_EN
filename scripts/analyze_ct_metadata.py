#!/usr/bin/env python3
"""
인하대병원 CT DICOM 메타데이터 분석
환자 수, 시리즈 정보, 품질 평가
"""

import pydicom
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

class CTMetadataAnalyzer:
    """DICOM 메타데이터 분석"""
    
    def __init__(self, dicom_dir: str):
        self.dicom_dir = Path(dicom_dir)
        self.patients = defaultdict(lambda: {
            'studies': defaultdict(lambda: {
                'series': defaultdict(list)
            })
        })
        
    def analyze(self):
        """전체 DICOM 파일 분석"""
        print(f"\nAnalyzing DICOM files in: {self.dicom_dir}")
        print("="*70)
        
        dcm_files = list(self.dicom_dir.glob('*.dcm'))
        total_files = len(dcm_files)
        
        print(f"Found {total_files} DICOM files")
        print("Processing...")
        
        for i, dcm_path in enumerate(dcm_files, 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{total_files}")
            
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                
                # Extract key metadata
                patient_id = getattr(ds, 'PatientID', 'Unknown')
                study_uid = getattr(ds, 'StudyInstanceUID', 'Unknown')
                series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
                series_desc = getattr(ds, 'SeriesDescription', 'Unknown')
                study_date = getattr(ds, 'StudyDate', 'Unknown')
                modality = getattr(ds, 'Modality', 'Unknown')
                
                # Organize by patient -> study -> series
                study = self.patients[patient_id]['studies'][study_uid]
                study['date'] = study_date
                study['series'][series_uid].append({
                    'file': dcm_path.name,
                    'description': series_desc,
                    'modality': modality,
                })
                
            except Exception as e:
                print(f"  Warning: Failed to read {dcm_path.name}: {e}")
        
        print("\nAnalysis complete!")
        return self.generate_summary()
    
    def generate_summary(self):
        """분석 결과 요약"""
        summary = {
            'total_patients': len(self.patients),
            'patients': []
        }
        
        for patient_id, patient_data in self.patients.items():
            patient_info = {
                'id': patient_id,
                'total_studies': len(patient_data['studies']),
                'studies': []
            }
            
            for study_uid, study_data in patient_data['studies'].items():
                study_info = {
                    'date': study_data.get('date', 'Unknown'),
                    'total_series': len(study_data['series']),
                    'series': []
                }
                
                for series_uid, series_files in study_data['series'].items():
                    series_info = {
                        'description': series_files[0]['description'] if series_files else 'Unknown',
                        'modality': series_files[0]['modality'] if series_files else 'Unknown',
                        'num_slices': len(series_files),
                    }
                    study_info['series'].append(series_info)
                
                patient_info['studies'].append(study_info)
            
            summary['patients'].append(patient_info)
        
        return summary
    
    def print_summary(self, summary):
        """요약 출력"""
        print("\n" + "="*70)
        print("CT DICOM Metadata Analysis Summary")
        print("="*70)
        
        print(f"\nTotal Patients: {summary['total_patients']}")
        
        for patient in summary['patients']:
            print(f"\n{'─'*70}")
            print(f"Patient ID: {patient['id']}")
            print(f"  Studies: {patient['total_studies']}")
            
            for i, study in enumerate(patient['studies'], 1):
                print(f"\n  Study {i}:")
                print(f"    Date: {study['date']}")
                print(f"    Series: {study['total_series']}")
                
                for j, series in enumerate(study['series'], 1):
                    print(f"      {j}. {series['description']}")
                    print(f"         Slices: {series['num_slices']}, Modality: {series['modality']}")
        
        print("\n" + "="*70)
        
        # Statistics
        total_slices = sum(
            series['num_slices']
            for patient in summary['patients']
            for study in patient['studies']
            for series in study['series']
        )
        
        total_series = sum(
            study['total_series']
            for patient in summary['patients']
            for study in patient['studies']
        )
        
        print("\nStatistics:")
        print(f"  Total Slices: {total_slices}")
        print(f"  Total Series: {total_series}")
        print(f"  Avg Slices per Patient: {total_slices / summary['total_patients']:.0f}" if summary['total_patients'] > 0 else "  N/A")
        print("="*70 + "\n")
    
    def save_report(self, summary, output_path: str):
        """결과 JSON 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Full report saved to: {output_path}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage: python analyze_ct_metadata.py <dicom_directory>")
        print("\nExample:")
        print("  python analyze_ct_metadata.py CTdcm")
        sys.exit(1)
    
    dicom_dir = sys.argv[1]
    output_json = Path(dicom_dir).parent / 'ct_metadata_report.json'
    
    analyzer = CTMetadataAnalyzer(dicom_dir)
    summary = analyzer.analyze()
    analyzer.print_summary(summary)
    analyzer.save_report(summary, str(output_json))
    
    print("\nNext Steps:")
    print("1. Review patient count and series descriptions")
    print("2. Check for multi-phase scans (arterial, venous, etc.)")
    print("3. Prepare for CT detection analysis")
