"""
ChEMBL Data Quick Analysis
이미 다운로드된 CSV 파일 분석
"""

import pandas as pd
from pathlib import Path
import numpy as np
import json

DB_DIR = Path("F:/ADDS/binding_database")


def analyze_csv_file(csv_path: Path):
    """CSV 파일 분석"""
    target_name = csv_path.stem.replace('chembl_', '').upper()
    
    df = pd.read_csv(csv_path)
    
    print(f"\n{target_name}:")
    print(f"  Total activities: {len(df)}")
    
    summary = {}
    
    for value_type in ['IC50', 'EC50', 'Ki', 'Kd']:
        # 해당 타입 필터
        mask = df['standard_type'].str.upper() == value_type
        values = df[mask]['standard_value']
        
        # 숫자 변환
        values = pd.to_numeric(values, errors='coerce').dropna()
        values = values[values > 0]
        
        summary[value_type] = {
            'count': len(values),
            'values': values.tolist()
        }
        
        if len(values) > 0:
            print(f"  {value_type}: {len(values)} values")
            print(f"    Mean: {values.mean():.2f} nM")
            print(f"    Median: {values.median():.2f} nM")
            print(f"    Range: [{values.min():.2f}, {values.max():.2f}]")
    
    return target_name, summary


def main():
    print("="*80)
    print("ChEMBL Data Analysis (from CSV)")
    print("="*80)
    
    all_summary = {}
    total_params = 0
    
    # 모든 CSV 파일 분석
    for csv_file in DB_DIR.glob("chembl_*.csv"):
        target_name, summary = analyze_csv_file(csv_file)
        all_summary[target_name] = summary
        
        target_params = sum(data['count'] for data in summary.values())
        total_params += target_params
    
    # 전체 통계
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total parameters extracted: {total_params}")
    
    # 타겟별 요약
    print(f"\nBy target:")
    for target, summary in all_summary.items():
        counts = [len(summary[t]['values']) for t in ['IC50', 'EC50', 'Ki', 'Kd']]
        total = sum(counts)
        print(f"  {target}: {total} ({summary['IC50']['count']} IC50, {summary['Ki']['count']} Ki, {summary['Kd']['count']} Kd)")
    
    # EGFR 상세
    if 'EGFR' in all_summary:
        print(f"\n{'='*80}")
        print("EGFR Detailed Statistics (for model)")
        print(f"{'='*80}")
        
        egfr =all_summary['EGFR']
        
        ic50_values = egfr['IC50']['values']
        ki_values = egfr['Ki']['values']
        kd_values = egfr['Kd']['values']
        
        print(f"\nIC50 ({len(ic50_values)} values):")
        if ic50_values:
            print(f"  Mean: {np.mean(ic50_values):.2f} nM")
            print(f"  Median: {np.median(ic50_values):.2f} nM")
            print(f"  Std: {np.std(ic50_values):.2f} nM")
        
        print(f"\nKi ({len(ki_values)} values):")
        if ki_values:
            print(f"  Mean: {np.mean(ki_values):.2f} nM")
            print(f"  Median: {np.median(ki_values):.2f} nM")
            print(f"  Std: {np.std(ki_values):.2f} nM")
        
        print(f"\nKd ({len(kd_values)} values):")
        if kd_values:
            print(f"  Mean: {np.mean(kd_values):.2f} nM")
            print(f"  Median: {np.median(kd_values):.2f} nM")
            print(f"  Std: {np.std(kd_values):.2f} nM")
    
    # JSON 저장
    output_file = DB_DIR / 'final_analysis.json'
    
    # 값 리스트 제거 (파일 크기 축소)
    summary_compact = {}
    for target, data in all_summary.items():
        summary_compact[target] = {}
        for param_type, info in data.items():
            values = info['values']
            if values:
                summary_compact[target][param_type] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
    
    with open(output_file, 'w') as f:
        json.dump(summary_compact, f, indent=2)
    
    print(f"\nAnalysis saved to {output_file}")


if __name__ == '__main__':
    main()
