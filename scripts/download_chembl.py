"""
ChEMBL Data Downloader
REST API로 생물활성 데이터 수집

타겟: EGFR, VEGFR2, mTOR, HER2
파라미터: IC50, EC50, Ki, Kd
"""

import requests
import pandas as pd
from pathlib import Path
import json
import time
from typing import List, Dict
import numpy as np

OUTPUT_DIR = Path("F:/ADDS/binding_database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ChEMBLDownloader:
    """ChEMBL REST API"""
    
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.session = requests.Session()
        
        # 주요 타겟 ChEMBL IDs
        self.targets = {
            'EGFR': 'CHEMBL203',
            'VEGFR2': 'CHEMBL279',
            'mTOR': 'CHEMBL2842',
            'HER2': 'CHEMBL1824',
            'BRAF': 'CHEMBL5145'
        }
    
    def get_target_activities(
        self, 
        chembl_id: str, 
        target_name: str,
        activity_types: List[str] = ['IC50', 'EC50', 'Ki', 'Kd']
    ) -> pd.DataFrame:
        """타겟의 활성 데이터 조회"""
        
        print(f"Querying ChEMBL for {target_name} ({chembl_id})...")
        
        all_activities = []
        
        for activity_type in activity_types:
            print(f"  Fetching {activity_type} data...")
            
            url = f"{self.base_url}/activity"
            params = {
                'target_chembl_id': chembl_id,
                'standard_type': activity_type,
                'limit': 1000,  # 최대 1000개
                'format': 'json'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                activities = data.get('activities', [])
                
                print(f"    Found {len(activities)} {activity_type} values")
                all_activities.extend(activities)
                
                time.sleep(0.5)  # Rate limit: 20 req/sec
            
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not all_activities:
            return pd.DataFrame()
        
        # DataFrame 변환
        df = pd.DataFrame(all_activities)
        df['target_name'] = target_name
        df['target_chembl_id'] = chembl_id
        
        return df
    
    def download_all_targets(self) -> Dict[str, pd.DataFrame]:
        """모든 타겟 데이터 다운로드"""
        all_data = {}
        
        for target_name, chembl_id in self.targets.items():
            df = self.get_target_activities(chembl_id, target_name)
            
            if not df.empty:
                all_data[target_name] = df
                
                # 저장
                output_file = OUTPUT_DIR / f'chembl_{target_name.lower()}.csv'
                df.to_csv(output_file, index=False)
                print(f"  Saved to {output_file}")
            
            time.sleep(1)
        
        return all_data
    
    def parse_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        """활성 데이터 파싱"""
        
        # 주요 컬럼 선택
        key_cols = [
            'standard_type',
            'standard_value',
            'standard_units',
            'standard_relation',
            'molecule_chembl_id',
            'assay_description',
            'target_name'
        ]
        
        available_cols = [col for col in key_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # 숫자 변환
        df_clean['standard_value'] = pd.to_numeric(
            df_clean['standard_value'], 
            errors='coerce'
        )
        
        # 양수만
        df_clean = df_clean[df_clean['standard_value'] > 0]
        
        return df_clean


def extract_values_by_type(df: pd.DataFrame, value_type: str) -> List[float]:
    """타입별 값 추출"""
    if 'standard_type' not in df.columns:
        return []
    
    mask = df['standard_type'].str.upper() == value_type.upper()
    values = df[mask]['standard_value']
    
    # 숫자로 변환
    values = pd.to_numeric(values, errors='coerce').dropna()
    
    # 양수만
    values = values[values > 0]
    
    return values.tolist()


def analyze_chembl_data(all_data: Dict[str, pd.DataFrame]):
    """ChEMBL 데이터 분석"""
    
    print("\n" + "="*80)
    print("ChEMBL DATA ANALYSIS")
    print("="*80)
    
    total_activities = 0
    
    summary = {}
    
    for target_name, df in all_data.items():
        print(f"\n{target_name}:")
        print(f"  Total activities: {len(df)}")
        
        target_summary = {
            'total': len(df),
            'IC50': [],
            'EC50': [],
            'Ki': [],
            'Kd': []
        }
        
        for value_type in ['IC50', 'EC50', 'Ki', 'Kd']:
            values = extract_values_by_type(df, value_type)
            target_summary[value_type] = values
            
            if values:
                print(f"  {value_type}: {len(values)} values")
                print(f"    Mean: {np.mean(values):.2f} nM")
                print(f"    Median: {np.median(values):.2f} nM")
                print(f"    Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
        
        summary[target_name] = target_summary
        total_activities += len(df)
    
    # 전체 통계
    print(f"\n{'='*80}")
    print(f"TOTAL ACTIVITIES: {total_activities}")
    
    total_params = sum(
        len(data[param_type])
        for data in summary.values()
        for param_type in ['IC50', 'EC50', 'Ki', 'Kd']
    )
    print(f"TOTAL PARAMETERS: {total_params}")
    
    # JSON 저장
    summary_file = OUTPUT_DIR / 'chembl_summary.json'
    
    # NumPy 배열 → 리스트 변환
    summary_serializable = {}
    for target, data in summary.items():
        summary_serializable[target] = {
            'total': data['total'],
            'IC50': data['IC50'],
            'EC50': data['EC50'],
            'Ki': data['Ki'],
            'Kd': data['Kd']
        }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_serializable, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")


def main():
    """메인 실행"""
    
    print("="*80)
    print("ChEMBL Data Download")
    print("="*80)
    print()
    
    downloader = ChEMBLDownloader()
    all_data = downloader.download_all_targets()
    
    if all_data:
        # 데이터 분석
        analyze_chembl_data(all_data)
    else:
        print("No data downloaded!")


if __name__ == '__main__':
    main()
