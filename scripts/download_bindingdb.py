"""
BindingDB Data Downloader
실제 약물-단백질 결합 상수 다운로드

타겟: EGFR, VEGFR2, mTOR, HER2
파라미터: Kd, Ki, IC50
"""

import requests
import pandas as pd
from pathlib import Path
import json
import time
from typing import List, Dict

OUTPUT_DIR = Path("F:/ADDS/binding_database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BindingDBDownloader:
    """BindingDB SOAP API 사용"""
    
    def __init__(self):
        self.base_url = "https://www.bindingdb.org/axis2/services/BDBService"
        self.session = requests.Session()
        
        # 주요 타겟 UniProt IDs
        self.targets = {
            'EGFR': 'P00533',
            'VEGFR2': 'P35968',
            'mTOR': 'P42345',
            'HER2': 'P04626',
            'BRAF': 'P15056'
        }
    
    def get_ligands_by_uniprot(self, uniprot_id: str, target_name: str) -> pd.DataFrame:
        """UniProt ID로 리간드 조회"""
        print(f"Querying BindingDB for {target_name} ({uniprot_id})...")
        
        url = f"{self.base_url}/getLigandsByUniprot"
        params = {
            'uniprot': uniprot_id,
            'response': 'application/json'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                print(f"  No data for {target_name}")
                return pd.DataFrame()
            
            # JSON → DataFrame
            df = pd.DataFrame(data)
            df['target_name'] = target_name
            df['uniprot_id'] = uniprot_id
            
            print(f"  Found {len(df)} ligands")
            return df
        
        except Exception as e:
            print(f"  Error: {e}")
            return pd.DataFrame()
    
    def download_all_targets(self) -> Dict[str, pd.DataFrame]:
        """모든 타겟 데이터 다운로드"""
        all_data = {}
        
        for target_name, uniprot_id in self.targets.items():
            df = self.get_ligands_by_uniprot(uniprot_id, target_name)
            
            if not df.empty:
                all_data[target_name] = df
                
                # 저장
                output_file = OUTPUT_DIR / f'bindingdb_{target_name.lower()}.csv'
                df.to_csv(output_file, index=False)
                print(f"  Saved to {output_file}")
            
            time.sleep(1)  # API 부담 방지
        
        return all_data
    
    def parse_binding_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """결합 상수 파싱 및 정리"""
        
        # 주요 컬럼 추출
        binding_cols = [col for col in df.columns if any(
            x in col.lower() for x in ['kd', 'ki', 'ic50', 'ec50', 'affinity']
        )]
        
        if not binding_cols:
            return df
        
        # 숫자 변환
        for col in binding_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


class AlternativeBindingDBDownloader:
    """웹 스크래핑 방식 (API 실패 시)"""
    
    def __init__(self):
        self.base_url = "https://www.bindingdb.org"
    
    def download_target_data(self, target_name: str, uniprot_id: str) -> pd.DataFrame:
        """타겟별 TSV 다운로드"""
        print(f"Downloading {target_name} via web interface...")
        
        # BindingDB TSV 다운로드 URL
        download_url = f"{self.base_url}/bind/BindingDB_All_{uniprot_id}.tsv"
        
        try:
            response = requests.get(download_url, timeout=60)
            
            if response.status_code == 200:
                # TSV 파일 저장
                output_file = OUTPUT_DIR / f'bindingdb_{target_name.lower()}_raw.tsv'
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                # DataFrame으로 읽기
                df = pd.read_csv(output_file, sep='\t', low_memory=False)
                print(f"  Downloaded {len(df)} rows")
                return df
            else:
                print(f"  Failed: HTTP {response.status_code}")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"  Error: {e}")
            return pd.DataFrame()


def extract_kd_values(df: pd.DataFrame) -> List[float]:
    """Kd 값 추출"""
    kd_cols = [col for col in df.columns if 'kd' in col.lower()]
    
    all_kd = []
    for col in kd_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        all_kd.extend(values.tolist())
    
    return [v for v in all_kd if v > 0]  # 양수만


def extract_ic50_values(df: pd.DataFrame) -> List[float]:
    """IC50 값 추출"""
    ic50_cols = [col for col in df.columns if 'ic50' in col.lower()]
    
    all_ic50 = []
    for col in ic50_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        all_ic50.extend(values.tolist())
    
    return [v for v in all_ic50 if v > 0]


def main():
    """메인 실행"""
    
    print("="*80)
    print("BindingDB Data Download")
    print("="*80)
    print()
    
    # Method 1: SOAP API 시도
    downloader = BindingDBDownloader()
    
    print("Method 1: SOAP API")
    all_data = downloader.download_all_targets()
    
    # API 실패 시 Method 2
    if not all_data:
        print("\nMethod 2: Web Download (fallback)")
        alt_downloader = AlternativeBindingDBDownloader()
        
        targets = {
            'EGFR': 'P00533',
            'VEGFR2': 'P35968',
            'mTOR': 'P42345'
        }
        
        all_data = {}
        for target_name, uniprot_id in targets.items():
            df = alt_downloader.download_target_data(target_name, uniprot_id)
            if not df.empty:
                all_data[target_name] = df
    
    # 통계 분석
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    total_kd = 0
    total_ic50 = 0
    
    for target_name, df in all_data.items():
        kd_values = extract_kd_values(df)
        ic50_values = extract_ic50_values(df)
        
        print(f"\n{target_name}:")
        print(f"  Total entries: {len(df)}")
        print(f"  Kd values: {len(kd_values)}")
        
        if kd_values:
            import numpy as np
            print(f"    Mean: {np.mean(kd_values):.2f} nM")
            print(f"    Median: {np.median(kd_values):.2f} nM")
            print(f"    Range: [{np.min(kd_values):.2f}, {np.max(kd_values):.2f}]")
        
        print(f"  IC50 values: {len(ic50_values)}")
        
        if ic50_values:
            print(f"    Mean: {np.mean(ic50_values):.2f} nM")
            print(f"    Median: {np.median(ic50_values):.2f} nM")
        
        total_kd += len(kd_values)
        total_ic50 += len(ic50_values)
    
    print(f"\n{'='*80}")
    print(f"TOTAL PARAMETERS EXTRACTED:")
    print(f"  Kd: {total_kd}")
    print(f"  IC50: {total_ic50}")
    print(f"  Total: {total_kd + total_ic50}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
