"""
Formula Extractor
논문에서 수식, 파라미터, 상수 추출

목표: 텍스트에서 물리학/수학 모델 추출
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class FormulaExtractor:
    """수식 및 파라미터 추출기"""
    
    def __init__(self):
        # 주요 파라미터 패턴
        self.parameter_patterns = {
            'energy': [
                r'ΔG\s*=\s*([0-9.\-]+)\s*(kcal/mol|kJ/mol)',
                r'activation energy\s*[=:]\s*([0-9.]+)\s*(kcal/mol|kJ/mol)',
                r'binding energy\s*[=:]\s*([0-9.\-]+)\s*(kcal/mol|kJ/mol)',
                r'E_?a\s*=\s*([0-9.]+)\s*(kcal/mol|kJ/mol)'
            ],
            'dissociation_constant': [
                r'K_?d\s*=\s*([0-9.]+)\s*([nμumM]+)',
                r'IC50\s*=\s*([0-9.]+)\s*([nμumM]+)',
                r'EC50\s*=\s*([0-9.]+)\s*([nμumM]+)',
                r'dissociation constant\s*[=:]\s*([0-9.]+)\s*([nμumM]+)'
            ],
            'rate_constant': [
                r'k_?on\s*=\s*([0-9.e\-]+)\s*([M\-1s\-1]+)',
                r'k_?off\s*=\s*([0-9.e\-]+)\s*(s\-1)',
                r'k_?cat\s*=\s*([0-9.]+)\s*(s\-1)',
                r'rate constant\s*[=:]\s*([0-9.e\-]+)'
            ],
            'hill_coefficient': [
                r'Hill coefficient\s*[=:]\s*([0-9.]+)',
                r'n_?H\s*=\s*([0-9.]+)',
                r'cooperativity\s*[=:]\s*([0-9.]+)'
            ],
            'tumor_doubling_time': [
                r'doubling time\s*[=:]\s*([0-9.]+)\s*(days?|hours?)',
                r'T_?d\s*=\s*([0-9.]+)\s*(days?)',
                r'tumor volume doubling\s*[=:]\s*([0-9.]+)'
            ]
        }
        
        # 수식 패턴
        self.equation_patterns = [
            r'E\s*=\s*.+',  # 에너지 방정식
            r'ΔG\s*=\s*.+',  # 자유 에너지
            r'\[D\]\s*=\s*.+',  # 농도
            r'V\(t\)\s*=\s*.+',  # 부피 시간 함수
            r'dV/dt\s*=\s*.+',  # 성장 속도
        ]
    
    def extract_from_text(self, text: str) -> Dict:
        """텍스트에서 수식 및 파라미터 추출"""
        results = {
            'parameters': {},
            'equations': [],
            'key_findings': []
        }
        
        # 파라미터 추출
        for param_type, patterns in self.parameter_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)
            
            if matches:
                results['parameters'][param_type] = matches
        
        # 수식 추출
        for pattern in self.equation_patterns:
            equations = re.findall(pattern, text)
            results['equations'].extend(equations)
        
        # 주요 발견사항 (숫자 + 단위)
        key_values = re.findall(
            r'([0-9.]+)\s*(kcal/mol|kJ/mol|nM|μM|mM|days?|hours?)',
            text
        )
        results['key_findings'] = key_values
        
        return results
    
    def extract_from_papers(self, papers_file: Path) -> Dict:
        """논문 JSON 파일에서 집단 추출"""
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        all_extractions = []
        
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            extraction = self.extract_from_text(text)
            
            if extraction['parameters'] or extraction['equations']:
                all_extractions.append({
                    'paper_id': paper.get('pmid') or paper.get('arxiv_id'),
                    'title': paper.get('title'),
                    'year': paper.get('year'),
                    'extraction': extraction
                })
        
        return {
            'total_papers': len(papers),
            'papers_with_data': len(all_extractions),
            'extractions': all_extractions
        }


class ParameterDatabase:
    """파라미터 데이터베이스 생성"""
    
    def __init__(self):
        self.parameters = {
            'binding_energies': [],
            'dissociation_constants': [],
            'rate_constants': [],
            'tumor_kinetics': []
        }
    
    def add_from_extraction(self, extractions: Dict):
        """추출 데이터를 DB에 추가"""
        for extraction_data in extractions['extractions']:
            params = extraction_data['extraction']['parameters']
            
            # 에너지
            if 'energy' in params:
                for value, unit in params['energy']:
                    self.parameters['binding_energies'].append({
                        'value': float(value),
                        'unit': unit,
                        'source': extraction_data['paper_id']
                    })
            
            # Kd
            if 'dissociation_constant' in params:
                for value, unit in params['dissociation_constant']:
                    self.parameters['dissociation_constants'].append({
                        'value': float(value),
                        'unit': unit,
                        'source': extraction_data['paper_id']
                    })
    
    def get_statistics(self) -> Dict:
        """통계 계산"""
        stats = {}
        
        for param_type, data in self.parameters.items():
            if data:
                values = [d['value'] for d in data]
                stats[param_type] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return stats
    
    def save(self, output_file: Path):
        """데이터베이스 저장"""
        data = {
            'parameters': self.parameters,
            'statistics': self.get_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Parameter database saved to {output_file}")


def main():
    """메인 처리"""
    literature_dir = Path("F:/ADDS/literature_database")
    
    extractor = FormulaExtractor()
    db = ParameterDatabase()
    
    # 모든 JSON 파일 처리
    for json_file in literature_dir.glob("*.json"):
        if json_file.name == 'parameter_database.json':
            continue
        
        print(f"Processing {json_file.name}...")
        
        extractions = extractor.extract_from_papers(json_file)
        db.add_from_extraction(extractions)
        
        # 추출 결과 저장
        output_file = json_file.parent / f"{json_file.stem}_extractions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extractions, f, indent=2)
    
    # 통합 파라미터 DB 저장
    db.save(literature_dir / 'parameter_database.json')
    
    # 통계 출력
    stats = db.get_statistics()
    print("\n" + "="*80)
    print("PARAMETER STATISTICS")
    print("="*80)
    for param_type, stat in stats.items():
        print(f"\n{param_type.upper()}:")
        print(f"  Count: {stat['count']}")
        print(f"  Mean: {stat['mean']:.4f}")
        print(f"  Std: {stat['std']:.4f}")
        print(f"  Range: [{stat['min']:.4f}, {stat['max']:.4f}]")


if __name__ == '__main__':
    main()
