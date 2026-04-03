"""
RAG System for Literature Database
벡터 데이터베이스 + LLM 기반 논문 검색 및 지식 추출
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class LiteratureRAG:
    """문헌 RAG 시스템"""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.papers = []
        self.embeddings = None
        self.index = None
        
        # 임베딩 모델 (논문용)
        print("Loading embedding model...")
        self.model = SentenceTransformer('allenai/specter')  # Scientific paper embeddings
        
        self.load_database()
    
    def load_database(self):
        """논문 데이터베이스 로드"""
        print(f"Loading papers from {self.db_path}...")
        
        # 모든 JSON 파일 로드
        for json_file in self.db_path.glob("*_pubmed.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 리스트인지 확인
                if isinstance(data, list):
                    self.papers.extend(data)
        
        for json_file in self.db_path.glob("arxiv_*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.papers.extend(data)
        
        print(f"Loaded {len(self.papers)} papers")
        
        if self.papers:
            self.build_index()
    
    def build_index(self):
        """FAISS 벡터 인덱스 구축"""
        print("Building vector index...")
        
        # 논문 텍스트 (제목 + 초록)
        texts = [
            f"{p.get('title', '')} {p.get('abstract', '')}"
            for p in self.papers
        ]
        
        # 임베딩 생성
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # FAISS 인덱스
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Index built with {len(self.papers)} papers")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """쿼리로 관련 논문 검색"""
        if self.index is None:
            return []
        
        # 쿼리 임베딩
        query_embedding = self.model.encode([query])[0]
        
        # 검색
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            top_k
        )
        
        # 결과 반환
        results = []
        for i, idx in enumerate(indices[0]):
            paper = self.papers[idx].copy()
            paper['relevance_score'] = float(1 / (1 + distances[0][i]))  # 거리 → 점수
            results.append(paper)
        
        return results
    
    def extract_knowledge(self, topic: str) -> Dict:
        """특정 주제에 대한 지식 집합 추출"""
        results = self.search(topic, top_k=20)
        
        knowledge = {
            'topic': topic,
            'paper_count': len(results),
            'papers': results,
            'key_insights': []
        }
        
        return knowledge
    
    def save_index(self, index_path: Path):
        """인덱스 저장 (재사용)"""
        faiss.write_index(self.index, str(index_path))
        print(f"Index saved to {index_path}")


class KnowledgeExtractor:
    """논문에서 핵심 지식 추출"""
    
    def __init__(self, rag: LiteratureRAG):
        self.rag = rag
    
    def extract_energy_parameters(self) -> Dict:
        """에너지 관련 파라미터 추출"""
        print("Extracting energy parameters...")
        
        topics = [
            "drug receptor binding free energy",
            "activation energy signal transduction",
            "thermodynamics of protein-ligand interaction"
        ]
        
        all_findings = []
        
        for topic in topics:
            papers = self.rag.search(topic, top_k=15)
            for paper in papers:
                abstract = paper.get('abstract', '') or ''
                
                if not abstract:
                    continue
                
                # ΔG 추출
                import re
                energy_matches = re.findall(
                    r'ΔG.*?([0-9.\-]+)\s*(kcal/mol|kJ/mol)',
                    abstract,
                    re.IGNORECASE
                )
                
                if energy_matches:
                    all_findings.append({
                        'paper_id': paper.get('pmid') or paper.get('arxiv_id'),
                        'title': paper.get('title'),
                        'values': energy_matches,
                        'topic': topic
                    })
        
        return {
            'parameter_type': 'binding_energy',
            'findings': all_findings,
            'count': len(all_findings)
        }
    
    def extract_dose_response_models(self) -> Dict:
        """용량-반응 모델 추출"""
        print("Extracting dose-response models...")
        
        papers = self.rag.search("dose response relationship Hill equation EC50", top_k=20)
        
        models = []
        for paper in papers:
            abstract = paper.get('abstract', '')
            
            # Hill equation 언급
            if 'Hill' in abstract or 'EC50' in abstract or 'IC50' in abstract:
                models.append({
                    'paper_id': paper.get('pmid'),
                    'title': paper.get('title'),
                    'abstract': abstract[:300]
                })
        
        return {
            'model_type': 'dose_response',
            'papers': models,
            'count': len(models)
        }
    
    def generate_summary_report(self, output_file: Path):
        """종합 요약 리포트 생성"""
        print("Generating summary report...")
        
        report = {
            'energy_parameters': self.extract_energy_parameters(),
            'dose_response_models': self.extract_dose_response_models()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 텍스트 버전
        text_file = output_file.with_suffix('.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LITERATURE KNOWLEDGE EXTRACTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. ENERGY PARAMETERS\n")
            f.write(f"   Found in {report['energy_parameters']['count']} papers\n\n")
            
            for finding in report['energy_parameters']['findings'][:10]:
                f.write(f"   • {finding['title']}\n")
                f.write(f"     Values: {finding['values']}\n\n")
            
            f.write("\n2. DOSE-RESPONSE MODELS\n")
            f.write(f"   Found in {report['dose_response_models']['count']} papers\n\n")
            
            for model in report['dose_response_models']['papers'][:10]:
                f.write(f"   • {model['title']}\n\n")
        
        print(f"Report saved to {output_file}")


def main():
    """메인 RAG 시스템"""
    
    db_path = Path("F:/ADDS/literature_database")
    
    # RAG 시스템 초기화
    rag = LiteratureRAG(db_path)
    
    # 인덱스 저장
    rag.save_index(db_path / 'faiss_index.bin')
    
    # 지식 추출
    extractor = KnowledgeExtractor(rag)
    extractor.generate_summary_report(db_path / 'knowledge_extraction_report.json')
    
    # 완료
    print("\n✓ RAG System ready!")
    print(f"  Papers indexed: {len(rag.papers)}")
    print(f"  Index saved: {db_path / 'faiss_index.bin'}")
    print(f"  Report saved: {db_path / 'knowledge_extraction_report.json'}")
    

if __name__ == '__main__':
    main()
