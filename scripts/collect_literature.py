"""
Literature Collection System
대규모 논문 수집 및 텍스트 데이터베이스 구축

목표: 400+ 논문에서 물리학 모델 구축을 위한 데이터 추출
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import requests
from xml.etree import ElementTree as ET

# PubMed API
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PUBMED_API_KEY = os.getenv('PUBMED_API_KEY', '')  # 선택사항

# ArXiv API
ARXIV_BASE = "http://export.arxiv.org/api/query"

# 저장 경로
OUTPUT_DIR = Path("F:/ADDS/literature_database")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class LiteratureCollector:
    """논문 수집기"""
    
    def __init__(self):
        self.papers = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ADDS-Research-Tool/1.0 (research; medical-ai)'
        })
    
    def search_pubmed(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        PubMed 검색
        
        주요 키워드:
        - drug thermodynamics
        - signal transduction energy
        - dose-response modeling
        - tumor growth kinetics
        """
        print(f"Searching PubMed: {query}")
        
        # 1. 검색 (논문 ID 가져오기)
        search_url = f"{PUBMED_BASE}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if PUBMED_API_KEY:
            params['api_key'] = PUBMED_API_KEY
        
        response = self.session.get(search_url, params=params)
        data = response.json()
        
        id_list = data.get('esearchresult', {}).get('idlist', [])
        print(f"Found {len(id_list)} papers")
        
        if not id_list:
            return []
        
        # 2. 상세 정보 가져오기
        time.sleep(0.34)  # API rate limit: 3 requests/sec
        
        fetch_url = f"{PUBMED_BASE}efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(id_list[:max_results]),
            'retmode': 'xml'
        }
        
        response = self.session.get(fetch_url, params=params)
        papers = self.parse_pubmed_xml(response.text)
        
        return papers
    
    def parse_pubmed_xml(self, xml_text: str) -> List[Dict]:
        """PubMed XML 파싱"""
        papers = []
        root = ET.fromstring(xml_text)
        
        for article in root.findall('.//PubmedArticle'):
            try:
                # PMID
                pmid = article.find('.//PMID').text
                
                # 제목
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else 'No title'
                
                # 초록
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                # 저자
                authors = []
                for author in article.findall('.//Author'):
                    lastname = author.find('.//LastName')
                    forename = author.find('.//ForeName')
                    if lastname is not None and forename is not None:
                        authors.append(f"{forename.text} {lastname.text}")
                
                # 저널
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else ''
                
                # 출판 연도
                year_elem = article.find('.//PubDate/Year')
                year = year_elem.text if year_elem is not None else ''
                
                # DOI
                doi_elem = article.find('.//ELocationID[@EIdType="doi"]')
                doi = doi_elem.text if doi_elem is not None else ''
                
                papers.append({
                    'pmid': pmid,
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'journal': journal,
                    'year': year,
                    'doi': doi,
                    'source': 'pubmed'
                })
            
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue
        
        return papers
    
    def search_arxiv(self, query: str, max_results: int = 100) -> List[Dict]:
        """ArXiv 검색 (물리/수학 논문)"""
        print(f"Searching arXiv: {query}")
        
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = self.session.get(ARXIV_BASE, params=params)
        papers = self.parse_arxiv_xml(response.text)
        
        return papers
    
    def parse_arxiv_xml(self, xml_text: str) -> List[Dict]:
        """ArXiv XML 파싱"""
        papers = []
        
        # Namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(xml_text)
        
        for entry in root.findall('atom:entry', ns):
            try:
                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                title = entry.find('atom:title', ns).text.strip()
                summary = entry.find('atom:summary', ns).text.strip()
                
                # 저자
                authors = [
                    author.find('atom:name', ns).text
                    for author in entry.findall('atom:author', ns)
                ]
                
                # 출판일
                published = entry.find('atom:published', ns).text[:4]
                
                # PDF 링크
                pdf_link = None
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        pdf_link = link.get('href')
                
                papers.append({
                    'arxiv_id': arxiv_id,
                    'title': title,
                    'abstract': summary,
                    'authors': authors,
                    'year': published,
                    'pdf_url': pdf_link,
                    'source': 'arxiv'
                })
            
            except Exception as e:
                print(f"Error parsing arXiv entry: {e}")
                continue
        
        return papers
    
    def save_papers(self, papers: List[Dict], filename: str):
        """논문 저장"""
        output_path = OUTPUT_DIR / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(papers)} papers to {output_path}")
        
        # 텍스트 버전도 저장 (RAG용)
        text_path = output_path.with_suffix('.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(f"="*80 + "\n")
                f.write(f"Title: {paper.get('title', 'N/A')}\n")
                f.write(f"Authors: {', '.join(paper.get('authors', []))}\n")
                f.write(f"Year: {paper.get('year', 'N/A')}\n")
                f.write(f"Journal: {paper.get('journal', paper.get('source', 'N/A'))}\n")
                f.write(f"DOI/ID: {paper.get('doi', paper.get('arxiv_id', 'N/A'))}\n")
                f.write(f"\nAbstract:\n{paper.get('abstract', 'N/A')}\n")
                f.write(f"="*80 + "\n\n")
        
        print(f"Saved text version to {text_path}")


def main():
    """메인 수집 스크립트"""
    
    collector = LiteratureCollector()
    
    # 검색 쿼리 (카테고리별)
    queries = {
        'drug_thermodynamics': [
            'drug receptor binding thermodynamics',
            'ligand binding free energy',
            'protein-drug interaction energetics',
            'dissociation constant Kd kinetics'
        ],
        'signal_pathways': [
            'EGFR signal transduction energy barrier',
            'VEGF pathway activation kinetics',
            'mTOR inhibitor mechanism thermodynamics',
            'tyrosine kinase inhibitor binding energy'
        ],
        'dose_response': [
            'dose response relationship mathematical model',
            'pharmacodynamics Hill equation',
            'EC50 IC50 drug concentration',
            'chemotherapy dose escalation model'
        ],
        'tumor_kinetics': [
            'tumor growth kinetics Gompertz',
            'cancer cell proliferation rate constant',
            'tumor volume doubling time',
            'logistic growth model cancer'
        ],
        'chemotherapy_physics': [
            'chemotherapy efficacy mathematical modeling',
            'cytotoxic drug mechanism physics',
            'tumor regression kinetics',
            'apoptosis rate constant chemotherapy'
        ]
    }
    
    all_papers = []
    
    # PubMed 검색
    print("\n" + "="*80)
    print("PHASE 1: PubMed Search")
    print("="*80 + "\n")
    
    for category, query_list in queries.items():
        print(f"\nCategory: {category}")
        category_papers = []
        
        for query in query_list:
            try:
                papers = collector.search_pubmed(query, max_results=50)
                category_papers.extend(papers)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error: {e}")
        
        # 중복 제거 (PMID 기준)
        seen_ids = set()
        unique_papers = []
        for paper in category_papers:
            paper_id = paper.get('pmid')
            if paper_id and paper_id not in seen_ids:
                seen_ids.add(paper_id)
                unique_papers.append(paper)
        
        print(f"Total unique papers for {category}: {len(unique_papers)}")
        
        # 저장
        collector.save_papers(unique_papers, f'{category}_pubmed.json')
        all_papers.extend(unique_papers)
    
    # ArXiv 검색
    print("\n" + "="*80)
    print("PHASE 2: ArXiv Search")
    print("="*80 + "\n")
    
    arxiv_queries = [
        'thermodynamics drug binding',
        'tumor growth mathematical model',
        'pharmacokinetics differential equations',
        'cancer treatment optimization'
    ]
    
    arxiv_papers = []
    for query in arxiv_queries:
        try:
            papers = collector.search_arxiv(query, max_results=25)
            arxiv_papers.extend(papers)
            time.sleep(3)  # ArXiv rate limit
        except Exception as e:
            print(f"Error: {e}")
    
    collector.save_papers(arxiv_papers, 'arxiv_physics_models.json')
    all_papers.extend(arxiv_papers)
    
    # 전체 저장
    collector.save_papers(all_papers, 'all_papers_combined.json')
    
    # 통계
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    print(f"Total papers collected: {len(all_papers)}")
    print(f"PubMed: {len([p for p in all_papers if p.get('source') == 'pubmed'])}")
    print(f"ArXiv: {len([p for p in all_papers if p.get('source') == 'arxiv'])}")
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
