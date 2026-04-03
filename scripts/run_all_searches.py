"""
Run All Literature Searches
============================
자동으로 모든 검색 쿼리를 실행하여 논문 메타데이터를 수집합니다.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.search_config import SEARCH_QUERIES, EMAIL, TOP_JOURNALS, MIN_PUBLICATION_YEAR
from scripts.pubmed_literature_search import PubMedSearcher

def main():
    print("="*70)
    print("  암 기전 지식 베이스 - 자동 문헌 검색")
    print("="*70)
    print()
    
    # 설정 확인
    if EMAIL == "your_email@example.com":
        print("❌ 오류: config/search_config.py에서 EMAIL을 설정하세요!")
        print("   NCBI 정책상 이메일 주소가 필수입니다.")
        return
    
    print(f"📧 Email: {EMAIL}")
    print(f"📚 총 검색 쿼리: {len(SEARCH_QUERIES)}개")
    print(f"📅 검색 기간: {MIN_PUBLICATION_YEAR}년 이후")
    print(f"🏛️  필터 저널: {len(TOP_JOURNALS)}개 고품질 저널")
    print()
    
    # 검색 실행
    searcher = PubMedSearcher()
    all_metadata = []
    
    for i, search_item in enumerate(SEARCH_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(SEARCH_QUERIES)}] {search_item['description']}")
        print(f"{'='*70}")
        print(f"Query: {search_item['query']}")
        print()
        
        # PubMed 검색
        pmids = searcher.search_pubmed(
            query=search_item['query'],
            max_results=search_item['max_results'],
            min_year=MIN_PUBLICATION_YEAR,
            journal_filter=TOP_JOURNALS
        )
        
        # 메타데이터 수집
        if pmids:
            metadata_list = searcher.fetch_metadata(pmids)
            
            # 검색 쿼리 정보 추가
            for meta in metadata_list:
                meta['search_query'] = search_item['description']
                meta['query_category'] = search_item['query']
            
            all_metadata.extend(metadata_list)
            print(f"✅ {len(metadata_list)}편 수집 완료")
        else:
            print("⚠️  검색 결과 없음")
    
    # 중복 제거 (같은 PMID)
    print(f"\n{'='*70}")
    print("📊 최종 결과 처리")
    print(f"{'='*70}")
    
    unique_metadata = {}
    for meta in all_metadata:
        pmid = meta['pmid']
        if pmid not in unique_metadata:
            unique_metadata[pmid] = meta
    
    final_metadata = list(unique_metadata.values())
    
    print(f"총 수집: {len(all_metadata)}편")
    print(f"중복 제거 후: {len(final_metadata)}편")
    print()
    
    # 저장
    searcher.save_metadata(final_metadata)
    
    # 통계 출력
    print("="*70)
    print("📈 수집 통계")
    print("="*70)
    
    # 연도별 분포
    year_dist = {}
    for meta in final_metadata:
        year = meta.get('publication_year', 'Unknown')
        year_dist[year] = year_dist.get(year, 0) + 1
    
    print("\n📅 연도별 분포:")
    for year in sorted(year_dist.keys(), reverse=True):
        print(f"  {year}: {year_dist[year]}편")
    
    # 저널별 분포 (상위 10개)
    journal_dist = {}
    for meta in final_metadata:
        journal = meta.get('journal', 'Unknown')
        journal_dist[journal] = journal_dist.get(journal, 0) + 1
    
    print("\n🏛️  저널별 분포 (상위 10개):")
    sorted_journals = sorted(journal_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    for journal, count in sorted_journals:
        print(f"  {journal}: {count}편")
    
    print(f"\n{'='*70}")
    print("✅ 모든 검색 완료!")
    print(f"{'='*70}")
    print(f"\n💾 저장 위치: data/literature/paper_metadata.json")
    print(f"📚 총 논문 수: {len(final_metadata)}편")
    print()
    print("다음 단계:")
    print("1. data/literature/paper_metadata.json 파일 확인")
    print("2. PDF 다운로드 (가능한 논문)")
    print("3. GPT-4로 지식 추출 시작")
    print()


if __name__ == "__main__":
    main()
