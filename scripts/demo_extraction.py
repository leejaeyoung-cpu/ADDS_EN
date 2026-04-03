"""
Demo: Knowledge Extraction from Sample Papers
==============================================
샘플 논문에서 지식 추출 프로세스 시연
"""

import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

def extract_from_abstract(paper_metadata: dict) -> dict:
    """논문 초록에서 구조화된 지식 추출"""
    
    abstract = paper_metadata['abstract']
    title = paper_metadata['title']
    
    extraction_prompt = f"""
다음 논문에서 구조화된 정보를 추출하세요:

제목: {title}
초록: {abstract}

다음 JSON 형식으로 반환:
{{
  "mechanisms": [
    {{
      "pathway_name": "경로 이름",
      "category": "카테고리 (예: Growth Signaling, DNA Damage, Angiogenesis)",
      "description": "상세 설명",
      "key_proteins": ["단백질 목록"],
      "regulation_type": "Activation 또는 Inhibition"
    }}
  ],
  "drugs": [
    {{
      "drug_name": "약물 이름",
      "generic_name": "일반명",
      "drug_class": "약물 분류",
      "mechanism_of_action": "작용 기전",
      "molecular_target": "분자 표적",
      "pathways_affected": ["영향받는 경로들"]
    }}
  ],
  "drug_interactions": [
    {{
      "drug1": "약물1",
      "drug2": "약물2",
      "interaction_type": "Synergy/Antagonism/Additive",
      "synergy_score": 0.65,
      "mechanism_basis": "상호작용 기전"
    }}
  ],
  "biomarkers": [
    {{
      "name": "바이오마커 이름",
      "type": "Genetic/Protein/Functional",
      "measurement": "측정 방법",
      "predictive_value": "예측 가치",
      "drug_associations": {{"약물명": "연관성"}}
    }}
  ],
  "clinical_findings": {{
    "key_result": "핵심 결과",
    "statistical_significance": "통계적 유의성",
    "clinical_relevance": "임상적 의미"
  }}
}}

중요: 초록에 명시된 정보만 추출하세요.
"""
    
    print(f"🤖 GPT-4로 추출 중: {title[:60]}...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert oncologist extracting structured cancer mechanism data from scientific literature. Always return valid JSON in Korean."
                },
                {
                    "role": "user",
                    "content": extraction_prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        extracted = json.loads(response.choices[0].message.content)
        
        # 메타데이터 추가
        extracted['source'] = {
            'pmid': paper_metadata['pmid'],
            'title': paper_metadata['title'],
            'journal': paper_metadata['journal'],
            'year': paper_metadata['publication_year'],
            'doi': paper_metadata.get('doi')
        }
        
        return extracted
        
    except Exception as e:
        print(f"❌ 추출 실패: {e}")
        return {}


def main():
    print("="*70)
    print("  GPT-4 Knowledge Extraction Demo")
    print("="*70)
    print()
    
    # Load sample papers
    metadata_file = Path("data/literature/paper_metadata.json")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"Total {len(papers)} sample papers")
    print()
    
    # 각 논문에서 지식 추출
    all_extractions = []
    
    for i, paper in enumerate(papers, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(papers)}] {paper['title']}")
        print(f"{'='*70}")
        
        extracted = extract_from_abstract(paper)
        
        if extracted:
            all_extractions.append(extracted)
            
            # 추출 결과 요약
            print(f"✅ 추출 완료:")
            print(f"   - 기전: {len(extracted.get('mechanisms', []))}개")
            print(f"   - 약물: {len(extracted.get('drugs', []))}개")
            print(f"   - 상호작용: {len(extracted.get('drug_interactions', []))}개")
            print(f"   - 바이오마커: {len(extracted.get('biomarkers', []))}개")
    
    # 결과 저장
    output_dir = Path("data/extracted")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "sample_extracted_knowledge.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_extractions, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("✅ 추출 완료!")
    print(f"{'='*70}")
    print(f"\n💾 저장 위치: {output_file}")
    print(f"📊 총 {len(all_extractions)}편 처리 완료")
    print()
    
    # 통계
    total_mechanisms = sum(len(e.get('mechanisms', [])) for e in all_extractions)
    total_drugs = sum(len(e.get('drugs', [])) for e in all_extractions)
    total_interactions = sum(len(e.get('drug_interactions', [])) for e in all_extractions)
    total_biomarkers = sum(len(e.get('biomarkers', [])) for e in all_extractions)
    
    print("📈 추출 통계:")
    print(f"   - 총 기전: {total_mechanisms}개")
    print(f"   - 총 약물: {total_drugs}개")
    print(f"   - 총 상호작용: {total_interactions}개")
    print(f"   - 총 바이오마커: {total_biomarkers}개")
    print()
    
    print("다음 단계:")
    print("1. data/extracted/sample_extracted_knowledge.json 확인")
    print("2. 데이터베이스에 로드")
    print("3. CDSS 통합 테스트")
    print()


if __name__ == "__main__":
    main()
