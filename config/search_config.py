# Cancer Knowledge Base - Search Configuration
# 암 기전 지식 베이스를 위한 PubMed 검색 설정

# ===================================
# STEP 1: PubMed 검색 설정
# ===================================

EMAIL = "your_email@example.com"  # ⚠️ 필수: 본인 이메일로 변경하세요!
API_KEY = None  # 선택: NCBI API key (더 빠른 검색을 원하면 https://www.ncbi.nlm.nih.gov/account/ 에서 발급)

# ===================================
# 검색 쿼리 목록
# ===================================

SEARCH_QUERIES = [
    # 1. 대장암 기본 기전
    {
        "query": "colorectal cancer signaling pathways mechanisms",
        "max_results": 50,
        "description": "대장암 신호전달 경로 기전"
    },
    
    # 2. 약물 내성
    {
        "query": "colorectal cancer drug resistance mechanisms acquired",
        "max_results": 30,
        "description": "대장암 약물 내성 메커니즘"
    },
    
    # 3. FOLFOX 시너지
    {
        "query": "FOLFOX combination therapy synergy colorectal cancer",
        "max_results": 20,
        "description": "FOLFOX 조합 치료 시너지"
    },
    
    # 4. 표적 치료
    {
        "query": "EGFR inhibitor cetuximab KRAS colorectal cancer",
        "max_results": 20,
        "description": "EGFR 억제제와 KRAS"
    },
    
    # 5. 혈관신생 억제
    {
        "query": "bevacizumab VEGF angiogenesis colorectal cancer",
        "max_results": 15,
        "description": "혈관신생 억제 치료"
    },
    
    # 6. 면역치료
    {
        "query": "PD-1 pembrolizumab MSI-H colorectal cancer immunotherapy",
        "max_results": 20,
        "description": "MSI-H 대장암 면역치료"
    },
    
    # 7. RAS 경로
    {
        "query": "RAS RAF MEK ERK pathway colorectal cancer therapy",
        "max_results": 15,
        "description": "RAS/RAF/MEK/ERK 경로"
    },
    
    # 8. PI3K/AKT 경로
    {
        "query": "PI3K AKT mTOR pathway colorectal cancer targeted therapy",
        "max_results": 15,
        "description": "PI3K/AKT/mTOR 경로 표적 치료"
    },
    
    # 9. 세포 주기 조절
    {
        "query": "cell cycle checkpoint inhibition cancer therapy",
        "max_results": 15,
        "description": "세포 주기 체크포인트 억제"
    },
    
    # 10. 약물 조합 최적화
    {
        "query": "chemotherapy combination optimization colorectal cancer",
        "max_results": 20,
        "description": "항암제 조합 최적화"
    }
]

# 총 예상 논문 수: 220편

# ===================================
# 고품질 저널 필터
# ===================================

TOP_JOURNALS = [
    # Nature 계열
    "Nature",
    "Nature Medicine", 
    "Nature Cancer",
    "Nature Reviews Cancer",
    "Nature Communications",
    
    # Cell 계열
    "Cell",
    "Cancer Cell",
    "Cell Reports",
    
    # 일반 의학
    "Science",
    "New England Journal of Medicine",
    "Lancet",
    "Lancet Oncology",
    "JAMA",
    "JAMA Oncology",
    
    # 종양학 전문
    "Journal of Clinical Oncology",
    "Cancer Research",
    "Clinical Cancer Research",
    "Molecular Cancer Therapeutics",
    "Cancer Discovery",
    "Annals of Oncology"
]

# ===================================
# 검색 필터 기준
# ===================================

MIN_PUBLICATION_YEAR = 2019  # 최근 5년 논문
MAX_RESULTS_TOTAL = 220      # 총 논문 수 목표
MIN_IMPACT_FACTOR = 5.0      # 최소 임팩트 팩터 (선택적)

# ===================================
# 실행 설정
# ===================================

# 자동으로 모든 검색 실행
AUTO_RUN_ALL = True

# PDF 다운로드 시도 (PubMed Central Open Access만 가능)
DOWNLOAD_PDFS = False  # True로 변경하면 자동 다운로드 시도

# ===================================
# 데이터 저장 경로
# ===================================

OUTPUT_DIR = "data/literature"
METADATA_FILE = "data/literature/paper_metadata.json"
PDF_DIR = "data/literature/pdfs"

# ===================================
# 다음 단계
# ===================================

"""
1. 위의 EMAIL 주소를 본인 이메일로 변경
2. scripts/pubmed_literature_search.py 실행:
   
   python scripts/pubmed_literature_search.py \
     --query "colorectal cancer signaling pathways" \
     --max_results 50
   
3. 또는 모든 검색을 한 번에:
   
   python scripts/run_all_searches.py
   
4. 결과 확인:
   - data/literature/paper_metadata.json
   - 총 220편 정도의 논문 메타데이터
"""
