# ADDS 출판 준비 - 이번 주 실행 계획

## 🎯 목표
Nature Communications 출판을 위한 첫 단계 실행

## 📋 이번 주 Task (우선순위 순)

### 1. K-BDS 데이터 재탐색 ⭐⭐⭐
**시간**: 2-3시간

- [ ] K-BDS 포털 재접속
  - URL: https://kbds.re.kr
  - 계정 확인
- [ ] 대장암/췌장암 데이터셋 검색
  - 병리 이미지 유무
  - PRNP 발현 데이터
  - 임상 정보
- [ ] 사용 가능 데이터 목록 작성
- [ ] 데이터 신청 절차 확인

**Output**: K-BDS 데이터 접근 계획

### 2. 내부 성능 측정 ⭐⭐⭐
**시간**: 3-4시간

- [ ] 기존 병리 분석 결과 수집
  - 저장된 분석 결과 검토
  - 샘플 수 확인 (N=?)
- [ ] Dice Coefficient 계산
  - Ground truth 필요
  - 또는 추정 방법
- [ ] 벤치마크 스크립트 실행
  ```bash
  python scripts/benchmark_performance.py
  ```
- [ ] 첫 벤치마크 리포트 작성

**Output**: `benchmark_report_v1.json`

### 3. 논문 Outline 초안 ⭐⭐
**시간**: 2-3시간

- [ ] Abstract 초안 (200 words)
  - Problem
  - Method
  - Results (예상)
  - Conclusion
- [ ] Introduction 구조
  - Background
  - Current challenges
  - Our approach
  - Contributions
- [ ] Methods 섹션 설계
  - System architecture
  - Data sources
  - Validation protocol

**Output**: `paper_outline_v1.md`

### 4. 데이터 정리 ⭐
**시간**: 1-2시간

- [ ] 기존 분석 결과 정리
  - `data/outputs/` 확인
  - 메타데이터 수집
- [ ] 샘플 데이터 선별
  - 대표 사례 10-20개
  - Figure용 고품질 이미지

**Output**: 정리된 샘플 데이터셋

## 📊 예상 성과

### 이번 주 말까지
- K-BDS 데이터 접근 계획 확정
- 내부 성능 첫 측정 완료
- 논문 구조 초안 완성
- 샘플 데이터셋 준비

### 다음 주 목표
- K-BDS 데이터 신청 제출
- 내부 데이터 N=50 목표
- 논문 Introduction 작성 시작

## 🚧 Blockers & Solutions

**Blocker 1**: Ground truth annotation 없음  
→ **Solution**: 기존 Cellpose 결과를 사용, 또는 간단한 annotation 수행

**Blocker 2**: K-BDS 데이터 접근 시간  
→ **Solution**: 공개 데이터셋(TCGA) 병행 탐색

**Blocker 3**: 협력자 부족  
→ **Solution**: 1저자로 진행, 리뷰 단계에서 협력 모색

## 📝 Notes

- 완벽함보다 진행이 중요
- 데이터 품질 > 데이터 양
- 정기적 진행 상황 리뷰 (주 1회)
