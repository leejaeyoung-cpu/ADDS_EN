# Contributing to ADDS

ADDS는 인하대학병원 의생명공학과의 내부 연구 프로젝트입니다.

## 코드 스타일

- Python: PEP 8
- Docstring: Google Style
- Type hints 사용 권장

## 테스트

새로운 기능 추가 시 반드시 테스트 작성:

```bash
pytest tests/ -v
```

## Commit 메시지

```
[Module] Brief description

Detailed description...
```

예:
```
[Models] Add GNN drug combination predictor

Implemented Graph Attention Network for modeling drug-drug
interactions and predicting combination efficacy.
```

## Pull Request

1. 기능 브랜치 생성
2. 변경사항 커밋
3. 테스트 통과 확인
4. PR 생성

## 문의

연구팀 내부 채널을 통해 문의해주세요.
