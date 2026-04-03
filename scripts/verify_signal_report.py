# -*- coding: utf-8 -*-
"""수정된 signal pathway 보고서 자동 검증 스크립트"""

with open(r'f:\ADDS\docs\pritamab_signal_pathway_energy_report.txt', 'r', encoding='utf-8') as f:
    txt = f.read()

lines = txt.split('\n')
total = len(lines)

checks = []

# 수정 1: ΔG 부호 체계
checks.append(('DeltaG_bind 용어', 'ΔG_bind' in txt))
checks.append(('DeltaG_barrier 용어', 'ΔG_barrier' in txt))
checks.append(('부호오류 = 18.0 제거', '= 18.0 kcal/mol' not in txt))
checks.append(('ΔG 표기규약 섹션', 'ΔG 표기 규약' in txt))

# 수정 2: Fold 기준 정의
checks.append(('baseline ~25% 정의', '25%' in txt and 'baseline' in txt.lower()))
checks.append(('Fold 계산식 명시', 'Fold' in txt and '25%' in txt))

# 수정 3: Irinotecan Bliss
checks.append(('Irinotecan Bliss +17.3', '+17.3' in txt))
checks.append(('구버전 오류범위 제거', '+18.4 ~ +21.7' not in txt))
checks.append(('수정 주석 존재', 'corrected 2026-03-03' in txt))

# 수정 4: Oxaliplatin ΔG
checks.append(('Oxali ΔG_bind -14.2', '−14.2 kcal/mol' in txt))

# 수정 5: TAS-102 EC50 주석
checks.append(('TAS-102 Trifluridine 주석', 'Trifluridine' in txt))
checks.append(('5-FU 대입 명시', '5-FU EC50' in txt or '5-FU class' in txt))

# 수정 6: RPSA 표기
checks.append(('YIGSR(LRP6) 오기 제거', 'YIGSR (LRP6)' not in txt))
checks.append(('RPSA/37LRP 사용', 'RPSA/37LRP' in txt))

# 수정 7: 3종 조합 투영값 명시
checks.append(('[에너지 모델 투영값] 3회 이상', txt.count('[에너지 모델 투영값]') >= 3))
checks.append(('Panel E 투영 명시', 'Panel E' in txt and '[에너지 모델 투영값]' in txt))

# 수정 8: Bliss 단위 구분
checks.append(('0~1 스케일 주석', '0~1 스케일' in txt))
checks.append(('0~100 스케일 선언', '0~100 스케일' in txt))

# 논문 원문 확인값 유지 검증
checks.append(('5-FU EC50 12000→9032 유지', '12,000' in txt and '9,032' in txt))
checks.append(('Oxali Bliss +21.7 유지', '+21.7' in txt))
checks.append(('5-FU Bliss +18.4 유지', '+18.4' in txt))
checks.append(('PFS mPFS 5.5→8.25 유지', '5.5' in txt and '8.25' in txt))
checks.append(('HR 0.667 유지', '0.667' in txt))
checks.append(('ddG_RLS=0.50 유지', '0.50' in txt and 'ddG_RLS' in txt))
checks.append(('EC50 감소 24.7% 유지', '24.7%' in txt))

print('=' * 65)
print('  수정된 보고서 자동 검증 결과')
print('  파일: pritamab_signal_pathway_energy_report.txt')
print('=' * 65)
passed = 0
failed = 0
for desc, result in checks:
    status = 'PASS' if result else 'FAIL'
    if result:
        passed += 1
    else:
        failed += 1
    mark = 'O' if result else 'X'
    print(f'[{status}] [{mark}]  {desc}')

print()
print(f'총 {len(checks)}개 항목: 통과 {passed} / 실패 {failed}')
print(f'파일 총 라인: {total}줄')
print('=' * 65)

if failed > 0:
    print('\n[실패 항목 상세]')
    for desc, result in checks:
        if not result:
            print(f'  - FAIL: {desc}')
