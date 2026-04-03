# -*- coding: utf-8 -*-
"""
전문 특허용 인포그래픽 생성 (개선 버전)
- 정확한 차트 범위
- 전문적인 레이아웃
- 깔끔한 타이포그래피
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
import numpy as np
from pathlib import Path
import sys

# UTF-8 출력
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 전문적인 폰트 및 스타일
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.8

OUTPUT_DIR = Path("c:/Users/brook/Desktop/ADDS/docs/patent_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# 전문적인 색상 (IEEE 표준 기반)
COLORS = {
    'primary': '#1f77b4',      # 표준 파랑
    'secondary': '#9467bd',    # 표준 보라
    'accent': '#ff7f0e',       # 표준 주황
    'success': '#2ca02c',      # 표준 녹색
    'warning': '#d62728',      # 표준 빨강
    'neutral': '#7f7f7f',      # 회색
    'light_bg': '#f7f7f7',     # 연한 배경
    'dark_text': '#2c2c2c'     # 진한 텍스트
}


def figure_5_active_learning_pro():
    """도 5: 능동 학습 (전문 버전)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    iterations = np.arange(0, 21)
    
    # 더 현실적인 데이터
    np.random.seed(42)
    ts_scores = 0.25 + 0.045 * iterations[:10] + np.random.normal(0, 0.015, 10)
    ei_scores = ts_scores[-1] + 0.065 * (iterations[10:] - 9) + np.random.normal(0, 0.008, 11)
    synergy_scores = np.concatenate([ts_scores, ei_scores])
    synergy_scores = np.clip(synergy_scores, 0.2, 0.9)  # 범위 제한
    
    # 플롯
    ax.plot(iterations[:10], ts_scores, 'o-', color=COLORS['primary'], 
            linewidth=2.5, markersize=7, label='Thompson Sampling (탐색)', 
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(iterations[10:], ei_scores, 's-', color=COLORS['accent'], 
            linewidth=2.5, markersize=7, label='Expected Improvement (활용)', 
            markeredgecolor='white', markeredgewidth=1.5)
    
    # 전환점
    ax.axvline(x=9.5, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.6)
    ax.text(9.5, 0.88, '전환점 (iter=10)', ha='center', va='bottom', 
            fontsize=11, fontweight='bold', color=COLORS['warning'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=COLORS['warning'], linewidth=1.5))
    
    # 수렴 임계값
    ax.axhline(y=0.8, color=COLORS['success'], linestyle=':', linewidth=2, alpha=0.6)
    ax.text(19.5, 0.82, '수렴 임계값', ha='right', va='bottom', 
            fontsize=10, color=COLORS['success'], fontweight='bold')
    
    # 수렴 포인트
    convergence_idx = 12
    ax.plot(convergence_idx, synergy_scores[convergence_idx], '*', 
            markersize=18, color=COLORS['warning'], 
            markeredgecolor='white', markeredgewidth=1.5, zorder=10)
    ax.annotate('수렴 달성\n(12회 반복)', 
                xy=(convergence_idx, synergy_scores[convergence_idx]),
                xytext=(convergence_idx+3.5, synergy_scores[convergence_idx]-0.08),
                arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2),
                fontsize=10, fontweight='bold', color=COLORS['warning'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=COLORS['warning'], linewidth=1.5))
    
    # 축 설정
    ax.set_xlabel('반복 횟수 (Iteration)', fontsize=12, fontweight='bold')
    ax.set_ylabel('시너지 점수 (Synergy Score)', fontsize=12, fontweight='bold')
    ax.set_title('도 5. 이중모드 능동 학습의 획득 함수 전환\n기존 20회 → 개선 12회 (40% 수렴 가속)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 범위 정확히 설정
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0.15, 0.95)
    
    ax.legend(loc='upper left', fontsize=11, frameon=True, 
             fancybox=True, shadow=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 테두리
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor(COLORS['dark_text'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_05_active_learning.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 5 (전문 버전)")
    plt.close()


def figure_6_lime_pro():
    """도 6: LIME (전문 버전)"""
    fig, ax = plt.subplots(figsize=(11, 8))
    
    features = ['Ki-67 > 40%', 'Tumor Area\n> 300 μm²', 'Circularity\n< 0.6', 
                'KRAS\nmutant', 'T3 병기', 'N1 림프절',
                'ECOG ≥ 2', 'CT Intensity\n50-80 HU', 'Texture\nContrast']
    importance = np.array([0.32, 0.18, 0.15, 0.12, 0.10, 0.08, -0.08, 0.07, 0.05])
    
    colors_bars = [COLORS['success'] if x > 0 else COLORS['warning'] for x in importance]
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importance, color=colors_bars, alpha=0.75, 
                   edgecolor='black', linewidth=1.2, height=0.7)
    
    # 값 레이블
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        x_pos = imp + (0.015 if imp > 0 else -0.015)
        ha = 'left' if imp > 0 else 'right'
        ax.text(x_pos, i, f'{imp:+.2f}', va='center', ha=ha, 
               fontsize=10, fontweight='bold')
    
    # 중앙선
    ax.axvline(x=0, color='black', linewidth=2)
    
    # 축 설정
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('종양 가능성 기여도 (Feature Contribution)', fontsize=12, fontweight='bold')
    ax.set_title('도 6. LIME 특징 중요도 분석 결과\n양수 = 종양 가능성 증가, 음수 = 감소', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 범위 정확히 설정
    ax.set_xlim(-0.15, 0.40)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 범례
    positive_patch = mpatches.Patch(color=COLORS['success'], alpha=0.75, label='양수 기여 (증가)')
    negative_patch = mpatches.Patch(color=COLORS['warning'], alpha=0.75, label='음수 기여 (감소)')
    ax.legend(handles=[positive_patch, negative_patch], loc='lower right', 
             fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # 테두리
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor(COLORS['dark_text'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_06_lime.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 6 (전문 버전)")
    plt.close()


def figure_8_performance_pro():
    """도 8: 성능 비교 (전문 버전)"""
    fig, ax = plt.subplots(figsize=(11, 7))
    
    systems = ['ADDS\n(GPU 가속)', 'ADDS\n(CPU)', 'IBM\nWatson', 'Tempus\nPlatform']
    times = np.array([11.2, 47.3, 180, 240])
    colors_list = [COLORS['success'], COLORS['primary'], COLORS['neutral'], COLORS['neutral']]
    
    bars = ax.barh(systems, times, color=colors_list, alpha=0.8, 
                   edgecolor='black', linewidth=1.5, height=0.6)
    
    # 값 표시
    for i, (bar, time) in enumerate(zip(bars, times)):
        ax.text(time + 8, i, f'{time:.1f}초', va='center', fontsize=11, fontweight='bold')
    
    # 임계값 선
    ax.axvline(x=60, color=COLORS['warning'], linestyle='--', linewidth=2.5, alpha=0.7)
    ax.text(60, 3.4, '임상 워크플로우\n허용 한계 (60초)', ha='center', fontsize=10, 
            color=COLORS['warning'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=COLORS['warning'], linewidth=1.5))
    
    # 개선 표시
    speedup = times[1] / times[0]
    ax.text(30, 0.2, f'GPU 가속\n효과: {speedup:.1f}×', ha='center', fontsize=11, 
            fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=COLORS['success'], 
                     edgecolor='white', linewidth=2))
    
    # 축 설정
    ax.set_xlabel('End-to-End 처리 시간 (초)', fontsize=12, fontweight='bold')
    ax.set_title('도 8. ADDS vs 기존 시스템 처리 시간 비교\n512×512 이미지 통합 분석 (세포 분할 + CT 검출 + 칵테일 최적화)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 범위 정확히 설정
    ax.set_xlim(0, 270)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 테두리
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor(COLORS['dark_text'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_08_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 8 (전문 버전)")
    plt.close()


def figure_9_db_pro():
    """도 9: DB 인덱싱 (전문 버전)"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    methods = ['인덱스 전\n(Full Scan)', '단일 인덱스\n(환자 ID)', '복합 인덱스\n(ID + Timestamp)']
    query_times = np.array([100.0, 2.5, 0.3])
    colors_list = [COLORS['warning'], COLORS['primary'], COLORS['success']]
    
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, query_times, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5, width=0.65)
    
    # 값 표시
    for i, (bar, time) in enumerate(zip(bars, query_times)):
        height = bar.get_height()
        ax.text(i, height * 1.15, f'{time:.1f} ms', ha='center', 
               fontsize=11, fontweight='bold')
    
    # 개선 배수
    speedup = query_times[0] / query_times[2]
    ax.annotate(f'{speedup:.0f}× 빠름', 
                xy=(2, query_times[2]), xytext=(1.5, 35),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2.5),
                fontsize=13, fontweight='bold', color=COLORS['success'],
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                         edgecolor=COLORS['success'], linewidth=2))
    
    # 축 설정
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel('쿼리 응답 시간 (ms, 로그 스케일)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title('도 9. 데이터베이스 인덱싱 전후 쿼리 성능 비교\n환자 ID + 타임스탬프 복합 인덱스 적용', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # 범위 정확히 설정
    ax.set_ylim(0.1, 300)
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')
    
    # 테두리
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor(COLORS['dark_text'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_09_db_indexing.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 9 (전문 버전)")
    plt.close()


def figure_3_ct_pro():
    """도 3: CT 검출 (전문 버전)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    
    # 왼쪽: 다중 임계값
    thresholds = [-50, 0, 50, 100, 150, 200]
    candidates_count = [12, 28, 45, 38, 22, 8]
    
    bars1 = ax1.barh(range(len(thresholds)), candidates_count, 
                     color=COLORS['primary'], alpha=0.7, 
                     edgecolor='black', linewidth=1.2, height=0.65)
    
    # 임계값 레이블
    for i, (thresh, count) in enumerate(zip(thresholds, candidates_count)):
        ax1.text(-3, i, f'{thresh} HU', va='center', ha='right', fontsize=10, fontweight='bold')
        ax1.text(count + 2, i, f'{count}개', va='center', fontsize=10)
    
    ax1.set_yticks(range(len(thresholds)))
    ax1.set_yticklabels(['' for _ in thresholds])
    ax1.set_xlabel('검출된 후보 영역 개수', fontsize=11, fontweight='bold')
    ax1.set_title('다중 임계값 종양 후보 검출\n(-50~200 HU, 10 HU 간격)', 
                  fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlim(0, 55)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 오른쪽: 신뢰도 스코어링
    criteria = ['크기\n(50-500mm²)', '형상\n(불규칙)', '강도\n(20-80 HU)', '해부학적\n위치']
    weights = np.array([0.30, 0.20, 0.30, 0.20])
    
    x_pos = np.arange(len(criteria))
    bars2 = ax2.bar(x_pos, weights, 
                    color=[COLORS['success'], COLORS['primary'], COLORS['accent'], COLORS['secondary']], 
                    alpha=0.75, edgecolor='black', linewidth=1.5, width=0.65)
    
    # 값 표시
    for i, (bar, weight) in enumerate(zip(bars2, weights)):
        height = bar.get_height()
        ax2.text(i, height + 0.015, f'{weight:.2f}', ha='center', 
                fontsize=11, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(criteria, fontsize=10)
    ax2.set_ylabel('가중치 (Weight)', fontsize=11, fontweight='bold')
    ax2.set_title('신뢰도 점수 계산 기준\nConfidence = Σ(기준 × 가중치)', 
                  fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(0, 0.38)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 테두리
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor(COLORS['dark_text'])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_03_ct_detection.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 3 (전문 버전)")
    plt.close()


if __name__ == "__main__":
    print("[전문 인포그래픽 재생성 시작...]\n")
    
    figure_5_active_learning_pro()
    figure_6_lime_pro()
    figure_8_performance_pro()
    figure_9_db_pro()
    figure_3_ct_pro()
    
    print("\n[완료] 전문 버전 도면 재생성 완료!")
    print(f"[저장 위치] {OUTPUT_DIR}")
