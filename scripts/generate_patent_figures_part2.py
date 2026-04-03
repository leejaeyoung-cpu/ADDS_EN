# -*- coding: utf-8 -*-
"""
특허 출원서용 추가 도면 생성 (Part 2)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path
import sys

# UTF-8 출력 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("c:/Users/brook/Desktop/ADDS/docs/patent_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'warning': '#D00000',
    'light': '#E8E9EB',
    'dark': '#2C2C2C'
}


def figure_2_cell_analysis_flowchart():
    """도 2: 세포 분석 모듈 순서도"""
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 제목
    ax.text(5, 13.5, '도 2. 세포 분석 모듈 처리 흐름', 
            ha='center', fontsize=18, fontweight='bold')
    
    y = 12.5
    box_width = 4
    box_height = 0.6
    x_center = 5
    
    # 단계 1: 입력
    ax.add_patch(FancyBboxPatch((x_center-box_width/2, y-box_height), box_width, box_height,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['primary'], alpha=0.3,
                                edgecolor=COLORS['primary'], linewidth=2))
    ax.text(x_center, y-box_height/2, '입력: 병리 이미지\n(2048x2048 TIFF)', 
            ha='center', va='center', fontsize=10)
    
    # 화살표
    y -= 1.2
    ax.arrow(x_center, y+0.5, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # 단계 2: 전처리
    ax.add_patch(FancyBboxPatch((x_center-box_width/2, y-box_height), box_width, box_height,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['secondary'], alpha=0.3,
                                edgecolor=COLORS['secondary'], linewidth=2))
    ax.text(x_center, y-box_height/2, '전처리 (121a)\n정규화 + CLAHE + 잡음 제거', 
            ha='center', va='center', fontsize=10)
    
    y -= 1.2
    ax.arrow(x_center, y+0.5, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # 단계 3: Cellpose 분할
    ax.add_patch(FancyBboxPatch((x_center-box_width/2, y-box_height*1.5), box_width, box_height*1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['accent'], alpha=0.4,
                                edgecolor=COLORS['accent'], linewidth=2))
    ax.text(x_center, y-box_height*0.75, 'Cellpose 분할 (121b)\n흐름 벡터 (Fx, Fy)\n세포 마스크 생성', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    y -= 2.0
    ax.arrow(x_center, y+0.5, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # 단계 4: 특징 추출
    ax.add_patch(FancyBboxPatch((x_center-box_width/2, y-box_height*2), box_width, box_height*2,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['success'], alpha=0.3,
                                edgecolor=COLORS['success'], linewidth=2))
    ax.text(x_center, y-box_height, '특징 추출 (121c)\n• 형태학적 (6개): 면적, 원형도...\n• 강도 (7개): 평균, 표준편차...\n• 텍스처 (4개): GLCM\n• 공간 (5개): 밀도, K-함수', 
            ha='center', va='center', fontsize=9)
    
    y -= 2.5
    ax.arrow(x_center, y+0.5, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # 단계 5: Ki-67 계산
    ax.add_patch(FancyBboxPatch((x_center-box_width/2, y-box_height), box_width, box_height,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['warning'], alpha=0.3,
                                edgecolor=COLORS['warning'], linewidth=2))
    ax.text(x_center, y-box_height/2, 'Ki-67 산출 (121d)\n고강도 핵 비율(%)', 
            ha='center', va='center', fontsize=10)
    
    y -= 1.2
    ax.arrow(x_center, y+0.5, 0, -0.3, head_width=0.2, head_length=0.1, fc='black', ec='black')
    
    # 출력
    ax.add_patch(FancyBboxPatch((x_center-box_width/2, y-box_height*1.5), box_width, box_height*1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['primary'], alpha=0.3,
                                edgecolor=COLORS['primary'], linewidth=2))
    ax.text(x_center, y-box_height*0.75, '출력\nKi-67: 38%\n25개 형태학적 특징', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 처리 시간 표시
    ax.text(8.5, 1.5, 'GPU 처리 시간:\n7.2초', 
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.3, edgecolor=COLORS['success']))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_02_cell_flowchart.png', dpi=300, bbox_inches='tight')
    print(f"[완료] 도 2 생성: {OUTPUT_DIR / 'figure_02_cell_flowchart.png'}")
    plt.close()


def figure_4_drug_cocktail_synergy():
    """도 4: 약물 칵테일 4-모델 시너지 계산"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 제목
    ax.text(7, 9.5, '도 4. 약물 칵테일 4-모델 시너지 계산 및 합의 알고리즘', 
            ha='center', fontsize=18, fontweight='bold')
    
    # 입력약물
    y = 8.5
    drug_a_x = 2
    drug_b_x = 5
    
    ax.add_patch(Circle((drug_a_x, y), 0.4, facecolor=COLORS['primary'], alpha=0.5, edgecolor='black', linewidth=2))
    ax.text(drug_a_x, y, '약물 A\n5-FU', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(Circle((drug_b_x, y), 0.4, facecolor=COLORS['secondary'], alpha=0.5, edgecolor='black', linewidth=2))
    ax.text(drug_b_x, y, '약물 B\nOxali', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 4개 모델
    y = 6.5
    model_y = y
    models = [
        ('Bliss\nIndependence', 2, '+0.18'),
        ('Loewe\nAdditivity', 5, '-0.12'),
        ('HSA', 8, '+0.25'),
        ('ZIP', 11, '+0.15')
    ]
    
    model_colors = [COLORS['success'], COLORS['warning'], COLORS['success'], COLORS['success']]
    
    for i, (name, x, score) in enumerate(models):
        color = model_colors[i]
        ax.add_patch(Rectangle((x-0.8, model_y-0.5), 1.6, 1, 
                               facecolor=color, alpha=0.3, 
                               edgecolor='black', linewidth=2))
        ax.text(x, model_y+0.2, name, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x, model_y-0.2, f'점수:\n{score}', ha='center', va='center', fontsize=8)
        
        # 입력에서 모델로 화살표
        ax.arrow(3.5, 8, x-3.5, model_y+0.5-8, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', alpha=0.5)
    
    # 합의 알고리즘
    y = 4.5
    ax.add_patch(FancyBboxPatch((4, y-1), 6, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['accent'], alpha=0.4,
                                edgecolor=COLORS['accent'], linewidth=3))
    ax.text(7, y-0.25, '합의 알고리즘\n양수 모델: 3/4 (75%)\n평균 점수: +0.115', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 모델에서 합의로 화살표
    for x in [2, 5, 8, 11]:
        ax.arrow(x, model_y-0.6, 7-x, y-0.5-(model_y-0.6), 
                head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    # 최종 판정
    y = 2.5
    ax.add_patch(FancyBboxPatch((4, y-0.8), 6, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['success'], alpha=0.5,
                                edgecolor=COLORS['success'], linewidth=3))
    ax.text(7, y-0.4, '최종 판정: "Synergy" (시너지 효과)', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    
    # 효능 가중치
    y = 1.2
    ax.text(7, y, '예상 효능: 50% × (1 + 0.15×0.115) = 50.86%', 
            ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['accent'], linewidth=2))
    
    # 범례
    ax.text(1, 0.3, '파란색: 양수 시너지 | 빨간색: 음수/길항', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_04_drug_synergy.png', dpi=300, bbox_inches='tight')
    print(f"[완료] 도 4 생성: {OUTPUT_DIR / 'figure_04_drug_synergy.png'}")
    plt.close()


def figure_10_docker_deployment():
    """도 10: Docker 배포 구조"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 제목
    ax.text(7, 9.5, '도 10. Docker 기반 배포 아키텍처', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Docker 컨테이너
    y = 7.5
    ax.add_patch(Rectangle((1, y-3), 12, 3,
                           facecolor=COLORS['light'], alpha=0.5,
                           edgecolor=COLORS['dark'], linewidth=3, linestyle='--'))
    ax.text(1.5, y+0.3, 'Docker Container (ADDS)', fontsize=12, fontweight='bold')
    
    # 내부 구성 요소
    components_y = y - 0.8
    
    # FastAPI
    ax.add_patch(Rectangle((2, components_y-0.5), 2.5, 0.6,
                           facecolor=COLORS['primary'], alpha=0.4,
                           edgecolor=COLORS['primary'], linewidth=2))
    ax.text(3.25, components_y-0.2, 'FastAPI\nBackend', ha='center', va='center', fontsize=10)
    
    # Streamlit
    ax.add_patch(Rectangle((5, components_y-0.5), 2.5, 0.6,
                           facecolor=COLORS['secondary'], alpha=0.4,
                           edgecolor=COLORS['secondary'], linewidth=2))
    ax.text(6.25, components_y-0.2, 'Streamlit\nFrontend', ha='center', va='center', fontsize=10)
    
    # Gunicorn
    ax.add_patch(Rectangle((8, components_y-0.5), 2.5, 0.6,
                           facecolor=COLORS['accent'], alpha=0.4,
                           edgecolor=COLORS['accent'], linewidth=2))
    ax.text(9.25, components_y-0.2, 'Gunicorn\n4 workers', ha='center', va='center', fontsize=10)
    
    # 모델 레이어
    components_y -= 1.2
    ax.add_patch(Rectangle((2, components_y-0.5), 8.5, 0.6,
                           facecolor=COLORS['success'], alpha=0.4,
                           edgecolor=COLORS['success'], linewidth=2))
    ax.text(6.25, components_y-0.2, 'AI Models: Cellpose (17M params) + CT Detector (25M params)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # NVIDIA Container Toolkit
    y = 3.5
    ax.add_patch(Rectangle((1, y-0.8), 12, 0.8,
                           facecolor='#76B900', alpha=0.3,
                           edgecolor='#76B900', linewidth=2))
    ax.text(7, y-0.4, 'NVIDIA Container Toolkit (GPU 접근)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 호스트 GPU
    y = 2.0
    ax.add_patch(Rectangle((1, y-0.8), 12, 0.8,
                           facecolor=COLORS['warning'], alpha=0.3,
                           edgecolor=COLORS['warning'], linewidth=2))
    ax.text(7, y-0.4, '호스트 GPU (CUDA 12.1)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # API 엔드포인트
    api_y = 0.5
    endpoints = ['/api/v1/segmentation', '/api/v1/tumor_detection', '/api/v1/cdss/integrate']
    for i, endpoint in enumerate(endpoints):
        x = 2 + i * 3.5
        ax.text(x, api_y, endpoint, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    # 화살표들
    ax.arrow(7, 7.5-3.1, 0, -0.3, head_width=0.3, head_length=0.1, fc='black', ec='black', linewidth=2)
    ax.arrow(7, 3.5-0.9, 0, -0.3, head_width=0.3, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # 처리량 표시
    ax.text(13.5, 7, '처리량:\n80 req/s', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_10_docker.png', dpi=300, bbox_inches='tight')
    print(f"[완료] 도 10 생성: {OUTPUT_DIR / 'figure_10_docker.png'}")
    plt.close()


def figure_3_ct_detection():
    """도 3: CT 검출 다중 임계값"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 왼쪽: 다중 임계값 과정
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('다중 임계값 종양 후보 검출', fontsize=14, fontweight='bold', pad=20)
    
    thresholds = [-50, 0, 50, 100, 150, 200]
    y_start = 9
    for i, thresh in enumerate(thresholds):
        y = y_start - i * 1.4
        color_alpha = 0.2 + (i * 0.1)
        
        ax1.add_patch(Rectangle((1, y-0.5), 3, 0.5,
                               facecolor=COLORS['primary'], alpha=color_alpha,
                               edgecolor='black', linewidth=1))
        ax1.text(2.5, y-0.25, f'{thresh} HU', ha='center', va='center', fontsize=10)
        
        # 후보 개수
        candidates = max(0, 50 - abs(thresh - 50) // 10)
        ax1.text(5.5, y-0.25, f'→ {candidates}개 후보', fontsize=9)
    
    # 오른쪽: 신뢰도 스코어링
    ax2.set_title('신뢰도 점수 계산', fontsize=14, fontweight='bold', pad=20)
    
    criteria = ['크기\n(50-500mm²)', '형상\n(불규칙)', '강도\n(20-80 HU)', '해부학적\n위치']
    weights = [0.3, 0.2, 0.3, 0.2]
    
    x_pos = np.arange(len(criteria))
    bars = ax2.bar(x_pos, weights, color=[COLORS['success'], COLORS['primary'], 
                                          COLORS['accent'], COLORS['secondary']], 
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        ax2.text(i, weight + 0.02, f'{weight:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(criteria, fontsize=10)
    ax2.set_ylabel('가중치', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 0.4)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_title('신뢰도 = Σ(기준 점수 × 가중치)', fontsize=11, pad=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_03_ct_detection.png', dpi=300, bbox_inches='tight')
    print(f"[완료] 도 3 생성: {OUTPUT_DIR / 'figure_03_ct_detection.png'}")
    plt.close()


if __name__ == "__main__":
    print("[추가 도면 생성 시작...]\n")
    
    figure_2_cell_analysis_flowchart()
    figure_3_ct_detection()
    figure_4_drug_cocktail_synergy()
    figure_10_docker_deployment()
    
    print("\n[완료] 모든 추가 도면 생성 완료!")
    print(f"[저장 위치] {OUTPUT_DIR}")
