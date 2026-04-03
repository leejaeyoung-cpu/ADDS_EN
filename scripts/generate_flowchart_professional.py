# -*- coding: utf-8 -*-
"""
세포 분석 순서도 (전문 버전)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = Path("c:/Users/brook/Desktop/ADDS/docs/patent_figures")

COLORS = {
    'input': '#2196F3',
    'process': '#9C27B0',
    'key': '#FF9800',
    'output': '#4CAF50',
    'border': '#424242'
}


def figure_2_cell_flowchart_pro():
    """도 2: 세포 분석 순서도 (전문 버전)"""
    fig = plt.figure(figsize=(11, 13))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # 제목
    ax.text(5.5, 12.6, '도 2. 세포 분석 모듈 처리 흐름', 
            ha='center', fontsize=17, fontweight='bold')
    ax.text(5.5, 12.2, 'Cell Analysis Module Processing Pipeline', 
            ha='center', fontsize=11, style='italic', color='gray')
    
    y = 11.5
    box_w = 5
    box_h = 0.7
    x = 5.5
    
    # 1. 입력
    ax.add_patch(FancyBboxPatch((x-box_w/2, y-box_h), box_w, box_h,
                                boxstyle="round,pad=0.15",
                                facecolor=COLORS['input'], alpha=0.3,
                                edgecolor=COLORS['input'], linewidth=2))
    ax.text(x, y-box_h/2, '입력: 병리 이미지\n2048×2048 TIFF/PNG', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 화살표
    y -= 1.3
    ax.add_patch(FancyArrowPatch((x, y+0.6), (x, y+0.1), 
                                arrowstyle='->', mutation_scale=25, 
                                linewidth=2, color=COLORS['border']))
    
    # 2. 전처리
    ax.add_patch(FancyBboxPatch((x-box_w/2, y-box_h), box_w, box_h,
                                boxstyle="round,pad=0.15",
                                facecolor=COLORS['process'], alpha=0.2,
                                edgecolor=COLORS['process'], linewidth=2))
    ax.text(x, y-box_h/2, '전처리 (121a)\n정규화 + CLAHE + 가우시안 필터', 
            ha='center', va='center', fontsize=9.5)
    
    y -= 1.3
    ax.add_patch(FancyArrowPatch((x, y+0.6), (x, y+0.1), 
                                arrowstyle='->', mutation_scale=25, 
                                linewidth=2, color=COLORS['border']))
    
    # 3. Cellpose 분할 (핵심!)
    box_h_large = 1.4
    ax.add_patch(FancyBboxPatch((x-box_w/2, y-box_h_large), box_w, box_h_large,
                                boxstyle="round,pad=0.15",
                                facecolor=COLORS['key'], alpha=0.25,
                                edgecolor=COLORS['key'], linewidth=2.5))
    ax.text(x, y-0.3, 'Cellpose 분할 (121b)', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    ax.text(x, y-0.7, '흐름 벡터 (Fx, Fy) 계산\n세포 마스크 재구성\nflow_threshold=0.4', 
            ha='center', va='center', fontsize=8.5)
    
    y -= 1.9
    ax.add_patch(FancyArrowPatch((x, y+0.5), (x, y+0.1), 
                                arrowstyle='->', mutation_scale=25, 
                                linewidth=2, color=COLORS['border']))
    
    # 4. 특징 추출
    box_h_large = 2.0
    ax.add_patch(FancyBboxPatch((x-box_w/2, y-box_h_large), box_w, box_h_large,
                                boxstyle="round,pad=0.15",
                                facecolor=COLORS['process'], alpha=0.2,
                                edgecolor=COLORS['process'], linewidth=2))
    ax.text(x, y-0.3, '특징 추출 (121c) - 25개 특징', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    ax.text(x, y-0.8, '• 형태학적 (6): 면적, 둘레, 원형도, 편심률...\n• 강도 (7): 평균, 표준편차, 분위수...\n• 텍스처 (4): GLCM 대비, 상관, 에너지...\n• 공간 (5): 밀도, 최근접 이웃, Ripley K...', 
            ha='center', va='center', fontsize=8)
    
    y -= 2.5
    ax.add_patch(FancyArrowPatch((x, y+0.5), (x, y+0.1), 
                                arrowstyle='->', mutation_scale=25, 
                                linewidth=2, color=COLORS['border']))
    
    # 5. Ki-67 계산
    ax.add_patch(FancyBboxPatch((x-box_w/2, y-box_h), box_w, box_h,
                                boxstyle="round,pad=0.15",
                                facecolor=COLORS['key'], alpha=0.3,
                                edgecolor=COLORS['key'], linewidth=2))
    ax.text(x, y-box_h/2, 'Ki-67 증식 지수 산출 (121d)\n고강도 핵 비율 (80th percentile 기준)', 
            ha='center', va='center', fontsize=9)
    
    y -= 1.3
    ax.add_patch(FancyArrowPatch((x, y+0.6), (x, y+0.1), 
                                arrowstyle='->', mutation_scale=25, 
                                linewidth=2, color=COLORS['border']))
    
    # 6. 출력
    box_h_large = 1.2
    ax.add_patch(FancyBboxPatch((x-box_w/2, y-box_h_large), box_w, box_h_large,
                                boxstyle="round,pad=0.15",
                                facecolor=COLORS['output'], alpha=0.3,
                                edgecolor=COLORS['output'], linewidth=2.5))
    ax.text(x, y-0.6, '출력 데이터\nKi-67: 38.2%\n25개 형태학적 특징 벡터', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 성능 표시
    ax.text(9.5, 10, 'GPU 가속\n처리 시간', ha='center', fontsize=9, fontweight='bold')
    ax.text(9.5, 9.4, '7.2초', ha='center', fontsize=14, fontweight='bold', 
            color=COLORS['output'])
    ax.add_patch(FancyBboxPatch((8.3, 9.8), 2.4, 1.5,
                                boxstyle="round,pad=0.2",
                                facecolor=COLORS['output'], alpha=0.15,
                                edgecolor=COLORS['output'], linewidth=2))
    
    ax.text(9.5, 8.9, 'vs CPU:\n30.4초', ha='center', fontsize=8, color='gray')
    
    plt.savefig(OUTPUT_DIR / 'figure_02_cell_flowchart.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 2 (전문 버전)")
    plt.close()


if __name__ == "__main__":
    print("[세포 순서도 재생성...]\n")
    figure_2_cell_flowchart_pro()
    print("\n[완료]")
