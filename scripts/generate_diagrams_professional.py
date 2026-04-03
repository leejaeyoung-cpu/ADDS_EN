# -*- coding: utf-8 -*-
"""
전문 특허용 복잡한 다이어그램 생성 (Part 2)
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Wedge
import numpy as np
from pathlib import Path
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = Path("c:/Users/brook/Desktop/ADDS/docs/patent_figures")

# 전문 색상
COLORS = {
    'layer1': '#E3F2FD',    # 연한 파랑 (데이터)
    'layer2': '#F3E5F5',    # 연한 보라 (처리)
    'layer3': '#FFF3E0',    # 연한 주황 (통합)
    'layer4': '#E8F5E9',    # 연한 녹색 (표현)
    'primary': '#1976D2',   # 진한 파랑
    'secondary': '#7B1FA2', # 진한 보라
    'accent': '#F57C00',    # 진한 주황
    'success': '#388E3C',   # 진한 녹색
    'warning': '#D32F2F',   # 빨강
    'border': '#424242'     # 진한 회색
}


def figure_1_architecture_pro():
    """도 1: 전체 아키텍처 (전문 버전)"""
    fig = plt.figure(figsize=(16, 14))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # 제목
    ax.text(8, 13.3, '도 1. ADDS 시스템 전체 아키텍처', 
            ha='center', fontsize=18, fontweight='bold')
    ax.text(8, 12.9, 'AI-Powered Multi-Modal Clinical Decision Support System', 
            ha='center', fontsize=11, style='italic', color='gray')
    
    y = 12.2
    layer_h = 2.2
    
    # === 계층 1: 데이터 획득 (110) ===
    ax.add_patch(Rectangle((0.5, y-layer_h), 15, layer_h, 
                           facecolor=COLORS['layer1'], edgecolor=COLORS['border'], linewidth=2))
    ax.text(1, y-0.3, '데이터 획득 계층 (110)', fontsize=13, fontweight='bold')
    
    boxes = [
        (1.5, '병리 이미지\n입력부 (111)\nTIFF/PNG\n2048×2048'),
        (5.8, 'CT DICOM\n입력부 (112)\n512×512×N\nslices'),
        (10.1, '임상 데이터\n입력부 (113)\nKRAS/TP53/MSI\nECOG/Lab')
    ]
    
    for x, text in boxes:
        ax.add_patch(Rectangle((x, y-1.8), 3.5, 1.3, 
                               facecolor='white', edgecolor=COLORS['primary'], linewidth=1.5))
        ax.text(x+1.75, y-1.15, text, ha='center', va='center', fontsize=9)
    
    # === 계층 2: 처리 계층 (120) ===
    y -= (layer_h + 0.4)
    ax.add_patch(Rectangle((0.5, y-layer_h), 15, layer_h, 
                           facecolor=COLORS['layer2'], edgecolor=COLORS['border'], linewidth=2))
    ax.text(1, y-0.3, '처리 계층 (120)', fontsize=13, fontweight='bold')
    
    # 세포 분석
    ax.add_patch(Rectangle((2, y-1.7), 5, 1.2, 
                           facecolor='white', edgecolor=COLORS['secondary'], linewidth=1.8))
    ax.text(4.5, y-1.1, '세포 분석 모듈 (121)\nCellpose 분할 + 25개 특징\nKi-67 증식 지수', 
            ha='center', va='center', fontsize=9.5, fontweight='bold')
    
    # CT 검출
    ax.add_patch(Rectangle((8.5, y-1.7), 5, 1.2, 
                           facecolor='white', edgecolor=COLORS['secondary'], linewidth=1.8))
    ax.text(11, y-1.1, 'CT 검출 모듈 (122)\n종양 후보 검출 + TNM 병기\n신뢰도 스코어링', 
            ha='center', va='center', fontsize=9.5, fontweight='bold')
    
    # 화살표 (데이터 → 처리)
    for x1, x2 in [(3, 4.5), (7.3, 11), (11.6, 11)]:
        ax.annotate('', xy=(x2, y+0.2), xytext=(x1, y+2.2+0.1),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['border']))
    
    # === 계층 3: 통합 계층 (130) - 핵심! ===
    y -= (layer_h + 0.5)
    layer_h = 2.8
    ax.add_patch(Rectangle((0.5, y-layer_h), 15, layer_h, 
                           facecolor=COLORS['layer3'], edgecolor=COLORS['accent'], linewidth=3))
    ax.text(1, y-0.3, '통합 계층 (130) ★ 핵심 혁신', fontsize=13, fontweight='bold', color=COLORS['accent'])
    
    # 통합 엔진
    ax.add_patch(Rectangle((1.5, y-1.2), 4, 0.9, 
                           facecolor='white', edgecolor=COLORS['accent'], linewidth=1.8))
    ax.text(3.5, y-0.75, '통합 엔진 (131)\n병기 + 위험도 + 예후', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 약물 칵테일 (신규 강조!)
    ax.add_patch(Rectangle((1.5, y-2.3), 4, 0.9, 
                           facecolor=COLORS['accent'], alpha=0.2, edgecolor=COLORS['accent'], linewidth=2.5))
    ax.text(3.5, y-1.85, '약물 칵테일 최적화 (131e)\n4-모델 시너지 계산 ★', 
            ha='center', va='center', fontsize=9, fontweight='bold', color=COLORS['accent'])
    
    # XAI
    ax.add_patch(Rectangle((6.3, y-1.75), 3.2, 1.4, 
                           facecolor='white', edgecolor=COLORS['accent'], linewidth=1.5))
    ax.text(7.9, y-1.05, 'XAI 모듈 (132)\nLIME + Grad-CAM\n+ Counterfactual', 
            ha='center', va='center', fontsize=8.5)
    
    # 능동 학습
    ax.add_patch(Rectangle((10.2, y-1.75), 4, 1.4, 
                           facecolor='white', edgecolor=COLORS['accent'], linewidth=1.5))
    ax.text(12.2, y-1.05, '능동 학습 모듈 (133)\nTS → EI 전환\n40% 수렴 가속', 
            ha='center', va='center', fontsize=8.5)
    
    # 화살표 (처리 → 통합)
    ax.annotate('', xy=(3.5, y+0.3), xytext=(4.5, y+2.8+0.1),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['border']))
    ax.annotate('', xy=(12.2, y+0.3), xytext=(11, y+2.8+0.1),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['border']))
    
    # === 계층 4: 표현 계층 (140) ===
    y -= (layer_h + 0.4)
    layer_h = 2.0
    ax.add_patch(Rectangle((0.5, y-layer_h), 15, layer_h, 
                           facecolor=COLORS['layer4'], edgecolor=COLORS['border'], linewidth=2))
    ax.text(1, y-0.3, '표현 계층 (140)', fontsize=13, fontweight='bold')
    
    outputs = [
        (2.5, '의사 인터페이스 (141)\n상세 분석 + 근거'),
        (6.5, '환자 인터페이스 (142)\n쉬운 설명 (6학년 수준)'),
        (10.5, 'RESTful API (143)\n3개 엔드포인트\n80 req/s')
    ]
    
    for x, text in outputs:
        ax.add_patch(Rectangle((x, y-1.5), 3, 1, 
                               facecolor='white', edgecolor=COLORS['success'], linewidth=1.5))
        ax.text(x+1.5, y-1, text, ha='center', va='center', fontsize=8.5)
    
    # 화살표 (통합 → 표현)
    ax.annotate('', xy=(8, y+0.2), xytext=(8, y+3.3),
               arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['border']))
    
    # 성능 지표
    ax.text(14.5, 3, 'GPU 가속:\n4.2× 빠름', ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['success'], 
                     alpha=0.3, edgecolor=COLORS['success'], linewidth=1.5))
    ax.text(14.5, 1.8, 'E2E 처리:\n11.2초', ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['primary'], 
                     alpha=0.3, edgecolor=COLORS['primary'], linewidth=1.5))
    
    plt.savefig(OUTPUT_DIR / 'figure_01_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 1 (전문 버전)")
    plt.close()


def figure_4_synergy_pro():
    """도 4: 약물 시너지 (전문 버전)"""
    fig = plt.figure(figsize=(14, 11))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # 제목
    ax.text(7, 10.5, '도 4. 약물 칵테일 4-모델 시너지 계산 및 합의 알고리즘', 
            ha='center', fontsize=17, fontweight='bold')
    
    # 입력 약물
    y = 9.2
    drug_specs = [
        (2.5, '약물 A\n5-Fluorouracil\n(5-FU)', COLORS['primary']),
        (5.5, '약물 B\nOxaliplatin\n(LOHP)', COLORS['secondary'])
    ]
    
    for x, label, color in drug_specs:
        ax.add_patch(Circle((x, y), 0.5, facecolor=color, alpha=0.3, 
                           edgecolor='black', linewidth=2))
        ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 효능 데이터
    y = 7.8
    ax.text(4, y, '효능 데이터', ha='center', fontsize=10, fontweight='bold')
    ax.add_patch(Rectangle((2.5, y-0.7), 3, 0.5, 
                           facecolor='white', edgecolor='black', linewidth=1))
    ax.text(4, y-0.45, 'E_A=0.40, E_B=0.30\nE_combined=0.75', 
            ha='center', va='center', fontsize=8)
    
    # 4개 모델
    y = 5.8
    models = [
        (1.5, 'Bliss\nIndependence', '+0.18', COLORS['success'], 
         'E = E_A + E_B\n- E_A×E_B'),
        (4.5, 'Loewe\nAdditivity', '-0.12', COLORS['warning'], 
         'CI = d_A/EC50_A\n+ d_B/EC50_B'),
        (7.5, 'HSA\n(Highest Single)', '+0.25', COLORS['success'], 
         'E = max\n(E_A, E_B)'),
        (10.5, 'ZIP\n(Zero Interaction)', '+0.15', COLORS['success'], 
         'Bliss + Loewe\n통합')
    ]
    
    for x, name, score, color, formula in models:
        # 박스
        ax.add_patch(Rectangle((x-0.9, y-1), 1.8, 1.6, 
                               facecolor='white', edgecolor=color, linewidth=2))
        ax.text(x, y+0.5, name, ha='center', va='center', 
               fontsize=9.5, fontweight='bold')
        ax.text(x, y, formula, ha='center', va='center', fontsize=7.5)
        
        # 점수
        ax.add_patch(Circle((x, y-0.6), 0.25, 
                           facecolor=color, alpha=0.3, edgecolor=color, linewidth=1.5))
        ax.text(x, y-0.6, score, ha='center', va='center', 
               fontsize=9, fontweight='bold')
        
        # 입력 화살표
        ax.annotate('', xy=(x, y+0.6), xytext=(4, 7.1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))
    
    # 합의 알고리즘
    y = 3.2
    ax.add_patch(Rectangle((3.5, y-1.2), 7, 1.6, 
                           facecolor=COLORS['accent'], alpha=0.15,
                           edgecolor=COLORS['accent'], linewidth=2.5))
    ax.text(7, y+0.1, '합의 알고리즘 (Consensus)', ha='center', fontsize=12, fontweight='bold')
    ax.text(7, y-0.3, '양수 모델: 3/4 (75%)', ha='center', fontsize=10)
    ax.text(7, y-0.6, '평균 점수: +0.115', ha='center', fontsize=10)
    
    # 모델에서 합의로 화살표
    for x in [1.5, 4.5, 7.5, 10.5]:
        ax.annotate('', xy=(7, y+0.4), xytext=(x, y+2.8-1),
                   arrowprops=dict(arrowstyle='->', lw=2, color=COLORS['accent']))
    
    # 최종 판정
    y = 1.2
    ax.add_patch(Rectangle((3, y-0.6), 8, 0.9, 
                           facecolor=COLORS['success'], alpha=0.7,
                           edgecolor='black', linewidth=2))
    ax.text(7, y-0.15, '최종 판정: "Synergy" (시너지 효과 확인)', 
            ha='center', va='center', fontsize=13, fontweight='bold', color='white')
    
    # 효능 계산
    ax.text(7, 0.2, '예상 효능 = 50% × (1 + 0.15×0.115) = 50.86%', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=COLORS['accent'], linewidth=1.5))
    
    plt.savefig(OUTPUT_DIR / 'figure_04_drug_synergy.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 4 (전문 버전)")
    plt.close()


def figure_10_docker_pro():
    """도 10: Docker 배포 (전문 버전)"""
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 제목
    ax.text(7, 9.7, '도 10. Docker 기반 배포 아키텍처', 
            ha='center', fontsize=17, fontweight='bold')
    
    # Docker 컨테이너 외곽
    ax.add_patch(Rectangle((1, 4.2), 12, 4.8, 
                           facecolor='#E3F2FD', alpha=0.3,
                           edgecolor='#0D47A1', linewidth=3, linestyle='--'))
    ax.text(1.5, 8.7, 'Docker Container: ADDS', fontsize=13, fontweight='bold', color='#0D47A1')
    
    y = 8
    # 애플리케이션 레이어
    apps = [
        (2.5, 'FastAPI\nBackend\n(Gunicorn\n4 workers)', COLORS['primary']),
        (5.5, 'Streamlit\nUI\nFrontend', COLORS['secondary']),
        (8.5, 'PostgreSQL\nDatabase\n(Indexed)', COLORS['accent'])
    ]
    
    for x, label, color in apps:
        ax.add_patch(Rectangle((x-1, y-1), 2, 1, 
                               facecolor='white', edgecolor=color, linewidth=2))
        ax.text(x, y-0.5, label, ha='center', va='center', fontsize=9)
    
    # 모델 레이어
    y = 6.3
    ax.add_patch(Rectangle((2, y-0.7), 9, 0.7, 
                           facecolor=COLORS['success'], alpha=0.2,
                           edgecolor=COLORS['success'], linewidth=2))
    ax.text(6.5, y-0.35, 'AI Models: Cellpose (17M params) + CT Detector (25M params)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # NVIDIA Container Toolkit
    y = 4.8
    ax.add_patch(Rectangle((1, y-0.5), 12, 0.5, 
                           facecolor='#76B900', alpha=0.3,
                           edgecolor='#76B900', linewidth=2))
    ax.text(7, y-0.25, 'NVIDIA Container Toolkit (GPU Resource Access)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 호스트 GPU
    y = 3.8
    ax.add_patch(Rectangle((1, y-0.6), 12, 0.6, 
                           facecolor=COLORS['warning'], alpha=0.2,
                           edgecolor=COLORS['warning'], linewidth=2))
    ax.text(7, y-0.3, 'Host GPU: NVIDIA RTX (CUDA 12.1, cuDNN 8.x)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # API 엔드포인트
    y = 2.5
    ax.text(7, y+0.3, 'RESTful API Endpoints', ha='center', fontsize=11, fontweight='bold')
    endpoints = [
        (2, '/api/v1/segmentation'),
        (5.5, '/api/v1/tumor_detection'),
        (9, '/api/v1/cdss/integrate')
    ]
    
    for x, endpoint in endpoints:
        ax.add_patch(Rectangle((x-1.3, y-0.4), 2.6, 0.4, 
                               facecolor='white', edgecolor='black', linewidth=1))
        ax.text(x, y-0.2, endpoint, ha='center', va='center', fontsize=8)
    
    # 성능 지표
    y = 1
    ax.add_patch(Rectangle((4, y-0.6), 6, 0.6, 
                           facecolor=COLORS['success'], alpha=0.3,
                           edgecolor=COLORS['success'], linewidth=2))
    ax.text(7, y-0.3, '처리량: 80 req/s | 응답 시간: 11.2s (E2E)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # 화살표들
    ax.annotate('', xy=(7, 5.6), xytext=(7, 6.3),
               arrowprops=dict(arrowstyle='<->', lw=2.5, color='black'))
    ax.annotate('', xy=(7, 4.3), xytext=(7, 4.8),
               arrowprops=dict(arrowstyle='<->', lw=2.5, color='black'))
    ax.annotate('', xy=(7, 3.2), xytext=(7, 3.8),
               arrowprops=dict(arrowstyle='<->', lw=2.5, color='black'))
    
    plt.savefig(OUTPUT_DIR / 'figure_10_docker.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[완료] 도 10 (전문 버전)")
    plt.close()


if __name__ == "__main__":
    print("[복잡한 다이어그램 재생성...]\n")
    
    figure_1_architecture_pro()
    figure_4_synergy_pro()
    figure_10_docker_pro()
    
    print("\n[완료] 모든 다이어그램 재생성 완료!")
