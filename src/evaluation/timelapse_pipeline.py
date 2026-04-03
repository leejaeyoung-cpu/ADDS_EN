"""
Time-lapse Analysis Pipeline
=============================
전체 Time-lapse 분석 파이프라인 통합

사용법:
    pipeline = TimeLapsePipeline()
    results = pipeline.run_full_analysis("experiment_video.avi")
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
import numpy as np

from src.preprocessing.timelapse_loader import TimeLapseLoader
from src.preprocessing.image_processor import CellposeProcessor
from src.pathology.cell_tracker import CellTracker
from src.evaluation.timelapse_analyzer import TimeLapseAnalyzer
from src.reporting.timelapse_visualizer import TimeLapseVisualizer

logger = logging.getLogger(__name__)


class TimeLapsePipeline:
    """Time-lapse 분석 전체 파이프라인"""
    
    def __init__(self,
                 frame_interval_minutes: float = 5.0,
                 cellpose_model: str = 'cyto2',
                 cellpose_diameter: Optional[int] = None,
                 iou_threshold: float = 0.3):
        """
        Args:
            frame_interval_minutes: 프레임 간 시간 간격 (분)
            cellpose_model: Cellpose 모델 ('cyto', 'cyto2', 'cyto3')
            cellpose_diameter: 세포 지름 (None이면 자동)
            iou_threshold: 추적 IoU임계값
        """
        self.frame_interval_minutes = frame_interval_minutes
        
        # 모듈 초기화
        self.loader = TimeLapseLoader()
        self.cellpose = CellposeProcessor(
            model_name=cellpose_model,
            diameter=cellpose_diameter
        )
        self.tracker = CellTracker(iou_threshold=iou_threshold)
        self.analyzer = TimeLapseAnalyzer(frame_interval_minutes=frame_interval_minutes)
        self.visualizer = TimeLapseVisualizer()
        
        logger.info("Time-lapse Pipeline 초기화 완료")
    
    def run_full_analysis(self,
                         input_path: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         experiment_name: str = "Experiment",
                         max_frames: Optional[int] = None,
                         generate_video: bool = True) -> Dict:
        """
        전체 분석 파이프라인 실행
        
        Args:
            input_path: 동영상 파일 또는 이미지 시퀀스 디렉토리
            output_dir: 결과 저장 디렉토리 (None이면 입력 경로와 같은 위치)
            experiment_name: 실험 이름
            max_frames: 분석할 최대 프레임 수
            generate_video: 추적 동영상 생성 여부
        
        Returns:
            종합 분석 결과 딕셔너리
        """
        logger.info(f"=== Time-lapse 분석 시작: {experiment_name} ===")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = Path(input_path).parent / "timelapse_results"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: 데이터 로딩
        logger.info("Step 1/5: 데이터 로딩...")
        data = self.loader.load(input_path, max_frames=max_frames)
        frames = data['frames']
        metadata = data['metadata']
        
        logger.info(f"로딩 완료: {len(frames)} 프레임, {metadata.get('format')}")
        
        # Step 2: Cellpose 세그멘테이션
        logger.info("Step 2/5: Cellpose 세그멘테이션...")
        masks_list = []
        features_list = []
        
        for i, frame in enumerate(frames):
            result = self.cellpose.process_image(frame)
            masks_list.append(result['masks'])
            features_list.append(result.get('cell_features', []))
            
            if (i + 1) % 50 == 0:
                logger.info(f"세그멘테이션 진행: {i+1}/{len(frames)}")
        
        logger.info(f"세그멘테이션 완료")
        
        # Step 3: 세포 추적
        logger.info("Step 3/5: 세포 추적...")
        tracks = self.tracker.track_cells(masks_list, features_list)
        track_stats = self.tracker.get_statistics()
        
        logger.info(f"추적 완료: {track_stats['total_tracks']} tracks, "
                   f"{track_stats['division_events']} divisions")
        
        # Step 4: 시계열 분석
        logger.info("Step 4/5: 시계열 분석...")
        report = self.analyzer.generate_comprehensive_report(tracks, experiment_name)
        
        logger.info(f"분석 완료: Doubling time = {report['proliferation'].get('doubling_time_hours', 0):.2f}h")
        
        # Step 5: 시각화 및 저장
        logger.info("Step 5/5: 시각화 및 저장...")
        
        # 추적 CSV 저장
        csv_path = output_dir / f"{experiment_name}_tracks.csv"
        self.tracker.export_tracks_to_csv(str(csv_path))
        
        # 추적 동영상 생성
        if generate_video:
            video_path = output_dir / f"{experiment_name}_tracking.mp4"
            self.visualizer.generate_tracking_video(
                frames, tracks, str(video_path),
                fps=10, show_tracks=True, show_ids=True
            )
        
        # HTML 대시보드 생성
        dashboard_path = output_dir / f"{experiment_name}_dashboard.html"
        self.visualizer.create_comprehensive_dashboard(report, str(dashboard_path))
        
        # 결과 종합
        results = {
            'experiment_name': experiment_name,
            'input_path': str(input_path),
            'output_dir': str(output_dir),
            'metadata': metadata,
            'track_statistics': track_stats,
            'analysis_report': report,
            'output_files': {
                'tracks_csv': str(csv_path),
                'tracking_video': str(video_path) if generate_video else None,
                'dashboard_html': str(dashboard_path)
            }
        }
        
        logger.info(f"=== 분석 완료 ===")
        logger.info(f"결과 저장: {output_dir}")
        
        return results
    
    def run_drug_comparison(self,
                           control_path: Union[str, Path],
                           treatment_paths: Dict[str, Union[str, Path]],
                           output_dir: Union[str, Path],
                           experiment_name: str = "Drug_Comparison") -> Dict:
        """
        여러 약물 조건 비교 분석
        
        Args:
            control_path: 대조군 경로
            treatment_paths: {'Drug A': 'path/to/video', ...}
            output_dir: 결과 저장 디렉토리
            experiment_name: 실험 이름
        
        Returns:
            비교 분석 결과
        """
        logger.info(f"=== 약물 비교 분석 시작: {experiment_name} ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 대조군 분석
        logger.info("대조군 분석 중...")
        control_results = self.run_full_analysis(
            control_path,
            output_dir / "control",
            "Control",
            generate_video=False
        )
        control_prolif = control_results['analysis_report']['proliferation']
        
        # 처리군 분석
        treatment_results = {}
        drug_responses = {}
        
        for drug_name, drug_path in treatment_paths.items():
            logger.info(f"{drug_name} 분석 중...")
            results = self.run_full_analysis(
                drug_path,
                output_dir / drug_name.lower().replace(' ', '_'),
                drug_name,
                generate_video=False
            )
            
            treatment_results[drug_name] = results
            
            # 약물 반응 계산
            treatment_prolif = results['analysis_report']['proliferation']
            response = self.analyzer.calculate_drug_response(
                control_prolif,
                treatment_prolif
            )
            drug_responses[drug_name] = response
            
            logger.info(f"{drug_name}: {response['growth_inhibition_percent']:.1f}% inhibition")
        
        # 비교 플롯 생성
        comparison_fig = self._plot_drug_comparison(
            control_prolif,
            {name: res['analysis_report']['proliferation'] 
             for name, res in treatment_results.items()},
            experiment_name
        )
        
        comparison_html = output_dir / f"{experiment_name}_comparison.html"
        comparison_fig.write_html(str(comparison_html))
        
        # 결과 종합
        comparison_results = {
            'experiment_name': experiment_name,
            'control': control_results,
            'treatments': treatment_results,
            'drug_responses': drug_responses,
            'comparison_plot': str(comparison_html)
        }
        
        logger.info("=== 약물 비교 분석 완료 ===")
        
        return comparison_results
    
    def run_cocktail_synergy_analysis(self,
                                     control_path: Union[str, Path],
                                     drug_a_path: Union[str, Path],
                                     drug_b_path: Union[str, Path],
                                     combo_path: Union[str, Path],
                                     output_dir: Union[str, Path],
                                     experiment_name: str = "Cocktail_Synergy",
                                     drug_a_name: str = "Drug A",
                                     drug_b_name: str = "Drug B") -> Dict:
        """
        칵테일 시너지 분석
        
        Args:
            control_path: 대조군
            drug_a_path: Drug A 단독
            drug_b_path: Drug B 단독
            combo_path: Drug A + B 조합
            output_dir: 결과 저장 디렉토리
            experiment_name: 실험 이름
            drug_a_name: Drug A 이름
            drug_b_name: Drug B 이름
        
        Returns:
            시너지 분석 결과
        """
        logger.info(f"=== 칵테일 시너지 분석 시작: {experiment_name} ===")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 조건 분석
        logger.info("대조군 분석...")
        control_results = self.run_full_analysis(control_path, output_dir / "control", "Control", generate_video=False)
        
        logger.info(f"{drug_a_name} 단독 분석...")
        drug_a_results = self.run_full_analysis(drug_a_path, output_dir / "drug_a", drug_a_name, generate_video=False)
        
        logger.info(f"{drug_b_name} 단독 분석...")
        drug_b_results = self.run_full_analysis(drug_b_path, output_dir / "drug_b", drug_b_name, generate_video=False)
        
        logger.info("조합 분석...")
        combo_results = self.run_full_analysis(combo_path, output_dir / "combo", f"{drug_a_name}+{drug_b_name}", generate_video=False)
        
        # 시너지 계산
        logger.info("시너지 계산 중...")
        synergy_data = self.analyzer.calculate_synergy_timelapse(
            drug_a_results['analysis_report']['proliferation'],
            drug_b_results['analysis_report']['proliferation'],
            combo_results['analysis_report']['proliferation'],
            control_results['analysis_report']['proliferation'],
            model='bliss'
        )
        
        logger.info(f"시너지 스코어: {synergy_data['mean_synergy_score']:.3f} "
                   f"({'Synergistic' if synergy_data['is_synergistic'] else 'Not synergistic'})")
        
        # 시너지 플롯 생성
        synergy_fig = self.visualizer.plot_synergy_over_time(
            synergy_data,
            f"{drug_a_name} + {drug_b_name} Synergy"
        )
        
        synergy_html = output_dir / f"{experiment_name}_synergy.html"
        synergy_fig.write_html(str(synergy_html))
        
        # 결과 종합
        results = {
            'experiment_name': experiment_name,
            'control': control_results,
            'drug_a': drug_a_results,
            'drug_b': drug_b_results,
            'combination': combo_results,
            'synergy_analysis': synergy_data,
            'synergy_plot': str(synergy_html)
        }
        
        logger.info("=== 칵테일 시너지 분석 완료 ===")
        
        return results
    
    def _plot_drug_comparison(self, control_data: Dict, 
                             treatment_data: Dict[str, Dict],
                             title: str) -> 'go.Figure':
        """약물 비교 플롯 생성"""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # 대조군
        time_points = control_data['time_points_hours']
        fig.add_trace(go.Scatter(
            x=time_points,
            y=control_data['cell_counts'],
            mode='lines+markers',
            name='Control',
            line=dict(color='black', width=3)
        ))
        
        #  처리군
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (drug_name, drug_data) in enumerate(treatment_data.items()):
            fig.add_trace(go.Scatter(
                x=drug_data['time_points_hours'],
                y=drug_data['cell_counts'],
                mode='lines+markers',
                name=drug_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (hours)",
            yaxis_title="Cell Count",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
