"""
Time-lapse Visualization Module
================================
Time-lapse 분석 결과 시각화

기능:
- 추적 동영상 생성
- 증식 곡선 플롯
- 이동 히트맵
- 이벤트 타임라인
- 시너지 스코어 플롯
"""

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TimeLapseVisualizer:
    """Time-lapse 분석 시각화"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Plotly
        
    def generate_tracking_video(self,
                               frames: List[np.ndarray],
                               tracks: Dict,
                               output_path: str,
                               fps: int = 10,
                               show_tracks: bool = True,
                               show_ids: bool = True) -> str:
        """
        추적 결과를 동영상으로 생성
        
        Args:
            frames: 원본 프레임 리스트
            tracks: CellTracker 결과
            output_path: 출력 경로
            fps: 프레임 레이트
            show_tracks: 이동 경로 표시 여부
            show_ids: Track ID 표시 여부
        
        Returns:
            생성된 동영상 경로
        """
        logger.info(f"추적 동영상 생성 중: {output_path}")
        
        if not frames:
            logger.warning("프레임이 없습니다")
            return ""
        
        # 동영상 설정
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track별 색상 할당
        track_colors = {}
        for i, track_id in enumerate(tracks.keys()):
            color_idx = i % len(self.colors)
            # Plotly 색상을 BGR로 변환
            color = self._plotly_to_bgr(self.colors[color_idx])
            track_colors[track_id] = color
        
        # 프레임별 처리
        for frame_idx, frame in enumerate(frames):
            # RGB to BGR 변환
            frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
            
            # 해당 프레임의 세포들 그리기
            for track_id, track in tracks.items():
                if frame_idx not in track.frames:
                    continue
                
                idx = track.frames.index(frame_idx)
                centroid = track.centroids[idx]
                color = track_colors[track_id]
                
                # 중심점 표시
                cv2.circle(frame_bgr, 
                          (int(centroid[0]), int(centroid[1])), 
                          5, color, -1)
                
                # Track ID 표시
                if show_ids:
                    cv2.putText(frame_bgr, 
                              f"#{track_id}", 
                              (int(centroid[0]) + 10, int(centroid[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 1)
                
                # 이동 경로 표시
                if show_tracks and idx > 0:
                    prev_idx = idx - 1
                    prev_centroid = track.centroids[prev_idx]
                    cv2.line(frame_bgr,
                            (int(prev_centroid[0]), int(prev_centroid[1])),
                            (int(centroid[0]), int(centroid[1])),
                            color, 2)
            
            # 프레임 정보 표시
            cv2.putText(frame_bgr,
                       f"Frame: {frame_idx}",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 2)
            
            out.write(frame_bgr)
            
            if (frame_idx + 1) % 50 == 0:
                logger.info(f"동영상 생성 진행: {frame_idx + 1}/{len(frames)}")
        
        out.release()
        logger.info(f"동영상 생성 완료: {output_path}")
        
        return output_path
    
    def _plotly_to_bgr(self, plotly_color: str) -> Tuple[int, int, int]:
        """Plotly RGB 문자열을 B GR 튜플로 변환"""
        # 'rgb(31, 119, 180)' -> (180, 119, 31)
        rgb_str = plotly_color.replace('rgb(', '').replace(')', '')
        r, g, b = [int(x) for x in rgb_str.split(',')]
        return (b, g, r)
    
    def plot_proliferation_curve(self,
                                proliferation_data: Dict,
                                title: str = "Cell Proliferation Curve") -> go.Figure:
        """
        세포 증식 곡선 플롯
        
        Args:
            proliferation_data: calculate_proliferation_curve 결과
            title: 그래프 제목
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        time_points = proliferation_data.get('time_points_hours', [])
        cell_counts = proliferation_data.get('cell_counts', [])
        alive_counts = proliferation_data.get('alive_counts', [])
        
        # 총 세포 수
        fig.add_trace(go.Scatter(
            x=time_points,
            y=cell_counts,
            mode='lines+markers',
            name='Total Cells',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # 생존 세포 수
        if alive_counts:
            fig.add_trace(go.Scatter(
                x=time_points,
                y=alive_counts,
                mode='lines+markers',
                name='Alive Cells',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=4)
            ))
        
        doubling_time = proliferation_data.get('doubling_time_hours')
        
        fig.update_layout(
            title=f"{title}<br><sub>Doubling time: {doubling_time:.2f} hours</sub>" if doubling_time else title,
            xaxis_title="Time (hours)",
            yaxis_title="Cell Count",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_migration_heatmap(self,
                               tracks: Dict,
                               image_shape: Tuple[int, int],
                               bins: int = 50) -> go.Figure:
        """
        세포 이동 히트맵
        
        Args:
            tracks: CellTracker 결과
            image_shape: (height, width)
            bins: 히트맵 빈 수
        
        Returns:
            Plotly Figure
        """
        height, width = image_shape
        
        # 모든 중심점 수집
        all_x = []
        all_y = []
        
        for track in tracks.values():
            for centroid in track.centroids:
                all_x.append(centroid[0])
                all_y.append(centroid[1])
        
        if not all_x:
            return go.Figure()
        
        # 2D 히스토그램 생성
        heatmap, xedges, yedges = np.histogram2d(
            all_x, all_y,
            bins=[bins, bins],
            range=[[0, width], [0, height]]
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            colorscale='Viridis',
            colorbar=dict(title="Cell Density")
        ))
        
        fig.update_layout(
            title="Cell Migration Heatmap",
            xaxis_title="X Position (pixels)",
            yaxis_title="Y Position (pixels)",
            height=600,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        return fig
    
    def plot_event_timeline(self,
                           events_data: Dict,
                           proliferation_data: Dict) -> go.Figure:
        """
        이벤트 타임라인 (분열, 사멸)
        
        Args:
            events_data: detect_events 결과
            proliferation_data: calculate_proliferation_curve 결과
        
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Cell Count Over Time", "Events Timeline"),
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # 상단: 증식 곡선
        time_points = proliferation_data.get('time_points_hours', [])
        cell_counts = proliferation_data.get('cell_counts', [])
        
        fig.add_trace(
            go.Scatter(x=time_points, y=cell_counts,
                      mode='lines', name='Cell Count',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # 하단: 이벤트
        division_times_frames = events_data.get('division_times_frames', [])
        death_times_frames = events_data.get('death_times_frames', [])
        
        # 프레임을 시간으로 변환 (가정: proliferation_data와 같은 간격)
        if time_points and proliferation_data.get('frames'):
            frames = proliferation_data['frames']
            frame_interval = time_points[1] - time_points[0] if len(time_points) > 1 else 1.0
            
            division_times_hours = [f * frame_interval for f in division_times_frames]
            death_times_hours = [f * frame_interval for f in death_times_frames]
            
            # 분열 이벤트
            fig.add_trace(
                go.Scatter(x=division_times_hours,
                          y=[1] * len(division_times_hours),
                          mode='markers',
                          name='Division',
                          marker=dict(color='green', size=10, symbol='circle')),
                row=2, col=1
            )
            
            # 사멸 이벤트
            fig.add_trace(
                go.Scatter(x=death_times_hours,
                          y=[0] * len(death_times_hours),
                          mode='markers',
                          name='Death',
                          marker=dict(color='red', size=10, symbol='x')),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
        fig.update_yaxes(title_text="Cell Count", row=1, col=1)
        fig.update_yaxes(title_text="Event Type", row=2, col=1,
                        tickvals=[0, 1], ticktext=['Death', 'Division'])
        
        fig.update_layout(height=700, template='plotly_white')
        
        return fig
    
    def plot_synergy_over_time(self,
                               synergy_data: Dict,
                               title: str = "Cocktail Synergy Over Time") -> go.Figure:
        """
        시간에 따른 시너지 스코어 플롯
        
        Args:
            synergy_data: calculate_synergy_timelapse 결과
            title: 그래프 제목
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        time_points = synergy_data.get('time_points_hours', [])
        synergy_scores = synergy_data.get('synergy_scores_over_time', [])
        
        # 시너지 스코어
        fig.add_trace(go.Scatter(
            x=time_points,
            y=synergy_scores,
            mode='lines+markers',
            name='Synergy Score',
            line=dict(color='purple', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ))
        
        # 0선 (additive 효과)
        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                     annotation_text="Additive")
        
        # Peak 표시
        peak_time = synergy_data.get('peak_synergy_time_hours')
        peak_score = synergy_data.get('peak_synergy_score')
        
        if peak_time and peak_score:
            fig.add_trace(go.Scatter(
                x=[peak_time],
                y=[peak_score],
                mode='markers',
                name='Peak Synergy',
                marker=dict(color='red', size=15, symbol='star')
            ))
        
        mean_synergy = synergy_data.get('mean_synergy_score', 0.0)
        is_synergistic = synergy_data.get('is_synergistic', False)
        
        subtitle = f"Mean synergy: {mean_synergy:.3f} ({'Synergistic' if is_synergistic else 'Not synergistic'})"
        
        fig.update_layout(
            title=f"{title}<br><sub>{subtitle}</sub>",
            xaxis_title="Time (hours)",
            yaxis_title="Synergy Score",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_comprehensive_dashboard(self,
                                      report: Dict,
                                      output_html: str) -> str:
        """
        종합 대시보드 HTML 생성
        
        Args:
            report: generate_comprehensive_report 결과
            output_html: 출력 HTML 경로
        
        Returns:
            생성된 HTML 경로
        """
        logger.info(f"종합 대시보드 생성 중: {output_html}")
        
        # 개별 플롯 생성
        prolif_fig = self.plot_proliferation_curve(report['proliferation'],
                                                   f"{report['experiment_name']} - Proliferation")
        
        event_fig = self.plot_event_timeline(report['events'], report['proliferation'])
        
        # HTML 구성
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report['experiment_name']} - Time-lapse Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.18.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f7fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .plot {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎥 {report['experiment_name']}</h1>
        <p>Time-lapse Analysis Report</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{report['total_tracks']}</div>
            <div class="stat-label">Total Tracks</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report['proliferation'].get('doubling_time_hours', 0):.1f}h</div>
            <div class="stat-label">Doubling Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report['events'].get('total_divisions', 0)}</div>
            <div class="stat-label">Division Events</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{report['migration'].get('mean_velocity_pixels_per_hour', 0):.1f}</div>
            <div class="stat-label">Avg Velocity (px/h)</div>
        </div>
    </div>
    
    <div class="plot" id="proliferation"></div>
    <div class="plot" id="events"></div>
    
    <script>
        var prolifData = {prolif_fig.to_json()};
        Plotly.newPlot('proliferation', prolifData.data, prolifData.layout);
        
        var eventsData = {event_fig.to_json()};
        Plotly.newPlot('events', eventsData.data, eventsData.layout);
    </script>
</body>
</html>
"""
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"대시보드 생성 완료: {output_html}")
        
        return output_html
