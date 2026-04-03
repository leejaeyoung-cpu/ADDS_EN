"""
Cell Tracking Engine
====================
프레임 간 세포를 추적하고 계보(lineage)를 구축하는 모듈

주요 기능:
- IoU 기반 세포 매칭
- Hungarian algorithm을 사용한 최적 할당
- 세포 분열 감지
- 세포 사멸 감지
- 계보 트리 구축
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import networkx as nx
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CellTrack:
    """개별 세포 추적 정보"""
    track_id: int
    frames: List[int] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    features: Dict = field(default_factory=dict)
    
    # 이벤트 정보
    birth_frame: Optional[int] = None
    death_frame: Optional[int] = None
    division_frame: Optional[int] = None
    parent_id: Optional[int] = None
    daughter_ids: List[int] = field(default_factory=list)
    
    def add_observation(self, frame_idx: int, centroid: Tuple[float, float], 
                       area: float, features: Optional[Dict] = None):
        """프레임별 관찰 데이터 추가"""
        self.frames.append(frame_idx)
        self.centroids.append(centroid)
        self.areas.append(area)
        
        if features:
            for key, value in features.items():
                if key not in self.features:
                    self.features[key] = []
                self.features[key].append(value)
    
    @property
    def lifetime(self) -> int:
        """세포 생존 시간 (프레임 수)"""
        if self.birth_frame is not None and self.death_frame is not None:
            return self.death_frame - self.birth_frame
        elif self.birth_frame is not None:
            return max(self.frames) - self.birth_frame if self.frames else 0
        return len(self.frames)
    
    @property
    def is_alive(self) -> bool:
        """세포가 살아있는지 (사멸하지 않음)"""
        return self.death_frame is None
    
    @property
    def total_displacement(self) -> float:
        """총 이동 거리"""
        if len(self.centroids) < 2:
            return 0.0
        
        displacement = 0.0
        for i in range(1, len(self.centroids)):
            dx = self.centroids[i][0] - self.centroids[i-1][0]
            dy = self.centroids[i][1] - self.centroids[i-1][1]
            displacement += np.sqrt(dx**2 + dy**2)
        
        return displacement
    
    @property
    def average_velocity(self) -> float:
        """평균 이동 속도 (픽셀/프레임)"""
        if len(self.frames) < 2:
            return 0.0
        return self.total_displacement / (len(self.frames) - 1)


class CellTracker:
    """세포 추적 엔진"""
    
    def __init__(self, 
                 iou_threshold: float = 0.3,
                 max_distance: float = 50.0,
                 division_iou_threshold: float = 0.15):
        """
        Args:
            iou_threshold: IoU 매칭 임계값 (0.3 권장)
            max_distance: 최대 이동 거리 (픽셀)
            division_iou_threshold: 분열 감지 임계값
        """
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.division_iou_threshold = division_iou_threshold
        
        self.tracks: Dict[int, CellTrack] = {}
        self.next_track_id = 0
        self.graph = nx.DiGraph()  # 계보 그래프
        
    def track_cells(self, 
                    masks_sequence: List[np.ndarray],
                    features_sequence: Optional[List[List[Dict]]] = None) -> Dict[int, CellTrack]:
        """
        전체 시퀀스에서 세포 추적
        
        Args:
            masks_sequence: 프레임별 마스크 리스트 [frames, height, width]
            features_sequence: 프레임별 세포 특징 리스트 (optional)
        
        Returns:
            track_id를 키로 하는 CellTrack 딕셔너리
        """
        logger.info(f"세포 추적 시작: {len(masks_sequence)} 프레임")
        
        # 첫 프레임 초기화
        self._initialize_tracks(masks_sequence[0], features_sequence[0] if features_sequence else None, 0)
        
        # 프레임별 추적
        for frame_idx in range(1, len(masks_sequence)):
            if frame_idx % 50 == 0:
                logger.info(f"추적 진행: {frame_idx}/{len(masks_sequence)} 프레임")
            
            prev_masks = masks_sequence[frame_idx - 1]
            curr_masks = masks_sequence[frame_idx]
            
            prev_features = features_sequence[frame_idx - 1] if features_sequence else None
            curr_features = features_sequence[frame_idx] if features_sequence else None
            
            self._track_frame(prev_masks, curr_masks, 
                            prev_features, curr_features, frame_idx)
        
        logger.info(f"추적 완료: {len(self.tracks)} 개의 track 생성")
        
        return self.tracks
    
    def _initialize_tracks(self, masks: np.ndarray, 
                          features: Optional[List[Dict]], 
                          frame_idx: int):
        """첫 프레임의 모든 세포를 새 트랙으로 초기화"""
        cell_ids = np.unique(masks)[1:]  # 0은 배경
        
        for cell_idx, cell_id in enumerate(cell_ids):
            cell_mask = (masks == cell_id)
            centroid = self._compute_centroid(cell_mask)
            area = np.sum(cell_mask)
            
            feature_dict = features[cell_idx] if features and cell_idx < len(features) else None
            
            track = CellTrack(
                track_id=self.next_track_id,
                birth_frame=frame_idx
            )
            track.add_observation(frame_idx, centroid, area, feature_dict)
            
            self.tracks[self.next_track_id] = track
            self.graph.add_node(self.next_track_id, frame=frame_idx)
            
            self.next_track_id += 1
    
    def _track_frame(self, 
                    prev_masks: np.ndarray, 
                    curr_masks: np.ndarray,
                    prev_features: Optional[List[Dict]],
                    curr_features: Optional[List[Dict]],
                    frame_idx: int):
        """연속된 두 프레임 간 세포 추적"""
        
        # 이전/현재 프레임의 세포 추출
        prev_cells = self._extract_cells(prev_masks)
        curr_cells = self._extract_cells(curr_masks)
        
        if not curr_cells:
            # 현재 프레임에 세포 없음 - 모든 트랙 종료
            active_tracks = self._get_active_tracks(frame_idx - 1)
            for track_id in active_tracks:
                self.tracks[track_id].death_frame = frame_idx
            return
        
        # IoU 행렬 계산
        iou_matrix = self._compute_iou_matrix(prev_cells, curr_cells)
        
        # 거리 행렬 계산 (centroid 기반)
        prev_centroids = np.array([self._compute_centroid(cell) for cell in prev_cells])
        curr_centroids = np.array([self._compute_centroid(cell) for cell in curr_cells])
        distance_matrix = cdist(prev_centroids, curr_centroids)
        
        # 결합 비용 행렬 (IoU가 높을수록, 거리가 가까울수록 비용 낮음)
        cost_matrix = -(iou_matrix * 2.0) + (distance_matrix / self.max_distance)
        
        # Hungarian algorithm으로 최적 할당
        row_indices, col_indices = self._assign_tracks(cost_matrix, iou_matrix)
        
        # 매칭 결과 처리
        matched_prev = set(row_indices)
        matched_curr = set(col_indices)
        
        # 1. 정상 매칭 (1:1)
        for prev_idx, curr_idx in zip(row_indices, col_indices):
            iou = iou_matrix[prev_idx, curr_idx]
            
            if iou >= self.iou_threshold:
                # 이전 프레임에서 활성 트랙 찾기
                track_id = self._find_track_at_frame(prev_idx, frame_idx - 1, prev_masks)
                
                if track_id is not None:
                    cell_mask = curr_cells[curr_idx]
                    centroid = curr_centroids[curr_idx]
                    area = np.sum(cell_mask)
                    feature_dict = curr_features[curr_idx] if curr_features and curr_idx < len(curr_features) else None
                    
                    self.tracks[track_id].add_observation(frame_idx, centroid, area, feature_dict)
        
        # 2. 분열 감지 (1 → 2)
        self._detect_divisions(prev_cells, curr_cells, iou_matrix, 
                              matched_prev, matched_curr, frame_idx, curr_centroids)
        
        # 3. 사멸 감지 (이전 프레임에 있었으나 매칭 안됨)
        unmatched_prev = set(range(len(prev_cells))) - matched_prev
        for prev_idx in unmatched_prev:
            track_id = self._find_track_at_frame(prev_idx, frame_idx - 1, prev_masks)
            if track_id is not None:
                self.tracks[track_id].death_frame = frame_idx
        
        # 4. 새 세포 등장 (현재 프레임에만 있음)
        unmatched_curr = set(range(len(curr_cells))) - matched_curr
        for curr_idx in unmatched_curr:
            cell_mask = curr_cells[curr_idx]
            centroid = curr_centroids[curr_idx]
            area = np.sum(cell_mask)
            feature_dict = curr_features[curr_idx] if curr_features and curr_idx < len(curr_features) else None
            
            track = CellTrack(
                track_id=self.next_track_id,
                birth_frame=frame_idx
            )
            track.add_observation(frame_idx, centroid, area, feature_dict)
            
            self.tracks[self.next_track_id] = track
            self.graph.add_node(self.next_track_id, frame=frame_idx)
            
            self.next_track_id += 1
    
    def _extract_cells(self, masks: np.ndarray) -> List[np.ndarray]:
        """마스크에서 개별 세포 추출"""
        cell_ids = np.unique(masks)[1:]  # 0은 배경
        return [(masks == cell_id) for cell_id in cell_ids]
    
    def _compute_centroid(self, cell_mask: np.ndarray) -> Tuple[float, float]:
        """세포 중심점 계산"""
        y_coords, x_coords = np.where(cell_mask)
        if len(y_coords) == 0:
            return (0.0, 0.0)
        return (float(np.mean(x_coords)), float(np.mean(y_coords)))
    
    def _compute_iou_matrix(self, 
                           cells_a: List[np.ndarray], 
                           cells_b: List[np.ndarray]) -> np.ndarray:
        """두 프레임 간 IoU 행렬 계산"""
        iou_matrix = np.zeros((len(cells_a), len(cells_b)))
        
        for i, cell_a in enumerate(cells_a):
            for j, cell_b in enumerate(cells_b):
                intersection = np.logical_and(cell_a, cell_b).sum()
                union = np.logical_or(cell_a, cell_b).sum()
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
        
        return iou_matrix
    
    def _assign_tracks(self, 
                      cost_matrix: np.ndarray,
                      iou_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Hungarian algorithm으로 세포 할당"""
        if cost_matrix.size == 0:
            return np.array([]), np.array([])
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # IoU 임계값 미만인 매칭 제거
        valid_matches = iou_matrix[row_indices, col_indices] >= self.iou_threshold
        
        return row_indices[valid_matches], col_indices[valid_matches]
    
    def _find_track_at_frame(self, 
                            cell_idx: int, 
                            frame_idx: int, 
                            masks: np.ndarray) -> Optional[int]:
        """특정 프레임의 특정 세포가 속한 트랙 찾기"""
        cell_ids = np.unique(masks)[1:]
        
        if cell_idx >= len(cell_ids):
            return None
        
        # 해당 프레임에서 활성화된 트랙 찾기
        for track_id, track in self.tracks.items():
            if frame_idx in track.frames:
                frame_position = track.frames.index(frame_idx)
                if frame_position == len(track.frames) - 1:  # 가장 최근 프레임
                    return track_id
        
        return None
    
    def _get_active_tracks(self, frame_idx: int) -> List[int]:
        """특정 프레임에서 활성화된 트랙 ID 리스트"""
        active = []
        for track_id, track in self.tracks.items():
            if track.frames and track.frames[-1] == frame_idx and track.is_alive:
                active.append(track_id)
        return active
    
    def _detect_divisions(self,
                         prev_cells: List[np.ndarray],
                         curr_cells: List[np.ndarray],
                         iou_matrix: np.ndarray,
                         matched_prev: Set[int],
                         matched_curr: Set[int],
                         frame_idx: int,
                         curr_centroids: np.ndarray):
        """
        세포 분열 감지
        1개의 이전 세포가 2개의 현재 세포와 겹치는 경우
        """
        for prev_idx in range(len(prev_cells)):
            # 이 세포와 겹치는 현재 세포들
            overlapping_curr = np.where(iou_matrix[prev_idx] >= self.division_iou_threshold)[0]
            
            if len(overlapping_curr) == 2:
                # 분열 후보
                curr_idx1, curr_idx2 = overlapping_curr
                
                # 두 딸세포의 IoU 합이 parent와 유사해야 함
                total_iou = iou_matrix[prev_idx, curr_idx1] + iou_matrix[prev_idx, curr_idx2]
                
                if total_iou >= 0.5:  # 분열로 판단
                    # 부모 트랙 찾기
                    parent_track_id = self._find_track_at_frame(prev_idx, frame_idx - 1, 
                                                                np.zeros_like(prev_cells[0]))
                    
                    if parent_track_id is not None:
                        # 부모 트랙 종료
                        self.tracks[parent_track_id].death_frame = frame_idx - 1
                        self.tracks[parent_track_id].division_frame = frame_idx
                        
                        # 딸세포 트랙 생성
                        for curr_idx in [curr_idx1, curr_idx2]:
                            centroid = curr_centroids[curr_idx]
                            area = np.sum(curr_cells[curr_idx])
                            
                            daughter_track = CellTrack(
                                track_id=self.next_track_id,
                                birth_frame=frame_idx,
                                parent_id=parent_track_id
                            )
                            daughter_track.add_observation(frame_idx, centroid, area)
                            
                            # 부모-딸 관계 기록
                            self.tracks[parent_track_id].daughter_ids.append(self.next_track_id)
                            self.tracks[self.next_track_id] = daughter_track
                            
                            # 계보 그래프에 추가
                            self.graph.add_node(self.next_track_id, frame=frame_idx)
                            self.graph.add_edge(parent_track_id, self.next_track_id, 
                                              event='division', frame=frame_idx)
                            
                            self.next_track_id += 1
                        
                        # 매칭된 것으로 표시
                        matched_prev.add(prev_idx)
                        matched_curr.update([curr_idx1, curr_idx2])
    
    def build_lineage_tree(self) -> nx.DiGraph:
        """
        세포 계보 트리 구축
        
        Returns:
            NetworkX DiGraph (부모 → 딸세포 방향)
        """
        # 이미 tracking 중에 graph 구축되었음
        return self.graph
    
    def get_statistics(self) -> Dict:
        """추적 통계 반환"""
        total_tracks = len(self.tracks)
        active_tracks = sum(1 for t in self.tracks.values() if t.is_alive)
        dead_tracks = total_tracks - active_tracks
        
        divisions = sum(1 for t in self.tracks.values() if t.division_frame is not None)
        
        avg_lifetime = np.mean([t.lifetime for t in self.tracks.values()]) if self.tracks else 0
        avg_displacement = np.mean([t.total_displacement for t in self.tracks.values()]) if self.tracks else 0
        
        return {
            'total_tracks': total_tracks,
            'active_tracks': active_tracks,
            'dead_tracks': dead_tracks,
            'division_events': divisions,
            'average_lifetime_frames': avg_lifetime,
            'average_displacement_pixels': avg_displacement
        }
    
    def export_tracks_to_csv(self, output_path: str):
        """트랙 데이터를 CSV로 저장"""
        import pandas as pd
        
        rows = []
        for track_id, track in self.tracks.items():
            for i, frame in enumerate(track.frames):
                row = {
                    'track_id': track_id,
                    'frame': frame,
                    'centroid_x': track.centroids[i][0],
                    'centroid_y': track.centroids[i][1],
                    'area': track.areas[i],
                    'parent_id': track.parent_id,
                    'birth_frame': track.birth_frame,
                    'death_frame': track.death_frame,
                    'division_frame': track.division_frame
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"트랙 데이터 저장: {output_path}")
