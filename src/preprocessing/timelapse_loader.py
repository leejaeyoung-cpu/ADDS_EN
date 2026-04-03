"""
Time-lapse Data Loader
======================
동영상 파일과 이미지 시퀀스를 자동으로 감지하고 로딩하는 모듈

지원 형식:
- 동영상: .avi, .mp4, .mov, .mkv
- 이미지: .tif, .tiff, .png, .jpg (시퀀스)
- TIFF 스택: 멀티프레임 TIFF
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import re
from datetime import datetime, timedelta
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class TimeLapseLoader:
    """Time-lapse 데이터 로더 - 다양한 형식 지원"""
    
    SUPPORTED_VIDEO = ['.avi', '.mp4', '.mov', '.mkv', '.wmv']
    SUPPORTED_IMAGES = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    
    def __init__(self):
        self.input_path = None
        self.input_type = None
        self.metadata = {}
        
    def auto_detect_format(self, input_path: Union[str, Path]) -> str:
        """
        입력 형식 자동 감지
        
        Returns:
            'video' | 'image_sequence' | 'tiff_stack' | 'unknown'
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"입력 경로가 존재하지 않습니다: {input_path}")
        
        # 파일인 경우
        if input_path.is_file():
            ext = input_path.suffix.lower()
            
            if ext in self.SUPPORTED_VIDEO:
                return 'video'
            elif ext in ['.tif', '.tiff']:
                # TIFF 스택인지 단일 이미지인지 확인
                try:
                    img = Image.open(input_path)
                    if hasattr(img, 'n_frames') and img.n_frames > 1:
                        return 'tiff_stack'
                    else:
                        return 'single_image'
                except:
                    return 'unknown'
            elif ext in self.SUPPORTED_IMAGES:
                return 'single_image'
            else:
                return 'unknown'
        
        # 디렉토리인 경우 - 이미지 시퀀스로 간주
        elif input_path.is_dir():
            image_files = self._find_image_sequence(input_path)
            if image_files:
                return 'image_sequence'
            else:
                return 'unknown'
        
        return 'unknown'
    
    def _find_image_sequence(self, directory: Path) -> List[Path]:
        """디렉토리에서 이미지 시퀀스 찾기"""
        image_files = []
        
        for ext in self.SUPPORTED_IMAGES:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper}'))
        
        # 파일명으로 정렬 (숫자 순서 고려)
        def natural_sort_key(path):
            return [int(c) if c.isdigit() else c.lower() 
                   for c in re.split(r'(\d+)', path.name)]
        
        return sorted(image_files, key=natural_sort_key)
    
    def load(self, input_path: Union[str, Path], 
             max_frames: Optional[int] = None) -> Dict:
        """
        Time-lapse 데이터 로딩 (자동 형식 감지)
        
        Args:
            input_path: 동영상 파일 또는 이미지 시퀀스 디렉토리
            max_frames: 최대 프레임 수 (None이면 전체)
        
        Returns:
            {
                'frames': List[np.ndarray],  # 프레임 리스트
                'metadata': {
                    'total_frames': int,
                    'fps': float,
                    'duration_seconds': float,
                    'resolution': (width, height),
                    'format': str,
                    'frame_interval': str  # 예: "5 minutes"
                }
            }
        """
        self.input_path = Path(input_path)
        self.input_type = self.auto_detect_format(self.input_path)
        
        logger.info(f"감지된 형식: {self.input_type}")
        
        if self.input_type == 'video':
            return self.load_video_frames(self.input_path, max_frames)
        elif self.input_type == 'image_sequence':
            return self.load_image_sequence(self.input_path, max_frames)
        elif self.input_type == 'tiff_stack':
            return self.load_tiff_stack(self.input_path, max_frames)
        else:
            raise ValueError(f"지원하지 않는 형식: {self.input_type}")
    
    def load_video_frames(self, video_path: Path, 
                         max_frames: Optional[int] = None) -> Dict:
        """
        동영상 파일에서 프레임 추출
        
        Args:
            video_path: 동영상 파일 경로
            max_frames: 최대 프레임 수
        
        Returns:
            frames와 metadata를 포함한 딕셔너리
        """
        logger.info(f"동영상 로딩 중: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise IOError(f"동영상을 열 수 없습니다: {video_path}")
        
        # 메타데이터 추출
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            # BGR to RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"프레임 로딩 진행: {frame_count}/{total_frames}")
        
        cap.release()
        
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        metadata = {
            'total_frames': len(frames),
            'fps': fps,
            'duration_seconds': duration_seconds,
            'resolution': (width, height),
            'format': 'video',
            'source_file': str(video_path),
            'frame_interval': self._estimate_interval(fps, duration_seconds, len(frames))
        }
        
        logger.info(f"동영상 로딩 완료: {len(frames)} 프레임")
        
        return {
            'frames': frames,
            'metadata': metadata
        }
    
    def load_image_sequence(self, directory: Path, 
                           max_frames: Optional[int] = None) -> Dict:
        """
        이미지 시퀀스 로딩
        
        Args:
            directory: 이미지 파일들이 있는 디렉토리
            max_frames: 최대 프레임 수
        
        Returns:
            frames와 metadata를 포함한 딕셔너리
        """
        logger.info(f"이미지 시퀀스 로딩 중: {directory}")
        
        image_files = self._find_image_sequence(directory)
        
        if not image_files:
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {directory}")
        
        if max_frames:
            image_files = image_files[:max_frames]
        
        frames = []
        
        for i, img_path in enumerate(image_files):
            try:
                img = Image.open(img_path)
                frame = np.array(img)
                
                # Grayscale to RGB 변환
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                
                frames.append(frame)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"이미지 로딩 진행: {i+1}/{len(image_files)}")
                    
            except Exception as e:
                logger.warning(f"이미지 로딩 실패 ({img_path}): {e}")
                continue
        
        # 해상도 추출 (첫 프레임 기준)
        height, width = frames[0].shape[:2] if frames else (0, 0)
        
        # 타임스탬프 파싱 시도
        time_interval = self._parse_timestamps(image_files)
        
        metadata = {
            'total_frames': len(frames),
            'fps': None,  # 이미지 시퀀스는 FPS 없음
            'duration_seconds': None,
            'resolution': (width, height),
            'format': 'image_sequence',
            'source_directory': str(directory),
            'frame_interval': time_interval or 'unknown'
        }
        
        logger.info(f"이미지 시퀀스 로딩 완료: {len(frames)} 프레임")
        
        return {
            'frames': frames,
            'metadata': metadata
        }
    
    def load_tiff_stack(self, tiff_path: Path, 
                       max_frames: Optional[int] = None) -> Dict:
        """
        멀티프레임 TIFF 스택 로딩
        
        Args:
            tiff_path: TIFF 파일 경로
            max_frames: 최대 프레임 수
        
        Returns:
            frames와 metadata를 포함한 딕셔너리
        """
        logger.info(f"TIFF 스택 로딩 중: {tiff_path}")
        
        img = Image.open(tiff_path)
        
        frames = []
        frame_count = 0
        
        try:
            while True:
                img.seek(frame_count)
                frame = np.array(img)
                
                # Grayscale to RGB 변환
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                
                frames.append(frame)
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                    
        except EOFError:
            pass  # 모든 프레임 읽기 완료
        
        # 해상도 추출
        height, width = frames[0].shape[:2] if frames else (0, 0)
        
        metadata = {
            'total_frames': len(frames),
            'fps': None,
            'duration_seconds': None,
            'resolution': (width, height),
            'format': 'tiff_stack',
            'source_file': str(tiff_path),
            'frame_interval': 'unknown'
        }
        
        logger.info(f"TIFF 스택 로딩 완료: {len(frames)} 프레임")
        
        return {
            'frames': frames,
            'metadata': metadata
        }
    
    def _estimate_interval(self, fps: float, duration: float, 
                          num_frames: int) -> str:
        """프레임 간 시간 간격 추정"""
        if fps > 0 and num_frames > 1:
            interval_seconds = 1.0 / fps
            
            if interval_seconds < 1:
                return f"{interval_seconds*1000:.0f} milliseconds"
            elif interval_seconds < 60:
                return f"{interval_seconds:.1f} seconds"
            elif interval_seconds < 3600:
                return f"{interval_seconds/60:.1f} minutes"
            else:
                return f"{interval_seconds/3600:.1f} hours"
        
        return "unknown"
    
    def _parse_timestamps(self, image_files: List[Path]) -> Optional[str]:
        """파일명에서 타임스탬프 파싱 시도"""
        # 파일명 패턴: frame_0001.tif, img_t0005.png 등
        # 간단한 휴리스틱: 파일 수와 일반적인 촬영 간격 추정
        
        num_files = len(image_files)
        
        # 3-4일 촬영 기준 추정
        if num_files < 100:
            return "~30-60 minutes (estimated)"
        elif num_files < 500:
            return "~5-15 minutes (estimated)"
        elif num_files < 2000:
            return "~1-5 minutes (estimated)"
        else:
            return f"High frequency ({num_files} frames)"
    
    def get_frame_generator(self, input_path: Union[str, Path], 
                           batch_size: int = 10):
        """
        메모리 효율적인 프레임 제너레이터
        전체 프레임을 메모리에 로딩하지 않고 배치 단위로 읽기
        
        Args:
            input_path: 입력 경로
            batch_size: 배치 크기
        
        Yields:
            (batch_frames, batch_indices)
        """
        input_path = Path(input_path)
        input_type = self.auto_detect_format(input_path)
        
        if input_type == 'video':
            yield from self._video_frame_generator(input_path, batch_size)
        elif input_type == 'image_sequence':
            yield from self._image_sequence_generator(input_path, batch_size)
        elif input_type == 'tiff_stack':
            yield from self._tiff_stack_generator(input_path, batch_size)
    
    def _video_frame_generator(self, video_path: Path, batch_size: int):
        """동영상 프레임 제너레이터"""
        cap = cv2.VideoCapture(str(video_path))
        
        batch_frames = []
        batch_indices = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb)
            batch_indices.append(frame_idx)
            frame_idx += 1
            
            if len(batch_frames) >= batch_size:
                yield batch_frames, batch_indices
                batch_frames = []
                batch_indices = []
        
        # 남은 프레임
        if batch_frames:
            yield batch_frames, batch_indices
        
        cap.release()
    
    def _image_sequence_generator(self, directory: Path, batch_size: int):
        """이미지 시퀀스 제너레이터"""
        image_files = self._find_image_sequence(directory)
        
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_frames = []
            batch_indices = []
            
            for idx, img_path in enumerate(batch_files):
                try:
                    img = Image.open(img_path)
                    frame = np.array(img)
                    
                    if len(frame.shape) == 2:
                        frame = np.stack([frame] * 3, axis=-1)
                    
                    batch_frames.append(frame)
                    batch_indices.append(i + idx)
                except:
                    continue
            
            if batch_frames:
                yield batch_frames, batch_indices
    
    def _tiff_stack_generator(self, tiff_path: Path, batch_size: int):
        """TIFF 스택 제너레이터"""
        img = Image.open(tiff_path)
        
        batch_frames = []
        batch_indices = []
        frame_idx = 0
        
        try:
            while True:
                img.seek(frame_idx)
                frame = np.array(img)
                
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                frame_idx += 1
                
                if len(batch_frames) >= batch_size:
                    yield batch_frames, batch_indices
                    batch_frames = []
                    batch_indices = []
        except EOFError:
            # 남은 프레임
            if batch_frames:
                yield batch_frames, batch_indices
