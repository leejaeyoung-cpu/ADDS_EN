"""
Batch Image Processor
다중 이미지 병렬 분석 엔진
"""

import numpy as np
from typing import List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class BatchImageProcessor:
    """
    다중 이미지 배치 처리 엔진
    
    Features:
    - 병렬 처리 (multi-threading)
    - 진행률 표시
    - 에러 핸들링
    - GPU 배치 처리 지원
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Args:
            max_workers: 최대 병렬 작업자 수
        """
        self.max_workers = max_workers
        logger.info(f"BatchImageProcessor initialized (max_workers: {max_workers})")
    
    def process_multiple_images(
        self,
        images: List,
        analyzer_func: Callable,
        analyzer_type: str,
        **kwargs
    ) -> List[Dict]:
        """
        여러 이미지를 병렬로 분석
        
        Args:
            images: 업로드된 이미지 파일 리스트
            analyzer_func: 분석 함수 (CTAnalyzer.analyze_ct_image 등)
            analyzer_type: 분석 타입 ('pathology', 'ct', 'mri')
            **kwargs: 분석 함수에 전달할 추가 인자
        
        Returns:
            분석 결과 리스트
        """
        if not images:
            return []
        
        logger.info(f"Processing {len(images)} {analyzer_type} images")
        
        results = []
        
        # 단일 이미지인 경우 단순 처리
        if len(images) == 1:
            result = self._process_single_image(
                images[0], analyzer_func, **kwargs
            )
            return [result]
        
        # 다중 이미지 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Future 생성
            future_to_image = {
                executor.submit(
                    self._process_single_image,
                    img,
                    analyzer_func,
                    **kwargs
                ): img for img in images
            }
            
            # 완료된 작업 수집
            for future in as_completed(future_to_image):
                img = future_to_image[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {img.name}: {str(e)}")
                    results.append({
                        'status': 'error',
                        'error': str(e),
                        'filename': img.name
                    })
        
        logger.info(f"Completed processing {len(results)}/{len(images)} images")
        return results
    
    def _process_single_image(
        self,
        image_file,
        analyzer_func: Callable,
        **kwargs
    ) -> Dict:
        """단일 이미지 처리"""
        import tempfile
        import os
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.name)[1]) as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 분석 실행
            result = analyzer_func(tmp_path, **kwargs)
            result['filename'] = image_file.name
            
            # 임시 파일 삭제
            try:
                os.unlink(tmp_path)
            except (PermissionError, OSError):
                pass
            
            return result
            
        except Exception as e:
            # 임시 파일 삭제 시도
            try:
                os.unlink(tmp_path)
            except:
                pass
            raise
    
    def aggregate_results(
        self,
        results: List[Dict],
        aggregation_type: str = 'average'
    ) -> Dict:
        """
        여러 이미지 분석 결과 통합
        
        Args:
            results: 개별 분석 결과 리스트
            aggregation_type: 통합 방법 ('average', 'max', 'consensus')
        
        Returns:
            통합 결과
        """
        if not results:
            return {}
        
        # 성공한 결과만 필터링
        successful_results = [r for r in results if r.get('status') == 'success']
        
        if not successful_results:
            return {
                'status': 'error',
                'error': 'No successful analyses',
                'total_images': len(results),
                'failed_images': len(results)
            }
        
        aggregated = {
            'status': 'success',
            'total_images': len(results),
            'successful_images': len(successful_results),
            'failed_images': len(results) - len(successful_results),
            'aggregation_type': aggregation_type
        }
        
        # Modality 확인
        modality = successful_results[0].get('modality', 'Unknown')
        aggregated['modality'] = modality
        
        # Aggregation 수행
        if aggregation_type == 'average':
            aggregated['aggregated_measurements'] = self._average_measurements(successful_results)
        elif aggregation_type == 'max':
            aggregated['aggregated_measurements'] = self._max_measurements(successful_results)
        elif aggregation_type == 'consensus':
            aggregated['aggregated_measurements'] = self._consensus_measurements(successful_results)
        
        # 개별 결과 요약
        aggregated['individual_summaries'] = [
            {
                'filename': r.get('filename', 'Unknown'),
                'status': r.get('status'),
                'key_finding': self._extract_key_finding(r)
            } for r in results
        ]
        
        return aggregated
    
    def _average_measurements(self, results: List[Dict]) -> Dict:
        """측정값 평균 계산"""
        measurements = {}
        
        # CT measurements
        if results[0].get('modality') == 'CT':
            diameters = [r.get('measurements', {}).get('longest_diameter_mm', 0) 
                        for r in results if r.get('measurements')]
            if diameters:
                measurements['average_longest_diameter_mm'] = float(np.mean(diameters))
                measurements['std_diameter_mm'] = float(np.std(diameters))
        
        # MRI measurements
        if results[0].get('modality') == 'MRI':
            volumes = [r.get('measurements', {}).get('tumor_volume_mm3', 0) 
                      for r in results if r.get('measurements')]
            if volumes:
                measurements['average_tumor_volume_mm3'] = float(np.mean(volumes))
                measurements['std_volume_mm3'] = float(np.std(volumes))
        
        return measurements
    
    def _max_measurements(self, results: List[Dict]) -> Dict:
        """최대값 측정"""
        measurements = {}
        
        if results[0].get('modality') == 'CT':
            diameters = [r.get('measurements', {}).get('longest_diameter_mm', 0) 
                        for r in results if r.get('measurements')]
            if diameters:
                measurements['max_longest_diameter_mm'] = float(max(diameters))
        
        return measurements
    
    def _consensus_measurements(self, results: List[Dict]) -> Dict:
        """Consensus 측정 (중앙값 기반)"""
        measurements = {}
        
        if results[0].get('modality') == 'CT':
            diameters = [r.get('measurements', {}).get('longest_diameter_mm', 0) 
                        for r in results if r.get('measurements')]
            if diameters:
                measurements['median_longest_diameter_mm'] = float(np.median(diameters))
        
        return measurements
    
    def _extract_key_finding(self, result: Dict) -> str:
        """개별 결과에서 핵심 소견 추출"""
        if result.get('status') != 'success':
            return f"Error: {result.get('error', 'Unknown')}"
        
        modality = result.get('modality')
        
        if modality == 'CT':
            diameter = result.get('measurements', {}).get('longest_diameter_mm', 0)
            return f"Longest diameter: {diameter:.1f}mm"
        
        elif modality == 'MRI':
            volume = result.get('measurements', {}).get('tumor_volume_mm3', 0)
            return f"Tumor volume: {volume:.1f}mm³"
        
        elif modality == 'Pathology':
            cell_count = result.get('cell_count', 0)
            return f"Cells detected: {cell_count}"
        
        return "Analyzed"
