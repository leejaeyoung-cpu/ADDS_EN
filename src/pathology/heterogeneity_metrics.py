"""
Tumor heterogeneity metrics calculator
Quantifies morphological and spatial heterogeneity in tumor populations
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, variation
from typing import Dict, List
from sklearn.preprocessing import StandardScaler


class HeterogeneityAnalyzer:
    """종양 이질성 지표 계산 (IP Module 1)"""
    
    def calculate_morphological_heterogeneity(
        self, 
        cell_features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        형태학적 이질성 종합 분석
        
        Args:
            cell_features: DataFrame with area, circularity, eccentricity, etc.
            
        Returns:
            Dictionary with heterogeneity metrics
        """
        results = {}
        
        # 1. Coefficient of Variation (CV) for each feature
        for feature in ['area', 'circularity', 'eccentricity', 'solidity']:
            if feature in cell_features.columns:
                cv = self._calculate_cv(cell_features[feature])
                results[f'cv_{feature}'] = cv
        
        # 2. Size distribution entropy
        if 'area' in cell_features.columns:
            size_entropy = self._calculate_size_entropy(cell_features['area'])
            results['size_entropy'] = size_entropy
        
        # 3. Shape diversity index
        if all(f in cell_features.columns for f in ['circularity', 'eccentricity', 'solidity']):
            shape_diversity = self._calculate_shape_diversity(cell_features)
            results['shape_diversity'] = shape_diversity
        
        # 4. Overall heterogeneity score (0-1)
        results['overall_heterogeneity'] = self._calculate_overall_score(results)
        
        # 5. Heterogeneity grade
        results['heterogeneity_grade'] = self._assign_grade(
            results['overall_heterogeneity']
        )
        
        return results
    
    def _calculate_cv(self, values: pd.Series) -> float:
        """변이계수 (CV = std/mean)"""
        mean = values.mean()
        if mean == 0:
            return 0.0
        return float(values.std() / mean)
    
    def _calculate_size_entropy(self, areas: pd.Series, num_bins: int = 20) -> float:
        """크기 분포의 엔트로피"""
        hist, _ = np.histogram(areas, bins=num_bins)
        # Add pseudocount to avoid log(0)
        prob = (hist + 1) / (hist.sum() + num_bins)
        return float(entropy(prob))
    
    def _calculate_shape_diversity(self, cell_features: pd.DataFrame) -> float:
        """
        형태 다양성 지수
        여러 형태 특징을 결합하여 얼마나 다양한 형태가 존재하는지 측정
        """
        shape_features = cell_features[['circularity', 'eccentricity', 'solidity']].copy()
        
        # Standardize features
        scaler = StandardScaler()
        shape_scaled = scaler.fit_transform(shape_features)
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(shape_scaled, metric='euclidean')
        
        # Average distance = diversity
        diversity = float(np.mean(distances))
        
        return diversity
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """종합 이질성 점수 계산 (0-1)"""
        scores = []
        
        # CV scores (normalize to 0-1)
        for key in ['cv_area', 'cv_circularity']:
            if key in metrics:
                # CV > 0.5 is high heterogeneity
                score = min(metrics[key] / 0.5, 1.0)
                scores.append(score)
        
        # Entropy score (normalize to 0-1)
        if 'size_entropy' in metrics:
            # Shannon entropy for 20 bins max ~ 3.0
            score = min(metrics['size_entropy'] / 3.0, 1.0)
            scores.append(score)
        
        # Shape diversity score (normalize to 0-1)
        if 'shape_diversity' in metrics:
            # Empirical max diversity ~ 5.0
            score = min(metrics['shape_diversity'] / 5.0, 1.0)
            scores.append(score)
        
        if not scores:
            return 0.0
        
        return float(np.mean(scores))
    
    def _assign_grade(self, score: float) -> str:
        """이질성 등급 부여"""
        if score < 0.3:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def interpret_heterogeneity(self, results: Dict[str, float]) -> Dict[str, str]:
        """이질성 결과 임상적 해석"""
        interpretation = {
            'overall_assessment': '',
            'clinical_implications': [],
            'treatment_considerations': []
        }
        
        score = results.get('overall_heterogeneity', 0)
        grade = results.get('heterogeneity_grade', 'Unknown')
        
        # Overall assessment
        interpretation['overall_assessment'] = (
            f"종양 이질성: {grade} (점수 {score:.2f})"
        )
        
        # Clinical implications
        if score > 0.7:
            interpretation['clinical_implications'].extend([
                "높은 종양 이질성으로 인한 치료 반응의 가변성 예상",
                "다양한 세포 아형 존재 가능성 높음",
                "단일 표적 치료의 효과 제한적일 수 있음"
            ])
        
        # Treatment considerations
        if score > 0.6:
            interpretation['treatment_considerations'].extend([
                "다제 병용 요법(Combination therapy) 고려 필요",
                "적응적 치료(Adaptive therapy) 전략 권장",
                "치료 반응 모니터링 강화 필요"
            ])
        
        return interpretation
    
    def calculate_subpopulation_metrics(
        self, 
        cell_features: pd.DataFrame,
        cluster_labels: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        세포 아집단별 이질성 분석
        
        Args:
            cell_features: Cell morphological features
            cluster_labels: Cluster assignment for each cell
            
        Returns:
            Dictionary mapping cluster_id -> heterogeneity metrics
        """
        subpop_metrics = {}
        
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_cells = cell_features[cluster_labels == cluster_id]
            
            if len(cluster_cells) < 5:  # Too few cells
                continue
            
            metrics = self.calculate_morphological_heterogeneity(cluster_cells)
            metrics['population_size'] = len(cluster_cells)
            metrics['population_fraction'] = len(cluster_cells) / len(cell_features)
            
            subpop_metrics[int(cluster_id)] = metrics
        
        return subpop_metrics
