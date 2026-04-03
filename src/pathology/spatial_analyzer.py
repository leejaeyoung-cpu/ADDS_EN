"""
Spatial statistics analyzer for pathology images
Analyzes cell distribution, clustering patterns, and spatial relationships
"""

import numpy as np
from scipy.spatial import distance_matrix, Voronoi, ConvexHull
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
import pandas as pd


class SpatialAnalyzer:
    """공간 통계 분석 - 세포 배열 및 군집 패턴 (IP Module 1)"""
    
    def __init__(self):
        self.dbscan_eps = 50  # pixels
        self.dbscan_min_samples = 5
    
    def analyze_spatial_distribution(self, cell_centroids: np.ndarray) -> Dict[str, float]:
        """
        종합 공간 분포 분석
        
        Args:
            cell_centroids: Nx2 array of (x, y) coordinates
            
        Returns:
            Dictionary with spatial metrics
        """
        if len(cell_centroids) < 3:
            return self._empty_results()
        
        results = {}
        
        # 1. Nearest neighbor distance
        nnd_metrics = self._calculate_nearest_neighbor_distance(cell_centroids)
        results.update(nnd_metrics)
        
        # 2. Clustering analysis
        cluster_metrics = self._analyze_clustering(cell_centroids)
        results.update(cluster_metrics)
        
        # 3. Spatial density
        density_metrics = self._calculate_spatial_density(cell_centroids)
        results.update(density_metrics)
        
        # 4. Spatial randomness (Clark-Evans index)
        results['clark_evans_index'] = self._calculate_clark_evans_index(
            nnd_metrics['mean_nnd'], 
            len(cell_centroids),
            self._estimate_study_area(cell_centroids)
        )
        
        return results
    
    def _calculate_nearest_neighbor_distance(self, centroids: np.ndarray) -> Dict[str, float]:
        """최근접 이웃 거리 계산"""
        dist_matrix = distance_matrix(centroids, centroids)
        np.fill_diagonal(dist_matrix, np.inf)
        nnd = np.min(dist_matrix, axis=1)
        
        return {
            'mean_nnd': float(np.mean(nnd)),
            'std_nnd': float(np.std(nnd)),
            'median_nnd': float(np.median(nnd)),
            'min_nnd': float(np.min(nnd)),
            'max_nnd': float(np.max(nnd))
        }
    
    def _analyze_clustering(self, centroids: np.ndarray) -> Dict[str, float]:
        """DBSCAN 기반 클러스터링 분석"""
        clustering = DBSCAN(
            eps=self.dbscan_eps, 
            min_samples=self.dbscan_min_samples
        ).fit(centroids)
        
        labels = clustering.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = np.sum(labels == -1)
        num_clustered = len(labels) - num_noise
        
        # Average cluster size
        if num_clusters > 0:
            cluster_sizes = [np.sum(labels == i) for i in range(num_clusters)]
            avg_cluster_size = np.mean(cluster_sizes)
            std_cluster_size = np.std(cluster_sizes)
        else:
            avg_cluster_size = 0
            std_cluster_size = 0
        
        return {
            'num_clusters': num_clusters,
            'clustered_ratio': float(num_clustered / len(labels)),
            'noise_ratio': float(num_noise / len(labels)),
            'avg_cluster_size': float(avg_cluster_size),
            'std_cluster_size': float(std_cluster_size)
        }
    
    def _calculate_spatial_density(self, centroids: np.ndarray) -> Dict[str, float]:
        """공간 밀도 계산"""
        # Convex hull area
        try:
            hull = ConvexHull(centroids)
            area = hull.volume  # In 2D, volume = area
            density = len(centroids) / area
        except:
            area = 0
            density = 0
        
        # Voronoi-based local density variance
        try:
            vor = Voronoi(centroids)
            # Calculate area of each Voronoi region (simplified)
            voronoi_areas = self._calculate_voronoi_areas(vor, centroids)
            density_variance = np.var(1.0 / np.array(voronoi_areas + [1]))  # Inverse area = local density
        except:
            density_variance = 0
        
        return {
            'global_density': float(density),
            'convex_hull_area': float(area),
            'density_variance': float(density_variance)
        }
    
    def _calculate_voronoi_areas(self, vor: Voronoi, centroids: np.ndarray) -> List[float]:
        """Voronoi 영역 면적 계산 (간소화)"""
        areas = []
        for region_index in vor.point_region[:min(len(centroids), 100)]:  # Limit for performance
            region = vor.regions[region_index]
            if not -1 in region and len(region) > 0:
                try:
                    polygon = [vor.vertices[i] for i in region]
                    hull = ConvexHull(polygon)
                    areas.append(hull.volume)
                except:
                    areas.append(100.0)  # Default area
        
        return areas if areas else [100.0]
    
    def _calculate_clark_evans_index(self, mean_nnd: float, n: int, area: float) -> float:
        """
        Clark-Evans index (R)
        R = 1: Random distribution
        R < 1: Clustered
        R > 1: Regular/uniform
        """
        if area == 0 or n == 0:
            return 1.0
        
        expected_nnd = 0.5 * np.sqrt(area / n)
        if expected_nnd == 0:
            return 1.0
        
        R = mean_nnd / expected_nnd
        return float(R)
    
    def _estimate_study_area(self, centroids: np.ndarray) -> float:
        """연구 영역 면적 추정"""
        x_range = np.ptp(centroids[:, 0])
        y_range = np.ptp(centroids[:, 1])
        return x_range * y_range
    
    def _empty_results(self) -> Dict[str, float]:
        """빈 결과 반환"""
        return {
            'mean_nnd': 0, 'std_nnd': 0, 'median_nnd': 0,
            'num_clusters': 0, 'clustered_ratio': 0, 'noise_ratio': 0,
            'global_density': 0, 'density_variance': 0,
            'clark_evans_index': 1.0
        }
    
    def interpret_results(self, results: Dict[str, float]) -> str:
        """결과 해석"""
        interpretation = []
        
        # Clark-Evans index interpretation
        R = results['clark_evans_index']
        if R < 0.8:
            interpretation.append(f"세포 분포가 군집화되어 있음 (R={R:.2f})")
        elif R > 1.2:
            interpretation.append(f"세포 분포가 균일함 (R={R:.2f})")
        else:
            interpretation.append(f"세포 분포가 무작위함 (R={R:.2f})")
        
        # Clustering interpretation
        if results['clustered_ratio'] > 0.7:
            interpretation.append(
                f"{results['clustered_ratio']*100:.0f}%의 세포가 "
                f"{results['num_clusters']}개 군집을 형성"
            )
        
        # Density variance interpretation
        if results['density_variance'] > 0.5:
            interpretation.append("세포 밀도의 공간적 불균일성이 높음")
        
        return " | ".join(interpretation)
