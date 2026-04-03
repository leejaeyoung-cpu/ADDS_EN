"""
Morphology Analyzer
Advanced morphological analysis including distribution analysis and abnormality detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class MorphologyAnalyzer:
    """Detailed morphological analysis of cells"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_area_distribution(
        self,
        cell_stats: pd.DataFrame
    ) -> Dict:
        """
        Comprehensive area distribution analysis
        
        Args:
            cell_stats: DataFrame with cell statistics
        
        Returns:
            Dictionary with distribution analysis
        """
        areas = cell_stats['area'].values
        
        # Basic statistics
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        median_area = np.median(areas)
        
        # Distribution shape
        skewness = stats.skew(areas)
        kurtosis = stats.kurtosis(areas)
        
        # Coefficient of variation
        cv = std_area / mean_area if mean_area > 0 else 0
        
        # Percentiles
        percentiles = {
            5: np.percentile(areas, 5),
            25: np.percentile(areas, 25),
            50: np.percentile(areas, 50),
            75: np.percentile(areas, 75),
            95: np.percentile(areas, 95),
            99: np.percentile(areas, 99)
        }
        
        # Detect outliers using IQR method
        q1 = percentiles[25]
        q3 = percentiles[75]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = np.where(
            (areas < lower_bound) | (areas > upper_bound)
        )[0]
        
        # Distribution type classification
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            dist_type = 'normal'
        elif abs(skewness) >= 0.5:
            dist_type = 'skewed'
        else:
            dist_type = 'bimodal'
        
        return {
            'statistics': {
                'mean': mean_area,
                'std': std_area,
                'median': median_area,
                'cv': cv,
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'percentiles': percentiles,
            'outliers': {
                'count': len(outlier_indices),
                'indices': outlier_indices.tolist(),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'distribution_type': dist_type
        }
    
    def detect_abnormal_cells(
        self,
        cell_stats: pd.DataFrame,
        metrics: List[str] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict:
        """
        Detect abnormal cells based on multiple metrics
        
        Args:
            cell_stats: DataFrame with cell statistics
            metrics: List of metrics to check (default: ['area', 'circularity'])
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for detection (IQR multiplier or Z-score)
        
        Returns:
            Dictionary with abnormal cell information
        """
        if metrics is None:
            metrics = ['area', 'circularity']
        
        abnormal_cells = []
        
        for metric in metrics:
            if metric not in cell_stats.columns:
                continue
            
            values = cell_stats[metric].values
            
            if method == 'iqr':
                outliers = self._detect_outliers_iqr(
                    values, threshold
                )
            elif method == 'zscore':
                outliers = self._detect_outliers_zscore(
                    values, threshold
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add to abnormal list
            for idx in outliers:
                abnormal_cells.append({
                    'cell_id': int(cell_stats.index[idx]),
                    'metric': metric,
                    'value': float(values[idx]),
                    'score': self._calculate_abnormality_score(
                        values[idx], values, method
                    ),
                    'reason': self._get_abnormality_reason(
                        metric, values[idx], values
                    )
                })
        
        # Remove duplicates, keep highest score
        unique_cells = {}
        for cell in abnormal_cells:
            cell_id = cell['cell_id']
            if cell_id not in unique_cells or cell['score'] > unique_cells[cell_id]['score']:
                unique_cells[cell_id] = cell
        
        return {
            'abnormal_cells': list(unique_cells.values()),
            'count': len(unique_cells),
            'percentage': len(unique_cells) / len(cell_stats) * 100
        }
    
    def _detect_outliers_iqr(
        self,
        values: np.ndarray,
        multiplier: float = 1.5
    ) -> List[int]:
        """Detect outliers using IQR method"""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outliers = np.where(
            (values < lower_bound) | (values > upper_bound)
        )[0]
        
        return outliers.tolist()
    
    def _detect_outliers_zscore(
        self,
        values: np.ndarray,
        threshold: float = 3.0
    ) -> List[int]:
        """Detect outliers using Z-score method"""
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values - mean) / std)
        outliers = np.where(z_scores > threshold)[0]
        
        return outliers.tolist()
    
    def _calculate_abnormality_score(
        self,
        value: float,
        all_values: np.ndarray,
        method: str
    ) -> float:
        """Calculate abnormality score (0-1)"""
        if method == 'iqr':
            q1 = np.percentile(all_values, 25)
            q3 = np.percentile(all_values, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                return 0.0
            
            # Distance from nearest quartile
            if value < q1:
                distance = (q1 - value) / iqr
            else:
                distance = (value - q3) / iqr
            
            return min(distance / 3.0, 1.0)  # Normalize to 0-1
        
        elif method == 'zscore':
            mean = np.mean(all_values)
            std = np.std(all_values)
            
            if std == 0:
                return 0.0
            
            z_score = abs((value - mean) / std)
            return min(z_score / 5.0, 1.0)  # Normalize to 0-1
        
        return 0.0
    
    def _get_abnormality_reason(
        self,
        metric: str,
        value: float,
        all_values: np.ndarray
    ) -> str:
        """Generate human-readable reason for abnormality"""
        mean = np.mean(all_values)
        
        if value > mean * 1.5:
            return f"Extremely high {metric} ({value:.1f} vs avg {mean:.1f})"
        elif value < mean * 0.5:
            return f"Extremely low {metric} ({value:.1f} vs avg {mean:.1f})"
        elif value > mean:
            return f"High {metric} ({value:.1f} vs avg {mean:.1f})"
        else:
            return f"Low {metric} ({value:.1f} vs avg {mean:.1f})"
    
    def classify_cell_morphology(
        self,
        cell_stats: pd.DataFrame
    ) -> Dict:
        """
        Classify cells by morphology
        
        Args:
            cell_stats: DataFrame with cell statistics
        
        Returns:
            Classification results
        """
        classifications = {
            'Round': [],
            'Elongated': [],
            'Irregular': [],
            'Normal': []
        }
        
        for idx, row in cell_stats.iterrows():
            circularity = row.get('circularity', 0)
            aspect_ratio = row.get('aspect_ratio', 1.0)
            
            if circularity > 0.8:
                category = 'Round'
            elif aspect_ratio > 2.0:
                category = 'Elongated'
            elif circularity < 0.5:
                category = 'Irregular'
            else:
                category = 'Normal'
            
            classifications[category].append(idx)
        
        return {
            'classification': classifications,
            'distribution': {
                k: len(v) for k, v in classifications.items()
            },
            'percentages': {
                k: len(v) / len(cell_stats) * 100
                for k, v in classifications.items()
            }
        }
    
    def find_size_clusters(
        self,
        areas: np.ndarray,
        n_clusters: int = 3
    ) -> Dict:
        """
        Cluster cells by size
        
        Args:
            areas: Array of cell areas
            n_clusters: Number of clusters (default: 3 for Small/Medium/Large)
        
        Returns:
            Clustering results
        """
        # Reshape for sklearn
        X = areas.reshape(-1, 1)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Get cluster centers in original scale
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        
        # Sort clusters by size
        sorted_indices = np.argsort(centers.flatten())
        cluster_names = ['Small', 'Medium', 'Large'][:n_clusters]
        
        # Map labels to names
        label_map = {sorted_indices[i]: cluster_names[i] for i in range(n_clusters)}
        
        # Group cell indices by cluster
        clusters = {name: [] for name in cluster_names}
        for idx, label in enumerate(labels):
            cluster_name = label_map[label]
            clusters[cluster_name].append(idx)
        
        return {
            'clusters': clusters,
            'centers': {
                cluster_names[i]: float(centers[sorted_indices[i]][0])
                for i in range(n_clusters)
            },
            'distribution': {
                name: len(indices) for name, indices in clusters.items()
            }
        }
    
    def generate_summary_report(
        self,
        cell_stats: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive morphology summary
        
        Args:
            cell_stats: DataFrame with cell statistics
        
        Returns:
            Complete analysis summary
        """
        results = {
            'distribution':self.analyze_area_distribution(cell_stats),
            'abnormalities': self.detect_abnormal_cells(cell_stats),
            'classification': self.classify_cell_morphology(cell_stats),
            'clusters': self.find_size_clusters(cell_stats['area'].values)
        }
        
        self.analysis_results = results
        return results
