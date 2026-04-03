"""
Analysis History Manager
Provides advanced filtering, statistics, and export capabilities for analysis records
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
from pathlib import Path
import json

from .analysis_db import AnalysisDatabase


class AnalysisHistoryManager:
    """
    Extended wrapper around AnalysisDatabase for history management
    Provides filtering, statistics, exports, and comparison utilities
    """
    
    def __init__(self, db_path: str = "data/analysis_results.db"):
        """
        Initialize history manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db = AnalysisDatabase(db_path)
    
    def get_history(
        self,
        filename_filter: Optional[str] = None,
        experiment_filter: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get filtered analysis history
        
        Args:
            filename_filter: Filter by image filename (partial match)
            experiment_filter: Filter by experiment name (partial match)
            date_from: Start date filter (inclusive)
            date_to: End date filter (inclusive)
            limit: Maximum number of records to return
            
        Returns:
            List of analysis records matching filters
        """
        # Get all analyses
        all_analyses = self.db.get_all_analyses()
        
        # Apply filters
        filtered = []
        for record in all_analyses:
            # Filename filter
            if filename_filter and record.get('image_name'):
                if filename_filter.lower() not in record['image_name'].lower():
                    continue
            
            # Experiment filter
            if experiment_filter and record.get('experiment_name'):
                if experiment_filter.lower() not in record['experiment_name'].lower():
                    continue
            
            # Date filters
            if date_from or date_to:
                try:
                    record_date = datetime.fromisoformat(record['timestamp']).date()
                    
                    if date_from and record_date < date_from:
                        continue
                    
                    if date_to and record_date > date_to:
                        continue
                except:
                    continue
            
            filtered.append(record)
        
        # Apply limit
        if limit:
            filtered = filtered[:limit]
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for all analyses
        
        Returns:
            Dictionary with statistical aggregates
        """
        all_analyses = self.db.get_all_analyses()
        
        if not all_analyses:
            return {
                'total_analyses': 0,
                'total_cells': 0,
                'avg_cells_per_analysis': 0,
                'avg_quality_score': 0,
                'unique_experiments': 0,
                'unique_cell_lines': 0,
                'date_range': None,
                'most_common_experiment': None,
                'most_common_cell_line': None
            }
        
        # Convert to DataFrame for easier computation
        df = pd.DataFrame(all_analyses)
        
        # Basic stats
        total_analyses = len(df)
        total_cells = df['num_cells'].sum() if 'num_cells' in df.columns else 0
        avg_cells = df['num_cells'].mean() if 'num_cells' in df.columns else 0
        avg_quality = df['quality_score'].mean() if 'quality_score' in df.columns else 0
        
        # Unique counts
        unique_experiments = df['experiment_name'].nunique() if 'experiment_name' in df.columns else 0
        unique_cell_lines = df['cell_line'].nunique() if 'cell_line' in df.columns else 0
        
        # Date range
        if 'timestamp' in df.columns:
            try:
                dates = pd.to_datetime(df['timestamp'])
                date_range = {
                    'earliest': dates.min().isoformat(),
                    'latest': dates.max().isoformat()
                }
            except:
                date_range = None
        else:
            date_range = None
        
        # Most common values
        most_common_exp = None
        if 'experiment_name' in df.columns:
            exp_counts = df['experiment_name'].value_counts()
            if len(exp_counts) > 0:
                most_common_exp = exp_counts.index[0]
        
        most_common_cell_line = None
        if 'cell_line' in df.columns:
            cell_counts = df['cell_line'].value_counts()
            if len(cell_counts) > 0:
                most_common_cell_line = cell_counts.index[0]
        
        return {
            'total_analyses': total_analyses,
            'total_cells': int(total_cells),
            'avg_cells_per_analysis': float(avg_cells),
            'avg_quality_score': float(avg_quality),
            'unique_experiments': unique_experiments,
            'unique_cell_lines': unique_cell_lines,
            'date_range': date_range,
            'most_common_experiment': most_common_exp,
            'most_common_cell_line': most_common_cell_line
        }
    
    def export_to_csv(self, output_path: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export analysis history to CSV
        
        Args:
            output_path: Path to output CSV file
            filters: Optional filters to apply (same as get_history params)
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Get filtered data
            if filters:
                records = self.get_history(**filters)
            else:
                records = self.db.get_all_analyses()
            
            if not records:
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Drop JSON column for CSV (too complex)
            if 'results_json' in df.columns:
                df = df.drop('results_json', axis=1)
            
            # Export
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def export_to_json(self, output_path: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export analysis history to JSON
        
        Args:
            output_path: Path to output JSON file
            filters: Optional filters to apply
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Get filtered data
            if filters:
                records = self.get_history(**filters)
            else:
                records = self.db.get_all_analyses()
            
            if not records:
                return False
            
            # Export
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    def export_to_excel(self, output_path: str, filters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export analysis history to Excel
        
        Args:
            output_path: Path to output Excel file
            filters: Optional filters to apply
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Get filtered data
            if filters:
                records = self.get_history(**filters)
            else:
                records = self.db.get_all_analyses()
            
            if not records:
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Drop JSON column
            if 'results_json' in df.columns:
                df = df.drop('results_json', axis=1)
            
            # Export with styling
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Analysis History')
            
            return True
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
            return False
    
    def compare_records(self, record_id_1: int, record_id_2: int) -> Optional[Dict[str, Any]]:
        """
        Compare two analysis records
        
        Args:
            record_id_1: ID of first record
            record_id_2: ID of second record
            
        Returns:
            Dictionary with comparison data, or None if records not found
        """
        rec1 = self.db.get_analysis_by_id(record_id_1)
        rec2 = self.db.get_analysis_by_id(record_id_2)
        
        if rec1 is None or rec2 is None:
            return None
        
        # Compute differences
        metrics_diff = {}
        for key in ['num_cells', 'mean_area', 'mean_circularity', 'quality_score']:
            if key in rec1 and key in rec2:
                val1 = rec1[key] or 0
                val2 = rec2[key] or 0
                diff = val2 - val1
                percent_change = (diff / val1 * 100) if val1 != 0 else 0
                
                metrics_diff[key] = {
                    'record_1': val1,
                    'record_2': val2,
                    'difference': diff,
                    'percent_change': percent_change
                }
        
        return {
            'record_1': rec1,
            'record_2': rec2,
            'metrics_comparison': metrics_diff
        }
    
    def get_trends(self, experiment_name: str, metric: str = 'num_cells') -> Optional[pd.DataFrame]:
        """
        Get trend data for a specific experiment
        
        Args:
            experiment_name: Name of experiment to analyze
            metric: Metric to track (e.g., 'num_cells', 'quality_score')
            
        Returns:
            DataFrame with timestamp and metric values, or None if no data
        """
        # Get all analyses for this experiment
        records = self.get_history(experiment_filter=experiment_name)
        
        if not records:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Filter to relevant columns
        if metric not in df.columns or 'timestamp' not in df.columns:
            return None
        
        # Sort by timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Return minimal trend data
        trend_df = df[['timestamp', metric, 'image_name']].copy()
        
        return trend_df
