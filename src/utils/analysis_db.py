"""
Database manager for storing analysis results
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


class AnalysisDatabase:
    """Manage analysis results in SQLite database"""
    
    def __init__(self, db_path: str = "data/analysis_results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create analysis results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT,
                num_cells INTEGER,
                mean_area REAL,
                std_area REAL,
                mean_circularity REAL,
                quality_score REAL,
                quality_grade TEXT,
                results_json TEXT,
                notes TEXT,
                tags TEXT,
                importance INTEGER DEFAULT 0,
                experiment_name TEXT,
                cell_line TEXT,
                treatment TEXT,
                concentration TEXT,
                condition TEXT,
                replicate_number INTEGER,
                processing_time_seconds REAL,
                gpu_memory_used_mb REAL,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add new columns to existing table if they don't exist (migration)
        try:
            cursor.execute("ALTER TABLE analysis_results ADD COLUMN processing_time_seconds REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE analysis_results ADD COLUMN gpu_memory_used_mb REAL")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE analysis_results ADD COLUMN model_version TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        conn.commit()
        conn.close()

    def create_indexes(self):
        """Create performance indexes for faster queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Timestamp index (most common sort field)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON analysis_results(timestamp DESC)
            """)
            
            # Image name index (for searches)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_image_name 
                ON analysis_results(image_name)
            """)
            
            # Experiment metadata composite index (for filtering)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment_metadata 
                ON analysis_results(experiment_name, cell_line, treatment)
            """)
            
            # Quality score index (for sorting by quality)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_score 
                ON analysis_results(quality_score DESC)
            """)
            
            # Created_at index (for date-based queries)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON analysis_results(created_at DESC)
            """)
            
            conn.commit()
        except sqlite3.OperationalError as e:
            # Indexes may already exist
            pass
        finally:
            conn.close()
    
    def save_analysis(self, results: Dict[str, Any], image_name: str = "Unknown", 
                     experiment_name: str = None, cell_line: str = None,
                     treatment: str = None, concentration: str = None,
                     condition: str = None, replicate_number: int = None,
                     processing_time: float = None, gpu_memory_mb: float = None,
                     model_version: str = None) -> int:
        """
        Save analysis results to database
        
        Args:
            results: Analysis results dictionary
            image_name: Name of the analyzed image
            experiment_name: Name of the experiment
            cell_line: Cell line used (e.g., HUVEC)
            treatment: Treatment applied (e.g., TNF-α)
            concentration: Concentration used (e.g., 10 ng/mL)
            condition: Experimental condition (e.g., Control, 6hr, 12hr)
            replicate_number: Replicate number (1, 2, 3, etc.)
            processing_time: Processing time in seconds
            gpu_memory_mb: GPU memory used in MB
            model_version: Model version string
            
        Returns:
            ID of the saved record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract key metrics
        seg_metadata = results.get('segmentation_metadata', {})
        metrics = results.get('metrics', {})
        quality = results.get('quality_assessment', {})
        
        # Convert numpy arrays and types to JSON-serializable format
        def convert_numpy_to_serializable(obj):
            """Recursively convert numpy arrays and types to JSON-serializable formats"""
            import numpy as np
            
            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle numpy scalar types
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.str_):
                return str(obj)
            # Handle dictionaries
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_serializable(value) for key, value in obj.items()}
            # Handle lists and tuples
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_to_serializable(item) for item in obj]
            # Handle sets (convert to list)
            elif isinstance(obj, set):
                return [convert_numpy_to_serializable(item) for item in obj]
            # Return as-is for other types
            else:
                return obj
        
        # Convert results to JSON-safe format
        serializable_results = convert_numpy_to_serializable(results)
        
        cursor.execute("""
            INSERT INTO analysis_results (
                timestamp, image_name, num_cells, mean_area, std_area,
                mean_circularity, quality_score, quality_grade, results_json,
                experiment_name, cell_line, treatment, concentration,
                condition, replicate_number, processing_time_seconds,
                gpu_memory_used_mb, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            image_name,
            seg_metadata.get('num_cells', 0),
            metrics.get('mean_area', 0.0),
            metrics.get('std_area', 0.0),
            metrics.get('mean_circularity', 0.0),
            quality.get('overall_score', 0.0),
            quality.get('overall_quality', 'N/A'),
            json.dumps(serializable_results, ensure_ascii=False),
            experiment_name,
            cell_line,
            treatment,
            concentration,
            condition,
            replicate_number,
            processing_time,
            gpu_memory_mb,
            model_version
        ))
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
    
    def get_all_analyses(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all analysis records
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of analysis records
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM analysis_results ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append(dict(row))
        
        conn.close()
        return results
    
    def get_analysis_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Get specific analysis by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM analysis_results WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def update_notes(self, record_id: int, notes: str):
        """Update notes for an analysis record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE analysis_results 
            SET notes = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (notes, record_id))
        
        conn.commit()
        conn.close()
    
    def update_tags(self, record_id: int, tags: str):
        """Update tags for an analysis record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE analysis_results 
            SET tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (tags, record_id))
        
        conn.commit()
        conn.close()
    
    def update_importance(self, record_id: int, importance: int):
        """Update importance rating (0-5)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE analysis_results 
            SET importance = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (importance, record_id))
        
        conn.commit()
        conn.close()
    
    def delete_analysis(self, record_id: int):
        """Delete an analysis record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM analysis_results WHERE id = ?", (record_id,))
        
        conn.commit()
        conn.close()
    
    def search_analyses(self, search_term: str) -> List[Dict[str, Any]]:
        """Search analyses by image name, notes, or tags"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM analysis_results 
            WHERE image_name LIKE ? OR notes LIKE ? OR tags LIKE ?
            ORDER BY timestamp DESC
        """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
        
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM analysis_results")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(num_cells) as avg_cells FROM analysis_results")
        avg_cells = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT AVG(quality_score) as avg_quality FROM analysis_results")
        avg_quality = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_analyses': total,
            'avg_cells': avg_cells,
            'avg_quality': avg_quality
        }
