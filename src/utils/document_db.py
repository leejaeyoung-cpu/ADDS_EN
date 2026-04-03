"""
Document Database for storing parsed documents and AI analysis
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class DocumentDatabase:
    """SQLite database for document storage and retrieval"""
    
    def __init__(self, db_path: str = "data/document_analysis.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT,
                upload_date TEXT,
                file_size INTEGER,
                file_path TEXT,
                
                -- Metadata
                title TEXT,
                author TEXT,
                subject TEXT,
                keywords TEXT,
                creation_date TEXT,
                
                -- Content
                full_text TEXT,
                summary TEXT,
                word_count INTEGER,
                page_count INTEGER,
                
                -- Analysis
                entities_json TEXT,
                relationships_json TEXT,
                analysis_json TEXT,
                
                -- Fine-tuning
                training_data_json TEXT,
                
                -- Status
                parsed BOOLEAN DEFAULT 0,
                analyzed BOOLEAN DEFAULT 0,
                
                -- Timestamps
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                section_title TEXT,
                section_content TEXT,
                section_order INTEGER,
                page_number INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        
        # Tables table (for extracted tables)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                table_data TEXT,
                table_caption TEXT,
                page_number INTEGER,
                table_index INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_document(
        self,
        parsed_doc: Dict[str, Any],
        file_path: str = None,
        analysis: Dict = None
    ) -> int:
        """
        Save parsed document to database
        
        Args:
            parsed_doc: Parsed document data
            file_path: Original file path
            analysis: AI analysis results
            
        Returns:
            Document ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata = parsed_doc.get('metadata', {})
        entities = parsed_doc.get('entities', {})
        full_text = parsed_doc.get('full_text', '')
        
        # Extract summary from analysis if available
        summary = ""
        if analysis and analysis.get('success'):
            if 'summary' in analysis:
                summary = analysis['summary']
            elif 'analysis' in analysis and isinstance(analysis['analysis'], dict):
                summary = analysis['analysis'].get('executive_summary', '')
        
        cursor.execute("""
            INSERT INTO documents (
                filename, file_type, upload_date, file_size, file_path,
                title, author, subject, keywords, creation_date,
                full_text, summary, word_count, page_count,
                entities_json, relationships_json, analysis_json,
                parsed, analyzed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed_doc.get('filename', 'Unknown'),
            parsed_doc.get('file_type', 'Unknown'),
            datetime.now().isoformat(),
            parsed_doc.get('file_size', 0),
            str(file_path) if file_path else None,
            metadata.get('title', ''),
            metadata.get('author', ''),
            metadata.get('subject', ''),
            metadata.get('keywords', ''),
            metadata.get('creation_date', ''),
            full_text,
            summary,
            len(full_text.split()),
            parsed_doc.get('page_count', parsed_doc.get('paragraph_count', 0)),
            json.dumps(entities, ensure_ascii=False),
            json.dumps(analysis.get('relationships', {}) if analysis else {}, ensure_ascii=False),
            json.dumps(analysis, ensure_ascii=False) if analysis else None,
            True,
            bool(analysis)
        ))
        
        document_id = cursor.lastrowid
        
        # Save sections
        for section in parsed_doc.get('sections', []):
            cursor.execute("""
                INSERT INTO document_sections (
                    document_id, section_title, section_content,
                    section_order, page_number
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                document_id,
                section.get('title', ''),
                section.get('content', ''),
                section.get('order', 0),
                section.get('page', 1)
            ))
        
        # Save tables
        for table in parsed_doc.get('tables', []):
            cursor.execute("""
                INSERT INTO document_tables (
                    document_id, table_data, table_caption,
                    page_number, table_index
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                document_id,
                json.dumps(table.get('data', []), ensure_ascii=False),
                table.get('caption', ''),
                table.get('page', 1),
                table.get('table_index', 0)
            ))
        
        conn.commit()
        conn.close()
        
        return document_id
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM documents
            ORDER BY upload_date DESC
        """)
        
        rows = cursor.fetchall()
        documents = [dict(row) for row in rows]
        
        conn.close()
        return documents
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Get single document by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        
        if row:
            doc = dict(row)
            
            # Get sections
            cursor.execute("""
                SELECT * FROM document_sections
                WHERE document_id = ?
                ORDER BY section_order
            """, (doc_id,))
            doc['sections'] = [dict(r) for r in cursor.fetchall()]
            
            # Get tables
            cursor.execute("""
                SELECT * FROM document_tables
                WHERE document_id = ?
                ORDER BY table_index
            """, (doc_id,))
            doc['tables'] = [dict(r) for r in cursor.fetchall()]
            
            conn.close()
            return doc
        
        conn.close()
        return None
    
    def search_documents(self, query: str) -> List[Dict]:
        """Search documents by text"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM documents
            WHERE filename LIKE ? OR title LIKE ? OR full_text LIKE ?
            ORDER BY upload_date DESC
        """, (f"%{query}%", f"%{query}%", f"%{query}%"))
        
        rows = cursor.fetchall()
        documents = [dict(row) for row in rows]
        
        conn.close()
        return documents
    
    def delete_document(self, doc_id: int):
        """Delete document and related data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        stats['total_documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM documents WHERE analyzed = 1")
        stats['analyzed_documents'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(word_count) FROM documents")
        stats['total_words'] = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM document_sections")
        stats['total_sections'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM document_tables")
        stats['total_tables'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def save_fine_tuning_data(self, doc_id: int, training_data: Dict):
        """Save fine-tuning data for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE documents
            SET training_data_json = ?, updated_at = ?
            WHERE id = ?
        """, (
            json.dumps(training_data, ensure_ascii=False),
            datetime.now().isoformat(),
            doc_id
        ))
        
        conn.commit()
        conn.close()
    
    def export_fine_tuning_dataset(self, output_path: str):
        """Export all fine-tuning data to JSONL format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT training_data_json FROM documents
            WHERE training_data_json IS NOT NULL
        """)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for row in cursor.fetchall():
                if row[0]:
                    f.write(row[0] + '\n')
        
        conn.close()
