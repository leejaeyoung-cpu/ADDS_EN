"""
Clinical data management for precision oncology pipeline
Handles patient records, genomic variants, and clinical metadata
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd


class ClinicalDatabase:
    """Extended database for clinical, genomic, and treatment data"""
    
    def __init__(self, db_path: str = "data/clinical/clinical_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize all clinical tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                cancer_type TEXT,
                stage TEXT,
                grade TEXT,
                primary_site TEXT,
                metastasis_sites TEXT,
                ecog_score INTEGER,
                diagnosis_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 1.5 Medical documents table (NEW)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medical_documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                document_type TEXT,
                file_path TEXT,
                original_filename TEXT,
                ai_analysis_result TEXT,
                upload_date TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        
        # 2. Genomic variants table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS genomic_variants (
                variant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                gene_name TEXT,
                variant_type TEXT,
                variant_detail TEXT,
                allele_frequency REAL,
                pathogenicity TEXT,
                test_date TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        
        # 3. Clinical metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_metadata (
                metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                pathology_image_id INTEGER,
                biopsy_site TEXT,
                biopsy_date TEXT,
                tumor_size_mm REAL,
                lymph_node_status TEXT,
                ki67_index REAL,
                her2_status TEXT,
                pdl1_tps REAL,
                microsatellite_status TEXT,
                additional_markers TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        
        # 4. Patient cohorts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_cohorts (
                cohort_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                cohort_name TEXT,
                classification_date TEXT,
                quantitative_features TEXT,
                clinical_features TEXT,
                genomic_features TEXT,
                confidence_score REAL,
                classifier_version TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
            )
        """)
        
        # 5. Treatment recommendations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS treatment_recommendations (
                recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                cohort_id INTEGER,
                recommendation_date TEXT,
                primary_regimen TEXT,
                alternative_regimens TEXT,
                dosage_plan TEXT,
                schedule_plan TEXT,
                evidence_summary TEXT,
                confidence_level TEXT,
                warnings TEXT,
                recommender_version TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (cohort_id) REFERENCES patient_cohorts(cohort_id)
            )
        """)
        
        # 6. Treatment outcomes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS treatment_outcomes (
                outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                recommendation_id INTEGER,
                treatment_start_date TEXT,
                treatment_end_date TEXT,
                actual_regimen TEXT,
                response_evaluation TEXT,
                survival_months REAL,
                progression_free_months REAL,
                adverse_events TEXT,
                outcome_notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
                FOREIGN KEY (recommendation_id) REFERENCES treatment_recommendations(recommendation_id)
            )
        """)
        
        # 7. Audit logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                user_role TEXT,
                action_type TEXT,
                resource_type TEXT,
                resource_id TEXT,
                ip_address TEXT,
                details TEXT,
                system_version TEXT
            )
        """)
        
        # Create indexes
        self._create_indexes(cursor)
        
        conn.commit()
        conn.close()
    
    def _create_indexes(self, cursor):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_patient_cancer_type ON patients(cancer_type)",
            "CREATE INDEX IF NOT EXISTS idx_patient_stage ON patients(stage)",
            "CREATE INDEX IF NOT EXISTS idx_genomic_gene ON genomic_variants(gene_name)",
            "CREATE INDEX IF NOT EXISTS idx_cohort_name ON patient_cohorts(cohort_name)",
            "CREATE INDEX IF NOT EXISTS idx_recommendation_date ON treatment_recommendations(recommendation_date DESC)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    # === PATIENT MANAGEMENT ===
    
    def save_patient(self, patient_data: Dict[str, Any]) -> str:
        """Save patient record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO patients (
                patient_id, age, gender, cancer_type, stage, grade,
                primary_site, metastasis_sites, ecog_score, diagnosis_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_data['patient_id'],
            patient_data.get('age'),
            patient_data.get('gender'),
            patient_data.get('cancer_type'),
            patient_data.get('stage'),
            patient_data.get('grade'),
            patient_data.get('primary_site'),
            json.dumps(patient_data.get('metastasis_sites', [])),
            patient_data.get('ecog_score'),
            patient_data.get('diagnosis_date')
        ))
        
        patient_id = patient_data['patient_id']
        conn.commit()
        conn.close()
        
        return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            patient = dict(row)
            patient['metastasis_sites'] = json.loads(patient['metastasis_sites']) if patient['metastasis_sites'] else []
            return patient
        return None
    
    # === GENOMIC VARIANTS ===
    
    def add_genomic_variant(self, patient_id: str, variant_data: Dict[str, Any]) -> int:
        """Add genomic variant"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO genomic_variants (
                patient_id, gene_name, variant_type, variant_detail,
                allele_frequency, pathogenicity, test_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            variant_data['gene_name'],
            variant_data.get('variant_type'),
            variant_data.get('variant_detail'),
            variant_data.get('allele_frequency'),
            variant_data.get('pathogenicity'),
            variant_data.get('test_date')
        ))
        
        variant_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return variant_id
    
    def get_patient_variants(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all genomic variants for a patient"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM genomic_variants WHERE patient_id = ?", (patient_id,))
        rows = cursor.fetchall()
        
        conn.close()
        return [dict(row) for row in rows]
    
    # === COHORT CLASSIFICATION ===
    
    def save_cohort_classification(self, patient_id: str, classification: Dict[str, Any]) -> int:
        """Save patient cohort classification"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO patient_cohorts (
                patient_id, cohort_name, classification_date,
                quantitative_features, clinical_features, genomic_features,
                confidence_score, classifier_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            classification['cohort_name'],
            datetime.now().isoformat(),
            json.dumps(classification.get('quantitative_features', {})),
            json.dumps(classification.get('clinical_features', {})),
            json.dumps(classification.get('genomic_features', {})),
            classification.get('confidence_score'),
            classification.get('classifier_version', '1.0.0')
        ))
        
        cohort_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return cohort_id
    
    # === TREATMENT RECOMMENDATIONS ===
    
    def save_recommendation(self, patient_id: str, cohort_id: int, 
                          recommendation: Dict[str, Any]) -> int:
        """Save treatment recommendation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO treatment_recommendations (
                patient_id, cohort_id, recommendation_date,
                primary_regimen, alternative_regimens, dosage_plan,
                schedule_plan, evidence_summary, confidence_level,
                warnings, recommender_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            cohort_id,
            datetime.now().isoformat(),
            json.dumps(recommendation['primary_regimen']),
            json.dumps(recommendation.get('alternative_regimens', [])),
            json.dumps(recommendation.get('dosage_plan', {})),
            json.dumps(recommendation.get('schedule_plan', {})),
            json.dumps(recommendation.get('evidence_summary', {})),
            recommendation.get('confidence_level', 'Medium'),
            json.dumps(recommendation.get('warnings', [])),
            '1.0.0'
        ))
        
        recommendation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return recommendation_id
    
    def get_patient_recommendations(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all recommendations for a patient"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM treatment_recommendations 
            WHERE patient_id = ? 
            ORDER BY recommendation_date DESC
        """, (patient_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        recommendations = []
        for row in rows:
            rec = dict(row)
            rec['primary_regimen'] = json.loads(rec['primary_regimen'])
            rec['alternative_regimens'] = json.loads(rec['alternative_regimens'])
            rec['evidence_summary'] = json.loads(rec['evidence_summary'])
            recommendations.append(rec)
        
        return recommendations
    
    # === MEDICAL DOCUMENTS ===
    
    def save_medical_document(
        self,
        patient_id: str,
        document_type: str,
        file_path: str,
        original_filename: str,
        ai_analysis_result: Optional[Dict] = None
    ) -> int:
        """
        의료 문서 저장
        
        Args:
            patient_id: 환자 ID
            document_type: 문서 타입 (pathology_image/radiology_image/report)
            file_path: 저장된 파일 경로
            original_filename: 원본 파일명
            ai_analysis_result: AI 분석 결과
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO medical_documents (
                patient_id, document_type, file_path, original_filename, ai_analysis_result
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            patient_id,
            document_type,
            file_path,
            original_filename,
            json.dumps(ai_analysis_result) if ai_analysis_result else None
        ))
        
        document_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return document_id
    
    def get_patient_documents(self, patient_id: str) -> List[Dict[str, Any]]:
        """환자의 모든 의료 문서 조회"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM medical_documents 
            WHERE patient_id = ? 
            ORDER BY upload_date DESC
        """, (patient_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            doc = dict(row)
            if doc['ai_analysis_result']:
                doc['ai_analysis_result'] = json.loads(doc['ai_analysis_result'])
            documents.append(doc)
        
        return documents
