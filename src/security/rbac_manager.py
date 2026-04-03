"""
Role-Based Access Control (RBAC) Manager
Implements hospital-grade security with role-based permissions
"""

from enum import Enum
from typing import List, Dict, Optional
import hashlib
import secrets


class UserRole(Enum):
    """사용자 역할 정의"""
    ADMIN = 'admin'
    PHYSICIAN = 'physician'
    PATHOLOGIST = 'pathologist'
    RESEARCHER = 'researcher'
    VIEWER = 'viewer'


class RBACManager:
    """역할 기반 접근 제어 (IP Module 4)"""
    
    def __init__(self):
        self.permissions = self._initialize_permissions()
        self.data_access_levels = self._initialize_data_access()
    
    def _initialize_permissions(self) -> Dict[UserRole, List[str]]:
        """역할별 권한 정의"""
        return {
            UserRole.ADMIN: ['*'],  # 모든 권한
            
            UserRole.PHYSICIAN: [
                'view_patient', 'edit_patient', 'add_patient',
                'view_pathology_image', 'view_quantitative_analysis',
                'view_recommendation', 'approve_treatment', 'modify_treatment',
                'view_clinical_report', 'download_report',
                'view_outcome_data', 'edit_outcome_data'
            ],
            
            UserRole.PATHOLOGIST: [
                'view_pathology_image', 'upload_pathology_image',
                'edit_image_annotation', 'run_quantitative_analysis',
                'view_quantitative_analysis', 'export_analysis',
                'view_patient_limited',  # Demographics only
                'generate_pathology_report'
            ],
            
            UserRole.RESEARCHER: [
                'view_anonymized_data', 'export_anonymized_statistics',
                'view_cohort_analysis', 'run_statistical_analysis',
                'download_anonymized_dataset'
            ],
            
            UserRole.VIEWER: [
                'view_patient', 'view_pathology_image',
                'view_recommendation', 'view_clinical_report'
            ]
        }
    
    def _initialize_data_access(self) -> Dict[UserRole, str]:
        """데이터 접근 수준 정의"""
        return {
            UserRole.ADMIN: 'full',
            UserRole.PHYSICIAN: 'identified',  # 환자 식별 정보 포함
            UserRole.PATHOLOGIST: 'limited_identified',  # 제한적 식별 정보
            UserRole.RESEARCHER: 'anonymized',  # 익명화만
            UserRole.VIEWER: 'identified'
        }
    
    def check_permission(self, user_role: UserRole, action: str) -> bool:
        """
        권한 확인
        
        Args:
            user_role: 사용자 역할
            action: 수행하려는 액션
            
        Returns:
            허용 여부
        """
        allowed_actions = self.permissions.get(user_role, [])
        
        # Admin has all permissions
        if '*' in allowed_actions:
            return True
        
        return action in allowed_actions
    
    def get_filtered_patient_data(
        self, 
        user_role: UserRole, 
        patient_data: Dict
    ) -> Dict:
        """
        역할별 환자 데이터 필터링
        
        Args:
            user_role: 사용자 역할
            patient_data: 원본 환자 데이터
            
        Returns:
            필터링된 데이터
        """
        access_level = self.data_access_levels.get(user_role, 'anonymized')
        
        if access_level == 'full' or access_level == 'identified':
            return patient_data
        
        elif access_level == 'limited_identified':
            # Pathologist - limited demographics
            return {
                'patient_id': patient_data['patient_id'],
                'age': patient_data.get('age'),
                'gender': patient_data.get('gender'),
                'cancer_type': patient_data.get('cancer_type'),
                'stage': patient_data.get('stage'),
                # Remove name, contact, etc.
            }
        
        elif access_level == 'anonymized':
            # Researcher - fully anonymized
            return self._anonymize_patient_data(patient_data)
        
        return {}
    
    def _anonymize_patient_data(self, data: Dict) -> Dict:
        """환자 정보 익명화"""
        anonymized = data.copy()
        
        # Remove identifiers
        fields_to_remove = [
            'patient_id', 'name', 'contact', 'address', 
            'email', 'phone', 'ssn', 'medical_record_number'
        ]
        
        for field in fields_to_remove:
            anonymized.pop(field, None)
        
        # Add anonymous hash ID
        if 'patient_id' in data:
            anonymized['patient_hash'] = self._hash_id(str(data['patient_id']))
        
        # Age binning
        if 'age' in data:
            age = data['age']
            if age < 50:
                anonymized['age_group'] = '<50'
            elif age < 70:
                anonymized['age_group'] = '50-69'
            else:
                anonymized['age_group'] = '>=70'
            anonymized.pop('age', None)
        
        return anonymized
    
    def _hash_id(self, patient_id: str) -> str:
        """환자 ID 해싱"""
        salt = "ADDS_SECURE_SALT_2024"  # In production, use secure random salt
        return hashlib.sha256(f"{patient_id}{salt}".encode()).hexdigest()[:16]
    
    def require_permission(self, user_role: UserRole, action: str):
        """
        권한 검증 데코레이터 (함수에 사용)
        
        Usage:
            @rbac.require_permission(UserRole.PHYSICIAN, 'view_patient')
            def view_patient_details(patient_id):
                ...
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.check_permission(user_role, action):
                    raise PermissionError(
                        f"Role {user_role.value} does not have permission for action '{action}'"
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_accessible_fields(
        self, 
        user_role: UserRole, 
        resource_type: str
    ) -> List[str]:
        """
        역할별 접근 가능 필드 리스트
        
        Args:
            user_role: 사용자 역할
            resource_type: 리소스 타입 ('patient', 'image', 'recommendation')
            
        Returns:
            접근 가능한 필드명 리스트
        """
        access_level = self.data_access_levels.get(user_role, 'anonymized')
        
        field_definitions = {
            'patient': {
                'full': ['*'],
                'identified': ['patient_id', 'name', 'age', 'gender', 'cancer_type', 'stage', 'diagnosis_date'],
                'limited_identified': ['patient_id', 'age', 'gender', 'cancer_type', 'stage'],
                'anonymized': ['age_group', 'gender', 'cancer_type', 'stage']
            },
            # Add more resource types...
        }
        
        fields = field_definitions.get(resource_type, {}).get(access_level, [])
        
        if '*' in fields:
            return ['*']  # All fields
        
        return fields


# === AUDIT LOGGING ===

class AuditLogger:
    """감사 로그 시스템 (IP Module 4)"""
    
    def __init__(self):
        from clinical.clinical_database import ClinicalDatabase
        self.db = ClinicalDatabase()
    
    def log_action(
        self,
        user_id: str,
        user_role: str,
        action_type: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict] = None,
        ip_address: Optional[str] = None
    ):
        """
        액션 로깅
        
        Args:
            user_id: 사용자 ID
            user_role: 사용자 역할
            action_type: 액션 (VIEW, EDIT, DELETE, RECOMMEND, APPROVE, etc.)
            resource_type: 리소스 타입 (PATIENT, IMAGE, RECOMMENDATION, etc.)
            resource_id: 리소스 ID
            details: 상세 내용 (dict)
            ip_address: IP 주소
        """
        import sqlite3
        import json
        from datetime import datetime
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO audit_logs (
                timestamp, user_id, user_role, action_type,
                resource_type, resource_id, ip_address, details, system_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            user_id,
            user_role,
            action_type,
            resource_type,
            resource_id,
            ip_address,
            json.dumps(details) if details else None,
            '1.0.0'
        ))
        
        conn.commit()
        conn.close()
    
    def get_audit_trail(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """감사 로그 조회"""
        import sqlite3
        
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []
        
        if resource_type:
            query += " AND resource_type = ?"
            params.append(resource_type)
        
        if resource_id:
            query += " AND resource_id = ?"
            params.append(resource_id)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        cursor.execute(query, params)
        logs = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return logs
