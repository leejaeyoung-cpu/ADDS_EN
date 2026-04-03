"""
Clinical Note Parser Service
의사 소견 NLP 파싱 및 구조화
"""

import os
import logging
import time
import re
from typing import Dict, List, Optional
from datetime import datetime

import psycopg2
from psycopg2.extras import Json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수
DB_HOST = os.getenv('DB_HOST', 'postgres')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'adds_clinical')
DB_USER = os.getenv('DB_USER', 'adds')
DB_PASSWORD = os.getenv('DB_PASSWORD')


class ClinicalNoteParser:
    """의사 소견 파서"""
    
    def __init__(self):
        self.db = self.connect_db()
        logger.info("Note Parser initialized")
    
    def connect_db(self):
        return psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    
    def parse_note(self, note_id: int):
        """
        의사 소견 파싱
        
        추출 정보:
        1. RECIST 점수 (CR, PR, SD, PD)
        2. 부작용 (Adverse Events)
        3. 치료 변경 사항
        4. 임상 소견 키워드
        """
        logger.info(f"Parsing clinical note: {note_id}")
        
        # 노트 조회
        note_info = self.get_note_info(note_id)
        raw_text = note_info['raw_text']
        
        if not raw_text:
            logger.warning(f"Empty note: {note_id}")
            return None
        
        # 파싱
        structured_data = self.extract_structured_data(raw_text)
        
        # RECIST 점수
        recist_score = self.extract_recist_score(raw_text)
        
        # 부작용
        adverse_events = self.extract_adverse_events(raw_text)
        
        # 저장
        self.save_parsed_note(note_id, structured_data, recist_score, adverse_events)
        
        logger.info(f"Note parsed: {note_id}")
        
        return {
            'structured_data': structured_data,
            'recist_score': recist_score,
            'adverse_events': adverse_events
        }
    
    def extract_structured_data(self, text: str) -> Dict:
        """구조화된 데이터 추출"""
        data = {}
        
        # 키워드 추출
        keywords = self.extract_keywords(text)
        data['keywords'] = keywords
        
        # 증상
        data['symptoms'] = self.extract_symptoms(text)
        
        # 치료 계획 변경
        data['treatment_changes'] = self.extract_treatment_changes(text)
        
        return data
    
    def extract_keywords(self, text: str) -> List[str]:
        """임상 키워드 추출"""
        keywords = []
        
        # 주요 키워드 리스트
        keyword_patterns = [
            r'호전|개선|감소|축소',
            r'악화|증가|진행|확대',
            r'안정|유지|변화없음',
            r'통증|불편|부작용',
            r'항암|화학요법|방사선',
            r'수술|절제',
        ]
        
        for pattern in keyword_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def extract_recist_score(self, text: str) -> Optional[str]:
        """
        RECIST 1.1 점수 추출
        
        - CR (Complete Response): 완전 관해
        - PR (Partial Response): 부분 관해
        - SD (Stable Disease): 안정 병변
        - PD (Progressive Disease): 진행성 병변
        """
        text_lower = text.lower()
        
        # CR
        if any(keyword in text_lower for keyword in ['완전 관해', 'complete response', 'cr']):
            return 'CR'
        
        # PR
        if any(keyword in text_lower for keyword in ['부분 관해', 'partial response', 'pr', '호전', '감소']):
            return 'PR'
        
        # PD
        if any(keyword in text_lower for keyword in ['진행성', 'progressive disease', 'pd', '악화', '증가']):
            return 'PD'
        
        # SD
        if any(keyword in text_lower for keyword in ['안정', 'stable disease', 'sd', '유지']):
            return 'SD'
        
        return None
    
    def extract_symptoms(self, text: str) -> List[str]:
        """증상 추출"""
        symptom_patterns = [
            r'(통증|pain)',
            r'(구토|nausea|vomiting)',
            r'(설사|diarrhea)',
            r'(피로|fatigue)',
            r'(식욕부진|anorexia)',
            r'(발열|fever)',
        ]
        
        symptoms = []
        for pattern in symptom_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                symptoms.append(pattern.replace('(', '').replace(')', '').split('|')[0])
        
        return symptoms
    
    def extract_adverse_events(self, text: str) -> List[Dict]:
        """부작용 추출"""
        events = []
        
        # 간단한 패턴 매칭
        ae_patterns = {
            'nausea': r'(구토|오심|nausea)',
            'diarrhea': r'(설사|diarrhea)',
            'neutropenia': r'(호중구 감소|neutropenia)',
            'neuropathy': r'(신경병증|neuropathy)',
            'fatigue': r'(피로|fatigue)',
        }
        
        for ae_name, pattern in ae_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # 등급 추출 (Grade 1-4)
                grade_match = re.search(r'(grade|등급)\s*([1-4])', text[match.start():], re.IGNORECASE)
                grade = int(grade_match.group(2)) if grade_match else None
                
                events.append({
                    'event': ae_name,
                    'grade': grade,
                    'description': match.group(0)
                })
        
        return events
    
    def extract_treatment_changes(self, text: str) -> List[str]:
        """치료 변경 사항"""
        changes = []
        
        change_patterns = [
            r'용량\s*(감소|증가|조정)',
            r'(중단|중지|discontinue)',
            r'(변경|switch|change)',
        ]
        
        for pattern in change_patterns:
            matches = re.findall(f'.{{0,30}}{pattern}.{{0,30}}', text, re.IGNORECASE)
            changes.extend(matches)
        
        return changes
    
    def get_note_info(self, note_id: int) -> Dict:
        """노트 정보 조회"""
        with self.db.cursor() as cur:
            cur.execute("""
                SELECT patient_id, note_type, raw_text
                FROM clinical_notes
                WHERE note_id = %s
            """, (note_id,))
            row = cur.fetchone()
        
        if not row:
            raise ValueError(f"Note {note_id} not found")
        
        return {
            'patient_id': row[0],
            'note_type': row[1],
            'raw_text': row[2]
        }
    
    def save_parsed_note(
        self,
        note_id: int,
        structured_data: Dict,
        recist_score: Optional[str],
        adverse_events: List[Dict]
    ):
        """파싱 결과 저장"""
        try:
            with self.db.cursor() as cur:
                cur.execute("""
                    UPDATE clinical_notes
                    SET 
                        structured_data = %s,
                        recist_score = %s,
                        adverse_events = %s
                    WHERE note_id = %s
                """, (
                    Json(structured_data),
                    recist_score,
                    Json(adverse_events),
                    note_id
                ))
            self.db.commit()
            logger.info(f"Parsed note saved: {note_id}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save parsed note: {e}")


def main():
    """메인 서비스"""
    logger.info("Starting Clinical Note Parser Service...")
    
    parser = ClinicalNoteParser()
    
    while True:
        try:
            # Pending 노트 조회
            with parser.db.cursor() as cur:
                cur.execute("""
                    SELECT note_id
                    FROM clinical_notes
                    WHERE structured_data IS NULL
                    AND raw_text IS NOT NULL
                    LIMIT 1
                """)
                row = cur.fetchone()
            
            if row:
                note_id = row[0]
                parser.parse_note(note_id)
            else:
                logger.debug("No pending notes")
                time.sleep(30)
        
        except KeyboardInterrupt:
            logger.info("Service stopped")
            break
        except Exception as e:
            logger.error(f"Service error: {e}")
            time.sleep(10)


if __name__ == '__main__':
    main()
