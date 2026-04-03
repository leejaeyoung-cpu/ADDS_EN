"""
Physician Notes Parser Service
Uses GPT-4 to extract structured clinical information from physician's notes
"""

import openai
import os
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PhysicianNotesParser:
    """Parse physician notes using GPT-4 to extract structured clinical data"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize parser with OpenAI API key
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found - parser will use mock data")
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def parse_notes(self, clinical_text: str, physician_name: Optional[str] = None) -> Dict:
        """
        Parse physician notes to extract structured clinical information
        
        Args:
            clinical_text: Raw physician notes text
            physician_name: Name of physician (optional)
        
        Returns:
            Structured clinical data dictionary
        """
        if not self.api_key:
            logger.warning("Using mock parsing - no OpenAI API key")
            return self._mock_parse(clinical_text)
        
        try:
            # GPT-4 prompt for clinical information extraction
            system_prompt = """You are a medical AI assistant that extracts structured clinical information from physician notes.

Extract the following information and return as JSON:
{
    "symptoms": [list of patient symptoms],
    "physical_exam_findings": [list of physical examination findings],
    "severity_score": integer from 1-10 (1=mild, 10=critical),
    "diagnosis_summary": "brief summary of diagnosis",
    "recommended_action": "recommended next steps or treatment plan",
    "clinical_concerns": [list of important concerns or red flags],
    "patient_history": "relevant patient history",
    "comorbidities": [list of comorbidities or complications]
}

Be precise and extract only factual information from the notes."""

            # Try new OpenAI v1.0+ API
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract structured clinical information from these physician notes:\n\n{clinical_text}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                gpt_output = response.choices[0].message.content
                
            except ImportError:
                # Fallback to old API (openai < 1.0)
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract structured clinical information from these physician notes:\n\n{clinical_text}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                gpt_output = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Try to find JSON in response
                start_idx = gpt_output.find('{')
                end_idx = gpt_output.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = gpt_output[start_idx:end_idx]
                    parsed_data = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in GPT response")
                
                logger.info(f"Successfully parsed physician notes using GPT-4")
                
                # Add metadata
                parsed_data['physician_name'] = physician_name
                parsed_data['raw_notes'] = clinical_text
                parsed_data['parsing_method'] = 'gpt-4'
                
                return parsed_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT JSON response: {e}")
                logger.error(f"GPT output: {gpt_output}")
                return self._mock_parse(clinical_text)
        
        except Exception as e:
            logger.error(f"GPT parsing failed: {e}")
            return self._mock_parse(clinical_text)
    
    def _mock_parse(self, clinical_text: str) -> Dict:
        """
        Mock parsing for when OpenAI is not available
        Uses simple keyword extraction
        """
        logger.info("Using mock parsing (no GPT)")
        
        # Simple keyword-based extraction
        text_lower = clinical_text.lower()
        
        # Extract symptoms (simple keyword matching)
        symptom_keywords = ['통증', '복통', '혈변', '체중', '감소', '피로', '발열', '오심', '구토', '설사']
        symptoms = [kw for kw in symptom_keywords if kw in text_lower]
        
        # Severity score based on text length and certain keywords
        severity_score = 5  # Default
        if '공격적' in text_lower or '심각' in text_lower or '중증' in text_lower:
            severity_score = 8
        elif '경미' in text_lower or '양호' in text_lower:
            severity_score = 3
        
        return {
            'symptoms': symptoms if symptoms else ['증상 정보 미추출'],
            'physical_exam_findings': ['Mock: 복부 압통'],
            'severity_score': severity_score,
            'diagnosis_summary': clinical_text[:100] + '...' if len(clinical_text) > 100 else clinical_text,
            'recommended_action': '정밀 검사 및 치료 계획 수립',
            'clinical_concerns': ['추가 평가 필요'],
            'patient_history': 'Mock: 기존 병력',
            'comorbidities': [],
            'physician_name': None,
            'raw_notes': clinical_text,
            'parsing_method': 'mock'
        }


def save_physician_note_to_db(
    patient_id: int,
    clinical_text: str,
    physician_name: Optional[str] = None,
    ct_analysis_id: Optional[int] = None
) -> int:
    """
    Parse and save physician note to database
    
    Args:
        patient_id: Database ID of patient
        clinical_text: Raw physician notes
        physician_name: Name of physician
        ct_analysis_id: Optional link to CT analysis
    
    Returns:
        Physician note ID
    """
    import json
    from patient_management_system.database.db_enhanced import get_session
    from patient_management_system.database.models_enhanced import PhysicianNote
    
    # Parse notes
    parser = PhysicianNotesParser()
    parsed_data = parser.parse_notes(clinical_text, physician_name)
    
    # Save to database
    db = get_session()
    
    # Ensure recommended_action is string (handle both string and list)
    recommended_action = parsed_data.get('recommended_action')
    if isinstance(recommended_action, list):
        recommended_action = ', '.join(recommended_action) if recommended_action else ''
    
    # Ensure clinical_assessment has value (NOT NULL constraint)
    clinical_assessment = parsed_data.get('diagnosis_summary')
    if not clinical_assessment:
        # Fallback: use first 100 chars of raw notes
        clinical_assessment = clinical_text[:100] if clinical_text else '진단 정보 없음'
    
    note = PhysicianNote(
        patient_id=patient_id,
        ct_analysis_id=ct_analysis_id,
        physician_name=parsed_data.get('physician_name'),
        clinical_assessment=clinical_assessment,
        severity_score=parsed_data.get('severity_score'),
        recommended_action=recommended_action,
        # Convert lists to JSON for SQLite
        symptoms=json.dumps(parsed_data.get('symptoms', []), ensure_ascii=False),
        physical_exam=json.dumps(parsed_data.get('physical_exam_findings', []), ensure_ascii=False)
    )
    
    db.add(note)
    db.commit()
    db.refresh(note)
    
    logger.info(f"Saved physician note #{note.id} for patient {patient_id}")
    
    return note.id


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    sample_notes = """
58세 남성 환자, 우측 대장에서 발견된 선암종 (T3N1M0).
3개월 전 복통과 혈변 증상 시작, 대장내시경 검사로 진단.
이전 치료력 없음. ECOG 1점으로 일상생활 가능.
가족력: 부친이 대장암 이력 있음.
현재 증상: 간헐적 복통, 체중 감소 5kg (3개월).
합병증: 경도 빈혈 (Hb 10.5), 혈압 정상.
환자 의지: 적극적 치료 원함, 항암 화학요법 동의.
    """
    
    parser = PhysicianNotesParser()
    result = parser.parse_notes(sample_notes, "Dr. Kim")
    
    print("Parsed Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
