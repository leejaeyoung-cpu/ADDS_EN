"""
OpenAI-powered medical document and image analyzer
Analyzes pathology images and clinical reports using GPT-4 Vision and GPT-4
"""

import os
from typing import Dict, List, Optional
import base64
from pathlib import Path
import json


class OpenAIAnalyzer:
    """OpenAI 기반 의료 문서 및 이미지 분석기"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI analyzer
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. "
                "Install with: pip install openai"
            )
    
    def analyze_pathology_image(
        self,
        image_path: str,
        cancer_type: str,
        additional_context: Optional[str] = None
    ) -> Dict:
        """
        병리 이미지 분석
        
        Args:
            image_path: 이미지 파일 경로
            cancer_type: 암 종류
            additional_context: 추가 컨텍스트
            
        Returns:
            구조화된 분석 결과
        """
        # Encode image to base64
        image_base64 = self._encode_image(image_path)
        
        # Prepare prompt
        prompt = f"""
당신은 전문 병리학자입니다. 다음 {cancer_type} 병리 이미지를 분석하여 다음 정보를 추출하세요:

1. **세포 형태학적 특징**:
   - 세포 크기 및 형태
   - 핵 특징 (크기, 형태, 염색질 패턴)
   - 세포질 특징
   
2. **조직학적 소견**:
   - 조직 구조
   - 세포 배열 패턴
   - 괴사 여부
   
3. **악성도 평가**:
   - 분화도 (well/moderate/poor)
   - 유사분열 활성도
   - 침습 패턴
   
4. **정량적 추정**:
   - 대략적인 세포 밀도
   - 이질성 정도 (low/moderate/high)
   
5. **진단 의견**:
   - 병리학적 진단
   - 추가 권장 검사

{f"추가 정보: {additional_context}" if additional_context else ""}

반드시 JSON 형식으로 응답하세요:
{{
  "morphology": {{"cell_size": "", "nuclear_features": "", "cytoplasm": ""}},
  "histology": {{"structure": "", "pattern": "", "necrosis": ""}},
  "grade": "",
  "quantitative": {{"cell_density": "", "heterogeneity": ""}},
  "diagnosis": "",
  "recommendations": []
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # GPT-4 with vision
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            
            # Try to extract JSON
            result = self._extract_json(result_text)
            
            return {
                'status': 'success',
                'analysis': result,
                'raw_response': result_text,
                'model': 'gpt-4o'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_medical_report(
        self,
        report_text: str,
        report_type: str = 'pathology',
        cancer_type: Optional[str] = None
    ) -> Dict:
        """
        의료 소견서 분석
        
        Args:
            report_text: 소견서 텍스트
            report_type: 소견서 종류 (pathology/radiology/clinical)
            cancer_type: 암 종류
            
        Returns:
            구조화된 분석 결과
        """
        prompt = f"""
당신은 전문 의료 데이터 분석가입니다. 다음 {report_type} 소견서를 분석하여 구조화된 데이터를 추출하세요.

소견서 내용:
{report_text}

다음 정보를 추출하세요:

1. **환자 기본 정보** (있는 경우):
   - 나이, 성별
   - 진단명
   
2. **주요 소견**:
   - 종양 크기 (mm)
   - 종양 위치
   - 병기 정보
   
3. **병리학적 특징**:
   - 조직학적 타입
   - 분화도 (grade)
   - 침습 깊이
   - 림프절 전이 여부
   
4. **바이오마커** (있는 경우):
   - Ki-67 지수
   - HER2 상태
   - PD-L1 발현
   - 기타 마커
   
5. **유전자 변이** (있는 경우):
   - 검출된 변이
   - 병원성 분류
   
6. **권장 사항**:
   - 추가 검사
   - 치료 권장사항

반드시 JSON 형식으로 응답하세요:
{{
  "patient_info": {{"age": null, "gender": "", "diagnosis": ""}},
  "tumor_info": {{"size_mm": null, "location": "", "stage": ""}},
  "pathology": {{"type": "", "grade": "", "invasion": "", "lymph_node": ""}},
  "biomarkers": {{"ki67": null, "her2": "", "pdl1": null}},
  "genomic_variants": [],
  "recommendations": []
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content
            result = self._extract_json(result_text)
            
            return {
                'status': 'success',
                'analysis': result,
                'raw_response': result_text
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_integrated_analysis(
        self,
        pathology_image_analysis: Dict,
        report_analysis: Dict,
        cancer_type: str
    ) -> Dict:
        """
        통합 분석 리포트 생성
        
        Args:
            pathology_image_analysis: 병리 이미지 분석 결과
            report_analysis: 소견서 분석 결과
            cancer_type: 암 종류
            
        Returns:
            통합 분석 결과
        """
        prompt = f"""
당신은 종양학 전문의입니다. 다음 정보를 통합하여 {cancer_type} 환자에 대한 종합 분석을 제공하세요.

병리 이미지 분석:
{json.dumps(pathology_image_analysis.get('analysis', {}), indent=2, ensure_ascii=False)}

소견서 분석:
{json.dumps(report_analysis.get('analysis', {}), indent=2, ensure_ascii=False)}

다음을 포함한 종합 분석을 제공하세요:

1. **환자 위험도 평가** (High/Intermediate/Low)
2. **예후 인자 분석**
3. **치료 전략 권장사항**
4. **추가 필요 검사**
5. **데이터 품질 평가** (이미지 및 소견서의 일관성)

JSON 형식으로 응답:
{{
  "risk_assessment": "",
  "prognostic_factors": [],
  "treatment_strategy": "",
  "additional_tests": [],
  "data_quality": {{"consistency": "", "completeness": ""}}
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            result = self._extract_json(result_text)
            
            return {
                'status': 'success',
                'integrated_analysis': result,
                'raw_response': result_text
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _encode_image(self, image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _extract_json(self, text: str) -> Dict:
        """텍스트에서 JSON 추출"""
        # Try to find JSON in markdown code blocks
        if '```json' in text:
            json_start = text.find('```json') + 7
            json_end = text.find('```', json_start)
            json_text = text[json_start:json_end].strip()
        elif '```' in text:
            json_start = text.find('```') + 3
            json_end = text.find('```', json_start)
            json_text = text[json_start:json_end].strip()
        else:
            # Try to find JSON object
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_text = text[json_start:json_end]
            else:
                json_text = text
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Return raw text if JSON parsing fails
            return {'raw_text': text}


class MedicalDocumentProcessor:
    """의료 문서 처리 및 저장"""
    
    def __init__(self, storage_dir: str = 'data/medical_documents'):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_dir / 'pathology_images').mkdir(exist_ok=True)
        (self.storage_dir / 'radiology_images').mkdir(exist_ok=True)
        (self.storage_dir / 'reports').mkdir(exist_ok=True)
        (self.storage_dir / 'analysis_results').mkdir(exist_ok=True)
    
    def save_uploaded_file(
        self,
        file_data: bytes,
        patient_id: str,
        file_type: str,
        original_filename: str
    ) -> str:
        """
        업로드된 파일 저장
        
        Args:
            file_data: 파일 데이터
            patient_id: 환자 ID
            file_type: 파일 타입 (pathology_image/radiology_image/report)
            original_filename: 원본 파일명
            
        Returns:
            저장된 파일 경로
        """
        # Determine subdirectory
        if file_type == 'pathology_image':
            subdir = 'pathology_images'
        elif file_type == 'radiology_image':
            subdir = 'radiology_images'
        else:
            subdir = 'reports'
        
        # Create filename
        file_ext = Path(original_filename).suffix
        safe_filename = f"{patient_id}_{file_type}{file_ext}"
        
        file_path = self.storage_dir / subdir / safe_filename
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        return str(file_path)
    
    def save_analysis_result(
        self,
        patient_id: str,
        analysis_result: Dict
    ) -> str:
        """AI 분석 결과 저장"""
        result_path = self.storage_dir / 'analysis_results' / f"{patient_id}_analysis.json"
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        return str(result_path)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """PDF에서 텍스트 추출"""
        try:
            import pypdf
            
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
            
            return text
        except ImportError:
            return "[PDF 텍스트 추출 실패: pypdf 라이브러리 필요]"
        except Exception as e:
            return f"[PDF 읽기 오류: {str(e)}]"
