"""
Fine-tuning dataset generator and trainer
Converts analysis data to training format for OpenAI fine-tuning
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime


class FineTuningDatasetGenerator:
    """파인튜닝 데이터셋 생성기"""
    
    def __init__(self, dataset_dir: str = 'data/integrated_datasets'):
        self.dataset_dir = Path(dataset_dir)
        self.finetune_dir = Path('data/finetuning')
        self.finetune_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_training_dataset(
        self,
        task_type: str = 'pathology_classification',
        min_quality: float = 0.6
    ) -> str:
        """
        학습 데이터셋 생성
        
        Args:
            task_type: 학습 목적 (pathology_classification/risk_prediction/grade_assessment)
            min_quality: 최소 데이터 품질 (0-1)
            
        Returns:
            생성된 JSONL 파일 경로
        """
        # Load master dataset
        master_path = self.dataset_dir / 'master_dataset.jsonl'
        
        if not master_path.exists():
            raise FileNotFoundError(f"Master dataset not found: {master_path}")
        
        # Read records
        records = []
        with open(master_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                # Filter by quality
                if record.get('data_quality', {}).get('completeness', 0) >= min_quality:
                    records.append(record)
        
        print(f"Loaded {len(records)} records (min quality: {min_quality})")
        
        # Convert to fine-tuning format
        if task_type == 'pathology_classification':
            training_data = self._format_pathology_classification(records)
        elif task_type == 'risk_prediction':
            training_data = self._format_risk_prediction(records)
        elif task_type == 'grade_assessment':
            training_data = self._format_grade_assessment(records)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Save as JSONL
        output_path = self.finetune_dir / f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Generated {len(training_data)} training examples")
        print(f"Saved to: {output_path}")
        
        return str(output_path)
    
    def _format_pathology_classification(self, records: List[Dict]) -> List[Dict]:
        """병리 분류 학습 데이터 포맷"""
        training_data = []
        
        for record in records:
            # Skip if missing critical data
            cellpose = record.get('cellpose_quantitative', {})
            ai_image = record.get('ai_image_interpretation', {})
            
            if not cellpose or not ai_image:
                continue
            
            # Create prompt
            prompt = self._create_pathology_prompt(record)
            
            # Create completion (expected output)
            completion = self._create_pathology_completion(record)
            
            # OpenAI fine-tuning format
            training_data.append({
                "messages": [
                    {"role": "system", "content": "당신은 전문 병리학자입니다. 세포 이미지의 정량 분석 결과를 해석하여 병리학적 진단과 권장사항을 제공합니다."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })
        
        return training_data
    
    def _create_pathology_prompt(self, record: Dict) -> str:
        """병리 분석 프롬프트 생성"""
        patient = record.get('patient_demographics', {})
        cellpose = record.get('cellpose_quantitative', {})
        
        prompt = f"""환자 정보:
- 나이: {patient.get('age')}세
- 성별: {patient.get('gender')}
- 암 종류: {patient.get('cancer_type')}
- 병기: {patient.get('stage')}

Cellpose 정량 분석 결과:
- 검출 세포 수: {cellpose.get('cell_count', 0)}개
- 평균 세포 면적: {cellpose.get('morphology', {}).get('mean_area', 0):.1f} px²
- 면적 변이도 (CV): {cellpose.get('morphology', {}).get('cv_area', 0):.2f}
- 이질성 점수: {cellpose.get('heterogeneity', {}).get('overall_score', 0):.2f}
- 이질성 등급: {cellpose.get('heterogeneity', {}).get('grade', 'Unknown')}
- Clark-Evans 지수: {cellpose.get('spatial_distribution', {}).get('clark_evans_index', 1.0):.2f}
- 군집화 비율: {cellpose.get('spatial_distribution', {}).get('clustered_ratio', 0)*100:.0f}%

위 정량 분석 결과를 해석하여 병리학적 소견을 제시하세요."""
        
        return prompt
    
    def _create_pathology_completion(self, record: Dict) -> str:
        """병리 분석 완성 답변 생성 (Ground truth)"""
        ai_image = record.get('ai_image_interpretation', {})
        integrated = record.get('integrated_analysis', {})
        
        # Use AI interpretation + integrated insights as ground truth
        completion_parts = []
        
        # Morphology
        morphology = ai_image.get('morphology', {})
        if morphology:
            completion_parts.append(f"**세포 형태학:**\n{json.dumps(morphology, ensure_ascii=False, indent=2)}")
        
        # Histology
        histology = ai_image.get('histology', {})
        if histology:
            completion_parts.append(f"**조직학적 소견:**\n{json.dumps(histology, ensure_ascii=False, indent=2)}")
        
        # Grade
        grade = ai_image.get('grade', '')
        if grade:
            completion_parts.append(f"**분화도:** {grade}")
        
        # Diagnosis
        diagnosis = ai_image.get('diagnosis', '')
        if diagnosis:
            completion_parts.append(f"**진단:** {diagnosis}")
        
        # Integrated insights
        insights = integrated.get('combined_insights', [])
        if insights:
            insight_text = "\n".join([f"- {i['finding']}: {i['clinical_significance']}" for i in insights])
            completion_parts.append(f"**통합 인사이트:**\n{insight_text}")
        
        # Recommendations
        recommendations = ai_image.get('recommendations', [])
        if recommendations:
            rec_text = "\n".join([f"- {r}" for r in recommendations])
            completion_parts.append(f"**권장사항:**\n{rec_text}")
        
        return "\n\n".join(completion_parts)
    
    def _format_risk_prediction(self, records: List[Dict]) -> List[Dict]:
        """위험도 예측 학습 데이터"""
        training_data = []
        
        for record in records:
            risk = record.get('integrated_analysis', {}).get('risk_assessment', {})
            
            if not risk.get('risk_level'):
                continue
            
            # Prompt with features
            patient = record.get('patient_demographics', {})
            cellpose = record.get('cellpose_quantitative', {})
            
            prompt = f"""환자 위험도 평가:

기본 정보:
- 나이: {patient.get('age')}
- 암 종류: {patient.get('cancer_type')}
- 병기: {patient.get('stage')}

정량 분석:
- 이질성: {cellpose.get('heterogeneity', {}).get('overall_score', 0):.2f}
- 세포 수: {cellpose.get('cell_count', 0)}

위험도를 평가하세요 (Low/Moderate/High)."""
            
            # Completion with risk level and factors
            risk_factors = risk.get('risk_factors', [])
            completion = f"""위험도: {risk.get('risk_level')}

위험 인자:
{chr(10).join([f'- {f}' for f in risk_factors])}"""
            
            training_data.append({
                "messages": [
                    {"role": "system", "content": "당신은 종양학 전문의입니다. 환자의 위험도를 평가합니다."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })
        
        return training_data
    
    def _format_grade_assessment(self, records: List[Dict]) -> List[Dict]:
        """등급 평가 학습 데이터"""
        training_data = []
        
        for record in records:
            grade = record.get('patient_demographics', {}).get('grade')
            ai_grade = record.get('ai_image_interpretation', {}).get('grade')
            
            if not grade or not ai_grade:
                continue
            
            cellpose = record.get('cellpose_quantitative', {})
            
            prompt = f"""세포 형태 분석 결과:
- 이질성: {cellpose.get('heterogeneity', {}).get('overall_score', 0):.2f}
- 면적 변이도: {cellpose.get('morphology', {}).get('cv_area', 0):.2f}
- 형태 다양성: {cellpose.get('heterogeneity', {}).get('shape_diversity', 0):.2f}

분화도를 평가하세요 (well/moderate/poor)."""
            
            completion = f"분화도: {grade}\n해석: {ai_grade}"
            
            training_data.append({
                "messages": [
                    {"role": "system", "content": "당신은 병리학자입니다. 종양의 분화도를 평가합니다."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })
        
        return training_data
    
    def validate_dataset(self, jsonl_path: str) -> Dict:
        """데이터셋 검증"""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        validation = {
            'total_examples': len(data),
            'valid_examples': 0,
            'invalid_examples': 0,
            'errors': []
        }
        
        for i, item in enumerate(data):
            try:
                # Check format
                assert 'messages' in item
                assert len(item['messages']) >= 2
                assert item['messages'][0]['role'] in ['system', 'user']
                assert item['messages'][-1]['role'] == 'assistant'
                
                validation['valid_examples'] += 1
            except AssertionError as e:
                validation['invalid_examples'] += 1
                validation['errors'].append(f"Example {i}: {str(e)}")
        
        return validation


class OpenAIFineTuner:
    """OpenAI 파인튜닝 관리자"""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
    
    def upload_training_file(self, jsonl_path: str) -> str:
        """학습 파일 업로드"""
        with open(jsonl_path, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        return response.id
    
    def create_fine_tuning_job(
        self,
        training_file_id: str,
        model: str = 'gpt-4o-mini-2024-07-18',
        suffix: Optional[str] = None
    ) -> Dict:
        """파인튜닝 작업 생성"""
        job = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            suffix=suffix or f"adds_{datetime.now().strftime('%Y%m%d')}"
        )
        
        return {
            'job_id': job.id,
            'model': job.model,
            'status': job.status,
            'created_at': job.created_at
        }
    
    def check_job_status(self, job_id: str) -> Dict:
        """작업 상태 확인"""
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        
        return {
            'job_id': job.id,
            'status': job.status,
            'trained_tokens': job.trained_tokens,
            'fine_tuned_model': job.fine_tuned_model,
            'finished_at': job.finished_at
        }
    
    def list_fine_tuned_models(self) -> List[Dict]:
        """파인튜닝된 모델 목록"""
        jobs = self.client.fine_tuning.jobs.list(limit=20)
        
        models = []
        for job in jobs.data:
            if job.fine_tuned_model:
                models.append({
                    'job_id': job.id,
                    'model_id': job.fine_tuned_model,
                    'base_model': job.model,
                    'created_at': job.created_at,
                    'status': job.status
                })
        
        return models
    
    def test_fine_tuned_model(self, model_id: str, test_prompt: str) -> str:
        """파인튜닝 모델 테스트"""
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
