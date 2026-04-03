"""
Fine-tuning management UI page
Export datasets and manage fine-tuning jobs
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import json


def show_finetuning_management():
    """파인튜닝 관리 페이지"""
    
    st.header("🎓 AI 모델 파인튜닝")
    st.info("분석 데이터를 활용하여 맞춤형 AI 모델을 학습시킵니다")
    
    tabs = st.tabs(["1️⃣ 데이터셋 생성", "2️⃣ 모델 학습", "3️⃣ 모델 관리", "4️⃣ 테스트"])
    
    # === TAB 1: 데이터셋 생성 ===
    with tabs[0]:
        st.markdown("### 📊 학습 데이터셋 생성")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 데이터셋 현황")
            
            # Check existing data
            dataset_path = Path('data/integrated_datasets/master_dataset.jsonl')
            
            if dataset_path.exists():
                # Count records
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    total_records = sum(1 for _ in f)
                
                st.metric("총 환자 레코드", f"{total_records}개")
                
                # Quality distribution
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    qualities = []
                    for line in f:
                        record = json.loads(line)
                        qualities.append(record.get('data_quality', {}).get('completeness', 0))
                
                avg_quality = sum(qualities) / len(qualities) if qualities else 0
                st.metric("평균 데이터 품질", f"{avg_quality*100:.0f}%")
                
                high_quality = sum(1 for q in qualities if q >= 0.8)
                st.metric("고품질 데이터 (≥80%)", f"{high_quality}개")
            else:
                st.warning("데이터셋이 없습니다. 먼저 환자 등록 및 분석을 진행하세요.")
        
        with col2:
            st.markdown("#### 생성 설정")
            
            task_type = st.selectbox(
                "학습 목적",
                [
                    "pathology_classification",
                    "risk_prediction",
                    "grade_assessment"
                ],
                format_func=lambda x: {
                    'pathology_classification': '병리 분류 & 진단',
                    'risk_prediction': '위험도 예측',
                    'grade_assessment': '등급 평가'
                }[x]
            )
            
            min_quality = st.slider(
                "최소 데이터 품질",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="이 품질 이상인 데이터만 학습에 사용"
            )
            
            if dataset_path.exists():
                estimated = sum(1 for q in qualities if q >= min_quality)
                st.info(f"📝 예상 학습 예제: {estimated}개")
        
        st.markdown("---")
        
        if st.button("🔨 데이터셋 생성", type="primary", disabled=not dataset_path.exists()):
            with st.spinner("학습 데이터셋 생성 중..."):
                try:
                    from src.ai.finetuning import FineTuningDatasetGenerator
                    
                    generator = FineTuningDatasetGenerator()
                    output_path = generator.generate_training_dataset(
                        task_type=task_type,
                        min_quality=min_quality
                    )
                    
                    # Validate
                    validation = generator.validate_dataset(output_path)
                    
                    st.success(f"✅ 데이터셋 생성 완료!")
                    st.write(f"**저장 위치:** {output_path}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 예제", validation['total_examples'])
                    with col2:
                        st.metric("유효", validation['valid_examples'])
                    with col3:
                        st.metric("오류", validation['invalid_examples'])
                    
                    if validation['errors']:
                        with st.expander("⚠️ 오류 상세"):
                            for error in validation['errors']:
                                st.text(error)
                    
                    # Download button
                    with open(output_path, 'r', encoding='utf-8') as f:
                        data = f.read()
                    
                    st.download_button(
                        "📥 데이터셋 다운로드",
                        data,
                        file_name=Path(output_path).name,
                        mime="application/jsonl"
                    )
                    
                    st.session_state['latest_dataset'] = output_path
                    
                except Exception as e:
                    st.error(f"오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # === TAB 2: 모델 학습 ===
    with tabs[1]:
        st.markdown("### 🚀 OpenAI 파인튜닝 시작")
        
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="파인튜닝 권한이 있는 API 키 필요"
        )
        
        if 'latest_dataset' in st.session_state:
            st.success(f"✓ 데이터셋 준비됨: {st.session_state['latest_dataset']}")
        else:
            st.warning("먼저 Tab 1에서 데이터셋을 생성하세요")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_model = st.selectbox(
                "Base Model",
                ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-3.5-turbo"],
                help="파인튜닝할 기본 모델"
            )
        
        with col2:
            model_suffix = st.text_input(
                "모델 접미사",
                value=f"adds_pathology",
                help="모델 식별용 이름"
            )
        
        st.markdown("---")
        
        if st.button("🎯 파인튜닝 시작", type="primary", disabled=not api_key or 'latest_dataset' not in st.session_state):
            with st.spinner("파일 업로드 및 학습 작업 생성 중..."):
                try:
                    from src.ai.finetuning import OpenAIFineTuner
                    
                    tuner = OpenAIFineTuner(api_key=api_key)
                    
                    # Upload file
                    st.info("📤 Step 1/2: 학습 파일 업로드 중...")
                    file_id = tuner.upload_training_file(st.session_state['latest_dataset'])
                    st.success(f"✓ 파일 업로드 완료: {file_id}")
                    
                    # Create job
                    st.info("🚀 Step 2/2: 파인튜닝 작업 생성 중...")
                    job = tuner.create_fine_tuning_job(
                        training_file_id=file_id,
                        model=base_model,
                        suffix=model_suffix
                    )
                    
                    st.success("✅ 파인튜닝 작업 시작!")
                    
                    st.json(job)
                    
                    st.info("""
                    📌 **다음 단계:**
                    1. 작업이 백그라운드에서 실행됩니다
                    2. Tab 3 '모델 관리'에서 진행 상황 확인
                    3. 완료되면 Tab 4 '테스트'에서 모델 테스트 가능
                    
                    ⏱ 예상 소요 시간: 10-60분 (데이터 크기에 따라)
                    """)
                    
                    st.session_state['latest_job_id'] = job['job_id']
                    
                except Exception as e:
                    st.error(f"오류: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # === TAB 3: 모델 관리 ===
    with tabs[2]:
        st.markdown("### 📋 파인튜닝 작업 관리")
        
        api_key_manage = st.text_input(
            "OpenAI API Key",
            type="password",
            key="api_key_manage"
        )
        
        if st.button("🔄 작업 목록 새로고침", disabled=not api_key_manage):
            with st.spinner("작업 목록 로딩 중..."):
                try:
                    from src.ai.finetuning import OpenAIFineTuner
                    
                    tuner = OpenAIFineTuner(api_key=api_key_manage)
                    models = tuner.list_fine_tuned_models()
                    
                    st.session_state['finetune_models'] = models
                    
                except Exception as e:
                    st.error(f"오류: {str(e)}")
        
        if 'finetune_models' in st.session_state:
            models = st.session_state['finetune_models']
            
            if models:
                st.success(f"✓ {len(models)}개 모델 발견")
                
                for model in models:
                    with st.expander(f"🤖 {model['model_id']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**작업 ID:** {model['job_id']}")
                            st.write(f"**기본 모델:** {model['base_model']}")
                            st.write(f"**상태:** {model['status']}")
                        
                        with col2:
                            st.write(f"**생성일:** {model['created_at']}")
                            
                            if st.button("상태 확인", key=f"check_{model['job_id']}"):
                                tuner = OpenAIFineTuner(api_key=api_key_manage)
                                status = tuner.check_job_status(model['job_id'])
                                st.json(status)
            else:
                st.info("파인튜닝된 모델이 없습니다")
    
    # === TAB 4: 테스트 ===
    with tabs[3]:
        st.markdown("### 🧪 모델 테스트")
        
        if 'finetune_models' in st.session_state and st.session_state['finetune_models']:
            model_ids = [m['model_id'] for m in st.session_state['finetune_models'] if m['status'] == 'succeeded']
            
            if model_ids:
                selected_model = st.selectbox("테스트할 모델", model_ids)
                
                test_prompt = st.text_area(
                    "테스트 프롬프트",
                    value="""환자 정보:
- 나이: 68세
- 성별: Male
- 암 종류: Colorectal
- 병기: III

Cellpose 정량 분석 결과:
- 검출 세포 수: 850개
- 평균 세포 면적: 245.3 px²
- 면적 변이도 (CV): 0.42
- 이질성 점수: 0.75
- 이질성 등급: High
- Clark-Evans 지수: 0.65
- 군집화 비율: 68%

위 정량 분석 결과를 해석하여 병리학적 소견을 제시하세요.""",
                    height=200
                )
                
                api_key_test = st.text_input("OpenAI API Key", type="password", key="api_key_test")
                
                if st.button("🎯 모델 테스트", disabled=not api_key_test):
                    with st.spinner("모델 추론 중..."):
                        try:
                            from src.ai.finetuning import OpenAIFineTuner
                            
                            tuner = OpenAIFineTuner(api_key=api_key_test)
                            result = tuner.test_fine_tuned_model(selected_model, test_prompt)
                            
                            st.success("✅ 추론 완료!")
                            st.markdown("### 모델 응답:")
                            st.markdown(result)
                            
                        except Exception as e:
                            st.error(f"오류: {str(e)}")
            else:
                st.warning("학습 완료된 모델이 없습니다")
        else:
            st.info("먼저 Tab 3에서 모델 목록을 로드하세요")


if __name__ == "__main__":
    show_finetuning_management()
