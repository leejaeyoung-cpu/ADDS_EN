"""
Precision Oncology UI Page
Integrated workflow for precision treatment recommendation

⚠️ DEPRECATED: This file is a duplicate of show_precision_oncology.py.
The canonical version used by app.py is:
    src/ui/show_precision_oncology.py
This file exists for standalone testing only. All new features
should be implemented in show_precision_oncology.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.clinical.clinical_database import ClinicalDatabase
from src.clinical.cohort_classifier import CohortClassifier
from src.pathology.spatial_analyzer import SpatialAnalyzer
from src.pathology.heterogeneity_metrics import HeterogeneityAnalyzer
from src.recommendation.drug_optimizer import DrugCombinationOptimizer
from src.recommendation.dosage_calculator import DosageCalculator
from src.recommendation.schedule_planner import SchedulePlanner
from src.reporting.clinical_report import ClinicalReportGenerator
from src.security.rbac_manager import RBACManager, UserRole, AuditLogger
from src.utils.ai_analyzer import generate_comprehensive_insights

# Import the literature-based prediction model from the canonical module
try:
    from src.ui.show_precision_oncology import _build_adds_predict_proba
    _HAS_LITERATURE_MODEL = True
except ImportError:
    _HAS_LITERATURE_MODEL = False


def show_precision_oncology():
    """정밀 종양학 추천 시스템 페이지"""
    
    st.header("🧬 정밀 항암 치료 추천 시스템")
    st.info("AI 기반 병리 분석 + 임상/유전자 정보 통합 → 근거 기반 맞춤형 치료 추천")
    
    # Tab structure - Expanded with XAI
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1️⃣ 환자 등록",
        "2️⃣ 병리 분석 결과",
        "3️⃣ 치료 추천",
        "4️⃣ 용량/스케줄",
        "5️⃣ 최종 리포트",
        "🔍 LIME 설명",
        "🔄 Counterfactual"
    ])
    
    # Initialize databases
    clinical_db = ClinicalDatabase()
    
    # === TAB 1: 환자 등록 ===
    with tab1:
        st.markdown("### 📋 환자 정보 입력")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 기본 정보")
            patient_id = st.text_input("환자 ID", value=f"PT-{random.randint(1000, 9999)}")
            age = st.number_input("나이", min_value=18, max_value=100, value=65)
            gender = st.selectbox("성별", ["Male", "Female"])
            
            st.markdown("#### 임상 정보")
            cancer_type = st.selectbox(
                "암 종류",
                ["Colorectal", "Lung", "Breast", "Pancreatic", "Gastric", "Prostate", 
                 "Ovarian", "Liver", "Bladder", "Renal", "Esophageal", "Head_Neck"]
            )
            stage = st.selectbox("병기", ["I", "II", "III", "IV"])
            grade = st.selectbox("등급", ["well", "moderate", "poor"])
            ecog_score = st.slider("ECOG 수행 상태", 0, 4, 1)
        
        with col2:
            st.markdown("#### 신체 계측")
            weight = st.number_input("체중 (kg)", min_value=30.0, max_value=200.0, value=70.0)
            height = st.number_input("키 (cm)", min_value=100.0, max_value=220.0, value=170.0)
            
            st.markdown("#### 바이오마커")
            ki67 = st.slider("Ki-67 지수 (%)", 0, 100, 20)
            pdl1 = st.slider("PD-L1 TPS (%)", 0, 100, 0)
            msi_status = st.selectbox("MSI 상태", ["MSS", "MSI-H"])
        
        st.markdown("---")
        st.markdown("#### 🧬 유전자 변이 정보")
        
        num_variants = st.number_input("변이 개수", min_value=0, max_value=10, value=2)
        
        genomic_variants = []
        for i in range(num_variants):
            with st.expander(f"변이 {i+1}"):
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    gene = st.selectbox(
                        "유전자",
                        ["KRAS", "EGFR", "BRAF", "TP53", "ALK", "ROS1", "NTRK", "PIK3CA"],
                        key=f"gene_{i}"
                    )
                    variant_type = st.selectbox(
                        "변이 타입",
                        ["SNV", "CNV", "Fusion", "Indel"],
                        key=f"vtype_{i}"
                    )
                with v_col2:
                    variant_detail = st.text_input("상세 변이", value="p.G12D", key=f"vdetail_{i}")
                    pathogenicity = st.selectbox(
                        "병원성",
                        ["Pathogenic", "Likely pathogenic", "VUS", "Benign"],
                        key=f"path_{i}"
                    )
                
                genomic_variants.append({
                    'gene_name': gene,
                    'variant_type': variant_type,
                    'variant_detail': variant_detail,
                    'pathogenicity': pathogenicity,
                    'allele_frequency': 0.45
                })
        
        st.markdown("---")
        st.markdown("#### 📤 의료 자료 업로드 & AI 분석")
        
        upload_col1, upload_col2 = st.columns(2)
        
        with upload_col1:
            st.markdown("**세포/병리 이미지**")
            pathology_image = st.file_uploader(
                "병리 이미지 업로드",
                type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                key='pathology_upload'
            )
            if pathology_image:
                st.image(pathology_image, caption="업로드된 병리 이미지", use_container_width=True)
            
            st.markdown("**영상 자료 (CT/MRI/PET)**")
            radiology_image = st.file_uploader(
                "영상 자료 업로드",
                type=['jpg', 'jpeg', 'png', 'dcm'],
                key='radiology_upload'
            )
            if radiology_image:
                st.image(radiology_image, caption="업로드된 영상 자료", use_container_width=True)
        
        with upload_col2:
            st.markdown("**소견서/진단서**")
            clinical_report = st.file_uploader(
                "소견서 업로드 (PDF/TXT)",
                type=['pdf', 'txt'],
                key='report_upload'
            )
            if clinical_report:
                st.success(f"✓ {clinical_report.name} 업로드 완료")
            
            # OpenAI API Key 입력
            st.markdown("**OpenAI API 설정**")
            
            # Load from environment or session state
            import os
            default_key = os.getenv('OPENAI_API_KEY', '')
            if 'saved_api_key' in st.session_state:
                default_key = st.session_state['saved_api_key']
            
            api_key = st.text_input(
                "OpenAI API Key",
                value=default_key,
                type="password",
                placeholder="sk-proj-...",
                help="AI 분석을 위한 OpenAI API 키 (환경 변수에서 자동 로드됨)"
            )
            
            # Save to session state
            if api_key:
                st.session_state['saved_api_key'] = api_key
                if default_key:
                    st.success("✓ API 키가 자동으로 로드되었습니다")
            
            enable_ai = st.checkbox("🤖 AI 자동 분석 활성화", value=True)
        
        # Show button status
        if pathology_image or clinical_report:
            # Debug info
            if not api_key:
                st.warning("⚠️ OpenAI API 키를 입력하세요")
            
            # AI 분석 버튼 - 조건 단순화
            button_disabled = not api_key or not enable_ai
            
            if st.button("🔍 AI 분석 실행", type="primary", disabled=button_disabled):
                with st.spinner("AI가 의료 자료를 분석 중입니다..."):
                    try:
                        from src.ai.openai_analyzer import OpenAIAnalyzer, MedicalDocumentProcessor
                        from src.ai.dataset_builder import IntegratedDatasetBuilder
                        
                        analyzer = OpenAIAnalyzer(api_key=api_key)
                        processor = MedicalDocumentProcessor()
                        dataset_builder = IntegratedDatasetBuilder()
                        
                        ai_results = {}
                        cellpose_results = None
                        
                        # === 1. Cellpose 자동 분석 (세포 이미지인 경우) ===
                        if pathology_image:
                            st.info("🔬 Step 1/3: Cellpose 정량 분석 실행 중...")
                            
                            # Save temporary file
                            import tempfile
                            import os
                            from src.preprocessing.image_processor import CellposeProcessor
                            from src.pathology.spatial_analyzer import SpatialAnalyzer
                            from src.pathology.heterogeneity_metrics import HeterogeneityAnalyzer
                            from PIL import Image
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                tmp_file.write(pathology_image.getvalue())
                                tmp_path = tmp_file.name
                            
                            try:
                                # Ensure numpy is available in this scope
                                import numpy as np
                                
                                # Run Cellpose
                                cellpose = CellposeProcessor(model_type='cyto', gpu=False)
                                image = Image.open(tmp_path)
                                image_array = np.array(image)
                                
                                # Get masks using correct method name
                                masks, flows, metadata = cellpose.segment_image(image_array)
                                
                                # Extract features using correct method name
                                features = cellpose.extract_morphological_features(image_array, masks)
                                
                                # Spatial analysis
                                if len(features) > 0:
                                    spatial = SpatialAnalyzer()
                                    centroids = features[['centroid_y', 'centroid_x']].values
                                    spatial_results = spatial.analyze_spatial_distribution(centroids)
                                    
                                    # Heterogeneity analysis
                                    het_analyzer = HeterogeneityAnalyzer()
                                    het_results = het_analyzer.calculate_morphological_heterogeneity(features)
                                    
                                    # Combine results
                                    cellpose_results = {
                                        'num_cells': len(features),
                                        'mean_area': float(features['area'].mean()),
                                        'std_area': float(features['area'].std()),
                                        'cv_area': float(features['area'].std() / features['area'].mean()) if features['area'].mean() > 0 else 0,
                                        'mean_circularity': float(features['circularity'].mean()) if 'circularity' in features else 0,
                                        'mean_eccentricity': float(features['eccentricity'].mean()) if 'eccentricity' in features else 0,
                                        **spatial_results,
                                        **het_results
                                    }
                                    
                                    st.session_state['cellpose_analysis'] = cellpose_results
                                    
                                    # Generate segmentation insights for Tab 2
                                    try:
                                        insights_input = {
                                            'segmentation_metadata': {
                                                'num_cells': cellpose_results['num_cells']
                                            },
                                            'metrics': {
                                                'mean_area': cellpose_results['mean_area'],
                                                'std_area': cellpose_results.get('std_area', 0),
                                                'mean_circularity': cellpose_results.get('mean_circularity', 0)
                                            },
                                            'quality_assessment': {
                                                'overall_score': 0.85,  # Estimated based on successful segmentation
                                                'overall_quality': 'Good'
                                            }
                                        }
                                        segmentation_insights = generate_comprehensive_insights(insights_input)
                                        st.session_state['segmentation_insights'] = segmentation_insights
                                    except Exception as e:
                                        # Don't fail if insights generation fails
                                        st.warning(f"인사이트 생성 중 오류 발생 (분석은 정상 완료): {str(e)}")
                                    
                                    st.success(f"✓ Cellpose 분석 완료: {len(features)}개 세포 검출")
                                else:
                                    st.warning("세포가 검출되지 않았습니다.")
                                    
                            except Exception as e:
                                st.warning(f"Cellpose 분석 실패: {str(e)}")
                                cellpose_results = None
                        
                        # === 2. OpenAI 이미지 분석 ===
                        if pathology_image:
                            st.info("🤖 Step 2/3: OpenAI 이미지 해석 중...")
                            
                            result = analyzer.analyze_pathology_image(
                                tmp_path,
                                cancer_type,
                                additional_context=f"Stage: {stage}, Grade: {grade}"
                            )
                            
                            ai_results['pathology_image'] = result
                            
                            # Clean up temporary file with error handling
                            try:
                                os.unlink(tmp_path)
                            except (PermissionError, OSError) as e:
                                # File might still be in use, schedule cleanup later
                                import atexit
                                atexit.register(lambda: os.unlink(tmp_path) if os.path.exists(tmp_path) else None)
                        
                        # === 3. OpenAI 소견서 분석 ===
                        if clinical_report:
                            st.info("📄 Step 3/3: OpenAI 소견서 분석 중...")
                            
                            report_text = ""
                            if clinical_report.type == "application/pdf":
                                # Save and extract PDF
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                    tmp_file.write(clinical_report.getvalue())
                                    tmp_path = tmp_file.name
                                
                                report_text = processor.extract_text_from_pdf(tmp_path)
                                
                                # Clean up with error handling
                                try:
                                    os.unlink(tmp_path)
                                except (PermissionError, OSError):
                                    pass  # File will be cleaned by OS on exit
                            else:
                                report_text = clinical_report.getvalue().decode('utf-8')
                            
                            result = analyzer.analyze_medical_report(
                                report_text,
                                report_type='pathology',
                                cancer_type=cancer_type
                            )
                            
                            ai_results['clinical_report'] = result
                        
                        # === 4. 통합 데이터셋 생성 ===
                        st.info("📊 통합 데이터셋 구축 중...")
                        
                        integrated_record = dataset_builder.build_comprehensive_patient_record(
                            patient_id=patient_id,
                            patient_info={
                                'age': age,
                                'gender': gender,
                                'cancer_type': cancer_type,
                                'stage': stage,
                                'grade': grade,
                                'diagnosis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
                            },
                            cellpose_analysis=cellpose_results,
                            openai_image_analysis=ai_results.get('pathology_image'),
                            openai_report_analysis=ai_results.get('clinical_report'),
                            clinical_files={
                                'pathology_image': pathology_image.name if pathology_image else None,
                                'radiology_image': radiology_image.name if radiology_image else None,
                                'clinical_report': clinical_report.name if clinical_report else None
                            }
                        )
                        
                        # Save to dataset
                        dataset_path = dataset_builder.save_to_dataset(integrated_record)
                        
                        # Store in session
                        st.session_state['ai_analysis'] = ai_results
                        st.session_state['integrated_record'] = integrated_record
                        
                        st.success("✅ 전체 분석 완료!")
                        
                        # === 결과 표시 ===
                        st.markdown("---")
                        st.markdown("### 📊 통합 분석 결과")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("검출 세포 수", f"{cellpose_results.get('num_cells', 0)}개" if cellpose_results else "N/A")
                            st.metric("이질성 점수", f"{cellpose_results.get('overall_heterogeneity', 0):.2f}" if cellpose_results else "N/A")
                        
                        with col2:
                            st.metric("AI 등급 평가", ai_results.get('pathology_image', {}).get('analysis', {}).get('grade', 'N/A'))
                            st.metric("데이터 완전성", f"{integrated_record['data_quality']['completeness']*100:.0f}%")
                        
                        with col3:
                            st.metric("위험도", integrated_record['integrated_analysis']['risk_assessment']['risk_level'])
                            st.metric("분석 신뢰도", f"{integrated_record['integrated_analysis']['confidence_score']*100:.0f}%")
                        
                        # 일관성 체크
                        if integrated_record['integrated_analysis'].get('consistency_check'):
                            with st.expander("🔍 Cellpose vs AI 일관성 체크"):
                                consistency = integrated_record['integrated_analysis']['consistency_check']
                                for key, value in consistency.items():
                                    if value == 'Consistent':
                                        st.success(f"✓ {key}: {value}")
                                    elif value == 'Inconsistent':
                                        st.warning(f"⚠ {key}: {value}")
                                    else:
                                        st.info(f"• {key}: {value}")
                        
                        # 통합 인사이트
                        if integrated_record['integrated_analysis'].get('combined_insights'):
                            with st.expander("💡 통합 인사이트"):
                                for insight in integrated_record['integrated_analysis']['combined_insights']:
                                    st.markdown(f"**[{insight['source']}]** {insight['finding']}")
                                    st.caption(f"→ {insight['clinical_significance']}")
                        
                        # 상세 결과
                        with st.expander("📋 전체 분석 데이터 (JSON)"):
                            st.json(integrated_record)
                        
                        st.info(f"💾 데이터셋 저장 위치: {dataset_path}")
                        
                    except Exception as e:
                        st.error(f"AI 분석 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        if st.button("💾 환자 정보 저장", type="primary"):
            # Save patient
            patient_data = {
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'cancer_type': cancer_type,
                'stage': stage,
                'grade': grade,
                'ecog_score': ecog_score,
                'diagnosis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
            clinical_db.save_patient(patient_data)
            
            # Save genomic variants
            for variant in genomic_variants:
                clinical_db.add_genomic_variant(patient_id, variant)
            
            # Store in session state
            st.session_state['precision_patient'] = {
                **patient_data,
                'weight': weight,
                'height': height,
                'ki67_index': ki67,
                'pdl1_tps': pdl1,
                'microsatellite_status': msi_status,
                'genomic_variants': genomic_variants
            }
            
            st.success(f"✅ 환자 {patient_id} 정보 저장 완료!")
    
    # === TAB 2: 병리 분석 결과 ===
    with tab2:
        st.markdown("### 🔬 병리 이미지 정량 분석 결과")
        
        if 'precision_patient' not in st.session_state:
            st.warning("먼저 Tab 1에서 환자 정보를 저장하세요.")
        else:
            # Check if AI analysis results exist from Tab 1
            has_ai_results = 'cellpose_analysis' in st.session_state
            
            if has_ai_results:
                # Auto-load AI analysis results
                cellpose_data = st.session_state['cellpose_analysis']
                
                # Convert cellpose format to quant_analysis format
                quant_data = {
                    'num_cells': cellpose_data.get('num_cells', 0),
                    'mean_area': cellpose_data.get('mean_area', 0),
                    'cv_area': cellpose_data.get('cv_area', 0),
                    'overall_heterogeneity': cellpose_data.get('overall_heterogeneity', 0),
                    'heterogeneity_grade': cellpose_data.get('heterogeneity_grade', 'N/A'),
                    'clustered_ratio': cellpose_data.get('clustered_ratio', 0),
                    'num_clusters': cellpose_data.get('num_clusters', 0),
                    'clark_evans_index': cellpose_data.get('clark_evans_index', 0),
                    'density_variance': cellpose_data.get('density_variance', 0),
                    'shape_diversity': cellpose_data.get('shape_diversity', 0)
                }
                
                # Save to quant_analysis for compatibility with downstream tabs
                st.session_state['quant_analysis'] = quant_data
                
                st.success("✅ Tab 1의 AI 분석 결과를 불러왔습니다")
                
            else:
                # No AI results - offer test data
                st.info("Tab 1에서 AI 분석을 실행하지 않았습니다. 테스트 데이터를 사용하거나 Tab 1로 돌아가세요.")
                
                use_test = st.checkbox("테스트 데이터 사용")
                
                if use_test:
                    # Use test quantitative data
                    quant_data = {
                        'num_cells': 850,
                        'mean_area': 245.3,
                        'cv_area': 0.42,
                        'overall_heterogeneity': 0.75,
                        'heterogeneity_grade': 'High',
                        'clustered_ratio': 0.68,
                        'num_clusters': 12,
                        'clark_evans_index': 0.65,
                        'density_variance': 0.58,
                        'shape_diversity': 3.2
                    }
                    
                    st.session_state['quant_analysis'] = quant_data
                    st.info("📊 테스트 데이터를 사용합니다")
            
            # Display results if quant_analysis exists
            if 'quant_analysis' in st.session_state:
                quant_data = st.session_state['quant_analysis']
                
                # === SECTION 1: Cellpose 정량 분석 ===
                st.markdown("---")
                st.markdown("### 1️⃣ Cellpose 정량 분석 결과")
                st.caption("🔬 컴퓨터 비전 기반 세포 형태 측정")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("총 세포 수", f"{quant_data.get('num_cells', 0):,}")
                    st.metric("평균 면적", f"{quant_data.get('mean_area', 0):.1f} px²")
                
                with col2:
                    st.metric("이질성 점수", f"{quant_data.get('overall_heterogeneity', 0):.2f}")
                    st.metric("이질성 등급", quant_data.get('heterogeneity_grade', 'N/A'))
                
                with col3:
                    st.metric("군집화 비율", f"{quant_data.get('clustered_ratio', 0)*100:.0f}%")
                    st.metric("군집 수", quant_data.get('num_clusters', 0))
                
                # === NEW SECTION 1.5: 세그멘테이션 심층 분석 ===
                st.markdown("---")
                st.markdown("### 🔬 세그멘테이션 심층 분석")
                st.caption("📋 형태학적 특징의 생물학적 해석 및 임상 의미")
                
                if 'segmentation_insights' in st.session_state:
                    insights = st.session_state['segmentation_insights']
                    
                    # Biological meaning
                    with st.expander("🧬 생물학적 의미", expanded=True):
                        st.markdown(insights.get('biological', '_분석 데이터 없음_'))
                    
                    # Key findings
                    with st.expander("🎯 주요 발견사항", expanded=True):
                        st.markdown(insights.get('key_findings', '_발견사항 없음_'))
                    
                    # Recommendations
                    with st.expander("💡 실험 및 임상 권장사항"):
                        st.markdown(insights.get('recommendations', '_권장사항 없음_'))
                    
                    # AI insights (if available)
                    ai_insights = insights.get('ai_insights')
                    if ai_insights:
                        with st.expander("🤖 AI 종합 인사이트"):
                            st.info(ai_insights)
                else:
                    st.info("💡 Tab 1에서 이미지를 분석하면 생물학적 해석이 표시됩니다")
                
                # === SECTION 2: 자체 데이터 기반 추론 ===
                st.markdown("---")
                st.markdown("### 2️⃣ 자체 데이터 기반 임상 추론")
                st.caption("📊 우리 데이터베이스 기반 통합 분석 및 위험도 평가")
                
                # Check if integrated analysis exists
                if 'integrated_record' in st.session_state:
                    integrated = st.session_state['integrated_record']
                    
                    # Risk assessment - safe nested access
                    integrated_analysis = integrated.get('integrated_analysis', {})
                    risk_info = integrated_analysis.get('risk_assessment', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        risk_level = risk_info.get('risk_level', 'Unknown')
                        risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Moderate" else "🟢"
                        st.metric("위험도 등급", f"{risk_color} {risk_level}")
                    
                    with col2:
                        confidence = integrated_analysis.get('confidence_score', 0)
                        st.metric("분석 신뢰도", f"{confidence*100:.0f}%")
                    
                    with col3:
                        data_quality = integrated.get('data_quality', {})
                        completeness = data_quality.get('completeness', 0)
                        st.metric("데이터 완전성", f"{completeness*100:.0f}%")
                    
                    # Risk factors
                    risk_factors = risk_info.get('risk_factors', [])
                    if risk_factors:
                        with st.expander("⚠️ 위험 요인 분석", expanded=True):
                            for factor in risk_factors:
                                st.warning(f"• {factor}")
                    
                    # Combined insights from integrated analysis
                    combined_insights = integrated_analysis.get('combined_insights', [])
                    if combined_insights:
                        with st.expander("💡 통합 임상 인사이트", expanded=True):
                            for insight in combined_insights:
                                source = insight.get('source', 'Unknown')
                                finding = insight.get('finding', 'N/A')
                                significance = insight.get('clinical_significance', 'N/A')
                                st.markdown(f"**[{source}]** {finding}")
                                st.caption(f"→ 임상적 의의: {significance}")
                                st.markdown("---")
                    
                    # Consistency check
                    consistency_check = integrated_analysis.get('consistency_check', {})
                    if consistency_check:
                        with st.expander("🔍 분석 일관성 검증"):
                            for key, value in consistency_check.items():
                                if value == 'Consistent':
                                    st.success(f"✓ {key}: {value}")
                                elif value == 'Inconsistent':
                                    st.warning(f"⚠ {key}: {value}")
                                else:
                                    st.info(f"• {key}: {value}")
                
                else:
                    st.info("💡 Tab 1에서 AI 분석을 실행하면 통합 추론 결과가 표시됩니다")
                
                # === SECTION 3: OpenAI 이미지 분석 ===
                st.markdown("---")
                st.markdown("### 3️⃣ OpenAI 이미지 분석")
                st.caption("🤖 GPT-4 Vision 기반 병리 이미지 해석")
                
                if 'ai_analysis' in st.session_state and st.session_state['ai_analysis'].get('pathology_image'):
                    openai_result = st.session_state['ai_analysis']['pathology_image']
                    
                    analysis = openai_result.get('analysis')
                    if analysis:
                        # Key findings
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            grade = analysis.get('grade', 'N/A')
                            st.metric("AI 등급 평가", grade)
                        
                        with col2:
                            cellularity = analysis.get('cellularity', 'N/A')
                            st.metric("세포 밀집도", cellularity)
                        
                        with col3:
                            necrosis = analysis.get('necrosis', 'N/A')
                            st.metric("괴사 여부", necrosis)
                        
                        # Detailed interpretation
                        interpretation = analysis.get('interpretation')
                        if interpretation:
                            with st.expander("📝 AI 상세 해석", expanded=True):
                                st.write(interpretation)
                        
                        # Morphological features
                        morphological_features = analysis.get('morphological_features')
                        if morphological_features:
                            with st.expander("🔬 형태학적 특징"):
                                for key, value in morphological_features.items():
                                    st.write(f"**{key}**: {value}")
                        
                        # Clinical recommendations from OpenAI
                        clinical_implications = analysis.get('clinical_implications')
                        if clinical_implications:
                            with st.expander("💊 AI 추천 치료 방향"):
                                st.info(clinical_implications)
                        
                        # Full OpenAI response
                        with st.expander("📋 전체 OpenAI 분석 데이터 (JSON)"):
                            st.json(openai_result)
                    
                    elif openai_result.get('error'):
                        st.error(f"OpenAI 분석 오류: {openai_result['error']}")
                
                else:
                    st.info("💡 Tab 1에서 이미지를 업로드하고 AI 분석을 실행하면 OpenAI 해석이 표시됩니다")
                
                # === SECTION 4: 임상적 해석 (기존 유지) ===
                st.markdown("---")
                st.markdown("### 📋 종합 임상적 해석")
                
                # Always show comprehensive interpretation
                st.markdown("#### 🔬 형태학적 종합 평가")
                
                # Cell count interpretation
                num_cells = quant_data.get('num_cells', 0)
                if num_cells < 200:
                    st.warning(f"⚠️ **낮은 세포 수** ({num_cells:,}개): 세포 생존율 감소 또는 실험 조건 재검토 필요")
                elif num_cells > 1500:
                    st.warning(f"⚠️ **매우 높은 세포 밀도** ({num_cells:,}개): Confluence 상태, contact inhibition 고려")
                else:
                    st.success(f"✅ **적정 세포 수** ({num_cells:,}개): 분석에 충분한 세포 검출")
                
                # Heterogeneity interpretation
                heterogeneity = quant_data.get('overall_heterogeneity', 0)
                if heterogeneity > 0.7:
                    st.warning(f"⚠️ **매우 높은 종양 이질성** (점수: {heterogeneity:.2f})")
                    st.info("💊 **치료 전략:** 다양한 세포 아형 존재 → 다제 병용 요법 우선 고려")
                elif heterogeneity > 0.5:
                    st.info(f"📊 **중간 수준 이질성** (점수: {heterogeneity:.2f}): 세포 집단 내 다양성 존재")
                else:
                    st.success(f"✅ **균일한 세포 집단** (점수: {heterogeneity:.2f}): 동질적 세포 특성")
                
                # Spatial distribution interpretation
                clark_evans = quant_data.get('clark_evans_index', 1.0)
                if clark_evans < 0.8:
                    st.info("📊 **공간 분포:** 군집 패턴 → 미세환경 표적 치료 중요")
                elif clark_evans > 1.2:
                    st.info("📊 **공간 분포:** 균등 분산 → 세포 간 경쟁 또는 억제 상호작용")
                else:
                    st.info("📊 **공간 분포:** 무작위 패턴 → 정상적인 세포 분포")
                
                # Clinical summary
                st.markdown("#### 💡 임상적 권장사항")
                
                recommendations = []
                
                # Based on heterogeneity
                if heterogeneity > 0.7:
                    recommendations.append("- 이질적 종양: 다중 바이오마커 검사 및 표적치료 병용 고려")
                
                # Based on cell count
                if num_cells < 300:
                    recommendations.append("- 저밀도: 세포 생존율 평가 및 배양 조건 최적화")
                elif num_cells > 1200:
                    recommendations.append("- 고밀도: 세포 주기 분석 및 증식 마커 확인")
                
                # Based on spatial pattern
                if clark_evans < 0.8:
                    recommendations.append("- 군집 패턴: 종양 미세환경 분석 및 면역치료 반응성 평가")
                
                # Always include general recommendation
                recommendations.append("- 시계열 분석을 통한 치료 반응 모니터링 권장")
                recommendations.append("- 대조군과의 정량적 비교를 통한 치료 효과 검증")
                
                for rec in recommendations:
                    st.markdown(rec)
    
    # === TAB 3: 치료 추천 ===
    with tab3:
        st.markdown("### 💊 정밀 치료 추천")
        
        if 'precision_patient' not in st.session_state or 'quant_analysis' not in st.session_state:
            st.warning("Tab 1과 Tab 2를 먼저 완료하세요.")
        else:
            if st.button("🤖 AI 추천 생성", type="primary"):
                with st.spinner("환자군 분류 및 치료 추천 생성 중..."):
                    patient = st.session_state['precision_patient']
                    quant = st.session_state['quant_analysis']
                    
                    # Classify cohort
                    classifier = CohortClassifier()
                    cohort = classifier.classify_patient(
                        quantitative_data=quant,
                        clinical_data=patient,
                        genomic_data=patient['genomic_variants']
                    )
                    
                    st.session_state['cohort'] = cohort
                    
                    # Generate recommendation
                    optimizer = DrugCombinationOptimizer()
                    recommendation = optimizer.recommend_regimen(
                        cohort_classification=cohort,
                        quantitative_results=quant,
                        clinical_profile=patient,
                        genomic_variants=patient['genomic_variants']
                    )
                    
                    st.session_state['recommendation'] = recommendation
                    
                    st.success("✅ 추천 생성 완료!")
            
            if 'recommendation' in st.session_state:
                rec = st.session_state['recommendation']
                cohort = st.session_state['cohort']
                
                # Display cohort
                st.markdown("#### 🎯 환자군 분류")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info(f"**{cohort['cohort_name']}**")
                    st.caption(cohort['cohort_description'])
                with col2:
                    st.metric("분류 신뢰도", f"{cohort['confidence_score']*100:.0f}%")
                
                # Primary regimen
                st.markdown("---")
                st.markdown("#### ⭐ 주 추천 조합")
                primary = rec['primary_regimen']
                
                st.success(f"**{primary.get('name', 'Custom Regimen')}**")
                st.write(f"**약물:** {', '.join(primary['drugs'])}")
                st.write(f"**치료 강도:** {primary.get('intensity', 'N/A')}")
                st.write(f"**신뢰도:** {rec['confidence_level']}")
                
                # Evidence
                st.markdown("#### 📋 추천 근거")
                evidence = rec['evidence_summary']
                
                st.write(f"**환자군 매칭:** {evidence.get('cohort_based_rationale', 'N/A')}")
                
                if evidence.get('quantitative_indicators'):
                    with st.expander("📊 정량 분석 근거"):
                        for ind in evidence['quantitative_indicators']:
                            st.write(f"- **{ind['metric']}** = {ind['value']}")
                            st.caption(f"  {ind['interpretation']}")
                            st.caption(f"  ➡️ {ind['decision_impact']}")
                
                if evidence.get('genomic_rationale'):
                    with st.expander("🧬 유전자 기반 근거"):
                        for g in evidence['genomic_rationale']:
                            st.write(f"- **{g['target_gene']} 변이** → {g['drug']}")
                            st.caption(f"  기전: {g['mechanism']}")
                
                # NEW: Literature Evidence Section
                # Must retrieve patient and define optimizer to avoid UnboundLocalError
                if 'precision_patient' in st.session_state:
                    patient = st.session_state['precision_patient']
                    optimizer = DrugCombinationOptimizer()  # Define here to avoid UnboundLocalError
                    
                    if optimizer.has_literature and patient.get('cancer_type'):
                        cancer_type = patient['cancer_type']
                        
                        # Try to enhance recommendation with literature
                        try:
                            enhanced_rec = optimizer.enhance_recommendation_with_literature(
                                rec,
                                patient,
                                patient.get('genomic_variants', [])
                            )
                            
                            if enhanced_rec.get('literature_evidence'):
                                with st.expander("📚 문헌 기반 증거", expanded=True):
                                    st.caption(f"💡 {cancer_type} 암에 대한 문헌 데이터")
                                    
                                    for lit_ev in enhanced_rec['literature_evidence']:
                                        st.markdown(f"**{lit_ev.get('title', 'N/A')}**")
                                        st.caption(f"Source: {lit_ev.get('source', 'N/A')}")
                                        if lit_ev.get('summary'):
                                            st.write(lit_ev['summary'])
                                        st.markdown("---")
                            
                            # Show biomarker prevalences
                            if enhanced_rec.get('biomarker_prevalences'):
                                with st.expander("📊 바이오마커 유병률"):
                                    for biomarker, prevalence in enhanced_rec['biomarker_prevalences'].items():
                                        st.write(f"• **{biomarker}**: {prevalence*100:.1f}%")
                            
                            # Show mutation frequencies
                            mutations_with_lit = [
                                alt for alt in enhanced_rec.get('genomic_alterations', [])
                                if alt.get('literature_frequency')
                            ]
                            
                            if mutations_with_lit:
                                with st.expander("📊 돌연변이 빈도"):
                                    for alt in mutations_with_lit:
                                        gene = alt.get('gene', 'Unknown')
                                        interp = alt.get('interpretation', 'N/A')
                                        st.write(f"• **{gene}**: {interp}")
                        
                        except Exception as e:
                            st.warning(f"문헌 데이터 로드 중 오류: {str(e)}")
                    
                    # Show literature summary for this cancer type
                    if optimizer.has_literature:
                        try:
                            lit_summary = optimizer.get_literature_summary(cancer_type.lower())
                            if lit_summary.get('available'):
                                with st.expander(f"ℹ️ {cancer_type} 암 문헌 정보"):
                                    if 'biomarkers' in lit_summary:
                                        st.markdown("**실행 가능 바이오마커:**")
                                        for bio, data in list(lit_summary['biomarkers'].items())[:5]:
                                            prev = data.get('prevalence')
                                            treatment = data.get('treatment', 'N/A')
                                            if prev:
                                                st.write(f"• {bio}: {prev*100:.0f}% (치료: {treatment})")
                                            else:
                                                st.write(f"• {bio}: (치료: {treatment})")
                                    
                                    if 'common_mutations' in lit_summary:
                                        st.markdown("**주요 돌연변이 (>10%):**")
                                        for gene, freq in list(lit_summary['common_mutations'].items())[:5]:
                                            st.write(f"• {gene}: {freq*100:.0f}%")
                                    
                                    if 'evidence_quality' in lit_summary:
                                        eq = lit_summary['evidence_quality']
                                        st.caption(f"📄 Literature: {eq.get('total_papers', 0)} papers, "
                                                 f"{eq.get('level_I_papers', 0)} Level I")
                        except Exception as e:
                            st.warning(f"문헌 증거 로드 실패: {str(e)}")
    
    # === TAB 4: 용량/스케줄 ===
    with tab4:
        st.markdown("### 💉 용량 계산 및 투여 스케줄")
        
        if 'recommendation' not in st.session_state:
            st.warning("먼저 Tab 3에서 추천을 생성하세요.")
        else:
            patient = st.session_state['precision_patient']
            rec = st.session_state['recommendation']
            
            # Calculate dosages
            dosage_calc = DosageCalculator()
            dosage_plan = dosage_calc.calculate_regimen_dosages(
                regimen=rec['primary_regimen'],
                patient_profile=patient
            )
            
            st.session_state['dosage_plan'] = dosage_plan
            
            # Display dosages
            st.markdown("#### 💊 개별 약물 용량")
            
            for drug_dose in dosage_plan['drugs']:
                drug_name = drug_dose.get('drug_name', drug_dose.get('drug', 'Unknown Drug'))
                final_dose = drug_dose.get('final_dose_mg', drug_dose.get('dose_mg', 0))
                
                with st.expander(f"{drug_name} - {final_dose} mg"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**기본 용량:** {drug_dose.get('base_dose_mg', 'N/A')} mg")
                        st.write(f"**최종 용량:** {final_dose} mg")
                        st.write(f"**투여 경로:** {drug_dose.get('administration_route', 'N/A')}")
                    with col2:
                        st.write(f"**BSA:** {drug_dose.get('bsa_m2', 'N/A')} m²")
                        if drug_dose.get('adjustments'):
                            st.write(f"**조정 인자:** {drug_dose['adjustments']}")
            
            # Generate schedule
            st.markdown("---")
            st.markdown("#### 📅 투여 스케줄")
            
            start_date = st.date_input("치료 시작일")
            num_cycles = st.number_input("사이클 수", min_value=1, max_value=12, value=6)
            
            if st.button("스케줄 생성"):
                planner = SchedulePlanner()
                schedule = planner.generate_schedule(
                    regimen=rec['primary_regimen'],
                    dosage_plan=dosage_plan,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    num_cycles=num_cycles
                )
                
                st.session_state['schedule'] = schedule
                
                # Display schedule
                st.success(f"✅ {num_cycles} 사이클 스케줄 생성 완료")
                
                # Calendar view
                calendar = planner.generate_calendar_view(schedule)
                calendar_df = pd.DataFrame(calendar)
                
                st.dataframe(calendar_df, use_container_width=True)
                
                # Patient instructions
                with st.expander("📋 환자 복약 안내"):
                    instructions = planner.generate_patient_instructions(schedule)
                    st.text(instructions)
    
    # === TAB 5: 최종 리포트 ===
    with tab5:
        st.markdown("### 📄 최종 임상 리포트")
        
        if all(key in st.session_state for key in ['recommendation', 'dosage_plan', 'schedule']):
            # Initialize report generator (needed for export)
            report_gen = ClinicalReportGenerator()
            
            if st.button("📊 리포트 생성", type="primary"):
                with st.spinner("종합 리포트 생성 중..."):
                    report = report_gen.generate_treatment_recommendation_report(
                        patient_data=st.session_state['precision_patient'],
                        quantitative_analysis=st.session_state['quant_analysis'],
                        clinical_metadata=st.session_state['precision_patient'],
                        genomic_data=st.session_state['precision_patient']['genomic_variants'],
                        cohort_classification=st.session_state['cohort'],
                        recommendation=st.session_state['recommendation'],
                        dosage_plan=st.session_state['dosage_plan'],
                        schedule=st.session_state['schedule']
                    )
                    
                    st.session_state['final_report'] = report
                    
                    st.success(f"✅ 리포트 생성 완료! (ID: {report['report_id']})")
            
            if 'final_report' in st.session_state:
                report = st.session_state['final_report']
                
                # Display report
                st.markdown(f"## {report['header']['title']}")
                st.caption(f"리포트 ID: {report['report_id']} | 생성일: {report['generation_date']}")
                
                st.markdown("---")
                
                # Patient Summary
                with st.expander("1. 환자 정보", expanded=True):
                    summary = report['patient_summary']
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**기본 정보**")
                        for k, v in summary['demographics'].items():
                            st.write(f"- {k}: {v}")
                    
                    with col2:
                        st.markdown("**임상 상태**")
                        for k, v in summary['clinical_status'].items():
                            st.write(f"- {k}: {v}")
                    
                    with col3:
                        st.markdown("**바이오마커**")
                        for k, v in summary['biomarkers'].items():
                            st.write(f"- {k}: {v}")
                
                # Treatment Recommendation
                with st.expander("2. 치료 추천", expanded=True):
                    rec_summary = report['treatment_recommendation']
                    st.success(f"**{rec_summary['추천 프로토콜']}**")
                    st.write(f"약물: {', '.join(rec_summary['약물 조합'])}")
                    st.write(f"신뢰도: {rec_summary['신뢰도']}")
                
                # Evidence
                with st.expander("3. 추천 근거", expanded=True):
                    evidence = report['evidence_rationale']
                    st.write(f"**환자군 매칭:** {evidence['환자군 기반 근거']}")
                    
                    if evidence['정량 분석 근거']:
                        st.markdown("**정량 지표:**")
                        for ind in evidence['정량 분석 근거']:
                            st.write(f"- {ind['지표']} = {ind['측정값']}: {ind['해석']}")
                
                # Download
                st.markdown("---")
                markdown_report = report_gen.export_to_markdown(report)
                st.download_button(
                    label="📥 마크다운 다운로드",
                    data=markdown_report,
                    file_name=f"{report['report_id']}.md",
                    mime="text/markdown"
                )
        else:
            st.warning("모든 단계를 완료하세요.")
    
    # === TAB 6: LIME 설명 ===
    with tab6:
        st.markdown("### 🔍 LIME - 예측 설명")
        st.info("AI 예측의 이유를 개별 특징별로 분석합니다")
        
        if 'recommendation' in st.session_state and 'quant_analysis' in st.session_state:
            try:
                from src.xai import LIMEExplainer
                import numpy as np
                
                st.markdown("#### 📊 특징별 기여도 분석")
                
                # Prepare feature vector
                quant = st.session_state['quant_analysis']
                patient = st.session_state['precision_patient']
                
                # Feature names
                feature_names = [
                    'num_cells', 'mean_area', 'cv_area', 'overall_heterogeneity',
                    'clustered_ratio', 'age', 'ki67_index', 'pdl1_tps'
                ]
                
                # Feature values
                feature_vector = np.array([
                    quant.get('num_cells', 0) / 1000,  # Normalize
                    quant.get('mean_area', 0) / 100,
                    quant.get('cv_area', 0),
                    quant.get('overall_heterogeneity', 0),
                    quant.get('clustered_ratio', 0),
                    patient.get('age', 0) / 100,
                    patient.get('ki67_index', 0) / 100,
                    patient.get('pdl1_tps', 0) / 100
                ])
                
                # Use literature-based prediction model
                if _HAS_LITERATURE_MODEL:
                    adds_predict_proba = _build_adds_predict_proba(
                        patient_data=patient
                    )
                else:
                    # Minimal fallback — clearly labeled
                    def adds_predict_proba(X):
                        prob_low = 0.3 + 0.4 * X[:, 3] + 0.2 * (1 - X[:, 5])
                        prob_low = np.clip(prob_low, 0, 1)
                        return np.column_stack([1 - prob_low, prob_low])
                
                # Create explainer (cached would be better)
                if 'lime_explainer' not in st.session_state:
                    explainer = LIMEExplainer(
                        feature_names=feature_names,
                        class_names=['High Response', 'Low Response']
                    )
                    # Clinically realistic CRC cohort for LIME training
                    n_cohort = 200
                    X_train = np.column_stack([
                        np.clip(np.random.normal(0.85, 0.3, n_cohort), 0.1, 3.0),   # num_cells/1000
                        np.clip(np.random.normal(2.5, 0.8, n_cohort), 0.5, 6.0),    # mean_area/100
                        np.clip(np.random.normal(0.35, 0.12, n_cohort), 0.05, 0.8), # cv_area
                        np.clip(np.random.normal(0.55, 0.2, n_cohort), 0.0, 1.0),   # heterogeneity
                        np.clip(np.random.normal(0.45, 0.2, n_cohort), 0.0, 1.0),   # clustered_ratio
                        np.clip(np.random.normal(0.62, 0.12, n_cohort), 0.2, 1.0),  # age/100
                        np.clip(np.random.normal(0.35, 0.15, n_cohort), 0.0, 1.0),  # ki67/100
                        np.clip(np.random.normal(0.15, 0.15, n_cohort), 0.0, 1.0),  # pdl1/100
                    ])
                    explainer.fit(X_train)
                    st.session_state['lime_explainer'] = explainer
                else:
                    explainer = st.session_state['lime_explainer']
                
                # Generate explanation
                with st.spinner("LIME 분석 중..."):
                    explanation_text = explainer.explain_clinical(
                        feature_vector,
                        adds_predict_proba,
                        feature_values={name: val for name, val in zip(feature_names, feature_vector)},
                        num_features=8
                    )
                
                st.markdown(explanation_text)
                
                # Visualization
                st.markdown("---")
                st.markdown("#### 📈 시각화")
                
                fig = explainer.plot_explanation(
                    feature_vector,
                    adds_predict_proba,
                    num_features=8
                )
                st.pyplot(fig)
                
                st.success("✓ LIME 설명 완료")
                
            except Exception as e:
                st.error(f"LIME 분석 오류: {str(e)}")
                st.info("💡 LIME 모듈 설치: pip install lime")
        else:
            st.warning("먼저 Tab 3에서 치료 추천을 생성하세요.")
    
    # === TAB 7: Counterfactual ===
    with tab7:
        st.markdown("### 🔄 Counterfactual - What-If 분석")
        st.info("치료 결과를 개선하기 위해 무엇을 변경해야 하는지 분석합니다")
        
        if 'recommendation' in st.session_state and 'quant_analysis' in st.session_state:
            try:
                from src.xai import CounterfactualGenerator
                import numpy as np
                
                quant = st.session_state['quant_analysis']
                patient = st.session_state['precision_patient']
                
                # Feature setup (same as LIME)
                feature_names = [
                    'num_cells', 'mean_area', 'cv_area', 'overall_heterogeneity',
                    'clustered_ratio', 'age', 'ki67_index', 'pdl1_tps'
                ]
                
                feature_vector = np.array([
                    quant.get('num_cells', 0) / 1000,
                    quant.get('mean_area', 0) / 100,
                    quant.get('cv_area', 0),
                    quant.get('overall_heterogeneity', 0),
                    quant.get('clustered_ratio', 0),
                    patient.get('age', 0) / 100,
                    patient.get('ki67_index', 0) / 100,
                    patient.get('pdl1_tps', 0) / 100
                ])
                
                # Mock prediction
                # Use literature-based prediction model
                if _HAS_LITERATURE_MODEL:
                    adds_predict_proba = _build_adds_predict_proba(
                        patient_data=patient
                    )
                else:
                    def adds_predict_proba(X):
                        prob_low = 0.3 + 0.4 * X[:, 3] + 0.2 * (1 - X[:, 5])
                        prob_low = np.clip(prob_low, 0, 1)
                        return np.column_stack([1 - prob_low, prob_low])
                
                current_pred = adds_predict_proba(feature_vector.reshape(1, -1))[0]
                
                st.markdown("#### 📊 현재 상태")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("예측 반응률", f"{current_pred[0]*100:.1f}%")
                with col2:
                    outcome = "High Response" if current_pred[0] > 0.5 else "Low Response"
                    st.metric("예상 결과", outcome)
                
                # Generate counterfactual
                if st.button("🔄 Counterfactual 생성", type="primary"):
                    with st.spinner("최적 변경 시나리오 탐색 중..."):
                        cf_gen = CounterfactualGenerator(
                            feature_names=feature_names,
                            continuous_features=feature_names[:5] + feature_names[6:],
                            immutable_features=['age']
                        )
                        
                        # Target High Response (class 1) if currently Low Response
                        # Class 0 = Low Response (index [0] in predictions)
                        # Class 1 = High Response (index [1] in predictions)
                        target_class = 1 if current_pred[0] < 0.5 else 1  # Always want High Response
                        
                        cfs = cf_gen.generate_simple_counterfactual(
                            feature_vector,
                            adds_predict_proba,
                            target_class=target_class,
                            max_iterations=500,
                            diversity=1
                        )
                        
                        if cfs and len(cfs) > 0:
                            cf = cfs[0]
                            
                            st.markdown("---")
                            # Class names must match prediction order:
                            # predictions[0] = Low Response probability
                            # predictions[1] = High Response probability  
                            explanation = cf_gen.explain_counterfactual_clinical(
                                cf,
                               class_names=['Low Response', 'High Response'],  # Match prediction indices!
                                outcome_metric="ORR"
                            )
                            
                            st.markdown(explanation)
                            
                            # Visualization
                            st.markdown("---")
                            st.markdown("#### 📈 변화 시각화")
                            fig = cf_gen.plot_counterfactual(cf, top_n=8)
                            st.pyplot(fig)
                            
                            st.success("✓ Counterfactual 분석 완료")
                        else:
                            st.warning("Counterfactual을 찾을 수 없습니다. 현재 상태가 이미 최적일 수 있습니다.")
                
            except Exception as e:
                st.error(f"Counterfactual 분석 오류: {str(e)}")
                st.info("💡 필요 패키지 설치: pip install -r requirements_xai.txt")
        else:
            st.warning("먼저 Tab 3에서 치료 추천을 생성하세요.")


if __name__ == "__main__":
    show_precision_oncology()
