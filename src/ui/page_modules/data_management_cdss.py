"""
CDSS Data Management Functions
Functions to show CDSS patient/CT data and ML performance in ADDS UI
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def show_patient_data():
    """Display patient management data overview"""
    st.header("🏥 Patient Data Management")
    
    db_path = "backend/patients.db"
    
    if not Path(db_path).exists():
        st.warning("Patient database not found. Initialize the system first.")
        return
    
    conn = sqlite3.connect(db_path)
    
    # Patients overview
    st.subheader("📊 Patients Overview")
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import (
            Patient, CTAnalysis, Treatment, TreatmentOutcome
        )
        
        db = get_session()
        
        # === METRICS ===
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = db.query(Patient).count()
        total_analyses = db.query(CTAnalysis).count()
        total_treatments = db.query(Treatment).count()
        total_outcomes = db.query(TreatmentOutcome).count()
        
        col1.metric("총 환자", total_patients)
        col2.metric("CT 분석", total_analyses)
        col3.metric("치료 기록", total_treatments)
        col4.metric("치료 결과", total_outcomes)
        
        st.markdown("---")
        
        # === DATA COMPLETENESS ===
        st.subheader("데이터 완성도")
        
        patients = db.query(Patient).all()
        if patients:
            completeness_data = []
            for p in patients:
                has_ct = db.query(CTAnalysis).filter(CTAnalysis.patient_id == p.id).first() is not None
                has_treatment = db.query(Treatment).filter(Treatment.patient_id == p.id).first() is not None
                has_outcome = (
                    db.query(TreatmentOutcome)
                    .join(Treatment)
                    .filter(Treatment.patient_id == p.id)
                    .first() is not None
                )
                
                score = sum([has_ct, has_treatment, has_outcome]) / 3
                
                completeness_data.append({
                    'Patient ID': p.patient_id,
                    'Name': p.name,
                    'CT': '✓' if has_ct else '✗',
                    'Treatment': '✓' if has_treatment else '✗',
                    'Outcome': '✓' if has_outcome else '✗',
                    'Completeness': f"{score*100:.0f}%"
                })
            
            df = pd.DataFrame(completeness_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("아직 환자 데이터가 없습니다.")
        
        st.markdown("---")
        
        # === RECENT ACTIVITY ===
        st.subheader("최근 활동")
        
        recent_analyses = (
            db.query(CTAnalysis)
            .order_by(CTAnalysis.analysis_date.desc())
            .limit(10)
            .all()
        )
        
        if recent_analyses:
            activity_data = []
            for a in recent_analyses:
                activity_data.append({
                    'Date': a.analysis_date.strftime('%Y-%m-%d %H:%M'),
                    'Patient': a.patient.patient_id,
                    'Type': 'CT Analysis',
                    'Status': a.status
                })
            
            st.dataframe(pd.DataFrame(activity_data), use_container_width=True, hide_index=True)
        
    except ImportError:
        st.warning("⚠️ CDSS 데이터베이스 모듈이 필요합니다.")
        st.info("patient_management_system 패키지를 확인하세요.")
    except Exception as e:
        st.error(f"CDSS 데이터 로드 실패: {str(e)}")


def show_ml_performance():
    """Display ML model performance metrics"""
    st.header("🤖 ML Performance Tracking")
    
    db_path = "backend/patients.db"
    
    if not Path(db_path).exists():
        st.info("ML tracking database not found. Train models first.")
        return
    
    conn = sqlite3.connect(db_path)
    
    # === Performance Metrics Overview ===
    col1, col2, col3, col4 = st.columns(4)
    
    # Get latest deployed model
    latest_query = """
    SELECT * FROM ml_training_runs 
    WHERE is_deployed = 1
    ORDER BY deployed_at DESC
    LIMIT 1
    """
    latest_df = pd.read_sql_query(latest_query, conn)
    
    if len(latest_df) > 0:
        latest = latest_df.iloc[0]
        
        # Parse metrics JSON if needed
        import json
        val_metrics = json.loads(latest['val_metrics']) if isinstance(latest['val_metrics'], str) else latest['val_metrics']
        
        with col1:
            st.metric(
                "Current Accuracy",
                f"{val_metrics.get('accuracy', 0)*100:.1f}%",
                delta=f"+{(val_metrics.get('accuracy', 0) - 0.75)*100:.1f}%"
            )
        
        with col2:
            st.metric(
                "AUC-ROC Score",
                f"{val_metrics.get('auc_roc', 0):.3f}",
                delta=f"+{val_metrics.get('auc_roc', 0) - 0.72:.3f}"
            )
        
        with col3:
            st.metric(
                "F1 Score",
                f"{val_metrics.get('f1_score', 0):.3f}"
            )
        
        with col4:
            st.metric(
                "Inference Count",
                f"{latest['inference_count']}"
            )
    
    st.divider()
    
    # === Training History ===
    st.subheader("📈 Training History")
    
    # Get training runs
    query = """
    SELECT * FROM ml_training_runs 
    ORDER BY run_date DESC 
    LIMIT 30
    """
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import MLTrainingRun
        
        db = get_session()
        
        # === TRAINING RUNS ===
        runs = db.query(MLTrainingRun).order_by(MLTrainingRun.run_date.desc()).all()
        
        if not runs:
            st.info("아직 ML 학습 기록이 없습니다.")
            st.caption("`daily_ml_trainer.py`를 실행하여 모델을 학습시키세요.")
            return
        
        # === DEPLOYED MODEL ===
        deployed = next((r for r in runs if r.is_deployed), None)
        
        if deployed:
            st.success(f"✓ 배포된 모델: v{deployed.id}")
            
            col1, col2, col3 = st.columns(3)
            
            val_metrics = deployed.val_metrics or {}
            col1.metric("검증 정확도", f"{val_metrics.get('val_accuracy', 0)*100:.1f}%")
            col2.metric("AUC-ROC", f"{val_metrics.get('auc_roc', 0):.3f}")
            col3.metric("학습 샘플", deployed.num_samples)
        
        st.markdown("---")
        
        # === TRAINING HISTORY ===
        st.subheader("학습 이력")
        
        history_data = []
        for r in reversed(runs[-20:]):
            val_metrics = r.val_metrics or {}
            history_data.append({
                'Date': r.run_date.date(),
                'Model': f"v{r.id}",
                'Samples': r.num_samples,
                'Val Accuracy': val_metrics.get('val_accuracy', 0),
                'Deployed': '✓' if r.is_deployed else ''
            })
        
        df_history = pd.DataFrame(history_data)
        
        # Plot
        fig = px.line(df_history, x='Date', y='Val Accuracy', 
                      title='모델 정확도 추이',
                      markers=True)
        fig.update_layout(yaxis_title='Validation Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(df_history, use_container_width=True, hide_index=True)
        
        # === IMPROVEMENT METRICS ===
        if len(runs) >= 2:
            st.subheader("개선 지표")
            
            first_run = runs[-1]
            latest_run = runs[0]
            
            first_acc = (first_run.val_metrics or {}).get('val_accuracy', 0)
            latest_acc = (latest_run.val_metrics or {}).get('val_accuracy', 0)
            
            improvement = ((latest_acc - first_acc) / first_acc * 100) if first_acc > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("초기 정확도", f"{first_acc*100:.1f}%")
            col2.metric("현재 정확도", f"{latest_acc*100:.1f}%")
            col3.metric("개선율", f"{improvement:+.1f}%", delta=f"{improvement:.1f}%")
        
    except ImportError:
        st.warning("⚠️ CDSS 데이터베이스 모듈이 필요합니다.")
    except Exception as e:
        st.error(f"ML 성과 데이터 로드 실패: {str(e)}")
        import traceback
        with st.expander("오류 상세"):
            st.code(traceback.format_exc())
