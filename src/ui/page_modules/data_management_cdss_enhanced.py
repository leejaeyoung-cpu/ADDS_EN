"""
Enhanced CDSS Data Management Dashboard

Provides comprehensive view of:
- Patient data completeness
- ML model performance
- System health monitoring
- Training history
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def show_cdss_dashboard():
    """Enhanced CDSS Data Management Dashboard"""
    st.title("🏥 CDSS Data Management Dashboard")
    
    st.markdown("""
    **메타데이터 학습 시스템 현황**
    
    이 대시보드는 CDSS 메타데이터 학습 시스템의 전반적인 상태를 모니터링합니다.
    """)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Patient Data",
        "🤖 ML Performance",
        "📈 System Health"
    ])
    
    with tab1:
        show_patient_overview()
    
    with tab2:
        show_ml_performance()
    
    with tab3:
        show_system_health()


def show_patient_overview():
    """Show patient data overview"""
    st.header("Patient Management")
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import (
            Patient, CTAnalysis, Treatment, TreatmentOutcome
        )
        
        db = get_session()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = db.query(Patient).count()
        total_analyses = db.query(CTAnalysis).count()
        total_treatments = db.query(Treatment).count()
        total_outcomes = db.query(TreatmentOutcome).count()
        
        col1.metric("Total Patients", total_patients)
        col2.metric("CT Analyses", total_analyses)
        col3.metric("Treatments", total_treatments)
        col4.metric("Outcomes Recorded", total_outcomes)
        
        st.divider()
        
        # Data completeness
        st.subheader("\ud83d\udcca Data Completeness")
        
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
                    'CT': '\u2713' if has_ct else '\u2717',
                    'Treatment': '\u2713' if has_treatment else '\u2717',
                    'Outcome': '\u2713' if has_outcome else '\u2717',
                    'Completeness': f"{score*100:.0f}%"
                })
            
            df = pd.DataFrame(completeness_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No patient data yet. Add patients to get started.")
        
    except Exception as e:
        st.error(f"Error loading patient data: {e}")


def show_ml_performance():
    """Show ML model performance tracking"""
    st.header("ML Performance & Training History")
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import MLTrainingRun
        
        db = get_session()
        
        # Get latest deployed model
        latest_run = (
            db.query(MLTrainingRun)
            .filter(MLTrainingRun.is_deployed == True)
            .order_by(MLTrainingRun.deployed_at.desc())
            .first()
        )
        
        if latest_run:
            st.subheader("\ud83c\udfaf Current Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Parse metrics
            val_metrics = latest_run.val_metrics if isinstance(latest_run.val_metrics, dict) else {}
            
            with col1:
                acc = val_metrics.get('accuracy', 0)
                st.metric(
                    "Validation Accuracy",
                    f"{acc*100:.1f}%",
                    delta=f"+{(acc - 0.70)*100:.1f}%" if acc > 0.70 else None
                )
            
            with col2:
                auc = val_metrics.get('auc_roc', 0)
                st.metric(
                    "AUC-ROC",
                    f"{auc:.3f}",
                    delta=f"+{auc - 0.72:.3f}" if auc > 0.72 else None
                )
            
            with col3:
                f1 = val_metrics.get('f1_score', 0)
                st.metric("F1 Score", f"{f1:.3f}")
            
            with col4:
                st.metric("Inference Count", f"{latest_run.inference_count}")
            
            # Model info
            st.info(f"""
            **Model**: {latest_run.model_type}  
            **Deployed**: {latest_run.deployed_at.strftime('%Y-%m-%d %H:%M') if latest_run.deployed_at else 'N/A'}  
            **Training Samples**: {latest_run.train_samples}  
            **Validation Samples**: {latest_run.val_samples}
            """)
        else:
            st.warning("No deployed model yet. Run training first.")
        
        st.divider()
        
        # Training history
        st.subheader("\ud83d\udcc8 Training History")
        
        training_runs = (
            db.query(MLTrainingRun)
            .order_by(MLTrainingRun.run_date.desc())
            .limit(20)
            .all()
        )
        
        if training_runs:
            history_data = []
            for run in training_runs:
                val_metrics = run.val_metrics if isinstance(run.val_metrics, dict) else {}
                
                history_data.append({
                    'Date': run.run_date.strftime('%Y-%m-%d %H:%M'),
                    'Model': run.model_type,
                    'Samples': run.num_samples,
                    'Val Acc': f"{val_metrics.get('accuracy', 0)*100:.1f}%",
                    'AUC': f"{val_metrics.get('auc_roc', 0):.3f}",
                    'Deployed': '\u2713' if run.is_deployed else '\u2717'
                })
            
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Plot training progress
            import plotly.graph_objects as go
            
            accuracies = [run.val_metrics.get('accuracy', 0) * 100 if isinstance(run.val_metrics, dict) else 0 
                         for run in reversed(training_runs)]
            dates = [run.run_date for run in reversed(training_runs)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=accuracies,
                mode='lines+markers',
                name='Validation Accuracy',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Training Date",
                yaxis_title="Accuracy (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training history yet.")
    
    except Exception as e:
        st.error(f"Error loading ML performance: {e}")
        import traceback
        st.code(traceback.format_exc())


def show_system_health():
    """Show system health metrics"""
    st.header("System Health & Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("\ud83d\udcbe Data Status")
        
        # Check database files
        db_files = {
            "Patient DB": Path("backend/patients.db"),
            "CDSS Records": Path("data/cdss_records.db"),
        }
        
        for name, path in db_files.items():
            status = "\u2713 Online" if path.exists() else "\u2717 Missing"
            color = "green" if path.exists() else "red"
            st.markdown(f"**{name}**: :{color}[{status}]")
    
    with col2:
        st.subheader("\ud83d\udee0\ufe0f Services")
        
        # Check key files
        services = {
            "Metadata Extractor": Path("patient_management_system/services/metadata_extraction.py"),
            "ML Trainer": Path("patient_management_system/services/daily_ml_trainer.py"),
            "Orchestrator": Path("patient_management_system/services/cdss_orchestrator.py"),
            "Dynamic Analysis": Path("patient_management_system/services/dynamic_analysis.py"),
        }
        
        for name, path in services.items():
            status = "\u2713 Ready" if path.exists() else "\u2717 Missing"
            color = "green" if path.exists() else "red"
            st.markdown(f"**{name}**: :{color}[{status}]")
    
    st.divider()
    
    # Quick actions
    st.subheader("\ud83d\ude80 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("\ud83d\udd04 Refresh Data", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("\ud83c\udfaf Train Model", use_container_width=True):
            st.info("Training initiated... (implement async job)")
    
    with col3:
        if st.button("\ud83d\udce5 Export Report", use_container_width=True):
            st.info("Report export... (implement PDF generation)")


if __name__ == "__main__":
    show_cdss_dashboard()
