"""
Treatment Outcome Collection System

Provides UI forms and backend APIs for recording treatment outcomes,
side effects, and quality of life metrics to enable AI fine-tuning.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def show_outcome_collection():
    """Treatment outcome collection interface"""
    st.title("📋 Treatment Outcome Collection")
    
    st.markdown("""
    Record treatment outcomes to improve AI model accuracy through continuous learning.
    """)
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import (
            Patient, Treatment, TreatmentOutcome, SideEffect
        )
        
        db = get_session()
        
        # === Patient Selection ===
        st.subheader("1️⃣ Select Patient")
        
        patients = db.query(Patient).all()
        
        if not patients:
            st.warning("No patients found. Add patients first.")
            return
        
        patient_options = {f"{p.patient_id} - {p.name}": p.id for p in patients}
        selected_patient_key = st.selectbox(
            "Patient",
            options=list(patient_options.keys())
        )
        patient_id = patient_options[selected_patient_key]
        
        # Get patient's treatments
        treatments = (
            db.query(Treatment)
            .filter(Treatment.patient_id == patient_id)
            .order_by(Treatment.start_date.desc())
            .all()
        )
        
        if not treatments:
            st.warning("No treatments found for this patient.")
            
            # Quick add treatment option
            with st.expander("➕ Add Treatment"):
                _add_treatment_form(db, patient_id)
            
            return
        
        st.divider()
        
        # === Treatment Selection ===
        st.subheader("2️⃣ Select Treatment")
        
        treatment_options = {
            f"{t.start_date.strftime('%Y-%m-%d')} - {t.drug_cocktail}": t.id 
            for t in treatments
        }
        selected_treatment_key = st.selectbox(
            "Treatment",
            options=list(treatment_options.keys())
        )
        treatment_id = treatment_options[selected_treatment_key]
        
        # Get selected treatment
        treatment = db.query(Treatment).filter(Treatment.id == treatment_id).first()
        
        # Show treatment details
        st.info(f"""
        **Drug Cocktail**: {treatment.drug_cocktail}  
        **Start Date**: {treatment.start_date.strftime('%Y-%m-%d')}  
        **Dosage**: {treatment.dosage}  
        **Cycle**: {treatment.cycle_number if treatment.cycle_number else 'N/A'}
        """)
        
        st.divider()
        
        # === Outcome Recording ===
        st.subheader("3️⃣ Record Outcome")
        
        # Check if outcome already exists
        existing_outcome = (
            db.query(TreatmentOutcome)
            .filter(TreatmentOutcome.treatment_id == treatment_id)
            .order_by(TreatmentOutcome.assessment_date.desc())
            .first()
        )
        
        if existing_outcome:
            st.success(f"✓ Outcome recorded on {existing_outcome.assessment_date.strftime('%Y-%m-%d')}")
            
            with st.expander("View Existing Outcome"):
                _display_outcome(existing_outcome)
            
            if st.checkbox("Update Outcome"):
                _outcome_form(db, treatment_id, existing_outcome)
        else:
            _outcome_form(db, treatment_id, None)
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


def _add_treatment_form(db, patient_id: int):
    """Quick add treatment form"""
    with st.form("add_treatment"):
        st.markdown("**Add New Treatment**")
        
        drug_cocktail = st.text_input(
            "Drug Cocktail",
            placeholder="e.g., FOLFOX, 5-FU+Oxaliplatin"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today())
        with col2:
            cycle_number = st.number_input("Cycle Number", min_value=1, value=1)
        
        dosage = st.text_input(
            "Dosage",
            placeholder="e.g., 85mg/m² Oxaliplatin + 400mg/m² 5-FU"
        )
        
        submitted = st.form_submit_button("Add Treatment")
        
        if submitted and drug_cocktail:
            from patient_management_system.database.models_enhanced import Treatment
            
            treatment = Treatment(
                patient_id=patient_id,
                drug_cocktail=drug_cocktail,
                start_date=datetime.combine(start_date, datetime.min.time()),
                dosage=dosage,
                cycle_number=cycle_number
            )
            
            db.add(treatment)
            db.commit()
            
            st.success("✓ Treatment added successfully!")
            st.rerun()


def _outcome_form(db, treatment_id: int, existing_outcome=None):
    """Outcome recording form"""
    
    with st.form("outcome_form"):
        st.markdown("**Clinical Assessment**")
        
        # Assessment date
        assessment_date = st.date_input(
            "Assessment Date",
            value=existing_outcome.assessment_date if existing_outcome else date.today()
        )
        
        # Response type (RECIST criteria)
        response_type = st.selectbox(
            "Response Type (RECIST)",
            options=["CR", "PR", "SD", "PD"],
            index=["CR", "PR", "SD", "PD"].index(existing_outcome.response_type) if existing_outcome else 2,
            help="""
            CR: Complete Response  
            PR: Partial Response  
            SD: Stable Disease  
            PD: Progressive Disease
            """
        )
        
        # Tumor size change
        col1, col2 = st.columns(2)
        
        with col1:
            tumor_size_change = st.number_input(
                "Tumor Size Change (%)",
                min_value=-100.0,
                max_value=100.0,
                value=float(existing_outcome.tumor_size_change_percent) if existing_outcome else 0.0,
                step=0.1,
                help="Negative = reduction, Positive = growth"
            )
        
        with col2:
            pfs_days = st.number_input(
                "Progression-Free Survival (days)",
                min_value=0,
                value=existing_outcome.pfs_days if existing_outcome else 0,
                step=1
            )
        
        st.markdown("---")
        
        # Quality of Life
        st.markdown("**Quality of Life Assessment**")
        
        qol_score = st.slider(
            "Overall QoL Score",
            min_value=0.0,
            max_value=10.0,
            value=float(existing_outcome.qol_score) if existing_outcome else 5.0,
            step=0.5,
            help="0 = Worst, 10 = Best"
        )
        
        st.markdown("---")
        
        # Side Effects
        st.markdown("**Side Effects**")
        
        side_effects = st.multiselect(
            "Experienced Side Effects",
            options=[
                "Nausea/Vomiting",
                "Fatigue",
                "Diarrhea",
                "Neuropathy",
                "Neutropenia",
                "Anemia",
                "Hand-foot syndrome",
                "Mucositis",
                "Hair loss",
                "Other"
            ]
        )
        
        severity = st.select_slider(
            "Overall Severity",
            options=["Grade 1 (Mild)", "Grade 2 (Moderate)", "Grade 3 (Severe)", "Grade 4 (Life-threatening)"],
            value="Grade 1 (Mild)"
        )
        
        notes = st.text_area(
            "Clinical Notes",
            height=100,
            placeholder="Additional observations, patient feedback, etc."
        )
        
        submitted = st.form_submit_button("💾 Save Outcome", use_container_width=True)
        
        if submitted:
            from patient_management_system.database.models_enhanced import TreatmentOutcome, SideEffect
            
            # Create or update outcome
            if existing_outcome:
                outcome = existing_outcome
                outcome.assessment_date = datetime.combine(assessment_date, datetime.min.time())
            else:
                outcome = TreatmentOutcome(
                    treatment_id=treatment_id,
                    assessment_date=datetime.combine(assessment_date, datetime.min.time())
                )
                db.add(outcome)
            
            outcome.response_type = response_type
            outcome.tumor_size_change_percent = tumor_size_change
            outcome.pfs_days = pfs_days
            outcome.qol_score = qol_score
            outcome.clinical_notes = notes
            
            db.commit()
            
            # Add side effects
            if side_effects:
                # Remove old side effects
                db.query(SideEffect).filter(SideEffect.outcome_id == outcome.id).delete()
                
                severity_grade = int(severity.split()[1][0])  # Extract grade number
                
                for effect in side_effects:
                    side_effect = SideEffect(
                        outcome_id=outcome.id,
                        side_effect_name=effect,
                        severity_grade=severity_grade,
                        onset_date=datetime.now()
                    )
                    db.add(side_effect)
                
                db.commit()
            
            st.success("✅ Outcome saved successfully!")
            st.balloons()
            
            # Trigger re-training suggestion
            st.info("💡 **Tip**: New outcome data can improve model accuracy. Consider triggering a training cycle.")


def _display_outcome(outcome):
    """Display existing outcome details"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Response Type", outcome.response_type)
    
    with col2:
        st.metric(
            "Tumor Change",
            f"{outcome.tumor_size_change_percent:+.1f}%",
            delta=f"{outcome.tumor_size_change_percent:.1f}%"
        )
    
    with col3:
        st.metric("PFS (days)", outcome.pfs_days)
    
    st.metric("Quality of Life", f"{outcome.qol_score}/10")
    
    if outcome.clinical_notes:
        st.text_area("Clinical Notes", value=outcome.clinical_notes, disabled=True, height=100)


def show_outcome_statistics():
    """Show outcome statistics and trends"""
    st.title("📊 Outcome Statistics")
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import TreatmentOutcome, Treatment
        import plotly.express as px
        import plotly.graph_objects as go
        
        db = get_session()
        
        # Get all outcomes
        outcomes = db.query(TreatmentOutcome).all()
        
        if not outcomes:
            st.info("No outcome data yet. Record outcomes to see statistics.")
            return
        
        # === Summary Metrics ===
        st.subheader("📈 Overall Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_outcomes = len(outcomes)
        response_counts = {}
        for outcome in outcomes:
            response_counts[outcome.response_type] = response_counts.get(outcome.response_type, 0) + 1
        
        col1.metric("Total Outcomes", total_outcomes)
        col2.metric("CR/PR Rate", f"{(response_counts.get('CR', 0) + response_counts.get('PR', 0)) / total_outcomes * 100:.1f}%")
        col3.metric("Avg QoL", f"{sum(o.qol_score for o in outcomes) / total_outcomes:.1f}/10")
        col4.metric("Avg PFS", f"{sum(o.pfs_days for o in outcomes) / total_outcomes:.0f} days")
        
        st.divider()
        
        # === Response Distribution ===
        st.subheader("🎯 Response Distribution")
        
        response_df = pd.DataFrame([
            {"Response": k, "Count": v} for k, v in response_counts.items()
        ])
        
        fig = px.pie(
            response_df,
            names="Response",
            values="Count",
            title="Treatment Response Distribution",
            color="Response",
            color_discrete_map={
                "CR": "#00cc66",
                "PR": "#66cc00",
                "SD": "#ffcc00",
                "PD": "#ff6666"
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # === Outcome Trends ===
        st.subheader("📉 Outcome Trends")
        
        outcome_data = []
        for outcome in outcomes:
            outcome_data.append({
                "Date": outcome.assessment_date,
                "Tumor Change (%)": outcome.tumor_size_change_percent,
                "QoL Score": outcome.qol_score,
                "PFS (days)": outcome.pfs_days
            })
        
        df = pd.DataFrame(outcome_data).sort_values("Date")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Tumor Change (%)"],
            mode='lines+markers',
            name='Tumor Change (%)',
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["QoL Score"],
            mode='lines+markers',
            name='QoL Score',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Outcome Trends Over Time",
            xaxis_title="Assessment Date",
            yaxis=dict(title="Tumor Change (%)", side="left"),
            yaxis2=dict(title="QoL Score", overlaying="y", side="right"),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    tab1, tab2 = st.tabs(["📋 Record Outcome", "📊 Statistics"])
    
    with tab1:
        show_outcome_collection()
    
    with tab2:
        show_outcome_statistics()
