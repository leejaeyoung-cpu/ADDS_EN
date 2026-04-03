"""
Physician Notes Entry Interface

UI for physicians to enter clinical notes which are automatically parsed
by the NLP system and can trigger re-analysis.
"""

import streamlit as st
from datetime import datetime, date
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def show_notes_entry():
    """Physician notes entry interface"""
    st.title("📝 Physician Notes Entry")
    
    st.markdown("""
    Enter clinical notes for automatic NLP parsing and intelligent re-analysis triggers.
    """)
    
    try:
        from patient_management_system.database.db_enhanced import get_session
        from patient_management_system.database.models_enhanced import Patient, PhysicianNote
        from patient_management_system.services.nlp_parser import PhysicianNotesParser
        
        db = get_session()
        
        # === Patient Selection ===
        st.subheader("1️⃣ Select Patient")
        
        patients = db.query(Patient).all()
        
        if not patients:
            st.warning("No patients found. Add patients first.")
            return
        
        patient_options = {f"{p.patient_id} - {p.name}": p for p in patients}
        selected_patient_key = st.selectbox(
            "Patient",
            options=list(patient_options.keys())
        )
        patient = patient_options[selected_patient_key]
        
        # Show patient info
        st.info(f"""
        **Patient ID**: {patient.patient_id}  
        **Name**: {patient.name}  
        **Age**: {(datetime.now() - patient.birthdate).days // 365} years  
        **Gender**: {patient.gender}
        """)
        
        st.divider()
        
        # === Notes Entry ===
        st.subheader("2️⃣ Enter Clinical Notes")
        
        with st.form("notes_entry"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                note_date = st.date_input(
                    "Assessment Date",
                    value=date.today()
                )
            
            with col2:
                physician_name = st.text_input(
                    "Physician Name",
                    value="Dr. "
                )
            
            # Notes text area
            clinical_notes = st.text_area(
                "Clinical Assessment",
                height=300,
                placeholder="""Example:
                
Patient presents for follow-up after 3 cycles of FOLFOX.

Physical Examination:
- General: Patient appears comfortable, alert and oriented
- Vital signs: BP 120/80, HR 72, Temp 36.8°C
- Abdomen: Soft, non-tender, no masses palpable

Recent Imaging:
- CT scan shows tumor reduced from 4.5 cm to 3.2 cm
- No new lesions identified
- No evidence of metastasis

Laboratory Results:
- CEA: 8.2 ng/mL (decreased from 14.5)
- CBC: WBC 6.2, Hgb 12.1, Platelets 185k

Symptoms:
- Mild fatigue, manageable with rest
- Occasional nausea, controlled with ondansetron
- No significant pain

Assessment:
- Partial response to chemotherapy
- Good tolerance to treatment
- Stable performance status

Plan:
- Continue FOLFOX regimen
- Schedule next cycle in 2 weeks
- Repeat CT scan after 2 more cycles
- Monitor CEA levels
"""
            )
            
            # Template buttons
            st.markdown("**Quick Templates:**")
            template_cols = st.columns(4)
            
            templates = {
                "Stable": "Patient stable. No significant changes. Continue current treatment.",
                "Improvement": "Patient showing improvement. Tumor reduction observed. Good response to treatment.",
                "Progression": "Disease progression noted. Tumor growth observed. Consider treatment modification.",
                "Side Effects": "Patient experiencing treatment-related side effects. Dose adjustment recommended."
            }
            
            selected_template = None
            for i, (name, text) in enumerate(templates.items()):
                with template_cols[i]:
                    if st.form_submit_button(name, use_container_width=True):
                        selected_template = text
            
            # Submit button
            st.markdown("---")
            submitted = st.form_submit_button("💾 Save Notes & Parse", type="primary", use_container_width=True)
            
            if submitted and clinical_notes:
                # Use template if selected
                final_notes = selected_template if selected_template else clinical_notes
                
                # Parse notes
                parser = PhysicianNotesParser()
                parsed_data = parser.parse(final_notes)
                
                # Save to database
                note = PhysicianNote(
                    patient_id=patient.id,
                    clinical_assessment=final_notes,
                    physician_name=physician_name,
                    severity_score=parsed_data['severity']['score'],
                    note_date=datetime.combine(note_date, datetime.min.time())
                )
                
                db.add(note)
                db.commit()
                
                st.success("✅ Notes saved successfully!")
                
                # Show parsing results
                st.subheader("🤖 NLP Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    severity_color = {
                        'mild': 'green',
                        'moderate': 'orange',
                        'critical': 'red',
                        'unknown': 'gray'
                    }
                    severity = parsed_data['severity']['level']
                    st.markdown(f"**Severity**: :{severity_color.get(severity, 'gray')}[{severity.upper()}] ({parsed_data['severity']['score']}/10)")
                
                with col2:
                    tumor_status = parsed_data['tumor_status']['status']
                    status_emoji = {
                        'growth': '📈',
                        'reduction': '📉',
                        'stable': '➡️',
                        'unknown': '❓'
                    }
                    st.markdown(f"**Tumor Status**: {status_emoji.get(tumor_status, '')} {tumor_status.capitalize()}")
                
                with col3:
                    reanalysis = parsed_data['requires_reanalysis']
                    st.markdown(f"**Re-analysis**: {'⚠️ REQUIRED' if reanalysis else '✓ Not needed'}")
                
                if parsed_data['symptoms']:
                    st.markdown(f"**Symptoms Detected**: {', '.join(parsed_data['symptoms'])}")
                
                if parsed_data['medications']:
                    st.markdown(f"**Medications Mentioned**: {', '.join(parsed_data['medications'])}")
                
                if parsed_data['key_findings']:
                    st.markdown("**Key Findings Extracted:**")
                    for finding in parsed_data['key_findings']:
                        st.markdown(f"- {finding}")
                
                # Show re-analysis warning
                if reanalysis:
                    st.warning("""
                    ⚠️ **Re-analysis Recommended**
                    
                    Based on the clinical notes, the system recommends triggering a re-analysis of patient data.
                    This may be due to:
                    - Suspected tumor growth
                    - Critical severity indicators
                    - Treatment changes mentioned
                    
                    The dynamic analysis system will automatically trigger re-analysis if configured.
                    """)
            
            elif submitted:
                st.error("Please enter clinical notes before saving.")
        
        # Show recent notes
        st.divider()
        st.subheader("📋 Recent Notes for This Patient")
        
        recent_notes = (
            db.query(PhysicianNote)
            .filter(PhysicianNote.patient_id == patient.id)
            .order_by(PhysicianNote.note_date.desc())
            .limit(5)
            .all()
        )
        
        if recent_notes:
            for note in recent_notes:
                with st.expander(f"{note.note_date.strftime('%Y-%m-%d')} - {note.physician_name} (Severity: {note.severity_score}/10)"):
                    st.text(note.clinical_assessment)
        else:
            st.info("No previous notes for this patient.")
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    show_notes_entry()
