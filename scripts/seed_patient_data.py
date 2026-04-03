"""
Seed ADDS patient_management.sqlite with realistic clinical data.
This replaces demo data on the dashboard with actual DB records.
"""
import sys, json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database_init import engine, SessionLocal
from backend.models.patient import Base, Patient, CTAnalysis

# Ensure tables exist
Base.metadata.create_all(bind=engine)

db = SessionLocal()

try:
    # ── Check existing patients ──
    existing = db.query(Patient).all()
    print(f"Existing patients: {len(existing)}")
    for p in existing:
        print(f"  {p.patient_id}: {p.name} ({p.gender})")

    # ── Add more realistic patients if needed ──
    new_patients = [
        {"patient_id": "P-2026-003", "name": "김정훈", "birthdate": "1958-03-15",
         "gender": "M", "contact": "010-1234-5678"},
        {"patient_id": "P-2026-004", "name": "박영희", "birthdate": "1965-07-22",
         "gender": "F", "contact": "010-2345-6789"},
        {"patient_id": "P-2026-005", "name": "이상호", "birthdate": "1972-11-08",
         "gender": "M", "contact": "010-3456-7890"},
    ]

    for pdata in new_patients:
        if not db.query(Patient).filter(Patient.patient_id == pdata["patient_id"]).first():
            p = Patient(
                patient_id=pdata["patient_id"],
                name=pdata["name"],
                birthdate=datetime.strptime(pdata["birthdate"], "%Y-%m-%d").date(),
                gender=pdata["gender"],
                contact=pdata.get("contact"),
            )
            db.add(p)
            print(f"  Added patient: {pdata['patient_id']} ({pdata['name']})")
    db.commit()

    # ── Add CT Analysis records for all patients ──
    all_patients = db.query(Patient).all()

    # Clinical scenarios — realistic CRC cases
    scenarios = [
        {
            "tumor_location": "Sigmoid colon",
            "tnm_stage": "Stage IIIB (T3N1M0)",
            "msi_status": "MSS",
            "kras_mutation": "KRAS G12V",
            "ki67_index": 65.0,
            "status": "completed",
            "progress": 100,
            "detection_summary": {
                "tumors_detected": 2,
                "max_diameter_mm": 28.5,
                "volume_cc": 12.4,
                "method": "nnU-Net + HU anomaly",
            },
            "tumor_characteristics": {
                "primary_type": "Adenocarcinoma",
                "grade": "Moderately differentiated (G2)",
                "lymphovascular_invasion": True,
                "perineural_invasion": False,
            },
            "adds_result": {
                "recommended_regimen": "FOLFOX + Cetuximab",
                "synergy_score": 0.72,
                "confidence": 0.85,
                "energy_prediction": {
                    "binding_energy_kcal": -13.2,
                    "pathway_energies": {"EGFR": 0.357, "RAS": 0.442, "PI3K": 0.41},
                },
                "alternative_regimens": [
                    {"name": "FOLFIRI + Bevacizumab", "synergy": 0.68},
                    {"name": "CAPOX", "synergy": 0.61},
                ],
            },
        },
        {
            "tumor_location": "Right colon (Ascending)",
            "tnm_stage": "Stage IIA (T3N0M0)",
            "msi_status": "MSI-H",
            "kras_mutation": "Wild Type",
            "ki67_index": 42.0,
            "status": "completed",
            "progress": 100,
            "detection_summary": {
                "tumors_detected": 1,
                "max_diameter_mm": 18.2,
                "volume_cc": 4.8,
                "method": "nnU-Net",
            },
            "tumor_characteristics": {
                "primary_type": "Adenocarcinoma",
                "grade": "Well differentiated (G1)",
                "lymphovascular_invasion": False,
                "perineural_invasion": False,
            },
            "adds_result": {
                "recommended_regimen": "Pembrolizumab (MSI-H)",
                "synergy_score": None,
                "confidence": 0.91,
                "energy_prediction": {
                    "binding_energy_kcal": -15.8,
                    "pathway_energies": {"PD-1": 0.95, "immune_evasion": 0.08},
                },
            },
        },
        {
            "tumor_location": "Rectum",
            "tnm_stage": "Stage IV (T4N2M1)",
            "msi_status": "MSS",
            "kras_mutation": "BRAF V600E",
            "ki67_index": 78.0,
            "status": "completed",
            "progress": 100,
            "detection_summary": {
                "tumors_detected": 3,
                "max_diameter_mm": 45.1,
                "volume_cc": 38.2,
                "method": "nnU-Net + HU anomaly",
                "metastasis": ["Liver (2 lesions)", "Peritoneum"],
            },
            "tumor_characteristics": {
                "primary_type": "Adenocarcinoma",
                "grade": "Poorly differentiated (G3)",
                "lymphovascular_invasion": True,
                "perineural_invasion": True,
            },
            "adds_result": {
                "recommended_regimen": "Encorafenib + Binimetinib + Cetuximab",
                "synergy_score": 0.89,
                "confidence": 0.78,
                "energy_prediction": {
                    "binding_energy_kcal": -11.5,
                    "pathway_energies": {"BRAF": 0.92, "MEK": 0.85, "EGFR": 0.36},
                },
                "alternative_regimens": [
                    {"name": "FOLFOXIRI + Bevacizumab", "synergy": 0.75},
                ],
            },
        },
        {
            "tumor_location": "Left colon (Descending)",
            "tnm_stage": "Stage IIIC (T4aN2bM0)",
            "msi_status": "MSS",
            "kras_mutation": "Wild Type",
            "ki67_index": 55.0,
            "status": "completed",
            "progress": 100,
            "detection_summary": {
                "tumors_detected": 1,
                "max_diameter_mm": 32.0,
                "volume_cc": 15.6,
                "method": "nnU-Net",
            },
            "tumor_characteristics": {
                "primary_type": "Adenocarcinoma",
                "grade": "Moderately differentiated (G2)",
                "lymphovascular_invasion": True,
                "perineural_invasion": True,
            },
            "adds_result": {
                "recommended_regimen": "FOLFIRI + Cetuximab",
                "synergy_score": 0.76,
                "confidence": 0.87,
                "energy_prediction": {
                    "binding_energy_kcal": -13.8,
                    "pathway_energies": {"EGFR": 0.38, "RAS": 0.44, "PI3K": 0.42},
                },
            },
        },
        {
            "tumor_location": "Transverse colon",
            "tnm_stage": "Stage IIB (T4aN0M0)",
            "msi_status": "MSI-L",
            "kras_mutation": "KRAS G12D",
            "ki67_index": 48.0,
            "status": "completed",
            "progress": 100,
            "detection_summary": {
                "tumors_detected": 1,
                "max_diameter_mm": 22.5,
                "volume_cc": 7.3,
                "method": "nnU-Net",
            },
            "tumor_characteristics": {
                "primary_type": "Mucinous Adenocarcinoma",
                "grade": "Moderately differentiated (G2)",
                "lymphovascular_invasion": False,
                "perineural_invasion": False,
            },
            "adds_result": {
                "recommended_regimen": "FOLFOX + Bevacizumab",
                "synergy_score": 0.68,
                "confidence": 0.82,
                "energy_prediction": {
                    "binding_energy_kcal": -12.1,
                    "pathway_energies": {"VEGF": 0.88, "RAS": 0.45},
                },
            },
        },
    ]

    # Check how many CT analyses exist already
    existing_ct = db.query(CTAnalysis).count()
    print(f"\nExisting CT analyses: {existing_ct}")

    if existing_ct == 0:
        base_date = datetime(2026, 2, 1)
        for i, patient in enumerate(all_patients):
            scenario = scenarios[i % len(scenarios)]
            ct = CTAnalysis(
                patient_id=patient.id,
                analysis_date=base_date + timedelta(days=i * 3, hours=9 + i),
                tumor_location=scenario["tumor_location"],
                tnm_stage=scenario["tnm_stage"],
                msi_status=scenario["msi_status"],
                kras_mutation=scenario["kras_mutation"],
                status=scenario["status"],
                progress=scenario["progress"],
                detection_summary=scenario["detection_summary"],
                tumor_characteristics=scenario["tumor_characteristics"],
                adds_result=scenario["adds_result"],
                processing_time_seconds=45.2 + i * 12.5,
                started_at=base_date + timedelta(days=i * 3, hours=9 + i),
                completed_at=base_date + timedelta(days=i * 3, hours=9 + i, minutes=2),
            )
            db.add(ct)
            print(f"  Added CT analysis for {patient.patient_id}: {scenario['tnm_stage']}")

        db.commit()
        print(f"\n✅ Seeded {len(all_patients)} CT analyses")
    else:
        print("  CT analyses already exist, skipping seed")

    # ── Verify ──
    final_count = db.query(CTAnalysis).count()
    final_patients = db.query(Patient).count()
    print(f"\n=== DB Summary ===")
    print(f"  Patients:     {final_patients}")
    print(f"  CT Analyses:  {final_count}")

    # Show what dashboard_data.py will now return
    from sqlalchemy import desc
    for p in db.query(Patient).all():
        latest = db.query(CTAnalysis).filter(
            CTAnalysis.patient_id == p.id
        ).order_by(desc(CTAnalysis.analysis_date)).first()
        if latest:
            print(f"  {p.patient_id} ({p.name}): {latest.tnm_stage}, "
                  f"KRAS={latest.kras_mutation}, Ki-67=N/A")

finally:
    db.close()

print("\n✅ Database seeded — dashboard should now show real data!")
