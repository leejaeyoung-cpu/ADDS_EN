"""
Dashboard Data Utility Module
Attempts to load real data from DB first, falls back to DEMO data if unavailable.

All return values include '_is_demo' flag to indicate data source.
"""

import pandas as pd
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)


def _try_get_db_session():
    """Attempt to get a database session. Returns None if unavailable."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from backend.database_init import get_db
        db = next(get_db())
        return db
    except Exception as e:
        logger.debug(f"DB unavailable, using demo data: {e}")
        return None


def _try_get_real_patient_cases(db, limit=5):
    """Try to load real patient cases from database."""
    try:
        from backend.models.patient import Patient, CTAnalysis
        from sqlalchemy import desc

        patients = db.query(Patient).order_by(
            desc(Patient.created_at)
        ).limit(limit).all()

        if not patients or len(patients) == 0:
            return None

        rows = []
        for p in patients:
            latest_ct = db.query(CTAnalysis).filter(
                CTAnalysis.patient_id == p.id
            ).order_by(desc(CTAnalysis.analysis_date)).first()

            # Extract Ki-67 safely from JSON fields (not a direct column)
            ki67 = 'N/A'
            treatment = 'N/A'
            if latest_ct:
                # Try tumor_characteristics JSON
                tc = latest_ct.tumor_characteristics or {}
                if isinstance(tc, dict):
                    ki67 = tc.get('ki67_index', tc.get('ki67', 'N/A'))
                # Try adds_result for treatment
                ar = latest_ct.adds_result or {}
                if isinstance(ar, dict):
                    treatment = ar.get('recommended_regimen', 'N/A')

            rows.append({
                'Patient ID': p.patient_id or f'PT-{p.id:05d}',
                'Name': p.name or f'Patient {p.id}',
                'Age': _calc_age(p.birthdate) if p.birthdate else 'N/A',
                'Gender': p.gender or 'N/A',
                'Stage': latest_ct.tnm_stage if latest_ct and latest_ct.tnm_stage else 'N/A',
                'Last Analysis': latest_ct.analysis_date.strftime('%Y-%m-%d %H:%M') if latest_ct and latest_ct.analysis_date else 'N/A',
                'Status': latest_ct.status.capitalize() if latest_ct and latest_ct.status else 'Active',
                'Ki-67': f'{ki67}%' if ki67 != 'N/A' else 'N/A',
                'Treatment': treatment,
            })

        df = pd.DataFrame(rows)
        df.attrs['_is_demo'] = False
        return df
    except Exception as e:
        logger.warning(f"Failed to load real patients: {e}")
        return None



def _try_get_real_metrics(db):
    """Try to load real metrics from PerformanceMetric table."""
    try:
        from backend.models.metadata_learning import PerformanceMetric
        from sqlalchemy import desc

        latest = db.query(PerformanceMetric).order_by(
            desc(PerformanceMetric.metric_date)
        ).first()

        if not latest:
            return None

        return {
            'active_patients': latest.total_patients or 0,
            'active_patients_delta': f'+{latest.new_patients_today or 0} this week',
            'pending_reviews': 0,
            'pending_priority': 'Normal',
            'cdss_analyses': latest.total_analyses or 0,
            'cdss_accuracy': f'{(latest.prediction_accuracy or 0) * 100:.0f}%',
            'critical_alerts': 0,
            'alerts_status': 'normal',
            'gpu_temperature': 'N/A',
            'processing_queue': 0,
            'model_version': latest.model_version or 'N/A',
            'last_update': latest.metric_date.strftime('%Y-%m-%d') if latest.metric_date else 'N/A',
            'accuracy_percent': (latest.prediction_accuracy or 0) * 100,
            'speed_seconds': latest.avg_analysis_time_seconds or 0,
            'throughput_per_day': latest.total_analyses or 0,
            'success_rate': (latest.prediction_accuracy or 0) * 100,
            'patient_satisfaction': 0,
            '_is_demo': False
        }
    except Exception as e:
        logger.debug(f"Failed to load real metrics: {e}")
        return None


def _calc_age(birthdate):
    """Calculate age from birthdate."""
    if not birthdate:
        return 'N/A'
    today = datetime.now().date()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))


def get_patient_cases(limit=5):
    """
    Get patient cases for dashboard display.
    Tries DB first, falls back to demo data.

    Returns:
        DataFrame with patient information. Has _is_demo attribute.
    """
    db = _try_get_db_session()
    if db:
        try:
            real = _try_get_real_patient_cases(db, limit)
            if real is not None:
                return real
        finally:
            db.close()

    # DEMO fallback
    patients = [
        {
            'Patient ID': 'DEMO-001',
            'Name': '[DEMO] John Smith',
            'Age': 54,
            'Gender': 'M',
            'Stage': 'Stage IIB',
            'Last Analysis': 'DEMO',
            'Status': 'Stable',
            'Ki-67': '35%',
            'Treatment': 'FOLFOX + Bevacizumab'
        },
        {
            'Patient ID': 'DEMO-002',
            'Name': '[DEMO] Emily Davis',
            'Age': 62,
            'Gender': 'F',
            'Stage': 'Stage IIIA',
            'Last Analysis': 'DEMO',
            'Status': 'Review',
            'Ki-67': '58%',
            'Treatment': 'FOLFIRI + Cetuximab'
        },
        {
            'Patient ID': 'DEMO-003',
            'Name': '[DEMO] Michael Brown',
            'Age': 48,
            'Gender': 'M',
            'Stage': 'Stage IV',
            'Last Analysis': 'DEMO',
            'Status': 'Urgent',
            'Ki-67': '72%',
            'Treatment': 'CAPOX + Nivolumab'
        },
        {
            'Patient ID': 'DEMO-004',
            'Name': '[DEMO] Sarah Wilson',
            'Age': 71,
            'Gender': 'F',
            'Stage': 'Stage IIB',
            'Last Analysis': 'DEMO',
            'Status': 'Stable',
            'Ki-67': '28%',
            'Treatment': 'FOLFOX'
        },
        {
            'Patient ID': 'DEMO-005',
            'Name': '[DEMO] David Lee',
            'Age': 59,
            'Gender': 'M',
            'Stage': 'Stage IIIC',
            'Last Analysis': 'DEMO',
            'Status': 'Review',
            'Ki-67': '64%',
            'Treatment': 'FOLFOXIRI'
        }
    ]

    df = pd.DataFrame(patients[:limit])
    df.attrs['_is_demo'] = True
    return df


def get_system_metrics():
    """
    Get system performance metrics.
    Tries DB first, falls back to demo data.

    Returns:
        Dictionary with system metrics. Includes '_is_demo' key.
    """
    db = _try_get_db_session()
    if db:
        try:
            real = _try_get_real_metrics(db)
            if real is not None:
                return real
        finally:
            db.close()

    # DEMO fallback
    return {
        'active_patients': 1247,
        'active_patients_delta': '+23 this week',
        'pending_reviews': 18,
        'pending_priority': 'High priority',
        'cdss_analyses': 34,
        'cdss_accuracy': '92%',
        'critical_alerts': 5,
        'alerts_status': 'urgent',
        'gpu_temperature': '45°C',
        'processing_queue': 3,
        'model_version': 'v2024.01',
        'last_update': '2 hours ago',
        'accuracy_percent': 92,
        'speed_seconds': 11.2,
        'throughput_per_day': 180,
        'success_rate': 89,
        'patient_satisfaction': 4.8,
        '_is_demo': True
    }


def get_analysis_trends(days=7):
    """
    Get analysis trend data for charts.
    Currently uses demo data with varying counts.

    Returns:
        DataFrame with date and analysis count. Has _is_demo attribute.
    """
    today = datetime.now()
    dates = []
    counts = []

    for i in range(days):
        date = today - timedelta(days=days-i-1)
        dates.append(date.strftime('%a'))
        base_count = 30
        variation = random.randint(-8, 12)
        counts.append(base_count + variation)

    df = pd.DataFrame({
        'Day': dates,
        'Analyses': counts
    })
    df.attrs['_is_demo'] = True
    return df


def get_team_activity():
    """
    Get team member activity data (demo only).

    Returns:
        List of dictionaries with team activity.
    """
    return [
        {
            'name': '이재영',
            'analyses_today': 15,
            'role': '연구원',
            'status': 'online',
            '_is_demo': True
        },
        {
            'name': '이상훈',
            'analyses_today': 12,
            'role': '연구원',
            'status': 'online',
            '_is_demo': True
        },
        {
            'name': '최문석',
            'analyses_today': 8,
            'role': '연구원',
            'status': 'online',
            '_is_demo': True
        }
    ]


def get_system_status():
    """
    Get detailed system status information (demo only).

    Returns:
        Dictionary with service statuses.
    """
    return {
        'services': {
            'Cellpose Engine': {'status': 'Active', 'color': 'green'},
            'CT Detection': {'status': 'Online', 'color': 'green'},
            'Drug Optimization': {'status': 'Ready', 'color': 'green'},
            'API Status': {'status': '200ms latency', 'color': 'green'}
        },
        'uptime': '99.9%',
        'response_time': '11.2s avg',
        '_is_demo': True
    }


def get_recent_results(limit=3):
    """
    Get recent analysis results (demo only).

    Returns:
        List of dictionaries with result information.
    """
    results = [
        {
            'id': 'DEMO-1234',
            'type': 'Lung Scan',
            'status': 'Completed',
            'time': '5 mins ago',
            '_is_demo': True
        },
        {
            'id': 'DEMO-5678',
            'type': 'Brain CT',
            'status': 'Completed',
            'time': '12 mins ago',
            '_is_demo': True
        },
        {
            'id': 'DEMO-9012',
            'type': 'Liver MRI',
            'status': 'Completed',
            'time': '30 mins ago',
            '_is_demo': True
        }
    ]
    return results[:limit]


def get_queue_items(limit=3):
    """
    Get items in processing queue (demo only).

    Returns:
        List of dictionaries with queue information.
    """
    queue = [
        {
            'patient': '[DEMO] Patient X',
            'time_remaining': '5 mins left',
            '_is_demo': True
        },
        {
            'patient': '[DEMO] Patient Y',
            'time_remaining': '12 mins left',
            '_is_demo': True
        },
        {
            'patient': '[DEMO] Patient Z',
            'time_remaining': '30 mins left',
            '_is_demo': True
        }
    ]
    return queue[:limit]
