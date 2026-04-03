"""
Daily ML Trainer for CDSS — Evidence-Based Learning Loop

Trains on treatment outcome data to improve tumor-drug interaction predictions.

Architecture:
    TreatmentOutcome → AnalysisResult → PatientMetadata → Patient
    TreatmentOutcome → Patient (direct FK)
    Patient → CTAnalysis (for clinical biomarkers: TNM, MSI, KRAS)

References:
    - André T et al., N Engl J Med 2020;383:2207-2218 (KEYNOTE-177)
    - Karapetis CS et al., N Engl J Med 2008;359:1757-1765 (KRAS/cetuximab)
    - McGranahan N & Swanton C, Cell 2017;168:613-628 (ITH and resistance)
    - Sargent DJ et al., J Clin Oncol 2001;19:4058-4065 (age and chemo)
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import json
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database_init import get_db, engine
from backend.models.metadata_learning import (
    TreatmentOutcome, AnalysisResult, PatientMetadata,
    TumorDrugInteraction, ModelTrainingHistory, PerformanceMetric
)
from backend.models.patient import Patient, CTAnalysis
from sqlalchemy.orm import Session
from sqlalchemy import desc

logger = logging.getLogger(__name__)


class DailyMLTrainer:
    """
    Daily learning pipeline for CDSS

    Learns tumor-drug associations from treatment outcomes:
    - Which drug cocktails work for which tumor profiles
    - Predicted vs actual efficacy calibration
    - Side effect pattern accumulation

    Data path:
        TreatmentOutcome.patient_id → Patient.id → CTAnalysis (latest)
        TreatmentOutcome.analysis_result_id → AnalysisResult (radiomics, tumor data)
    """

    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self.model_version = datetime.now().strftime("v%Y%m%d")
        self._current_model = None

    # ─── Main Pipeline ─────────────────────────────────────────────────

    def run_daily_training(self, db: Session) -> Dict[str, Any]:
        """
        Main daily training pipeline.

        Returns:
            Training report dict with status, metrics, and step details.
        """
        logger.info(f"Starting daily ML training ({self.model_version})")
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
            "steps": []
        }

        # Step 1: Fetch treatment outcomes with non-null actual data
        outcomes = self._fetch_outcomes(db)
        report["total_outcomes"] = len(outcomes)

        if len(outcomes) < self.min_samples:
            report["status"] = "skipped"
            report["reason"] = (
                f"Insufficient data ({len(outcomes)} < {self.min_samples})"
            )
            logger.warning(f"Skipping training: only {len(outcomes)} outcomes")
            return report

        # Step 2: Build feature matrix
        X, y, meta = self._build_feature_matrix(outcomes, db)
        report["steps"].append({
            "step": "feature_extraction",
            "samples": len(X),
            "features": X.shape[1] if len(X) > 0 else 0
        })

        if len(X) < self.min_samples:
            report["status"] = "skipped"
            report["reason"] = "Insufficient valid feature rows"
            return report

        # Step 3: Train efficacy prediction model
        model_metrics = self._train_efficacy_model(X, y)
        report["steps"].append({
            "step": "model_training",
            "metrics": model_metrics
        })

        # Step 4: Update tumor-drug interaction knowledge base
        interactions_updated = self._update_drug_interactions(outcomes, db)
        report["steps"].append({
            "step": "knowledge_update",
            "interactions_updated": interactions_updated
        })

        # Step 5: Record training history
        self._record_training_history(db, model_metrics, len(X))

        # Step 6: Record daily performance metrics
        self._record_performance_metrics(db, outcomes, model_metrics)

        report["status"] = "completed"
        logger.info(f"Daily training complete: {model_metrics}")
        return report

    # ─── Step 1: Fetch Outcomes ────────────────────────────────────────

    def _fetch_outcomes(self, db: Session) -> List[TreatmentOutcome]:
        """
        Fetch treatment outcomes that have actual efficacy data.

        Schema: TreatmentOutcome columns used:
            - tumor_response (str): CR, PR, SD, PD
            - actual_efficacy (float): 0-1
            - predicted_efficacy (float): 0-1
            - prescribed_cocktail (JSON): {"drugs": [...], "dosages": [...]}
            - baseline_tumor_size (float)
            - patient_id (FK → patients.id)
            - analysis_result_id (FK → analysis_results.id)
        """
        return db.query(TreatmentOutcome).filter(
            TreatmentOutcome.actual_efficacy.isnot(None)
        ).all()

    # ─── Step 2: Feature Matrix ────────────────────────────────────────

    def _build_feature_matrix(
        self, outcomes: List[TreatmentOutcome], db: Session
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Build feature matrix from outcomes + patient clinical data.

        Feature extraction path:
            TreatmentOutcome → Patient → CTAnalysis (latest, for TNM/MSI/KRAS)
            TreatmentOutcome → AnalysisResult (for radiomics features)

        Target:
            actual_efficacy (0-1) from TreatmentOutcome
        """
        features = []
        targets = []
        metadata_list = []

        for outcome in outcomes:
            try:
                feat_vec = self._extract_features(outcome, db)
                if feat_vec is None:
                    continue

                target = outcome.actual_efficacy
                if target is None:
                    continue

                features.append(feat_vec)
                targets.append(float(target))
                metadata_list.append({
                    "outcome_id": outcome.id,
                    "patient_id": outcome.patient_id,
                    "tumor_response": outcome.tumor_response
                })

            except Exception as e:
                logger.warning(f"Error processing outcome {outcome.id}: {e}")
                continue

        if not features:
            return np.array([]), np.array([]), []

        return np.array(features), np.array(targets), metadata_list

    def _extract_features(
        self, outcome: TreatmentOutcome, db: Session
    ) -> Optional[List[float]]:
        """
        Extract 10-dimensional feature vector per outcome.

        Features (all normalized to ~[0, 1]):
            [0] age_norm: patient age / 100
            [1] stage_val: TNM stage ordinal (I=0.2, II=0.4, III=0.6, IV=0.8)
            [2] msi_flag: 1.0 if MSI-H, else 0.0
            [3] kras_wt_flag: 1.0 if KRAS wild-type, else 0.0
            [4] drug_count_norm: number of drugs in cocktail / 5
            [5] predicted_eff: model's prior prediction (0-1)
            [6] baseline_tumor_norm: baseline tumor size / 200 mm
            [7] tumor_reduction: tumor_size_change_percent / 100 (signed)
            [8] heterogeneity: from analysis (0-1)
            [9] tumor_volume_norm: from analysis / 200 cm³
        """
        features = []

        # ── Patient demographics ──
        patient = db.query(Patient).filter(
            Patient.id == outcome.patient_id
        ).first()

        if not patient:
            return None

        # Age
        age = 60.0
        if patient.birthdate:
            age = (datetime.now().date() - patient.birthdate).days / 365.25
        features.append(age / 100.0)  # [0]

        # ── Clinical biomarkers from latest CTAnalysis ──
        latest_ct = db.query(CTAnalysis).filter(
            CTAnalysis.patient_id == patient.id
        ).order_by(desc(CTAnalysis.analysis_date)).first()

        # TNM Stage
        stage_val = 0.5  # default
        if latest_ct and latest_ct.tnm_stage:
            stage_map = {'I': 0.2, 'II': 0.4, 'IIA': 0.4, 'IIB': 0.45,
                         'III': 0.6, 'IIIA': 0.55, 'IIIB': 0.6, 'IIIC': 0.65,
                         'IV': 0.8, 'IVA': 0.8, 'IVB': 0.9}
            for k, v in stage_map.items():
                if k in latest_ct.tnm_stage:
                    stage_val = v
                    break
        features.append(stage_val)  # [1]

        # MSI status
        msi_flag = 0.0
        if latest_ct and latest_ct.msi_status:
            msi_flag = 1.0 if 'MSI-H' in latest_ct.msi_status else 0.0
        features.append(msi_flag)  # [2]

        # KRAS wild-type flag
        kras_wt = 0.0
        if latest_ct and latest_ct.kras_mutation:
            kras_wt = 1.0 if 'WT' in latest_ct.kras_mutation.upper() or \
                             'WILD' in latest_ct.kras_mutation.upper() else 0.0
        features.append(kras_wt)  # [3]

        # ── Treatment regimen ──
        cocktail = outcome.prescribed_cocktail or {}
        if isinstance(cocktail, str):
            try:
                cocktail = json.loads(cocktail)
            except (json.JSONDecodeError, TypeError):
                cocktail = {}
        drug_count = len(cocktail.get('drugs', [])) if isinstance(cocktail, dict) else 1
        features.append(drug_count / 5.0)  # [4]

        # Predicted efficacy (prior prediction)
        pred_eff = outcome.predicted_efficacy if outcome.predicted_efficacy else 0.5
        features.append(float(pred_eff))  # [5]

        # Baseline tumor size
        baseline = outcome.baseline_tumor_size or 50.0
        features.append(float(baseline) / 200.0)  # [6]

        # Tumor size change
        change = outcome.tumor_size_change_percent or 0.0
        features.append(float(change) / 100.0)  # [7]

        # ── Analysis-derived features ──
        heterogeneity = 0.5
        tumor_volume = 50.0

        if outcome.analysis_result_id:
            analysis = db.query(AnalysisResult).filter(
                AnalysisResult.id == outcome.analysis_result_id
            ).first()
            if analysis:
                # Tumors JSON may contain feature data
                tumors = analysis.tumors or []
                if isinstance(tumors, str):
                    try:
                        tumors = json.loads(tumors)
                    except (json.JSONDecodeError, TypeError):
                        tumors = []
                if tumors and isinstance(tumors, list) and len(tumors) > 0:
                    first_tumor = tumors[0] if isinstance(tumors[0], dict) else {}
                    heterogeneity = first_tumor.get('heterogeneity', 0.5)
                    tumor_volume = first_tumor.get('volume_cm3',
                                   first_tumor.get('size', 50.0))

        features.append(float(heterogeneity))  # [8]
        features.append(float(tumor_volume) / 200.0)  # [9]

        return features

    # ─── Step 3: Model Training ────────────────────────────────────────

    def _train_efficacy_model(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train efficacy prediction model.

        Strategy:
            n >= 20: GradientBoostingRegressor (handles nonlinear interactions)
            n < 20:  Ridge regression (regularized linear, avoids overfitting)

        Ref: Hastie T, Tibshirani R, Friedman J. The Elements of Statistical
             Learning. 2nd ed. Springer; 2009. Ch16 (ensemble methods).
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, r2_score

        n_samples = len(X)

        if n_samples >= 20:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=min(100, n_samples * 2),
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)

        # Cross-validation
        n_folds = min(5, n_samples)
        if n_folds >= 2:
            cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring='r2')
            cv_r2 = float(np.mean(cv_scores))
        else:
            cv_r2 = 0.0

        # Full training
        model.fit(X, y)
        y_pred = model.predict(X)

        mse = float(mean_squared_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))
        mae = float(np.mean(np.abs(y - y_pred)))

        self._current_model = model

        return {
            "mse": round(mse, 4),
            "mae": round(mae, 4),
            "r2_train": round(r2, 4),
            "r2_cv": round(cv_r2, 4),
            "n_samples": n_samples,
            "model_type": type(model).__name__
        }

    # ─── Step 4: Update Drug Interaction Knowledge ─────────────────────

    def _update_drug_interactions(
        self, outcomes: List[TreatmentOutcome], db: Session
    ) -> int:
        """
        Aggregate treatment outcomes into TumorDrugInteraction knowledge base.

        Schema mapping (TumorDrugInteraction columns):
            tumor_type (str): cancer type, e.g. "colon"
            tumor_stage (str): TNM stage
            drug_cocktail (JSON): {"drugs": [...], "dosages": [...]}
            drug_cocktail_hash (str): SHA256 for lookup
            efficacy_score (float): weighted mean efficacy
            confidence (float): min(0.95, evidence_count / 20)
            evidence_count (int): number of outcomes
            avg_tumor_reduction (float): mean tumor_size_change_percent
        """
        interactions_updated = 0

        # Group outcomes by (tumor_type, cocktail_hash)
        groups: Dict[str, List[TreatmentOutcome]] = {}
        for outcome in outcomes:
            # Get tumor stage from linked CTAnalysis
            ct = db.query(CTAnalysis).filter(
                CTAnalysis.patient_id == outcome.patient_id
            ).order_by(desc(CTAnalysis.analysis_date)).first()

            tumor_stage = ct.tnm_stage if ct and ct.tnm_stage else "Unknown"

            # Hash the drug cocktail for grouping
            cocktail = outcome.prescribed_cocktail or {}
            if isinstance(cocktail, str):
                try:
                    cocktail = json.loads(cocktail)
                except (json.JSONDecodeError, TypeError):
                    cocktail = {}

            drugs = sorted(cocktail.get('drugs', [])) if isinstance(cocktail, dict) else []
            cocktail_hash = hashlib.sha256(
                json.dumps(drugs).encode()
            ).hexdigest()[:16]

            key = f"{tumor_stage}|{cocktail_hash}"
            if key not in groups:
                groups[key] = {
                    "outcomes": [],
                    "tumor_stage": tumor_stage,
                    "cocktail": cocktail,
                    "cocktail_hash": cocktail_hash,
                    "drugs": drugs
                }
            groups[key]["outcomes"].append(outcome)

        # Update or create TumorDrugInteraction records
        for key, group in groups.items():
            group_outcomes = group["outcomes"]
            if len(group_outcomes) < 2:
                continue

            # Calculate aggregate metrics
            efficacies = [
                o.actual_efficacy for o in group_outcomes
                if o.actual_efficacy is not None
            ]
            reductions = [
                o.tumor_size_change_percent for o in group_outcomes
                if o.tumor_size_change_percent is not None
            ]
            survivals = [
                o.survival_months for o in group_outcomes
                if o.survival_months is not None
            ]

            if not efficacies:
                continue

            avg_efficacy = float(np.mean(efficacies))
            avg_reduction = float(np.mean(reductions)) if reductions else None
            avg_survival = float(np.mean(survivals)) if survivals else None
            n = len(group_outcomes)

            # Upsert
            existing = db.query(TumorDrugInteraction).filter(
                TumorDrugInteraction.drug_cocktail_hash == group["cocktail_hash"],
                TumorDrugInteraction.tumor_stage == group["tumor_stage"]
            ).first()

            if existing:
                existing.efficacy_score = avg_efficacy
                existing.confidence = min(0.95, n / 20.0)
                existing.evidence_count = n
                existing.avg_tumor_reduction = avg_reduction
                existing.avg_survival_months = avg_survival
                existing.last_updated = datetime.now()
            else:
                interaction = TumorDrugInteraction(
                    tumor_type="colon",  # Default for ADDS
                    tumor_stage=group["tumor_stage"],
                    drug_cocktail=group["cocktail"],
                    drug_cocktail_hash=group["cocktail_hash"],
                    efficacy_score=avg_efficacy,
                    confidence=min(0.95, n / 20.0),
                    evidence_count=n,
                    avg_tumor_reduction=avg_reduction,
                    avg_survival_months=avg_survival
                )
                db.add(interaction)

            interactions_updated += 1

        db.commit()
        return interactions_updated

    # ─── Step 5: Record Training History ───────────────────────────────

    def _record_training_history(
        self, db: Session, metrics: Dict, n_samples: int
    ):
        """
        Record this training run in ModelTrainingHistory.

        Schema mapping (ModelTrainingHistory columns):
            version (str, unique): model version string
            training_date (DateTime)
            dataset_size (int): number of training samples
            training_loss (float): MSE on training set
            validation_loss (float): not used (CV-based)
            test_accuracy (float): R² from cross-validation
            prediction_mae (float): mean absolute error
            parameters (JSON): hyperparameters snapshot
        """
        history = ModelTrainingHistory(
            version=self.model_version,
            training_date=datetime.now(),
            dataset_size=n_samples,
            training_loss=metrics.get("mse"),
            test_accuracy=metrics.get("r2_cv"),
            prediction_mae=metrics.get("mae"),
            parameters={
                "model_type": metrics.get("model_type"),
                "r2_train": metrics.get("r2_train"),
                "r2_cv": metrics.get("r2_cv"),
                "n_features": 10
            }
        )
        db.add(history)
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            # version uniqueness conflict — append timestamp
            history.version = f"{self.model_version}_{datetime.now().strftime('%H%M%S')}"
            db.add(history)
            db.commit()

    # ─── Step 6: Record Performance Metrics ────────────────────────────

    def _record_performance_metrics(
        self, db: Session, outcomes: List, metrics: Dict
    ):
        """
        Record daily performance metrics in PerformanceMetric.

        Schema mapping (PerformanceMetric columns):
            metric_date (Date, unique)
            total_analyses (int)
            prediction_accuracy (float): 1 - mean|pred - actual|
            avg_prediction_error (float)
            model_version (str)
        """
        pred_errors = []
        for o in outcomes:
            if o.predicted_efficacy is not None and o.actual_efficacy is not None:
                pred_errors.append(abs(o.predicted_efficacy - o.actual_efficacy))

        avg_error = float(np.mean(pred_errors)) if pred_errors else None

        today = datetime.now().date()

        # Check if today's record already exists
        existing = db.query(PerformanceMetric).filter(
            PerformanceMetric.metric_date == today
        ).first()

        if existing:
            existing.total_analyses = len(outcomes)
            existing.prediction_accuracy = (1.0 - avg_error) if avg_error else None
            existing.avg_prediction_error = avg_error
            existing.model_version = self.model_version
        else:
            perf = PerformanceMetric(
                metric_date=today,
                total_analyses=len(outcomes),
                prediction_accuracy=(1.0 - avg_error) if avg_error else None,
                avg_prediction_error=avg_error,
                model_version=self.model_version
            )
            db.add(perf)

        db.commit()


# ─── Entry Point ───────────────────────────────────────────────────────

def run_daily_training():
    """Entry point for daily training cron job / scheduled task."""
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        trainer = DailyMLTrainer(min_samples=3)
        report = trainer.run_daily_training(db)

        print("\n" + "=" * 60)
        print("DAILY ML TRAINING REPORT")
        print("=" * 60)
        print(f"Status: {report['status']}")
        print(f"Total outcomes: {report.get('total_outcomes', 0)}")

        if report['status'] == 'completed':
            for step in report.get('steps', []):
                print(f"\n  {step['step']}:")
                for k, v in step.items():
                    if k != 'step':
                        print(f"    {k}: {v}")
        elif report.get('reason'):
            print(f"Reason: {report['reason']}")

        print("=" * 60)
        return report

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_training()
