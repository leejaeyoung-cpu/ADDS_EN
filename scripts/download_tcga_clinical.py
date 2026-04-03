"""TCGA-COAD/READ 임상 데이터 다운로드 및 처리"""

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training/tcga")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_tcga():
    base_url = "https://api.gdc.cancer.gov/cases"
    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-COAD", "TCGA-READ"],
        },
    }
    fields = ",".join([
        "submitter_id",
        "project.project_id",
        "demographic.gender",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.age_at_diagnosis",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.ajcc_pathologic_t",
        "diagnoses.ajcc_pathologic_n",
        "diagnoses.ajcc_pathologic_m",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.treatments.treatment_type",
        "diagnoses.treatments.therapeutic_agents",
    ])

    params = urllib.parse.urlencode({
        "filters": json.dumps(filters),
        "fields": fields,
        "size": "1000",
        "format": "json",
    })

    url = base_url + "?" + params
    logger.info("Fetching TCGA-COAD/READ from GDC API...")

    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    cases = data.get("data", {}).get("hits", [])
    logger.info("Total cases: %d", len(cases))

    with open(DATA_DIR / "tcga_coad_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

    return cases


def process_cases(cases):
    records = []
    for case in cases:
        rec = {"case_id": case.get("submitter_id", "")}
        proj = case.get("project", {})
        rec["project"] = proj.get("project_id", "")
        demo = case.get("demographic", {})
        rec["gender"] = demo.get("gender", "")
        rec["vital_status"] = demo.get("vital_status", "")
        rec["days_to_death"] = demo.get("days_to_death")

        diags = case.get("diagnoses", [])
        if diags:
            d = diags[0]
            rec["age_at_diagnosis"] = d.get("age_at_diagnosis")
            rec["ajcc_stage"] = d.get("ajcc_pathologic_stage", "")
            rec["ajcc_t"] = d.get("ajcc_pathologic_t", "")
            rec["ajcc_n"] = d.get("ajcc_pathologic_n", "")
            rec["ajcc_m"] = d.get("ajcc_pathologic_m", "")
            rec["days_to_followup"] = d.get("days_to_last_follow_up")

            treatments = d.get("treatments", [])
            chemo = any(
                t.get("treatment_type") == "Pharmaceutical Therapy" for t in treatments
            )
            rec["chemo"] = chemo
            agents = [
                t.get("therapeutic_agents", "")
                for t in treatments
                if t.get("therapeutic_agents")
            ]
            rec["agents"] = "; ".join(agents)

        records.append(rec)

    df = pd.DataFrame(records)
    n_coad = (df["project"] == "TCGA-COAD").sum()
    n_read = (df["project"] == "TCGA-READ").sum()
    logger.info("Total: %d, COAD: %d, READ: %d", len(df), n_coad, n_read)

    # Chemo patients
    chemo = df[df["chemo"] == True].copy()
    logger.info("Chemo patients: %d", len(chemo))

    # OS
    chemo["os_days"] = chemo.apply(
        lambda r: r["days_to_death"]
        if pd.notna(r["days_to_death"])
        else r.get("days_to_followup"),
        axis=1,
    )
    chemo_surv = chemo[chemo["os_days"].notna()].copy()
    chemo_surv["label"] = (
        (chemo_surv["vital_status"] == "Alive") | (chemo_surv["os_days"] > 1095)
    ).astype(int)

    resp = chemo_surv["label"].sum()
    nonr = len(chemo_surv) - resp
    logger.info("With survival: %d, Resp: %d, NonResp: %d", len(chemo_surv), resp, nonr)

    # Stage distribution
    logger.info("Stage distribution:")
    for s in sorted(chemo_surv["ajcc_stage"].unique()):
        n = (chemo_surv["ajcc_stage"] == s).sum()
        logger.info("  %s: %d", s, n)

    # Agent distribution
    logger.info("Chemo agents:")
    all_agents = {}
    for a in chemo_surv["agents"]:
        for ag in str(a).split(";"):
            ag = ag.strip()
            if ag and ag != "nan":
                all_agents[ag] = all_agents.get(ag, 0) + 1
    for ag, n in sorted(all_agents.items(), key=lambda x: -x[1])[:10]:
        logger.info("  %s: %d", ag, n)

    # Clinical-only CV
    if len(chemo_surv) > 30:
        stage_map = {}
        for s in chemo_surv["ajcc_stage"].unique():
            st = str(s)
            if "IV" in st:
                stage_map[s] = 4
            elif "III" in st:
                stage_map[s] = 3
            elif "II" in st:
                stage_map[s] = 2
            elif "I" in st:
                stage_map[s] = 1
            else:
                stage_map[s] = np.nan

        chemo_surv["stage_num"] = chemo_surv["ajcc_stage"].map(stage_map)
        chemo_surv["is_male"] = (chemo_surv["gender"] == "male").astype(float)
        chemo_surv["age_years"] = chemo_surv["age_at_diagnosis"].apply(
            lambda x: x / 365.25 if pd.notna(x) and x > 0 else np.nan
        )

        feat_cols = ["stage_num", "is_male", "age_years"]
        X = chemo_surv[feat_cols].values.astype(np.float32)
        y = chemo_surv["label"].values.astype(int)

        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = np.nanmedian(X[:, j])

        if y.sum() > 5 and (len(y) - y.sum()) > 5:
            n_splits = min(5, min(y.sum(), len(y) - y.sum()))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            aucs = []
            for ti, vi in skf.split(X, y):
                clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
                clf.fit(X[ti], y[ti])
                prob = clf.predict_proba(X[vi])[:, 1]
                try:
                    auc = roc_auc_score(y[vi], prob)
                    aucs.append(auc)
                except Exception:
                    pass

            if aucs:
                avg = np.mean(aucs)
                std = np.std(aucs)
                logger.info("TCGA clinical-only CV AUC: %.4f +/- %.4f", avg, std)

    chemo_surv.to_csv(DATA_DIR / "tcga_coad_chemo_survival.csv", index=False)
    logger.info("Saved TCGA data")
    return chemo_surv


def main():
    cases = download_tcga()
    if cases:
        process_cases(cases)


if __name__ == "__main__":
    main()
