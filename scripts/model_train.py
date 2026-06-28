"""
Train credit-default models, evaluate, select the best, and register it in the model bank.

Design (leakage-safe):
  - Prediction is made at the point of loan APPLICATION (feature snapshot = application month A).
  - The label (30dpd / 6mob) matures 6 months later, so it lives in the label store at month A+6.
  - We therefore join label_store(snapshot=T)  <->  feature_store(snapshot = T - 6 months)
    on Customer_ID. This guarantees only information available at application time is used.

Train / test / OOT split follows Lab 5 conventions, indexed by LABEL snapshot date:
  oot_end_date        = model_train_date - 1 day
  oot_start_date      = model_train_date - oot_period_months
  train_test_end_date = oot_start_date - 1 day
  train_test_start    = oot_start_date - train_test_period_months

Two candidate models are trained (Logistic Regression + XGBoost). The one with the
highest OOT AUC is registered as the production artefact in model_bank/.

Call: python scripts/model_train.py --snapshotdate "2024-09-01"
(working directory = project root, e.g. /opt/airflow)
"""
import argparse
import glob
import os
import pickle
import pprint
from datetime import datetime, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta

import pyspark
from pyspark.sql.functions import col

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# how many months after application the label matures (30dpd / 6mob)
LABEL_MOB = 6

ENG_FEATURES = [f"click_{i}m" for i in range(1, 7)]
FIN_FEATURES = [
    "Credit_History_Age", "Num_Fin_Pdts", "EMI_to_Salary", "Debt_to_Salary",
    "Repayment_Ability", "Loans_per_Credit_Item", "Loan_Extent", "Outstanding_Debt",
    "Interest_Rate", "Delay_from_due_date", "Changed_Credit_Limit",
]
FEATURE_COLS = ENG_FEATURES + FIN_FEATURES


def _read_all_parquet(spark, folder_path):
    files = [os.path.join(folder_path, os.path.basename(f))
             for f in glob.glob(os.path.join(folder_path, "*"))]
    if not files:
        raise FileNotFoundError(f"No partitions found in {folder_path}")
    return spark.read.option("header", "true").parquet(*files)


def build_labelled_dataset(spark):
    """Join matured labels to application-time features (T-6 offset). Returns pandas df.

    Financial-risk features are always available for an applicant (inner join), while
    clickstream engagement is an optional enrichment (left join) — many applicants have
    no clickstream history, so those engagement columns are left null and imputed later.
    """
    labels = _read_all_parquet(spark, "datamart/gold/label_store/").toPandas()
    eng = _read_all_parquet(spark, "datamart/gold/feature_store/eng/").toPandas()
    risk = _read_all_parquet(spark, "datamart/gold/feature_store/cust_fin_risk/").toPandas()

    for df in (labels, eng, risk):
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    # application month = label month minus the maturity window
    labels["feature_date"] = labels["snapshot_date"] - pd.DateOffset(months=LABEL_MOB)

    eng = eng.rename(columns={"snapshot_date": "feature_date"})[["Customer_ID", "feature_date"] + ENG_FEATURES]
    risk = risk.rename(columns={"snapshot_date": "feature_date"})[["Customer_ID", "feature_date"] + FIN_FEATURES]

    data = labels.merge(risk, on=["Customer_ID", "feature_date"], how="inner")   # financials required
    data = data.merge(eng, on=["Customer_ID", "feature_date"], how="left")       # clickstream optional
    return data


def make_pipeline(kind):
    pre = ColumnTransformer(
        [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), FEATURE_COLS)]
    )
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=88)
    elif kind == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=88,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def main(snapshotdate):
    print("\n\n--- starting model training job (train date", snapshotdate, ") ---\n\n")

    spark = (pyspark.sql.SparkSession.builder.appName("model_train")
             .master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # ---------- config (windows indexed by LABEL snapshot date) ----------
    cfg = {
        "model_train_date_str": snapshotdate,
        "train_test_period_months": 12,
        "oot_period_months": 2,
        "train_test_ratio": 0.8,
    }
    cfg["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    cfg["oot_end_date"] = cfg["model_train_date"] - timedelta(days=1)
    cfg["oot_start_date"] = cfg["model_train_date"] - relativedelta(months=cfg["oot_period_months"])
    cfg["train_test_end_date"] = cfg["oot_start_date"] - timedelta(days=1)
    cfg["train_test_start_date"] = cfg["oot_start_date"] - relativedelta(months=cfg["train_test_period_months"])
    pprint.pprint(cfg)

    # ---------- assemble labelled dataset ----------
    data = build_labelled_dataset(spark)
    data = data[(data["snapshot_date"] >= cfg["train_test_start_date"]) &
                (data["snapshot_date"] <= cfg["oot_end_date"])]
    print("labelled rows in window:", len(data), "default rate:", round(data["label"].mean(), 3))

    oot = data[(data["snapshot_date"] >= cfg["oot_start_date"]) &
               (data["snapshot_date"] <= cfg["oot_end_date"])]
    tt = data[(data["snapshot_date"] >= cfg["train_test_start_date"]) &
              (data["snapshot_date"] <= cfg["train_test_end_date"])]

    X_oot, y_oot = oot[FEATURE_COLS], oot["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        tt[FEATURE_COLS], tt["label"],
        test_size=1 - cfg["train_test_ratio"], random_state=88,
        shuffle=True, stratify=tt["label"],
    )
    print(f"train={len(X_train)} test={len(X_test)} oot={len(X_oot)}")

    # ---------- train + evaluate candidate models ----------
    candidates = {}
    for kind in ["logreg", "xgboost"]:
        pipe = make_pipeline(kind)
        pipe.fit(X_train, y_train)
        res = {
            "auc_train": roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1]),
            "auc_test": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
            "auc_oot": roc_auc_score(y_oot, pipe.predict_proba(X_oot)[:, 1]),
        }
        res["gini_oot"] = round(2 * res["auc_oot"] - 1, 3)
        candidates[kind] = {"pipeline": pipe, "results": res}
        print(f"[{kind}] AUC train={res['auc_train']:.3f} test={res['auc_test']:.3f} oot={res['auc_oot']:.3f}")

    # ---------- select best by OOT AUC ----------
    best_kind = max(candidates, key=lambda k: candidates[k]["results"]["auc_oot"])
    best = candidates[best_kind]
    print(f"\nSelected best model: {best_kind} (OOT AUC={best['results']['auc_oot']:.3f})")

    # baseline score distribution on training data -> used by monitoring for PSI
    train_scores = best["pipeline"].predict_proba(X_train)[:, 1]

    # ---------- build + save artefact ----------
    model_version = "credit_model_" + snapshotdate.replace("-", "_")
    artefact = {
        "model": best["pipeline"],            # full sklearn pipeline (impute+scale+clf)
        "model_type": best_kind,
        "model_version": model_version,
        "feature_cols": FEATURE_COLS,
        "label_mob": LABEL_MOB,
        "data_dates": {k: (v.strftime("%Y-%m-%d") if isinstance(v, datetime) else v)
                       for k, v in cfg.items()},
        "data_stats": {"X_train": len(X_train), "X_test": len(X_test), "X_oot": len(X_oot),
                       "default_rate_train": round(float(y_train.mean()), 3)},
        "results": {k: candidates[k]["results"] for k in candidates},
        "score_baseline": train_scores.tolist(),
    }

    os.makedirs("model_bank/", exist_ok=True)
    out_path = os.path.join("model_bank/", model_version + ".pkl")
    with open(out_path, "wb") as f:
        pickle.dump(artefact, f)
    print("Model registered to model bank:", out_path)

    spark.stop()
    print("\n\n--- completed model training job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train + register credit default model")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (model train date)")
    args = parser.parse_args()
    main(args.snapshotdate)
