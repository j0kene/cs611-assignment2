"""
Monitor model performance and stability for one EVALUATION month.

At evaluation month E we can finally judge the cohort scored 6 months earlier
(application month A = E - label_mob), because their 30dpd/6mob labels mature at E
and land in the label store at snapshot = E.

We compute and persist, per evaluation month:
  performance : AUC, Gini, sample size, observed default rate (vs predicted)
  stability   : PSI of the application-month score distribution vs the training baseline

Output: datamart/gold/model_monitoring/<model_version>/<model_version>_monitor_<E>.parquet

Call: python scripts/model_monitor.py --snapshotdate "2024-10-01" --modelname "credit_model_2024_03_01.pkl"
(working directory = project root, e.g. /opt/airflow)
"""
import argparse
import glob
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import pyspark
from sklearn.metrics import roc_auc_score


def _read_exact(spark, filepath):
    """Read a single known parquet partition (returns pandas df or None if absent/empty)."""
    if not os.path.exists(filepath):
        return None
    pdf = spark.read.parquet(filepath).toPandas()
    return pdf if len(pdf) else None


def calculate_psi(expected, actual, n_bins=10):
    """Population Stability Index of `actual` scores vs the `expected` baseline scores."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    # quantile bin edges from the baseline distribution
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    act_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    # avoid div-by-zero / log(0)
    eps = 1e-6
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main(snapshotdate, modelname):
    print("\n\n--- starting monitoring job for eval month", snapshotdate, "---\n\n")

    spark = (pyspark.sql.SparkSession.builder.appName("model_monitor")
             .master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    eval_date = datetime.strptime(snapshotdate, "%Y-%m-%d")

    # ---------- load artefact (for baseline + label_mob) ----------
    with open(os.path.join("model_bank/", modelname), "rb") as f:
        artefact = pickle.load(f)
    model_version = artefact["model_version"]
    label_mob = artefact.get("label_mob", 6)
    baseline_scores = artefact.get("score_baseline", [])

    # application month whose labels mature now
    app_date = eval_date - relativedelta(months=label_mob)
    app_date_str = app_date.strftime("%Y-%m-%d")
    app_us = app_date_str.replace("-", "_")
    eval_us = snapshotdate.replace("-", "_")

    # ---------- load predictions (made at application month) ----------
    preds = _read_exact(
        spark,
        f"datamart/gold/model_predictions/{model_version}/{model_version}_predictions_{app_us}.parquet")
    if preds is None:
        print("no predictions for application month", app_date_str, "- skipping monitoring")
        spark.stop()
        return

    # ---------- load matured labels (label snapshot = eval month) ----------
    labels = _read_exact(spark, f"datamart/gold/label_store/gold_label_store_{eval_us}.parquet")

    auc = gini = obs_default_rate = float("nan")
    n_eval = 0
    if labels is not None:
        joined = preds.merge(labels[["Customer_ID", "label"]], on="Customer_ID", how="inner")
        n_eval = len(joined)
        if n_eval > 0 and joined["label"].nunique() == 2:
            auc = roc_auc_score(joined["label"], joined["model_predictions"])
            gini = round(2 * auc - 1, 4)
            obs_default_rate = round(float(joined["label"].mean()), 4)

    # ---------- stability (PSI of this cohort's scores vs training baseline) ----------
    psi = calculate_psi(baseline_scores, preds["model_predictions"].values)

    monitor_row = pd.DataFrame([{
        "model_version": model_version,
        "eval_date": snapshotdate,
        "application_date": app_date_str,
        "n_scored": len(preds),
        "n_labelled": n_eval,
        "auc": None if np.isnan(auc) else round(float(auc), 4),
        "gini": None if (isinstance(gini, float) and np.isnan(gini)) else gini,
        "observed_default_rate": None if np.isnan(obs_default_rate) else obs_default_rate,
        "avg_predicted_score": round(float(preds["model_predictions"].mean()), 4),
        "psi": None if np.isnan(psi) else round(psi, 4),
    }])
    print(monitor_row.to_string(index=False))

    out_dir = f"datamart/gold/model_monitoring/{model_version}/"
    os.makedirs(out_dir, exist_ok=True)
    partition = f"{model_version}_monitor_{snapshotdate.replace('-', '_')}.parquet"
    filepath = os.path.join(out_dir, partition)
    spark.createDataFrame(monitor_row).write.mode("overwrite").parquet(filepath)
    print("saved monitoring to:", filepath)

    spark.stop()
    print("\n\n--- completed monitoring job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run model monitoring for one eval month")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="artefact filename in model_bank/")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
