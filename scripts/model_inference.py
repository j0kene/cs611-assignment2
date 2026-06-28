"""
Score one application month with the registered production model.

For --snapshotdate A (an application month) we assemble application-time features
(engagement[A] + cust_fin_risk[A]), load the model artefact from the model bank,
and write the predicted default probabilities to the gold datamart:

    datamart/gold/model_predictions/<model_version>/<model_version>_predictions_<A>.parquet

Call: python scripts/model_inference.py --snapshotdate "2024-04-01" --modelname "credit_model_2024_03_01.pkl"
(working directory = project root, e.g. /opt/airflow)
"""
import argparse
import glob
import os
import pickle
from datetime import datetime

import pandas as pd

import pyspark


def _read_exact(spark, filepath):
    """Read a single known parquet partition (returns pandas df or None if absent/empty)."""
    if not os.path.exists(filepath):
        return None
    pdf = spark.read.parquet(filepath).toPandas()
    return pdf if len(pdf) else None


def main(snapshotdate, modelname):
    print("\n\n--- starting inference job for", snapshotdate, "model", modelname, "---\n\n")

    spark = (pyspark.sql.SparkSession.builder.appName("model_inference")
             .master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    ds_us = snapshotdate.replace("-", "_")

    # ---------- load model artefact ----------
    model_path = os.path.join("model_bank/", modelname)
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model = artefact["model"]
    feature_cols = artefact["feature_cols"]
    model_version = artefact["model_version"]
    print("loaded model:", model_version, "type:", artefact.get("model_type"))

    # ---------- assemble application-time features ----------
    # financial-risk features define the scorable applicant population; clickstream
    # engagement is left-joined as optional enrichment (null -> imputed by the model).
    risk = _read_exact(spark, f"datamart/gold/feature_store/cust_fin_risk/gold_ft_store_cust_fin_risk_{ds_us}.parquet")
    if risk is None:
        print("no financial feature data for", snapshotdate, "- nothing to score")
        spark.stop()
        return
    risk["snapshot_date"] = pd.to_datetime(risk["snapshot_date"])

    eng = _read_exact(spark, f"datamart/gold/feature_store/eng/gold_ft_store_engagement_{ds_us}.parquet")
    if eng is not None:
        eng["snapshot_date"] = pd.to_datetime(eng["snapshot_date"])
        features = risk.merge(eng, on=["Customer_ID", "snapshot_date"], how="left")
    else:
        features = risk.copy()
    # guarantee all model columns exist even when clickstream is entirely absent
    for c in feature_cols:
        if c not in features.columns:
            features[c] = float("nan")
    n_with_clicks = int(features[[c for c in feature_cols if c.startswith("click_")]].notna().any(axis=1).sum())
    print(f"scorable population: {len(features)} ({n_with_clicks} with clickstream)")

    X = features[feature_cols]
    scores = model.predict_proba(X)[:, 1]

    out = features[["Customer_ID", "snapshot_date"]].copy()
    out["model_name"] = model_version
    out["model_predictions"] = scores

    # ---------- save to gold ----------
    gold_dir = f"datamart/gold/model_predictions/{model_version}/"
    os.makedirs(gold_dir, exist_ok=True)
    partition = f"{model_version}_predictions_{snapshotdate.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_dir, partition)
    spark.createDataFrame(out).write.mode("overwrite").parquet(filepath)
    print("saved predictions to:", filepath)

    spark.stop()
    print("\n\n--- completed inference job ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run model inference for one snapshot")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="artefact filename in model_bank/")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
