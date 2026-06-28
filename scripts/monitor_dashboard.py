"""
Aggregate all monitoring partitions and render performance + stability visualisations.

Reads datamart/gold/model_monitoring/<model_version>/* for the given model and writes:
  datamart/gold/model_monitoring/plots/<model_version>_performance.png   (AUC / Gini over time)
  datamart/gold/model_monitoring/plots/<model_version>_stability.png     (PSI over time)
  datamart/gold/model_monitoring/plots/<model_version>_monitoring.csv    (tidy summary)

This task is idempotent and safe to run on every backfill month: it always
re-aggregates whatever monitoring partitions exist so far.

Call: python scripts/monitor_dashboard.py --modelname "credit_model_2024_03_01.pkl"
(working directory = project root, e.g. /opt/airflow)
"""
import argparse
import glob
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import pyspark


def main(modelname):
    print("\n\n--- starting monitoring dashboard for", modelname, "---\n\n")

    with open(os.path.join("model_bank/", modelname), "rb") as f:
        artefact = pickle.load(f)
    model_version = artefact["model_version"]

    spark = (pyspark.sql.SparkSession.builder.appName("monitor_dashboard")
             .master("local[*]").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    folder = f"datamart/gold/model_monitoring/{model_version}/"
    files = glob.glob(os.path.join(folder, "*.parquet"))
    if not files:
        print("no monitoring partitions yet - nothing to plot")
        spark.stop()
        return

    df = spark.read.parquet(*files).toPandas()
    df["eval_date"] = pd.to_datetime(df["eval_date"])
    df = df.sort_values("eval_date")

    plots_dir = "datamart/gold/model_monitoring/plots/"
    os.makedirs(plots_dir, exist_ok=True)
    df.to_csv(os.path.join(plots_dir, f"{model_version}_monitoring.csv"), index=False)

    # ---------- performance over time ----------
    perf = df.dropna(subset=["auc"])
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(perf):
        ax.plot(perf["eval_date"], perf["auc"], marker="o", label="AUC")
        ax.plot(perf["eval_date"], perf["gini"], marker="s", label="Gini")
        ax.axhline(0.5, ls="--", color="grey", lw=1, label="random (AUC=0.5)")
    ax.set_title(f"Model performance over time — {model_version}")
    ax.set_xlabel("Evaluation month")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    perf_path = os.path.join(plots_dir, f"{model_version}_performance.png")
    fig.savefig(perf_path, dpi=120)
    plt.close(fig)
    print("saved:", perf_path)

    # ---------- stability (PSI) over time ----------
    stab = df.dropna(subset=["psi"])
    fig, ax = plt.subplots(figsize=(9, 5))
    if len(stab):
        ax.bar(stab["eval_date"].dt.strftime("%Y-%m"), stab["psi"], color="#4C78A8")
        ax.axhline(0.1, ls="--", color="orange", lw=1, label="PSI=0.1 (moderate shift)")
        ax.axhline(0.25, ls="--", color="red", lw=1, label="PSI=0.25 (major shift)")
        ax.legend()
    ax.set_title(f"Population Stability Index over time — {model_version}")
    ax.set_xlabel("Evaluation month")
    ax.set_ylabel("PSI")
    ax.grid(alpha=0.3, axis="y")
    fig.autofmt_xdate()
    fig.tight_layout()
    stab_path = os.path.join(plots_dir, f"{model_version}_stability.png")
    fig.savefig(stab_path, dpi=120)
    plt.close(fig)
    print("saved:", stab_path)

    spark.stop()
    print("\n\n--- completed monitoring dashboard ---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="render monitoring visualisations")
    parser.add_argument("--modelname", type=str, required=True, help="artefact filename in model_bank/")
    args = parser.parse_args()
    main(args.modelname)
