"""
End-to-end Medallion data pipeline for ONE snapshot date.

Builds, for the given --snapshotdate:
  bronze -> silver -> gold  for  label store + feature stores
    - label store        (gold/label_store)              from lms loan data, dpd=30 / mob=6
    - engagement table   (gold/feature_store/eng)        clickstream last-6-month pivot
    - cust_fin_risk      (gold/feature_store/cust_fin_risk) engineered financial-risk features

This script is designed to be triggered once per month by Airflow with
--snapshotdate "{{ ds }}" so the whole datamart can be backfilled across time.

Call: python scripts/data_pipeline.py --snapshotdate "2023-01-01"
(run with working directory = project root, e.g. /opt/airflow)
"""
import argparse
import os

import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


def main(snapshotdate):
    print("\n\n--- starting data pipeline job for", snapshotdate, "---\n\n")

    spark = (
        pyspark.sql.SparkSession.builder
        .appName("data_pipeline")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    date_str = snapshotdate

    # ---------- directory layout ----------
    dirs = {
        "bronze_lms":  "datamart/bronze/lms/",
        "bronze_clks": "datamart/bronze/clks/",
        "bronze_attr": "datamart/bronze/attr/",
        "bronze_fin":  "datamart/bronze/fin/",
        "silver_lms":  "datamart/silver/lms/",
        "silver_clks": "datamart/silver/clks/",
        "silver_attr": "datamart/silver/attr/",
        "silver_fin":  "datamart/silver/fin/",
        "gold_label":  "datamart/gold/label_store/",
        "gold_eng":    "datamart/gold/feature_store/eng/",
        "gold_risk":   "datamart/gold/feature_store/cust_fin_risk/",
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # ---------- BRONZE ----------
    print("=== BRONZE ===")
    bronze.process_bronze_loan_table(date_str, dirs["bronze_lms"], spark)
    bronze.process_bronze_clickstream_table(date_str, dirs["bronze_clks"], spark)
    bronze.process_bronze_attributes_table(date_str, dirs["bronze_attr"], spark)
    bronze.process_bronze_financials_table(date_str, dirs["bronze_fin"], spark)

    # ---------- SILVER ----------
    print("=== SILVER ===")
    silver.process_silver_loan_table(date_str, dirs["bronze_lms"], dirs["silver_lms"], spark)
    silver.process_silver_clickstream_table(date_str, dirs["bronze_clks"], dirs["silver_clks"], spark)
    silver.process_silver_attributes_table(date_str, dirs["bronze_attr"], dirs["silver_attr"], spark)
    silver.process_silver_financials_table(date_str, dirs["bronze_fin"], dirs["silver_fin"], spark)

    # ---------- GOLD ----------
    print("=== GOLD ===")
    # label store: loans reaching mob=6 in this snapshot month (applied 6 months earlier)
    gold.process_labels_gold_table(date_str, dirs["silver_lms"], dirs["gold_label"], spark, dpd=30, mob=6)
    # engagement features: clickstream pivot over the 6 months BEFORE this snapshot
    gold.process_fts_gold_engag_table(date_str, dirs["silver_clks"], dirs["gold_eng"], spark)
    # customer financial-risk features at this snapshot
    gold.process_fts_gold_cust_risk_table(date_str, dirs["silver_fin"], dirs["gold_risk"], spark)

    spark.stop()
    print("\n\n--- completed data pipeline job for", snapshotdate, "---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run monthly data pipeline")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
