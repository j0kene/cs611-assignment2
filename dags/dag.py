"""
End-to-end ML pipeline DAG (Assignment 2).

Runs once per month and is fully backfillable (catchup=True) across the data history.
For each monthly snapshot {{ ds }} it executes, in order:

  1. data pipeline      : bronze -> silver -> gold (feature + label stores)
  2. model training     : (only on TRAIN_DATE) train candidates, select best, register in model_bank
  3. model inference     : (INFER window) score the application cohort -> gold predictions
  4. model monitoring    : (MONITOR window) evaluate matured cohort + PSI -> gold monitoring
  5. monitoring dashboard: render performance / stability visualisations

Timeline rationale (features end 2024-12, labels mature 6 months later):
  TRAIN_DATE   = 2024-07-01                 -> model trained on earliest matured window
  INFER window = 2024-07-01 .. 2024-12-01   -> score 6 out-of-time application cohorts
  MONITOR win  = 2025-01-01 .. 2025-06-01   -> those cohorts' labels mature here

Date gating is done in-line with bash conditionals so the python scripts stay pure
and reusable. Lexicographic comparison of YYYY-MM-DD strings is order-correct.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# ---------------- pipeline configuration ----------------
TRAIN_DATE = "2024-07-01"
INFER_START, INFER_END = "2024-07-01", "2024-12-01"
MONITOR_START = "2025-01-01"
MODEL_NAME = "credit_model_2024_07_01.pkl"   # artefact produced on TRAIN_DATE

PROJECT_DIR = "/opt/airflow"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="train -> inference -> monitoring ML pipeline, run monthly",
    schedule_interval="0 0 1 * *",          # 00:00 on the 1st of each month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------- 1. data pipeline (runs every month) ----------
    run_data_pipeline = BashOperator(
        task_id="run_data_pipeline",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f'python3 scripts/data_pipeline.py --snapshotdate "{{{{ ds }}}}"'
        ),
    )
    data_pipeline_completed = EmptyOperator(task_id="data_pipeline_completed")

    # ---------- 2. model training (only on TRAIN_DATE) ----------
    model_train = BashOperator(
        task_id="model_train",
        bash_command=(
            f'if [ "{{{{ ds }}}}" = "{TRAIN_DATE}" ]; then '
            f"cd {PROJECT_DIR} && "
            f'python3 scripts/model_train.py --snapshotdate "{{{{ ds }}}}"; '
            f'else echo "[{{{{ ds }}}}] not a training month - skipping"; fi'
        ),
    )

    # ---------- 3. model inference (INFER window) ----------
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=(
            f'if [[ "{{{{ ds }}}}" > "{INFER_START}" || "{{{{ ds }}}}" == "{INFER_START}" ]] && '
            f'[[ "{{{{ ds }}}}" < "{INFER_END}" || "{{{{ ds }}}}" == "{INFER_END}" ]]; then '
            f"cd {PROJECT_DIR} && "
            f'python3 scripts/model_inference.py --snapshotdate "{{{{ ds }}}}" --modelname "{MODEL_NAME}"; '
            f'else echo "[{{{{ ds }}}}] outside inference window - skipping"; fi'
        ),
    )

    # ---------- 4. model monitoring (MONITOR window) ----------
    model_monitor = BashOperator(
        task_id="model_monitor",
        bash_command=(
            f'if [[ "{{{{ ds }}}}" > "{MONITOR_START}" || "{{{{ ds }}}}" == "{MONITOR_START}" ]]; then '
            f"cd {PROJECT_DIR} && "
            f'python3 scripts/model_monitor.py --snapshotdate "{{{{ ds }}}}" --modelname "{MODEL_NAME}"; '
            f'else echo "[{{{{ ds }}}}] before monitoring window - skipping"; fi'
        ),
    )

    # ---------- 5. monitoring dashboard ----------
    monitor_dashboard = BashOperator(
        task_id="monitor_dashboard",
        bash_command=(
            f'if [[ "{{{{ ds }}}}" > "{MONITOR_START}" || "{{{{ ds }}}}" == "{MONITOR_START}" ]]; then '
            f"cd {PROJECT_DIR} && "
            f'python3 scripts/monitor_dashboard.py --modelname "{MODEL_NAME}"; '
            f'else echo "[{{{{ ds }}}}] before monitoring window - skipping"; fi'
        ),
    )

    end = EmptyOperator(task_id="end")

    # ---------- dependencies ----------
    start >> run_data_pipeline >> data_pipeline_completed
    data_pipeline_completed >> model_train >> model_inference >> model_monitor >> monitor_dashboard >> end
