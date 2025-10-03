from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from src.data.email_alerts import save_new_emails
from src.features.anamoly_detection import process_transactions
from src.models.text_splitting import update_monthly_chunks
from datetime import datetime

def fetch_emails():
    print("Fetching emails...")  # runs only when task executes

# DAG must be at global scope
with DAG(
    dag_id="test_pipeline",  # new DAG ID
    start_date=datetime(2025, 10, 1),
    schedule="@daily",
    catchup=False
) as dag:
    task1 = PythonOperator(
        task_id="fetch_and_save_emails",
        python_callable=fetch_emails
    )

    fetch_emails_task = PythonOperator(
        task_id="fetch_emails",
        python_callable=save_new_emails
    )

    detect_anomalies_task = PythonOperator(
        task_id="process_transactions",
        python_callable=process_transactions
    )

    text_splitting_task = PythonOperator(
        task_id="update_monthly_chunks",
        python_callable=update_monthly_chunks
    )

    task1 >> fetch_emails_task >> detect_anomalies_task >> text_splitting_task
print("DAG test_pipeline loaded")