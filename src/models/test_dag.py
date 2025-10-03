from airflow.models.dagbag import DagBag

dagbag = DagBag(dag_folder="/home/sehar/INTELLIGENT-EXPENSE-TRACKER/dags")
if dagbag.import_errors:
    print("Errors:", dagbag.import_errors)
else:
    print("DAGs loaded successfully:", dagbag.dags.keys())
