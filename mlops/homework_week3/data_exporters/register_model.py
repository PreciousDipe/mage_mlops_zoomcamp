import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import subprocess

MLFLOW_TRACKING_URI = "http://mlflow:5000"

# Start MLflow UI
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("homework3")

mlflow.sklearn.autolog(log_datasets=False)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
    # Initialize MLflow client
    client = MlflowClient()

    # List all experiments
    experiments = client.list_experiments()

    # Find the experiment ID for the given experiment name
    experiment_id = next((exp.experiment_id for exp in experiments if exp.name == EXPERIMENT_NAME), None)

    # List all runs in the experiment
    runs = client.list_run_infos(experiment_id)

    # Get the latest run
    latest_run = runs[0]

    # Get the run ID
    run_id = latest_run.run_id

    # Retrieve the model artifacts
    artifacts = client.list_artifacts(run_id, "model")

    # Find the MLModel file
    mlmodel_file_info = next((artifact for artifact in artifacts if artifact.path.endswith("MLmodel")), None)

    # Retrieve the model size
    model_size_bytes = mlmodel_file_info.file_size
    print(f"Model size (bytes): {model_size_bytes}")

    return model_size_bytes
