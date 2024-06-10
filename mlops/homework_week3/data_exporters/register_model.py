import mlflow
import mlflow.sklearn
import pickle
import os

MLFLOW_TRACKING_URI = "http://mlflow:5000"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("homework3")

# Enable autologging
mlflow.sklearn.autolog(log_datasets=False)

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export_data(data, *args, **kwargs):
    lr, dv = data

    with mlflow.start_run() as run:
        # Log the intercept as a metric
        intercept = lr.intercept_
        mlflow.log_metric("intercept", intercept)
        print("Intercept:", intercept)

        # Log the linear regression model
        mlflow.sklearn.log_model(lr, artifact_path="model")

        # Save and log the DictVectorizer
        dv_path = "dict_vectorizer.pkl"
        with open(dv_path, 'wb') as f:
            pickle.dump(dv, f)
        mlflow.log_artifact(dv_path, artifact_path="preprocessors")
        
        # Clean up
        os.remove(dv_path)
        
        print(f"Model and DictVectorizer logged with run_id: {run.info.run_id}")

if name == 'main':
    export_data()