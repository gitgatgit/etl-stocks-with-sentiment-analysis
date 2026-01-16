"""MLflow utilities for experiment tracking and model registry."""
import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manage MLflow experiment tracking and model registry."""

    def __init__(self, tracking_uri: Optional[str] = None, experiment_name: str = "volatility-prediction"):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI (defaults to env var MLFLOW_TRACKING_URI)
            experiment_name: Name of the MLflow experiment
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start an MLflow run.

        Args:
            run_name: Name for this run
            tags: Dictionary of tags to apply to the run

        Returns:
            MLflow run object
        """
        return mlflow.start_run(run_name=run_name, tags=tags or {})

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current MLflow run."""
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to current MLflow run.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.info(f"Logged {len(metrics)} metrics to MLflow")

    def log_model(self, model, artifact_path: str = "model",
                  registered_model_name: Optional[str] = None,
                  signature=None, input_example=None):
        """
        Log model to MLflow.

        Args:
            model: Trained model object
            artifact_path: Path within run artifacts to log model
            registered_model_name: Name to register model under (for model registry)
            signature: MLflow model signature
            input_example: Example input for model inference

        Returns:
            Model info object
        """
        import sklearn

        # Determine model flavor
        if hasattr(model, 'get_params'):  # scikit-learn style
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )
        else:
            # Generic python model
            model_info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example
            )

        logger.info(f"Logged model to MLflow: {artifact_path}")
        if registered_model_name:
            logger.info(f"Registered model: {registered_model_name}")

        return model_info

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact file to MLflow.

        Args:
            local_path: Path to local file
            artifact_path: Destination path within run artifacts
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """
        Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Filename for the artifact
        """
        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"Logged dictionary as {artifact_file}")

    def load_model(self, model_uri: str):
        """
        Load model from MLflow.

        Args:
            model_uri: MLflow model URI (e.g., "models:/model_name/production", "runs:/run_id/model")

        Returns:
            Loaded model
        """
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded model from {model_uri}")
        return model

    def get_best_model(self, metric: str = "f1_score", ascending: bool = False) -> Optional[str]:
        """
        Get the best model run based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: If True, lower is better; if False, higher is better

        Returns:
            Run ID of best model, or None if no runs found
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            logger.warning(f"Experiment '{self.experiment_name}' not found")
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )

        if runs.empty:
            logger.warning("No runs found in experiment")
            return None

        best_run_id = runs.iloc[0]['run_id']
        best_metric_value = runs.iloc[0][f'metrics.{metric}']
        logger.info(f"Best model run: {best_run_id} with {metric}={best_metric_value}")

        return best_run_id

    def register_model(self, run_id: str, model_name: str, stage: str = "None"):
        """
        Register a model version and optionally transition to a stage.

        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            stage: Model stage (None, Staging, Production, Archived)

        Returns:
            ModelVersion object
        """
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(model_uri, model_name)

        if stage != "None":
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            logger.info(f"Transitioned model {model_name} v{model_version.version} to {stage}")

        return model_version

    def get_production_model_uri(self, model_name: str) -> str:
        """
        Get URI for production version of a registered model.

        Args:
            model_name: Registered model name

        Returns:
            Model URI string
        """
        return f"models:/{model_name}/Production"

    def compare_runs(self, run_ids: list) -> dict:
        """
        Compare metrics across multiple runs.

        Args:
            run_ids: List of MLflow run IDs

        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            comparison[run_id] = {
                'metrics': run.data.metrics,
                'params': run.data.params,
                'status': run.info.status
            }

        return comparison

    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")


def create_model_signature(input_df, output_array=None):
    """
    Create MLflow model signature from input/output examples.

    Args:
        input_df: Input DataFrame or example
        output_array: Output array or example (optional)

    Returns:
        MLflow ModelSignature
    """
    from mlflow.models.signature import infer_signature
    return infer_signature(input_df, output_array)
