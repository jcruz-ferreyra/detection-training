"""
MLflow logging utilities for model training and evaluation.
Framework-agnostic functions that can be called from any module in the project.
"""

import mlflow
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd

logger = logging.getLogger(__name__)


def log_experiment_info(
    model_family: str,
    model_architecture: str,
    environment: str,
    experiment_name: str,
    checkpoint_source: str,
    task_type: str = "object_detection",
):
    """
    Log high-level experiment information and tags.

    Args:
        model_family: Type of model (e.g., 'yolo', 'rf-detr', 'faster-rcnn')
        model_architecture: Specific architecture (e.g., 'yolov8n', 'rf-detr-r50')
        environment: Training environment ('local', 'colab')
        experiment_name: Name of the experiment
        checkpoint_source: Path to model checkpoint
        task_type: Type of ML task
    """
    # Tags for organization and filtering
    mlflow.set_tag("model_family", model_family)
    mlflow.set_tag("model_architecture", model_architecture)
    mlflow.set_tag("environment", environment)
    mlflow.set_tag("training_status", "running")
    mlflow.set_tag("task_type", task_type)

    # High-level experiment info
    mlflow.log_param("experiment_name", experiment_name)
    mlflow.log_param("checkpoint_source", str(checkpoint_source))


def log_model_config(
    model_name: str,
    dataset_dir: Path,
    num_parameters: Optional[int] = None,
    model_size_mb: Optional[float] = None,
    **additional_config,
):
    """
    Log model configuration details.

    Args:
        model_name: Name/identifier of the model
        dataset_dir: Path to dataset directory
        num_parameters: Number of model parameters
        model_size_mb: Model size in MB
        **additional_config: Any additional model configuration parameters
    """
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("dataset_path", str(dataset_dir))

    if num_parameters:
        mlflow.log_param("num_parameters", num_parameters)
    if model_size_mb:
        mlflow.log_param("model_size_mb", model_size_mb)

    # Log additional configuration
    for key, value in additional_config.items():
        if isinstance(value, (dict, list)):
            mlflow.log_param(key, json.dumps(value))
        else:
            mlflow.log_param(key, value)


def log_dataset_config(
    num_classes: int,
    class_names: List[str],
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    **dataset_info,
):
    """
    Log dataset configuration information.

    Args:
        num_classes: Number of classes in dataset
        class_names: List of class names
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        **dataset_info: Additional dataset information
    """
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("class_names", class_names)

    if train_samples:
        mlflow.log_param("train_samples", train_samples)
    if val_samples:
        mlflow.log_param("val_samples", val_samples)
    if test_samples:
        mlflow.log_param("test_samples", test_samples)

    # Log additional dataset info
    for key, value in dataset_info.items():
        if isinstance(value, (dict, list)):
            mlflow.log_param(key, json.dumps(value))
        else:
            mlflow.log_param(key, value)


def log_training_config(training_params: Dict[str, Any], output_dir: Optional[Path] = None):
    """
    Log training hyperparameters and configuration.

    Args:
        training_params: Dictionary of training parameters
        output_dir: Optional output directory for training
    """
    # Log all training parameters
    for key, value in training_params.items():
        if isinstance(value, (dict, list)):
            mlflow.log_param(key, json.dumps(value))
        else:
            mlflow.log_param(key, value)

    if output_dir:
        mlflow.log_param("output_dir", str(output_dir))


def log_training_metrics(
    metrics: Dict[str, Union[float, int]], epoch: Optional[int] = None, prefix: str = "train"
):
    """
    Log training metrics for a single epoch or final results.

    Args:
        metrics: Dictionary of metric names and values
        epoch: Optional epoch number for time-series metrics
        prefix: Prefix for metric names (e.g., 'train', 'val', 'final')
    """
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            full_metric_name = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(full_metric_name, value, step=epoch)
        else:
            logger.warning(f"Skipping non-numeric metric {metric_name}: {value}")


def log_validation_metrics(metrics: Dict[str, Union[float, int]], epoch: Optional[int] = None):
    """
    Log validation metrics.

    Args:
        metrics: Dictionary of metric names and values
        epoch: Optional epoch number for time-series metrics
    """
    log_training_metrics(metrics, epoch, prefix="val")


def log_final_metrics(metrics: Dict[str, Union[float, int]]):
    """
    Log final training/validation metrics.

    Args:
        metrics: Dictionary of final metric names and values
    """
    log_training_metrics(metrics, epoch=None, prefix="final")


def log_evaluation_metrics(metrics: Dict[str, Union[float, int]], dataset_split: str = "test"):
    """
    Log evaluation metrics on a specific dataset split.

    Args:
        metrics: Dictionary of metric names and values
        dataset_split: Name of dataset split ('test', 'val', 'holdout', etc.)
    """
    log_training_metrics(metrics, epoch=None, prefix=f"eval_{dataset_split}")


def log_epoch_metrics(
    train_metrics: Dict[str, Union[float, int]],
    val_metrics: Dict[str, Union[float, int]],
    epoch: int,
):
    """
    Log both training and validation metrics for a specific epoch.

    Args:
        train_metrics: Training metrics for this epoch
        val_metrics: Validation metrics for this epoch
        epoch: Epoch number
    """
    log_training_metrics(train_metrics, epoch, prefix="train")
    log_training_metrics(val_metrics, epoch, prefix="val")


def log_metrics_from_dataframe(
    df: pd.DataFrame, epoch_column: str = "epoch", metric_prefix: str = ""
):
    """
    Log metrics from a pandas DataFrame (framework-agnostic).

    Args:
        df: DataFrame containing metrics
        epoch_column: Name of column containing epoch numbers
        metric_prefix: Optional prefix for all metric names
    """
    try:
        for _, row in df.iterrows():
            epoch = int(row[epoch_column]) if epoch_column in df.columns else None

            for col in df.columns:
                if col != epoch_column and pd.notna(row[col]):
                    try:
                        value = float(row[col])
                        metric_name = f"{metric_prefix}_{col}" if metric_prefix else col
                        # Clean metric name
                        metric_name = metric_name.replace(" ", "_").replace("/", "_per_")
                        mlflow.log_metric(metric_name, value, step=epoch)
                    except (ValueError, TypeError):
                        continue

    except Exception as e:
        logger.warning(f"Could not log metrics from DataFrame: {e}")


def log_artifact_file(file_path: Path, artifact_folder: str = "outputs"):
    """
    Log a single file as an artifact.

    Args:
        file_path: Path to file to log
        artifact_folder: Folder name in MLflow artifacts
    """
    if file_path.exists():
        try:
            mlflow.log_artifact(str(file_path), artifact_folder)
        except Exception as e:
            logger.warning(f"Could not log artifact {file_path.name}: {e}")
    else:
        logger.warning(f"Artifact file not found: {file_path}")


def log_artifacts_from_directory(
    directory: Path, artifact_patterns: List[str] = None, artifact_folder: str = "outputs"
):
    """
    Log multiple artifacts from a directory based on patterns.

    Args:
        directory: Directory containing artifacts
        artifact_patterns: List of file patterns to match (e.g., ['*.png', '*.csv'])
        artifact_folder: Folder name in MLflow artifacts
    """
    if not directory.exists():
        logger.warning(f"Artifact directory not found: {directory}")
        return

    if artifact_patterns is None:
        artifact_patterns = ["*.png", "*.jpg", "*.csv", "*.json", "*.txt"]

    logged_count = 0
    for pattern in artifact_patterns:
        for file_path in directory.glob(pattern):
            try:
                mlflow.log_artifact(str(file_path), artifact_folder)
                logged_count += 1
            except Exception as e:
                logger.warning(f"Could not log artifact {file_path.name}: {e}")

    logger.info(f"Logged {logged_count} artifacts from {directory}")


def log_model_artifacts(model_path: Path, model_size_bytes: Optional[int] = None):
    """
    Log model-related information without uploading the actual model file.

    Args:
        model_path: Path to the trained model file
        model_size_bytes: Size of model file in bytes
    """
    if model_path.exists():
        mlflow.log_param("model_path", str(model_path))
        if model_size_bytes is None:
            model_size_bytes = model_path.stat().st_size
        mlflow.log_param("model_size_bytes", model_size_bytes)
        mlflow.log_param("model_size_mb", model_size_bytes / (1024**2))
    else:
        logger.warning(f"Model file not found: {model_path}")


def log_dataset_version_info(
    dataset_dir: Path, dataset_version: Optional[str] = None, **additional_info
):
    """
    Create and log dataset version information as an artifact.

    Args:
        dataset_dir: Path to dataset directory
        dataset_version: Optional version string for the dataset
        **additional_info: Additional dataset information to include
    """
    try:
        dataset_info = {
            "dataset_directory": str(dataset_dir),
            "dataset_version": dataset_version or "unknown",
            "created_at": str(pd.Timestamp.now()),
            **additional_info,
        }

        # Save as JSON artifact
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dataset_info, f, indent=2)
            temp_path = f.name

        mlflow.log_artifact(temp_path, "dataset_info")
        os.unlink(temp_path)  # Clean up temp file

    except Exception as e:
        logger.warning(f"Could not log dataset version info: {e}")


def log_system_info(**system_params):
    """
    Log system information like GPU, CPU, memory, etc.

    Args:
        **system_params: System information parameters
    """
    for key, value in system_params.items():
        mlflow.log_param(f"system_{key}", value)


def set_training_status(status: str):
    """
    Update training status tag.

    Args:
        status: Status to set ('running', 'completed', 'failed')
    """
    mlflow.set_tag("training_status", status)


def log_error(error_message: str, error_type: Optional[str] = None):
    """
    Log error information when training fails.

    Args:
        error_message: Error message to log
        error_type: Optional error type/category
    """
    set_training_status("failed")
    mlflow.log_param("error_message", str(error_message))
    if error_type:
        mlflow.log_param("error_type", error_type)


def log_custom_metric(metric_name: str, value: Union[float, int], step: Optional[int] = None):
    """
    Log a custom metric with optional step.

    Args:
        metric_name: Name of the metric
        value: Metric value
        step: Optional step number (e.g., epoch)
    """
    if isinstance(value, (int, float)):
        mlflow.log_metric(metric_name, value, step=step)
    else:
        logger.warning(f"Cannot log non-numeric metric {metric_name}: {value}")


def log_custom_param(param_name: str, param_value: Any):
    """
    Log a custom parameter.

    Args:
        param_name: Parameter name
        param_value: Parameter value
    """
    if isinstance(param_value, (dict, list)):
        mlflow.log_param(param_name, json.dumps(param_value))
    else:
        mlflow.log_param(param_name, param_value)


def extract_model_architecture(checkpoint_path: Union[str, Path]) -> str:
    """
    Extract model architecture from checkpoint path.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        Extracted architecture name
    """
    checkpoint_name = Path(checkpoint_path).name.lower()

    # YOLO models
    if "yolo" in checkpoint_name:
        parts = checkpoint_name.split(".")
        for part in parts:
            if "yolo" in part:
                return part

    # RF-DETR models
    if "detr" in checkpoint_name:
        if "rf-detr" in checkpoint_name:
            return "rf-detr"
        return "detr"

    # Faster R-CNN models
    if "faster" in checkpoint_name and "rcnn" in checkpoint_name:
        return "faster-rcnn"

    # Default: return filename without extension
    return Path(checkpoint_path).stem
