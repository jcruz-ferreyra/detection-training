import logging
from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MeanAveragePrecisionResult
from ultralytics import YOLO
import yaml

from detection_training.tracking import log_metrics
from detection_training.utils import get_ultralytics_detections

from .types import ModelEvaluationContext

logger = logging.getLogger(__name__)


def _get_mlflow_run_info(ctx: ModelEvaluationContext) -> None:
    """Discover and populate context with latest MLflow run information."""
    logger.info(f"Searching for latest run in experiment: {ctx.experiment}")

    try:
        # Get experiment by name
        experiment = mlflow.get_experiment_by_name(ctx.experiment)
        if experiment is None:
            raise ValueError(f"Experiment '{ctx.experiment}' not found in MLflow")

        # Search for runs in the experiment, ordered by start time (latest first)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1
        )

        if runs.empty:
            raise ValueError(f"No runs found in experiment '{ctx.experiment}'")

        latest_run = runs.iloc[0]
        run_id = latest_run["run_id"]

        logger.info(f"Found latest run: {run_id}")
        logger.debug(f"Run start time: {latest_run['start_time']}")
        logger.debug(f"Run status: {latest_run['status']}")

        # Store run_id in context to log evaluation metrics
        ctx.run_id = run_id

        # Get run details to access parameters
        run_details = mlflow.get_run(run_id)
        params = run_details.data.params

        # Extract required parameters
        ctx.dataset_folder = _extract_parameter(params, "dataset", "dataset folder")
        ctx.data_yaml = _extract_parameter(params, "data_yaml", "data yaml", "data.yaml")
        ctx.model_name = _extract_parameter(params, "best_model", "best model checkpoint")
        ctx.artifacts = _extract_parameter(params, "artifacts", "artifacts directory")

        logger.info(f"Populated context with run info:")
        logger.info(f"  Dataset folder: {ctx.dataset_folder}")
        logger.info(f"  Data YAML: {ctx.data_yaml}")
        logger.info(f"  Model checkpoint: {ctx.model_name}")
        logger.info(f"  Artifacts dir: {ctx.artifacts}")

    except Exception as e:
        logger.error(f"Failed to discover MLflow run info: {e}")
        raise


def _extract_parameter(
    params: dict, param_name: str, description: str, fallback: str = None
) -> str:
    """Extract a parameter from MLflow run parameters with optional fallback."""
    if param_name not in params:
        if fallback is not None:
            logger.warning(f"Parameter '{param_name}' not found, using fallback: {fallback}")
            return fallback
        raise ValueError(
            f"Required parameter '{param_name}' ({description}) not found in MLflow run"
        )

    value = params[param_name]
    logger.debug(f"Extracted {description}: {value}")
    return value


def _validate_discovered_paths(ctx: ModelEvaluationContext) -> None:
    """Validate that all discovered paths exist."""
    logger.info("Validating discovered paths")

    if not ctx.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {ctx.dataset_dir}")

    if not ctx.data_yaml_path.exists():
        raise FileNotFoundError(f"Data YAML file not found: {ctx.data_yaml_path}")

    if not ctx.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {ctx.model_path}")

    if not ctx.artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {ctx.artifacts_dir}")

    logger.info("All paths validated successfully")


def _load_model(ctx: ModelEvaluationContext) -> None:
    """Load YOLO model from the discovered model path."""
    logger.info(f"Loading YOLO model from: {ctx.model_path}")

    try:
        ctx.model = YOLO(str(ctx.model_path))
        logger.info("YOLO model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def _load_dataset(ctx: ModelEvaluationContext):
    """Load dataset for evaluation using supervision DetectionDataset."""
    logger.info(f"Loading dataset for split: {ctx.split}")

    try:
        # Load data YAML to get split paths
        with open(ctx.data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        if ctx.split not in data_config:
            raise ValueError(f"Split '{ctx.split}' not found in data YAML")

        # Get images path for the split
        split_images_path = data_config[ctx.split]
        logger.debug(f"Split images path from YAML: {split_images_path}")

        # Build full paths relative to data YAML parent directory
        images_dir = ctx.data_yaml_path.parent / split_images_path

        # Convert 'images' to 'labels' for annotations directory
        annotations_dir = Path(str(images_dir).replace("images", "labels"))

        logger.info(f"Images directory: {images_dir}")
        logger.info(f"Annotations directory: {annotations_dir}")

        # Validate directories exist
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

        # Load dataset using supervision
        ds = sv.DetectionDataset.from_yolo(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(annotations_dir),
            data_yaml_path=str(ctx.data_yaml_path),
        )

        logger.info(f"Dataset loaded successfully with {len(ds)} samples")
        return ds

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def _get_image_detections(
    ctx: ModelEvaluationContext, frame: np.ndarray, filename: str
) -> Optional[sv.Detections]:
    """Get YOLO detections for single image."""
    logger.debug(f"Running detection on image {filename}")

    try:
        dets = get_ultralytics_detections(
            frame, ctx.model, ctx.yolo_params, ctx.class_confidence, bgr=True
        )
        logger.debug(f"Found {len(dets)} detections in frame {filename}")
        return dets

    except Exception as e:
        logger.warning(f"Detection failed on frame {filename}: {e}")
        return sv.Detections.empty()


def _run_evaluation(
    ctx: ModelEvaluationContext, ds: sv.DetectionDataset
) -> Tuple[MeanAveragePrecisionResult, Optional[Dict[str, float]]]:
    """Run model evaluation on dataset with timing metrics."""
    logger.info(f"Starting evaluation on {len(ds)} images")

    map_metric = MeanAveragePrecision()
    detection_times = []

    for image_path, image, ann in ds:
        image_filename = Path(image_path).name

        start_time = time.time()
        dets = _get_image_detections(ctx, image, image_filename)
        end_time = time.time()

        detection_time = end_time - start_time
        detection_times.append(detection_time)

        map_metric.update(dets, ann)

    map_result = map_metric.compute()

    if detection_times:
        time_result = {
            "avg_time": sum(detection_times) / len(detection_times),
            "min_time": min(detection_times),
            "max_time": max(detection_times),
        }

        logger.info(
            f"Detection timing results: "
            f"Avg: {time_result['avg_time']:.4f}s, "
            f"Min: {time_result['min_time']:.4f}s, "
            f"Max: {time_result['max_time']:.4f}s"
        )

    else:
        time_result = None

    return map_result, time_result


def _log_evaluation_metrics(
    ctx: ModelEvaluationContext,
    map_results: MeanAveragePrecisionResult,
    time_result: Optional[Dict[str, float]],
) -> None:
    """Log evaluation metrics to MLflow with split prefix."""
    logger.info("Logging evaluation metrics to MLflow")

    # Log mAP metrics
    map_metrics = {
        "mAP_50_95": map_results.map50_95,
        "mAP_50": map_results.map50,
        "mAP_75": map_results.map75,
    }

    log_metrics(split=ctx.split, **map_metrics)

    # Log per-class AP metrics (average across IoU thresholds)
    if map_results.ap_per_class.size > 0:
        per_class_metrics = {}
        for class_id, ap_of_class in zip(map_results.matched_classes, map_results.ap_per_class):
            # Average AP across all IoU thresholds for this class
            avg_ap = ap_of_class.mean()
            label = ctx.class_label[class_id]
            per_class_metrics[f"AP_{label}"] = avg_ap

        log_metrics(split=ctx.split, **per_class_metrics)

    # Log object size-specific metrics if available
    if map_results.small_objects is not None:
        small_metrics = {
            "mAP_50_95_small": map_results.small_objects.map50_95,
            "mAP_50_small": map_results.small_objects.map50,
            "mAP_75_small": map_results.small_objects.map75,
        }
        log_metrics(split=ctx.split, **small_metrics)

    if map_results.medium_objects is not None:
        medium_metrics = {
            "mAP_50_95_medium": map_results.medium_objects.map50_95,
            "mAP_50_medium": map_results.medium_objects.map50,
            "mAP_75_medium": map_results.medium_objects.map75,
        }
        log_metrics(split=ctx.split, **medium_metrics)

    if map_results.large_objects is not None:
        large_metrics = {
            "mAP_50_95_large": map_results.large_objects.map50_95,
            "mAP_50_large": map_results.large_objects.map50,
            "mAP_75_large": map_results.large_objects.map75,
        }
        log_metrics(split=ctx.split, **large_metrics)

    # Log timing metrics if available
    if time_result is not None:
        log_metrics(split=ctx.split, **time_result)

    logger.info("Evaluation metrics logged successfully")


def _save_results(
    ctx: ModelEvaluationContext,
    map_results: MeanAveragePrecisionResult,
    time_result: Optional[Dict[str, float]],
) -> None:
    """Save evaluation results to CSV file."""
    logger.info("Saving evaluation results to CSV")

    # Convert mAP results to pandas DataFrame
    results_df = map_results.to_pandas()

    # Add timing results if available
    if time_result is not None:
        for metric_name, metric_value in time_result.items():
            results_df[metric_name] = metric_value

    # Add context information
    results_df["split"] = ctx.split
    results_df["model_family"] = ctx.model_family
    results_df["experiment"] = ctx.experiment
    results_df["run_id"] = ctx.run_id

    # Save to CSV
    csv_path = ctx.artifacts_dir / f"{ctx.split}_results.csv"
    results_df.to_csv(csv_path, index=False)

    logger.info(f"Results saved to: {csv_path}")


def evaluate_model(ctx: ModelEvaluationContext):
    logger.info("Starting model evaluation")

    # Populates None fields from MLflow
    _get_mlflow_run_info(ctx)

    # Ensure files actually exist
    _validate_discovered_paths(ctx)

    # Load model and store it in context
    _load_model(ctx)

    # Load evaluation dataset
    ds = _load_dataset(ctx)

    # Continue the same MLflow run for logging evaluation metrics
    with mlflow.start_run(run_id=ctx.run_id):
        map_result, time_result = _run_evaluation(ctx, ds)

        _log_evaluation_metrics(ctx, map_result, time_result)

        _save_results(ctx, map_result, time_result)

    logger.info("Evaluation completed successfully")
