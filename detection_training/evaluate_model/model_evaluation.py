import logging
from pathlib import Path

import mlflow
import supervision as sv
from ultralytics import YOLO
import yaml

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

        results = _run_evaluation(ctx)

        # Log evaluation metrics to the same run
        mlflow.log_metrics(
            {
                f"eval_{ctx.split}_mAP": results.map,
                f"eval_{ctx.split}_precision": results.precision,
                f"eval_{ctx.split}_recall": results.recall,
                # etc.
            }
        )

        _save_results(ctx, results)

    logger.info("Evaluation completed successfully")
