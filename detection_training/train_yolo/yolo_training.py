from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict

import mlflow
import pytz
from ultralytics import YOLO
import yaml

from detection_training.tracking import log_metrics, log_params, log_tags

from .types import YoloTrainingContext

logger = logging.getLogger(__name__)


def _retrieve_and_unzip_data(ctx: YoloTrainingContext):
    """Download and extract dataset from drive to colab local storage."""
    logger.info("Starting dataset retrieval and extraction for Colab environment")

    import shutil
    import time
    import zipfile

    zipfile_name = "yolo.zip"
    colab_data_dir = Path("/content/data")

    colab_dataset_dir = colab_data_dir / ctx.dataset_folder
    colab_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Copy zip file to colab
    src = ctx.dataset_dir / zipfile_name
    dst = colab_dataset_dir / zipfile_name

    # Validate source file exists
    if not src.exists():
        logger.error(f"Source zip file not found: {src}")
        raise FileNotFoundError(f"Dataset zip file not found: {src}")

    try:
        logger.info(f"Copying dataset from drive: {src}")
        start = time.time()
        shutil.copy2(src, dst)
        copy_time = time.time() - start

        # Log copy statistics
        file_size_mb = src.stat().st_size / (1024 * 1024)
        logger.info(
            f"Copied {file_size_mb:.1f} MB in {copy_time:.2f} seconds "
            f"({file_size_mb/copy_time:.1f} MB/s)"
        )

    except Exception as e:
        logger.error(f"Failed to copy dataset zip file: {e}")
        raise

    # Create extraction directory named after the dataset format
    extraction_dir = colab_dataset_dir / "yolo"  # "yolo"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Extracting dataset to: {colab_data_dir}")
        start = time.time()

        with zipfile.ZipFile(dst, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)

        extract_time = time.time() - start

        # Count extracted files
        extracted_files = len(list(colab_data_dir.rglob("*")))
        logger.info(f"Extracted {extracted_files} files in {extract_time:.2f} seconds")

    except zipfile.BadZipFile as e:
        logger.error(f"Corrupted zip file: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        raise

    try:
        # Clean up zip file to save space
        dst.unlink()
        logger.info("Cleaned up zip file to save space")

    except Exception as e:
        logger.warning(f"Failed to clean up zip file: {e}")

    # Update context dataset directory
    ctx.data_dir = colab_data_dir
    logger.info(f"Updated dataset directory to: {ctx.dataset_dir}")
    logger.info("Dataset retrieval and extraction completed successfully")


def _update_dataset_yaml_path(ctx: YoloTrainingContext):
    """Update the 'path' value in a YOLO dataset YAML file."""
    logger.info(f"Updating {ctx.data_yaml_path.name} main path value")

    # Read the existing YAML file
    with open(ctx.data_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Update the path value
    data["path"] = str(ctx.data_yaml_path.parent)

    # Write back to the file
    with open(ctx.data_yaml_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def _get_model_architecture(ctx: YoloTrainingContext) -> str:
    """Extract YOLO architecture name from checkpoint filename."""
    try:
        # Get filename without extension
        checkpoint_stem = Path(ctx.checkpoint).stem
        parts = checkpoint_stem.split("_")

        # Look for YOLO architecture patterns
        for part in parts:
            part_lower = part.lower()
            if "yolo" in part_lower:
                return part

        # Fallback: return the checkpoint stem or "unknown"
        logger.warning(f"Could not extract architecture from checkpoint: {ctx.checkpoint}")
        return checkpoint_stem if checkpoint_stem else "unknown"

    except Exception as e:
        logger.warning(f"Failed to extract architecture from checkpoint {ctx.checkpoint}: {e}")
        return "unknown"


def _load_dataset_info(ctx: YoloTrainingContext) -> Dict[str, Any]:
    """Load dataset information from YOLO data.yaml file and count samples."""
    try:
        with open(ctx.data_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Count samples for each split
        dataset_dir = ctx.data_yaml_path.parent
        for split in ["train", "val", "test"]:
            split_folder = data_config.get(split)
            if split_folder:
                split_path = dataset_dir / split_folder
                if split_path.exists() and split_path.is_dir():
                    try:
                        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
                        sample_count = len(
                            [
                                f
                                for f in split_path.iterdir()
                                if f.is_file() and f.suffix.lower() in image_extensions
                            ]
                        )
                        data_config[f"{split}_samples"] = sample_count
                    except Exception as e:
                        logger.warning(f"Failed to count {split} samples: {e}")
                        data_config[f"{split}_samples"] = 0
                else:
                    logger.warning(f"Split directory not found: {split_path}")

        desired_keys = [
            "names",
            "nc",
            "train",
            "train_samples",
            "val",
            "val_samples",
            "test",
            "test_samples",
        ]
        filtered_config = {key: data_config[key] for key in desired_keys if key in data_config}

        return filtered_config

    except Exception as e:
        logger.warning(f"Failed to load dataset config from {ctx.data_yaml_path}: {e}")
        return {}


def _train_yolo(ctx: YoloTrainingContext):
    """Train YOLO model with specified configuration and return training results."""
    logger.info(f"Starting YOLO model training with checkpoint: {ctx.checkpoint_path}")

    mlflow.set_experiment(ctx.project_name)
    logger.info(f"Set MLflow experiment: {ctx.project_name}")

    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        project_name_with_run = f"{ctx.project_name}/{run_id}"
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"Artifacts will be saved to: {project_name_with_run}")

        try:
            # Log experiment metadata
            log_tags(
                experiment=ctx.project_name,
                model_family="yolo",
                model_arch=_get_model_architecture(ctx),
                environment=ctx.environment,
                task="detection",
                purpose="active_learning",
                run_id=run_id,
            )

            log_params(
                model_arch=_get_model_architecture(ctx),
                timestamp=datetime.now(pytz.UTC).isoformat(),
                run_id=run_id,
            )

            # Log dataset configuration
            dataset_info = _load_dataset_info(ctx)
            log_params(dataset=ctx.dataset_folder, data_yaml=ctx.data_yaml, **dataset_info)

            # Load YOLO model
            model = YOLO(ctx.checkpoint_path)

            # Log model configuration
            log_params(
                checkpoint=str(ctx.checkpoint),
                model_params=(
                    sum(p.numel() for p in model.model.parameters())
                    if hasattr(model, "model")
                    else None
                ),
            )

            # Log training configuration
            log_params(**ctx.training_params)

            # Training with custom callback for epoch metrics
            results = model.train(
                data=str(ctx.data_yaml_path),
                val=True,
                save=True,
                project=str(ctx.models_dir),
                name=project_name_with_run,
                exist_ok=True,
                plots=True,
                **ctx.training_params,
            )

            # Log final metrics (extract from results object)
            if hasattr(results, "results_dict"):
                log_metrics(split="val", **results.results_dict)

            # Log model info (without uploading the file)
            log_params(
                best_model=Path(project_name_with_run) / "weights" / "best.pt",
                artifacts=project_name_with_run,
            )

            log_tags(status="completed")
            return results

        except Exception as e:
            log_tags(status="failed")
            log_params(error_message=str(e), error_type=type(e).__name__)
            raise


def train_yolo(ctx: YoloTrainingContext):
    """
    Train YOLO model with dataset preprocessing for different environments.

    Args:
        ctx: YoloTrainingContext containing training configuration
    """
    logger.info("Starting YOLO training process")

    if ctx.environment == "colab":
        # Set up dataset in working directory and change dataset dir attribute in ctx
        _retrieve_and_unzip_data(ctx)

    # update yaml main path to dataset directory
    _update_dataset_yaml_path(ctx)

    # Train model
    _train_yolo(ctx)

    logger.info("YOLO training process completed successfully")
