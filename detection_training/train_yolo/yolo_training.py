import logging
from pathlib import Path

import mlflow
from ultralytics import YOLO
import yaml

from detection_training.tracking import (
    extract_model_architecture,
    log_artifacts_from_directory,
    log_dataset_config,
    log_error,
    log_experiment_info,
    log_final_metrics,
    log_model_artifacts,
    log_model_config,
    log_training_config,
    set_training_status,
)

from .types import YoloTrainingContext

logger = logging.getLogger(__name__)


def _retrieve_and_unzip_data(ctx: YoloTrainingContext):
    """Download and extract dataset from drive to colab local storage."""
    logger.info("Starting dataset retrieval and extraction for Colab environment")

    import shutil
    import time
    import zipfile

    zipfile_name = "yolo.zip"
    colab_dataset_dir = Path("/content/dataset")

    # Create extraction directory named after the zip file (without .zip extension)
    extraction_dir = colab_dataset_dir / Path(zipfile_name).stem  # "yolo"

    # Create colab dataset directory
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

    try:
        logger.info(f"Extracting dataset to: {colab_dataset_dir}")
        start = time.time()

        # Create extraction directory
        extraction_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(dst, "r") as zip_ref:
            zip_ref.extractall(extraction_dir)

        extract_time = time.time() - start

        # Count extracted files
        extracted_files = len(list(colab_dataset_dir.rglob("*")))
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
    ctx.dataset_dir = colab_dataset_dir
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


# def _train_yolo(ctx: YoloTrainingContext):
#     """Train YOLO model with specified configuration and return training results."""
#     logger.info(f"Starting YOLO model training with checkpoint: {ctx.checkpoint_dir}")

#     try:
#         # Load YOLO model
#         model = YOLO(ctx.checkpoint_dir)
#         logger.info("YOLO model loaded successfully")

#         # Log training configuration
#         logger.info("Training configuration:")
#         logger.info(f"  Dataset: {ctx.dataset_dir}")
#         logger.info(f"  Project: {ctx.project_dir}")
#         logger.info(f"  Name: {ctx.project_name}")
#         logger.info(f"  Training params: {ctx.training_params}")

#         # Start training
#         logger.info("Starting model training...")
#         results = model.train(
#             data=str(ctx.data_yaml_path),
#             val=True,
#             save=True,
#             project=str(ctx.project_dir),
#             name=ctx.project_name,
#             exist_ok=True,
#             plots=True,
#             **ctx.training_params,
#         )

#         logger.info("YOLO training completed successfully")
#         return results

#     except Exception as e:
#         logger.error(f"YOLO training failed: {e}")
#         raise


def _train_yolo(ctx: YoloTrainingContext):
    """Train YOLO model with specified configuration and return training results."""
    logger.info(f"Starting YOLO model training with checkpoint: {ctx.checkpoint_dir}")

    with mlflow.start_run():
        try:
            # Log experiment metadata
            log_experiment_info(
                model_family="yolo",
                model_architecture=extract_model_architecture(ctx.checkpoint_dir),
                environment=ctx.environment,
                experiment_name=ctx.project_name,
                checkpoint_source=str(ctx.checkpoint_dir),
            )

            # Load YOLO model
            model = YOLO(ctx.checkpoint_dir)

            # Log model configuration (extract what you can from model)
            log_model_config(
                model_name=f"yolo_{ctx.project_name}",
                dataset_dir=ctx.dataset_dir,
                num_parameters=(
                    sum(p.numel() for p in model.model.parameters())
                    if hasattr(model, "model")
                    else None
                ),
            )

            # Log dataset configuration (extract from your YAML or data)
            # You'd extract this info from your dataset YAML or inspection
            # TODO
            log_dataset_config(
                num_classes=80,  # Extract from your dataset
                class_names=["person", "car", ...],  # Extract from your dataset
                train_samples=5000,  # Extract from your dataset stats
                val_samples=1000,
            )

            # Log training configuration
            log_training_config(ctx.training_params, ctx.project_dir)

            # Training with custom callback for epoch metrics
            results = model.train(
                data=str(ctx.data_yaml_path),
                val=True,
                save=True,
                project=str(ctx.project_dir),
                name=ctx.project_name,
                exist_ok=True,
                plots=True,
                **ctx.training_params,
            )

            # Log final metrics (extract from results object)
            if hasattr(results, "results_dict"):
                log_final_metrics(results.results_dict)

            # Log artifacts (generic approach)
            output_dir = Path(ctx.project_dir) / ctx.project_name
            log_artifacts_from_directory(
                output_dir,
                artifact_patterns=["*.png", "*.csv", "*.jpg"],
                artifact_folder="training_outputs",
            )

            # Log model info (without uploading the file)
            best_model_path = output_dir / "weights" / "best.pt"
            log_model_artifacts(best_model_path)

            set_training_status("completed")
            return results

        except Exception as e:
            log_error(str(e), type(e).__name__)
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
