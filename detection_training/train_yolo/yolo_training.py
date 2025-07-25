import logging
from pathlib import Path

from ultralytics import YOLO

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

        with zipfile.ZipFile(dst, "r") as zip_ref:
            zip_ref.extractall(colab_dataset_dir)

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


def _train_yolo(ctx: YoloTrainingContext):
    """Train YOLO model with specified configuration and return training results."""
    logger.info(f"Starting YOLO model training with checkpoint: {ctx.checkpoint_dir}")

    try:
        # Load YOLO model
        model = YOLO(ctx.checkpoint_dir)
        logger.info("YOLO model loaded successfully")

        # Log training configuration
        logger.info("Training configuration:")
        logger.info(f"  Dataset: {ctx.dataset_dir}")
        logger.info(f"  Project: {ctx.project_dir}")
        logger.info(f"  Name: {ctx.project_name}")
        logger.info(f"  Training params: {ctx.training_params}")

        # Start training
        logger.info("Starting model training...")
        results = model.train(
            data=str(ctx.dataset_dir),
            val=True,
            save=True,
            device=0,
            project=str(ctx.project_dir),
            name=ctx.project_name,
            exist_ok=True,
            plots=True,
            **ctx.training_params,
        )

        logger.info("YOLO training completed successfully")
        return results

    except Exception as e:
        logger.error(f"YOLO training failed: {e}")
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

    # Train model
    _train_yolo(ctx)

    logger.info("YOLO training process completed successfully")
