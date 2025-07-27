from pathlib import Path

import mlflow

from detection_training.config import (
    DRIVE_DATA_DIR,
    DRIVE_MODELS_DIR,
    LOCAL_DATA_DIR,
    LOCAL_MODELS_DIR,
)
from detection_training.tracking import get_mlflow_uri, start_mlflow, stop_mlflow
from detection_training.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, LOCAL_DATA_DIR)

from detection_training.train_yolo import YoloTrainingContext, train_yolo

logger.info("Starting dataset generation pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "dataset_folder",
    "data_yaml",
    "checkpoint",
    "project_name",
    "training_params",
    "environment",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

TRAINING_PARAMS = script_config["training_params"]

# Create paths for dataset input (local or drive)
ENVIRONMENT = script_config["environment"]
valid_environments = ["local", "colab"]

if ENVIRONMENT not in valid_environments:
    raise ValueError(f"output_storage configuration should be one of {valid_environments}")
elif ENVIRONMENT == "drive" and (DRIVE_DATA_DIR is None or DRIVE_MODELS_DIR is None):
    raise ValueError(
        "Error accesing Drive directory. Try setting local storage or check provided drive path."
    )

DATASET_FOLDER = script_config["dataset_folder"]
DATA_YAML = script_config["data_yaml"]

CHECKPOINT = script_config["checkpoint"]
PROJECT_NAME = script_config["project_name"]

# Select data and models dir based on environment
DATA_DIR = DRIVE_DATA_DIR if ENVIRONMENT == "colab" else LOCAL_DATA_DIR
MODELS_DIR = DRIVE_MODELS_DIR if ENVIRONMENT == "colab" else LOCAL_MODELS_DIR

logger.info(f"Dataset directory: {DATA_DIR}")
logger.info(f"Project directory: {MODELS_DIR}")

# Select initial labelling batch
context = YoloTrainingContext(
    data_dir=DATA_DIR,
    dataset_folder=DATASET_FOLDER,
    data_yaml=DATA_YAML,
    models_dir=MODELS_DIR,
    checkpoint=CHECKPOINT,
    project_name=PROJECT_NAME,
    training_params=TRAINING_PARAMS,
    environment=ENVIRONMENT,
)

# Task main function
try:
    start_mlflow(context.environment)
    mlflow.set_tracking_uri(get_mlflow_uri())

    train_yolo(context)

except Exception as e:
    logger.error(e)
    raise

finally:
    stop_mlflow()
