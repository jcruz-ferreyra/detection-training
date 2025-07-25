from pathlib import Path

from detection_training.config import DATA_DIR, DRIVE_DATA_DIR, MODELS_DIR, DRIVE_MODELS_DIR
from detection_training.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_training.train_yolo import (
    YoloTrainingContext,
    train_yolo,
)

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

if ENVIRONMENT == "local":
    DATASET_DIR = DATA_DIR / DATASET_FOLDER
elif ENVIRONMENT == "colab":
    DATASET_DIR = DRIVE_DATA_DIR / DATASET_FOLDER

logger.info(f"Dataset directory: {DATASET_DIR}")

# Create paths for model checkpoint and project output (local or drive)
CHECKPOINT = script_config["checkpoint"]
PROJECT_NAME = script_config["project_name"]

if ENVIRONMENT == "local":
    CHECKPOINT_DIR = MODELS_DIR / CHECKPOINT
    PROJECT_DIR = MODELS_DIR
elif ENVIRONMENT == "colab":
    CHECKPOINT_DIR = DRIVE_MODELS_DIR / CHECKPOINT
    PROJECT_DIR = DRIVE_MODELS_DIR

logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
logger.info(f"Project directory: {PROJECT_DIR}")

# Select initial labelling batch
context = YoloTrainingContext(
    dataset_dir=DATASET_DIR,
    data_yaml=DATA_YAML,
    checkpoint_dir=CHECKPOINT_DIR,
    project_dir=PROJECT_DIR,
    project_name=PROJECT_NAME,
    training_params=TRAINING_PARAMS,
    environment=ENVIRONMENT,
)

# Task main function
train_yolo(context)
