from pathlib import Path

from detection_training.config import DATA_DIR, DRIVE_DATA_DIR
from detection_training.utils import load_config, setup_logging

script_name = Path(__file__).parent.name
logger = setup_logging(script_name, DATA_DIR)

from detection_training.generate_dataset import (
    DatasetGenerationContext,
    generate_dataset,
)

logger.info("Starting dataset generation pipeline")

# Get script specific configs
CONFIG_PATH = Path(__file__).parent.resolve() / "config.yaml"

logger.info(f"Loading config from: {CONFIG_PATH}")
script_config = load_config(CONFIG_PATH)

required_keys = [
    "input_folder",
    "output_folder",
    "output_format",
    "split_datasets",
    "target_classes",
    "output_storage",
]
missing_keys = [key for key in required_keys if key not in script_config]
if missing_keys:
    logger.error(f"Missing required config keys: {missing_keys}")
    raise ValueError(f"Missing required config keys: {missing_keys}")

OUTPUT_FORMAT = script_config["output_format"]

SPLIT_DATASETS = script_config["split_datasets"]

TARGET_CLASSES = script_config["target_classes"]

# Create paths for frames directory
INPUT_DIR = DATA_DIR / script_config["input_folder"]

# Create paths for output directory (local or drive)
OUTPUT_STORAGE = script_config["output_storage"]
valid_storages = ["local", "drive"]

if OUTPUT_STORAGE not in valid_storages:
    raise ValueError(f"output_storage configuration should be one of {valid_storages}")
elif OUTPUT_STORAGE == "drive" and DRIVE_DATA_DIR is None:
    raise ValueError(
        "Error accesing Drive directory. Try setting local storage or check provided drive path."
    )

if OUTPUT_STORAGE == "local":
    OUTPUT_DIR = DATA_DIR / script_config["output_folder"] / OUTPUT_FORMAT
elif OUTPUT_STORAGE == "drive":
    OUTPUT_DIR = DRIVE_DATA_DIR / script_config["output_folder"] / OUTPUT_FORMAT

logger.info(f"Input main directory: {INPUT_DIR}")
logger.info(f"Output main directory: {OUTPUT_DIR}")

# Select initial labelling batch
context = DatasetGenerationContext(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    output_format=OUTPUT_FORMAT,
    split_datasets=SPLIT_DATASETS,
    target_classes=TARGET_CLASSES,
    output_storage=OUTPUT_STORAGE,
)

# Task main function
generate_dataset(context)
