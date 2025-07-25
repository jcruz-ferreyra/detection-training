import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the HOME and DATA paths from the .env file and convert to a Path object
HOME_DIR = Path(os.getenv("HOME_DIR")).resolve()
LOCAL_DIR = Path(os.getenv("LOCAL_DIR")).resolve()

DATA_FOLDER = Path(os.getenv("DATA_FOLDER"))
MODELS_FOLDER = Path(os.getenv("MODELS_FOLDER"))

DATA_DIR = LOCAL_DIR / DATA_FOLDER
MODELS_DIR = LOCAL_DIR / MODELS_FOLDER

for x in ["HOME_DIR", "LOCAL_DIR", "DATA_DIR", "MODELS_DIR"]:
    path = locals()[x]
    if not path.exists():
        raise ValueError(f"{x} path '{path}' from .env does not exist.")

# Prepare Drive paths, and check correct access to them
DRIVE_DIR = os.getenv("DRIVE_DIR")

if DRIVE_DIR is not None:
    DRIVE_DIR = Path(DRIVE_DIR).resolve()
    DRIVE_DATA_DIR = DRIVE_DIR / DATA_FOLDER
    DRIVE_MODELS_DIR = DRIVE_DIR / MODELS_FOLDER

    if not DRIVE_DATA_DIR.exists():
        DRIVE_DATA_DIR = None

    if not DRIVE_MODELS_DIR.exists():
        DRIVE_MODELS_DIR = None

else:
    DRIVE_DATA_DIR = None
    DRIVE_MODELS_DIR = None
