from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO


@dataclass
class ModelEvaluationContext:
    data_dir: Path
    models_dir: Path

    experiment: str
    model_family: str
    split: str

    environment: str

    dataset_folder: Optional[str] = None
    data_yaml: Optional[str] = None

    model_name: Optional[str] = None
    model: Union[YOLO, YOLO] = None

    artifacts: Optional[str] = None

    run_id: Optional[str] = None

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def data_yaml_path(self) -> Path:
        return self.dataset_dir / self.model_family / self.data_yaml

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_name

    @property
    def artifacts_dir(self) -> Path:
        return self.models_dir / self.artifacts
