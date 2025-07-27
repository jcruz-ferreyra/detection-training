from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class YoloTrainingContext:
    data_dir: Path
    dataset_folder: str
    data_yaml: str

    models_dir: Path
    checkpoint: str
    project_name: str

    training_params: Dict[str, Any]

    environment: str

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset_folder

    @property
    def checkpoint_dir(self) -> Path:
        return self.models_dir / self.checkpoint

    @property
    def project_dir(self) -> Path:
        return self.models_dir / self.project_name

    @property
    def data_yaml_path(self) -> Path:
        return self.dataset_dir / "yolo" / self.data_yaml
