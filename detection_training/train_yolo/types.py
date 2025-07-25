from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class YoloTrainingContext:
    dataset_dir: Path
    data_yaml: str

    checkpoint_dir: Path

    project_dir: Path
    project_name: str

    training_params: Dict[str, Any]

    environment: str

    @property
    def data_yaml_path(self) -> Path:
        return self.dataset_dir / "yolo" / self.data_yaml