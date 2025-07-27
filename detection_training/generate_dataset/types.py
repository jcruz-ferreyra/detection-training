from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class DatasetGenerationContext:
    input_data_dir: Path
    input_folder: str
    output_data_dir: Path
    output_folder: str

    output_format: str

    split_datasets: Dict[str, List[str]]
    target_classes: List[str]

    output_storage: str = "local"

    def __post_init__(self):
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        supported_formats = ["yolo", "coco"]
        if self.output_format.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported output format: '{self.output_format}'. "
                f"Supported formats: {supported_formats}. "
            )

    @property
    def input_dir(self) -> Path:
        return self.input_data_dir / self.input_folder

    @property
    def output_dir(self) -> Path:
        return self.output_data_dir / self.output_folder / self.output_format
