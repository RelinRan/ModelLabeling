from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset_detector import DetectedDataset


@dataclass
class DatasetSession:
    """Canonical dataset context shared by UI services.

    Keeping these paths together prevents a format switch from leaving the
    image directory, annotation directory, and task type out of sync.
    """

    root: Path
    image_dir: Path
    annotation_dir: Path
    format_name: str
    task_name: str
    current_path: Path | None = None
    total_images: int = 0

    @classmethod
    def from_detected(cls, detected: DetectedDataset) -> "DatasetSession":
        return cls(
            root=Path(detected.root).resolve(),
            image_dir=Path(detected.image_dir).resolve(),
            annotation_dir=Path(detected.annotation_dir).resolve(),
            format_name=detected.format_name,
            task_name=detected.task_name or detected.format_name,
        )

