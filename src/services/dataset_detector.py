from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectedDataset:
    format_name: str
    root: Path
    image_dir: Path
    annotation_dir: Path
    task_name: str | None = None


class DatasetDetector:
    """Detect supported dataset layouts without depending on any widget."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    @classmethod
    def detect(cls, root: Path) -> DetectedDataset:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"dataset directory does not exist: {root}")

        result = cls._detect_voc(root) or cls._detect_coco(root) or cls._detect_yolo(root)
        if result is None:
            raise ValueError(f"unsupported dataset format: {root}")
        return result

    @classmethod
    def _detect_voc(cls, root: Path) -> DetectedDataset | None:
        candidates = (
            (root / "JPEGImages", root / "Annotations"),
            (root / "images", root / "Annotations"),
            (root, root / "Annotations"),
        )
        for image_dir, annotation_dir in candidates:
            if image_dir.is_dir() and annotation_dir.is_dir() and any(annotation_dir.glob("*.xml")):
                return DetectedDataset("voc", root, image_dir, annotation_dir)
        return None

    @classmethod
    def _detect_coco(cls, root: Path) -> DetectedDataset | None:
        candidates = (
            (root / "images", root / "annotations"),
            (root, root / "annotations"),
            (root / "images", root / "Annotations"),
            (root / "images", root),
            (root, root),
        )
        for image_dir, annotation_dir in candidates:
            json_files = sorted(annotation_dir.glob("*.json")) if annotation_dir.is_dir() else []
            for json_path in json_files:
                try:
                    document = json.loads(json_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if {"images", "annotations", "categories"}.issubset(document):
                    return DetectedDataset("coco", root, image_dir, annotation_dir)
        return None

    @classmethod
    def _detect_yolo(cls, root: Path) -> DetectedDataset | None:
        candidates = (
            (root / "images", root / "labels"),
            (root / "train" / "images", root / "train" / "labels"),
            (root / "images" / "train", root / "labels" / "train"),
            (root, root / "labels"),
        )
        has_yolo_marker = (root / "data.yaml").is_file() or (root / "data.yml").is_file() or (root / "classes.txt").is_file()
        for image_dir, annotation_dir in candidates:
            if not image_dir.is_dir() or not annotation_dir.is_dir():
                continue
            label_files = [
                path for path in annotation_dir.rglob("*.txt")
                if path.name.lower() not in {"classes.txt", "train.txt", "val.txt", "test.txt"}
            ]
            if label_files or has_yolo_marker:
                return DetectedDataset("yolo", root, cls._shared_yolo_image_dir(root, image_dir), cls._shared_yolo_label_dir(root, annotation_dir), cls._yolo_task(root, label_files))
        return None

    @staticmethod
    def _yolo_task(root: Path, label_files: list[Path]) -> str:
        for yaml_path in (root / "data.yaml", root / "data.yml"):
            if yaml_path.is_file():
                content = yaml_path.read_text(encoding="utf-8", errors="ignore").lower()
                if "kpt_shape" in content or "task: pose" in content:
                    return "yolo_pose"
        for path in label_files[:20]:
            try:
                lengths = [len(line.split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except OSError:
                continue
            if any(length >= 8 and (length - 5) % 3 == 0 for length in lengths):
                return "yolo_pose"
            if any(length >= 7 and length % 2 == 1 for length in lengths):
                return "yolo_segmentation"
        return "yolo_detection"

    @staticmethod
    def _shared_yolo_image_dir(root: Path, detected: Path) -> Path:
        if (root / "images").is_dir():
            return root / "images"
        return detected

    @staticmethod
    def _shared_yolo_label_dir(root: Path, detected: Path) -> Path:
        if (root / "labels").is_dir():
            return root / "labels"
        return detected
