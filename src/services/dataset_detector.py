from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .yolo_metadata import load_yolo_metadata


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
    def detect(cls, root: Path, allow_plain_images: bool = True) -> DetectedDataset:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"dataset directory does not exist: {root}")

        result = cls._detect_voc(root) or cls._detect_coco(root) or cls._detect_yolo(root)
        if result is None and allow_plain_images:
            result = cls._detect_image_only(root)
        if result is None:
            # Permit a test/project container directory that contains one
            # actual dataset directory, e.g. test/voc-action-test. Do not
            # guess when multiple child datasets exist.
            candidates: list[DetectedDataset] = []
            for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
                if not child.is_dir():
                    continue
                try:
                    candidates.append(cls.detect(child))
                except ValueError:
                    continue
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError(f"multiple datasets found under: {root}")
            raise ValueError(f"unsupported dataset format: {root}")
        return result

    @classmethod
    def _detect_image_only(cls, root: Path) -> DetectedDataset | None:
        """A folder of images with no annotation layout yet defaults to YOLO.

        The default dataset format is YOLO and saving creates the labels
        directory beside the images, so a completely unannotated folder can
        be opened and annotated right away.
        """
        image_dir = root / "images" if (root / "images").is_dir() else root
        if not any(
            path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS
            for path in image_dir.iterdir()
        ):
            return None
        return DetectedDataset("yolo", root, image_dir, root / "labels", "yolo_detection")

    @classmethod
    def _detect_voc(cls, root: Path) -> DetectedDataset | None:
        candidates = (
            (root / "JPEGImages", root / "Annotations"),
            (root / "images", root / "Annotations"),
            (root, root / "Annotations"),
        )
        for image_dir, annotation_dir in candidates:
            if not image_dir.is_dir():
                continue
            # A dataset can legitimately have no annotations at all yet:
            # JPEGImages or an existing Annotations directory alone are
            # enough to identify the VOC layout; saving recreates the folder.
            # Compare real on-disk names so a COCO "annotations" folder is
            # not matched on case-insensitive Windows paths.
            voc_marker = cls._dir_exists_named(root, "JPEGImages") or cls._dir_exists_named(root, "Annotations")
            if voc_marker or (annotation_dir.is_dir() and any(annotation_dir.glob("*.xml"))):
                return DetectedDataset("voc", root, image_dir, annotation_dir)
        return None

    @staticmethod
    def _dir_exists_named(root: Path, name: str) -> bool:
        try:
            return any(child.is_dir() and child.name == name for child in root.iterdir())
        except OSError:
            return False

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
        for image_dir, annotation_dir in candidates:
            # The images/+labels/ pair is itself the YOLO layout: a fresh
            # dataset may not have a single annotation or marker file yet.
            if not image_dir.is_dir() or not annotation_dir.is_dir():
                continue
            label_files = [
                path for path in annotation_dir.rglob("*.txt")
                if path.name.lower() not in {"classes.txt", "train.txt", "val.txt", "test.txt"}
            ]
            return DetectedDataset("yolo", root, cls._shared_yolo_image_dir(root, image_dir), cls._shared_yolo_label_dir(root, annotation_dir), cls._yolo_task(root, label_files))
        return None

    @staticmethod
    def _yolo_task(root: Path, label_files: list[Path]) -> str:
        metadata = load_yolo_metadata(root)
        task = str(metadata.get("task", "")).strip().lower()
        if metadata.get("kpt_shape") is not None or task == "pose":
            return "yolo_pose"
        if task in {"segment", "segmentation"}:
            return "yolo_segmentation"
        if task in {"obb", "obb detection", "oriented", "rotated", "rotate"}:
            return "yolo_obb"
        lengths: list[int] = []
        for path in label_files[:20]:
            try:
                lengths.extend(
                    len(line.split()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                )
            except OSError:
                continue
        if lengths and all(length == 9 for length in lengths):
            # class + 8 corner coordinates. A quadrilateral-only segmentation
            # dataset is indistinguishable by shape alone; users can switch
            # the task in Application Settings.
            return "yolo_obb"
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
