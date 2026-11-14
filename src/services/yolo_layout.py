"""Normalize a plain image folder into the standard YOLO dataset layout.

Ultralytics derives a label path by replacing the last ``/images/`` segment of
the image path with ``/labels/`` (``img2label_paths``). When the images live
directly in the selected folder there is no such segment, so the trainer looks
for ``<image stem>.txt`` *beside* the image and silently ignores a sibling
``labels/`` directory: every image is treated as a background image.

This module moves the images into ``images/`` once, preserving their relative
structure, and records the class table in ``data.yaml``. Moving (not copying)
keeps the user's folder the dataset root and does not duplicate large image
sets. The migration is planned in full before anything is moved and rolled
back if a move fails, so a failure never leaves a half-migrated folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.models.annotation import LabelPreset

from .dataset_detector import DatasetDetector


@dataclass(frozen=True)
class LayoutMigration:
    root: Path
    image_dir: Path
    annotation_dir: Path
    moved_images: int = 0
    moved_labels: int = 0


def needs_standard_layout(
    dataset_root: Path | None,
    image_dir: Path | None,
    annotation_dir: Path | None,
    annotation_format: str | None,
) -> bool:
    """True when the open dataset is a plain folder that YOLO cannot train on.

    Only the YOLO family is affected: VOC and COCO resolve labels by their own
    layout rules. A dataset whose images already live in ``images/`` (the
    ``root/images`` + ``root/labels`` case) is standard and left alone.
    """
    if str(annotation_format or "").strip().lower() != "yolo":
        return False
    if dataset_root is None or image_dir is None:
        return False
    root = Path(dataset_root).resolve()
    images = Path(image_dir).resolve()
    if images != root:
        return False
    return not (root / "images").is_dir()


def plan_migration(root: Path) -> list[tuple[Path, Path]]:
    """Return every ``(source, destination)`` move for the standard layout.

    The plan is validated before the caller moves anything: a destination that
    already exists is a conflict, not something to overwrite. Labels stored
    beside their image (a common YOLO layout) move into ``labels/`` too.
    """
    root = Path(root).resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.exists() and not images_dir.is_dir():
        raise ValueError(f"cannot create images/ because a file with that name exists: {images_dir}")
    if labels_dir.exists() and not labels_dir.is_dir():
        raise ValueError(f"cannot create labels/ because a file with that name exists: {labels_dir}")

    moves: list[tuple[Path, Path]] = []
    targets: set[str] = set()
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix().casefold(),
    ):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {"images", "labels"}:
            continue
        if path.suffix.lower() not in DatasetDetector.IMAGE_EXTENSIONS:
            continue
        candidates = [(path, images_dir / relative)]
        beside = path.with_suffix(".txt")
        if beside.is_file():
            candidates.append((beside, labels_dir / relative.with_suffix(".txt")))
        for source, target in candidates:
            key = str(target).casefold()
            if key in targets:
                raise ValueError(f"two files would be moved to the same destination: {target}")
            if target.exists():
                raise ValueError(
                    f"cannot migrate {source.name}: destination already exists: {target}"
                )
            targets.add(key)
            moves.append((source, target))
    return moves


def migrate_plain_folder(
    root: Path,
    *,
    presets: list[LabelPreset] | None = None,
    task_name: str | None = None,
) -> LayoutMigration:
    """Move images into ``images/`` and write the dataset's class table.

    The caller has already confirmed with the user; this function performs the
    moves atomically per file and rolls back the moves it completed when one
    fails.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")
    moves = plan_migration(root)
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    completed: list[tuple[Path, Path]] = []
    try:
        for source, target in moves:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
            completed.append((source, target))
    except OSError:
        for source, target in reversed(completed):
            try:
                if target.exists() and not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, source)
            except OSError:
                pass
        raise

    moved_images = sum(1 for _source, destination in moves if images_dir in destination.parents)
    moved_labels = len(moves) - moved_images
    _write_class_table(root, presets, task_name)
    return LayoutMigration(
        root, images_dir, labels_dir, moved_images=moved_images, moved_labels=moved_labels,
    )


def _existing_names(path: Path) -> list[str]:
    """Class names declared by an existing data.yaml, or an empty list."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    value = loaded.get("names") if isinstance(loaded, dict) else None
    if isinstance(value, list):
        return [str(name) for name in value if str(name).strip()]
    if isinstance(value, dict):
        try:
            ordered = {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError):
            return [str(name) for name in value.values() if str(name).strip()]
        return [ordered[index] for index in sorted(ordered) if ordered[index].strip()]
    return []


def _write_class_table(root: Path, presets: list[LabelPreset] | None, task_name: str | None) -> None:
    """Give the folder a self-describing ``classes.txt`` and ``data.yaml``.

    Written during the migration rather than only by the annotation save, so
    the folder is a complete dataset even when the pending annotation fails
    validation afterwards.
    """
    classes_path = root / "classes.txt"
    names: list[str] = []
    # A class table already on disk is the dataset's own truth: label files
    # are keyed by class id, so replacing it with the app's presets would
    # rename every existing object. Some datasets keep the table inside
    # labels/; accept it there too.
    for candidate in (classes_path, root / "labels" / "classes.txt"):
        if candidate.is_file():
            names = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if names:
                break
    existing = root / "data.yaml"
    if not names and existing.is_file():
        names = _existing_names(existing)
    if not names:
        names = [
            preset.name
            for preset in sorted(presets or [], key=lambda preset: preset.class_id)
            if preset.name
        ]
        if not names:
            # With no class table anywhere there is nothing to describe yet;
            # the annotation save writes one when the first label is created.
            return
        classes_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    document = {
        "path": root.as_posix(),
        "train": "images",
        "val": "images",
        "task": {
            "yolo_segmentation": "segment",
            "yolo_pose": "pose",
            "yolo_obb": "obb",
        }.get(str(task_name or ""), "detect"),
        "names": {index: name for index, name in enumerate(names)},
    }
    if existing.is_file():
        try:
            loaded = yaml.safe_load(existing.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            loaded = None
        if isinstance(loaded, dict):
            for key in ("kpt_shape", "kpt_names", "flip_idx"):
                if key in loaded:
                    document[key] = loaded[key]
        # A YAML written by an earlier non-standard layout can point elsewhere.
        # The folder is the dataset root now, so replace those fields.
    existing.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False, default_flow_style=None),
        encoding="utf-8",
    )