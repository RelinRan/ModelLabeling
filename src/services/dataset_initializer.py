from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .dataset_detector import DatasetDetector

IMAGE_EXTENSIONS = DatasetDetector.IMAGE_EXTENSIONS

_INVALID_NAME = re.compile(r'[\\/:*?"<>|]')


@dataclass(frozen=True)
class InitializedDataset:
    root: Path
    image_dir: Path
    annotation_dir: Path
    task_name: str
    created: tuple[Path, ...]
    copied_images: int = 0
    imported_annotations: int = 0


class DatasetInitializer:
    """Turn a plain folder of images into a standard dataset layout.

    The images are never moved or copied: the marker structure (labels/,
    classes.txt, data.yaml, Annotations/, annotations.json) is created beside
    them so DatasetDetector recognizes the folder immediately afterwards.
    """

    TASKS = ("yolo_detection", "yolo_segmentation", "yolo_pose", "yolo_obb", "voc", "coco")

    @classmethod
    def count_images(cls, root: Path) -> int:
        root = Path(root)
        sources = [root, root / "images", root / "JPEGImages"]
        return sum(
            1
            for source in sources
            if source.is_dir()
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @classmethod
    def image_directory(cls, root: Path) -> Path:
        """Images stay where they are; report the folder the detector will use."""
        root = Path(root)
        for candidate in (root / "images", root / "JPEGImages"):
            if candidate.is_dir() and cls.count_images(candidate):
                return candidate
        return root

    @classmethod
    def initialize(cls, root: Path, task_name: str, class_names: list[str] | None = None, keypoint_count: int = 17, allow_empty: bool = False) -> InitializedDataset:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"folder does not exist: {root}")
        if task_name not in cls.TASKS:
            raise ValueError(f"unsupported task: {task_name}")
        created: list[Path] = []
        image_dir = cls.image_directory(root)
        if not allow_empty and cls.count_images(root) == 0:
            raise ValueError(f"no images found in: {root}")

        if task_name == "voc":
            annotation_dir = root / "Annotations"
            annotation_dir.mkdir(exist_ok=True)
            if not any(annotation_dir.iterdir()) or not annotation_dir.exists():
                created.append(annotation_dir)
            return InitializedDataset(root, image_dir, annotation_dir, task_name, tuple(created))

        if task_name == "coco":
            annotation_dir = root / "annotations"
            annotation_dir.mkdir(exist_ok=True)
            document = annotation_dir / "annotations.json"
            if not document.exists():
                document.write_text(
                    json.dumps({"images": [], "annotations": [], "categories": []}, indent=2),
                    encoding="utf-8",
                )
                created.append(document)
            return InitializedDataset(root, image_dir, annotation_dir, task_name, tuple(created))

        # YOLO family: labels/ plus the task-describing data.yaml.
        annotation_dir = root / "labels"
        annotation_dir.mkdir(exist_ok=True)
        yaml_path = root / "data.yaml"
        if not yaml_path.exists():
            lines = [f"task: {cls._yaml_task(task_name)}"]
            if task_name == "yolo_pose":
                lines.append(f"kpt_shape: [{max(1, int(keypoint_count))}, 3]")
            if class_names:
                lines.append("names: [" + ", ".join(str(name).strip() for name in class_names if str(name).strip()) + "]")
            yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            created.append(yaml_path)
        classes_path = root / "classes.txt"
        if class_names and not classes_path.exists():
            classes_path.write_text(
                "\n".join(str(name).strip() for name in class_names if str(name).strip()) + "\n",
                encoding="utf-8",
            )
            created.append(classes_path)
        if not annotation_dir.exists():
            created.append(annotation_dir)
        return InitializedDataset(root, image_dir, annotation_dir, task_name, tuple(created))

    @staticmethod
    def _yaml_task(task_name: str) -> str:
        return {
            "yolo_detection": "detect",
            "yolo_segmentation": "segment",
            "yolo_pose": "pose",
            "yolo_obb": "obb",
        }.get(task_name, "detect")

    # ---- Workspace creation (the programming-style project flow) ---------

    @staticmethod
    def validate_name(name: str) -> str | None:
        """Return an error message for an invalid dataset name, else None."""
        name = str(name).strip()
        if not name:
            return "数据集名称不能为空"
        if _INVALID_NAME.search(name):
            return '名称不能包含 \\ / : * ? " < > | 等字符'
        if name in {".", ".."}:
            return "名称无效"
        return None

    @classmethod
    def create_in_workspace(
        cls,
        workspace: Path,
        name: str,
        task_name: str,
        class_names: list[str] | None = None,
        keypoint_count: int = 17,
        source_dir: Path | None = None,
    ) -> InitializedDataset:
        """Create a fresh dataset under workspace/<name> like an IDE project.

        The source image folder is only material: its images are copied into
        the new dataset's images/ directory. If the source is itself a
        recognizable dataset, its annotation files are imported as well.
        """
        error = cls.validate_name(name)
        if error:
            raise ValueError(error)
        workspace = Path(workspace)
        target = workspace / str(name).strip()
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"工作空间中已存在同名数据集: {target}")
        if task_name not in cls.TASKS:
            raise ValueError(f"unsupported task: {task_name}")
        workspace.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        (target / "images").mkdir(exist_ok=True)

        copied = 0
        if source_dir is not None:
            copied = cls._copy_images(Path(source_dir), target / "images")

        info = cls.initialize(target, task_name, class_names, keypoint_count, allow_empty=True)
        imported = 0
        if source_dir is not None:
            imported = cls._import_annotations(Path(source_dir), info)
        return InitializedDataset(info.root, info.image_dir, info.annotation_dir, info.task_name, info.created, copied, imported)

    @staticmethod
    def _copy_images(source: Path, destination: Path) -> int:
        """Copy every supported image below the source into destination/.

        Nested folders are flattened; name collisions get the parent folder
        prefixed so no image is silently overwritten.
        """
        images = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in images:
            name = path.name
            candidate = destination / name
            if candidate.exists():
                prefix = path.parent.name.strip() or "img"
                candidate = destination / f"{prefix}_{name}"
                if candidate.exists():
                    continue
            shutil.copy2(path, candidate)
        return len(images)

    @staticmethod
    def _import_annotations(source: Path, info: InitializedDataset) -> int:
        """Import annotation files when the source folder is a dataset.

        YOLO/VOC annotation files are matched per image stem; a COCO source
        contributes its JSON document directly.
        """
        try:
            detected = DatasetDetector.detect(source, allow_plain_images=False)
        except ValueError:
            return 0
        imported = 0
        if detected.format_name == "coco":
            for json_path in detected.annotation_dir.glob("*.json"):
                shutil.copy2(json_path, info.annotation_dir / json_path.name)
                imported += 1
            return imported
        suffix = ".xml" if detected.format_name == "voc" else ".txt"
        for image in sorted(info.image_dir.glob("*")):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            matches = [p for p in detected.annotation_dir.rglob(f"{image.stem}{suffix}")]
            if matches:
                shutil.copy2(matches[0], info.annotation_dir / f"{image.stem}{suffix}")
                imported += 1
        return imported

    @staticmethod
    def structure_summary(task_name: str) -> str:
        return {
            "yolo_detection": "labels/ + data.yaml + classes.txt（YOLO 检测）",
            "yolo_segmentation": "labels/ + data.yaml（task: segment）",
            "yolo_pose": "labels/ + data.yaml（task: pose + kpt_shape）",
            "yolo_obb": "labels/ + data.yaml（task: obb）",
            "voc": "Annotations/（Pascal VOC）",
            "coco": "annotations/annotations.json（COCO）",
        }.get(task_name, "")

    @classmethod
    def verify(cls, root: Path) -> tuple[str, str] | None:
        """Confirm the folder is now detectable; returns (format, task)."""
        try:
            detected = DatasetDetector.detect(Path(root))
        except ValueError:
            return None
        return detected.format_name, detected.task_name or ""
