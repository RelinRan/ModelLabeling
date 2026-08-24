from __future__ import annotations

import json
import os
import tempfile
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QPointF, QRectF

from src.models.annotation import Annotation, Keypoint, LabelPreset, ShapeType, label_color
from src.models.keypoint import COCO_PERSON_SKELETON
from src.services.format_capabilities import task_for_format, validate_annotations
from src.models.project import ProjectSettings
from src.utils.geometry import rect_from_points, rect_to_yolo, yolo_to_rect
from .coco_store import CocoAnnotationStore
from .format_adapters import adapter_for


@dataclass
class LoadResult:
    annotations: list[Annotation] = field(default_factory=list)
    error: str | None = None


@dataclass
class SaveResult:
    ok: bool
    error: str | None = None


@dataclass
class DatasetAnnotationIndex:
    """Read-only lookup tables used while scanning a dataset."""

    files_by_stem: dict[str, list[Path]] = field(default_factory=dict)
    coco_images: dict[str, dict] = field(default_factory=dict)
    coco_annotations: dict[object, list[dict]] = field(default_factory=dict)
    coco_categories: dict[object, str] = field(default_factory=dict)
    coco_keypoint_names: dict[object, list[str]] = field(default_factory=dict)


class AnnotationService:
    def build_index(self, annotation_dir: Path, annotation_format: str, cancel_callback=None) -> DatasetAnnotationIndex:
        index = DatasetAnnotationIndex()
        if not annotation_dir.exists():
            return index
        if annotation_format == "coco":
            path = self._coco_json_path(annotation_dir)
            if path is None:
                return index
            document = self._load_coco_document(annotation_dir)
            index.coco_images = {
                str(item.get("file_name", "")): item
                for item in document.get("images", [])
                if item.get("file_name")
            }
            index.coco_images.update({
                Path(str(item.get("file_name", ""))).name: item
                for item in document.get("images", [])
                if item.get("file_name")
            })
            for item in document.get("annotations", []):
                index.coco_annotations.setdefault(item.get("image_id"), []).append(item)
            index.coco_categories = {
                item.get("id"): item.get("name", f"class_{item.get('id')}")
                for item in document.get("categories", [])
            }
            index.coco_keypoint_names = {
                item.get("id"): [str(name) for name in item.get("keypoints", [])]
                for item in document.get("categories", [])
                if item.get("keypoints")
            }
            return index
        suffix = ".xml" if annotation_format == "voc" else ".txt"
        paths = annotation_dir.rglob(f"*{suffix}")
        if cancel_callback is None:
            paths = sorted(paths)
        for path in paths:
            if cancel_callback and cancel_callback():
                break
            index.files_by_stem.setdefault(path.stem, []).append(path)
        return index

    def load(
        self,
        image_path: Path,
        annotation_dir: Path,
        settings: ProjectSettings,
        index: DatasetAnnotationIndex | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> LoadResult:
        try:
            return adapter_for(settings.annotation_format, settings.dataset_task).load(
                self, image_path, annotation_dir, settings, index, image_size
            )
        except (OSError, ValueError, ET.ParseError, KeyError) as exc:
            return LoadResult(error=str(exc))

    def save(
        self,
        image_path: Path,
        annotations: list[Annotation],
        annotation_dir: Path,
        settings: ProjectSettings,
    ) -> SaveResult:
        try:
            adapter = adapter_for(settings.annotation_format, settings.dataset_task)
            validate_annotations(annotations, settings.annotation_format, settings.dataset_task)
            annotation_dir.mkdir(parents=True, exist_ok=True)
            return adapter.save(self, image_path, annotations, annotation_dir, settings)
        except (OSError, ValueError, KeyError) as exc:
            return SaveResult(False, str(exc))

    def load_internal_metadata(self, metadata_path: Path) -> dict[str, list[Annotation]]:
        if not metadata_path.exists():
            return {}
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            key: [Annotation.from_dict(item) for item in values]
            for key, values in payload.items()
        }

    def save_internal_metadata(
        self,
        metadata_path: Path,
        data: dict[str, list[Annotation]],
    ) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: [item.to_dict() for item in values] for key, values in data.items()}
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _preset_maps(self, presets: list[LabelPreset]) -> tuple[dict[str, int], dict[int, LabelPreset]]:
        by_name = {preset.name: preset.class_id for preset in presets}
        by_id = {preset.class_id: preset for preset in presets}
        return by_name, by_id

    def _load_by_adapter(self, image_path: Path, directory: Path, settings: ProjectSettings, adapter, index=None, image_size=None) -> LoadResult:
        if adapter.format_name == "voc":
            return LoadResult(self._load_voc(image_path, directory, settings.label_presets, index))
        if adapter.format_name == "coco":
            return LoadResult(self._load_coco(image_path, directory, settings.label_presets, index))
        task = adapter.task.value
        if task == "yolo_pose":
            return LoadResult(self._load_yolo_pose(image_path, directory, settings.label_presets, image_size, index))
        if task == "yolo_segmentation":
            return LoadResult(self._load_yolo_segmentation(image_path, directory, settings.label_presets, image_size, index))
        return LoadResult(self._load_yolo(image_path, directory, settings.label_presets, image_size, index))

    def _save_by_adapter(self, image_path: Path, annotations: list[Annotation], directory: Path, settings: ProjectSettings, adapter) -> SaveResult:
        if adapter.format_name == "voc":
            self._save_voc(image_path, annotations, directory, settings.label_presets)
        elif adapter.format_name == "coco":
            self._save_coco(image_path, annotations, directory, settings.label_presets)
        elif adapter.task.value == "yolo_pose":
            self._save_yolo_pose(image_path, annotations, directory, settings.label_presets, settings)
        elif adapter.task.value == "yolo_segmentation":
            self._save_yolo_segmentation(image_path, annotations, directory, settings.label_presets)
        else:
            self._save_yolo(image_path, annotations, directory, settings.label_presets)
        return SaveResult(True)

    @staticmethod
    def _coco_json_path(directory: Path) -> Path | None:
        if directory.is_file() and directory.suffix.lower() == ".json":
            return directory
        preferred = (directory / "annotations.json", directory / "instances.json")
        for path in preferred:
            if path.exists():
                return path
        candidates = sorted(directory.glob("*.json")) if directory.exists() else []
        return candidates[0] if candidates else None

    def _load_coco_document(self, directory: Path) -> dict:
        store = CocoAnnotationStore(directory)
        if store.is_initialized():
            return store.read_document()
        path = self._coco_json_path(directory)
        if path is None:
            return {"images": [], "annotations": [], "categories": []}
        document = json.loads(path.read_text(encoding="utf-8"))
        store.replace_document(document)
        return document

    def _load_coco(self, image_path: Path, directory: Path, presets: list[LabelPreset], index: DatasetAnnotationIndex | None = None) -> list[Annotation]:
        if index is None:
            document = self._load_coco_document(directory)
            image_record = next(
                (item for item in document.get("images", [])
                 if Path(str(item.get("file_name", ""))).name == image_path.name
                 or str(item.get("file_name", "")) == image_path.name),
                None,
            )
            annotation_items = document.get("annotations", [])
            categories = {item.get("id"): item.get("name", f"class_{item.get('id')}") for item in document.get("categories", [])}
            keypoint_names = {item.get("id"): list(item.get("keypoints", [])) for item in document.get("categories", [])}
        else:
            image_record = index.coco_images.get(image_path.name)
            annotation_items = index.coco_annotations.get(image_record.get("id"), []) if image_record else []
            categories = index.coco_categories
            keypoint_names = index.coco_keypoint_names
        if image_record is None:
            return []
        image_id = image_record.get("id")
        preset_by_name = {preset.name: preset for preset in presets}
        annotations: list[Annotation] = []
        for item in annotation_items:
            if item.get("image_id") != image_id:
                continue
            category_name = categories.get(item.get("category_id"), f"class_{item.get('category_id')}")
            preset = preset_by_name.get(category_name)
            color = label_color(category_name)
            bbox = item.get("bbox", [])
            bbox_points = []
            if len(bbox) >= 4:
                x, y, width, height = (float(value) for value in bbox[:4])
                bbox_points = [QPointF(x, y), QPointF(x + width, y + height)]
            raw_keypoints = item.get("keypoints", [])
            parsed_keypoints: list[Keypoint] = []
            if raw_keypoints and len(raw_keypoints) % 3 == 0:
                names = keypoint_names.get(item.get("category_id"), [])
                parsed_keypoints = [
                    Keypoint(names[index // 3] if index // 3 < len(names) else f"keypoint_{index // 3}", QPointF(float(raw_keypoints[index]), float(raw_keypoints[index + 1])), int(raw_keypoints[index + 2]))
                    for index in range(0, len(raw_keypoints), 3)
                ]
            if parsed_keypoints:
                annotations.append(Annotation(ShapeType.KEYPOINT, category_name, bbox_points, color=color, keypoints=parsed_keypoints, schema_name="COCO Keypoints"))
                continue
            segmentation = item.get("segmentation")
            if isinstance(segmentation, dict) and segmentation.get("counts") is not None:
                raise ValueError(
                    "COCO RLE mask annotations are not supported; convert masks to polygon segmentation first"
                )
            if isinstance(segmentation, list) and segmentation and isinstance(segmentation[0], list):
                parts = [
                    [QPointF(float(values[index]), float(values[index + 1])) for index in range(0, len(values) - 1, 2)]
                    for values in segmentation
                    if len(values) >= 6
                ]
                if parts:
                    annotations.append(Annotation(
                        ShapeType.POLYGON, category_name, parts[0], color=color,
                        polygon_parts=parts,
                    ))
                    continue
            if bbox_points:
                annotations.append(Annotation(ShapeType.RECTANGLE, category_name, bbox_points, color=color))
        return annotations

    def _save_coco(self, image_path: Path, annotations: list[Annotation], directory: Path, presets: list[LabelPreset]) -> None:
        with Image.open(image_path) as source_image:
            image_width, image_height = source_image.size
        categories = [
            {"name": preset.name, "supercategory": "object"}
            for preset in presets
        ]
        category_by_name = {item["name"]: item for item in categories}
        drafts: list[dict] = []
        for annotation in annotations:
            if annotation.label not in category_by_name:
                category = {"name": annotation.label, "supercategory": "object"}
                categories.append(category)
                category_by_name[annotation.label] = category
            points = annotation.points
            if annotation.shape_type == ShapeType.KEYPOINT and len(points) < 2:
                points = [keypoint.point for keypoint in annotation.keypoints if keypoint.visibility > 0]
            rect = rect_from_points(points)
            item = {
                "category_name": annotation.label,
                "bbox": [rect.left(), rect.top(), rect.width(), rect.height()],
                "area": rect.width() * rect.height(),
                "iscrowd": 0,
                "segmentation": [],
            }
            if annotation.shape_type == ShapeType.POLYGON:
                parts = annotation.polygon_parts or [annotation.points]
                item["segmentation"] = [
                    [coordinate for point in part for coordinate in (point.x(), point.y())]
                    for part in parts
                ]
                item["area"] = sum(self._polygon_area(part) for part in parts)
            if annotation.keypoints:
                item["keypoints"] = [
                    coordinate
                    for keypoint in annotation.keypoints
                    for coordinate in (
                        keypoint.point.x(), keypoint.point.y(), keypoint.visibility,
                    )
                ]
                item["num_keypoints"] = sum(
                    keypoint.visibility > 0 for keypoint in annotation.keypoints
                )
                category = category_by_name[annotation.label]
                category["keypoints"] = [keypoint.name for keypoint in annotation.keypoints]
                category["skeleton"] = (
                    [[start + 1, end + 1] for start, end in COCO_PERSON_SKELETON]
                    if len(annotation.keypoints) == 17 else []
                )
            drafts.append(item)
        CocoAnnotationStore(directory).upsert_image(
            image_path.name, image_width, image_height, categories, drafts
        )

    @staticmethod
    def export_coco(directory: Path) -> Path | None:
        store = CocoAnnotationStore(directory)
        if not store.is_initialized():
            return None
        service = AnnotationService()
        return store.export_json(service._coco_json_path(directory))


    def save_coco_batch(self, records: list[tuple[Path, list[Annotation]]], directory: Path, presets: list[LabelPreset]) -> None:
        """Write one complete COCO document for a batch conversion."""
        validate_annotations(
            [annotation for _image_path, annotations in records for annotation in annotations],
            "coco",
            "coco",
        )
        document = {"info": {"description": "ModelLabeling dataset"}, "licenses": [], "images": [], "annotations": [], "categories": []}
        category_by_name: dict[str, int] = {}
        for preset in presets:
            category_id = int(preset.class_id) + 1
            document["categories"].append({"id": category_id, "name": preset.name, "supercategory": "object"})
            category_by_name[preset.name] = category_id
        next_annotation_id = 1
        for image_id, (image_path, annotations) in enumerate(records, start=1):
            with Image.open(image_path) as source_image:
                width, height = source_image.size
            document["images"].append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
            for annotation in annotations:
                if annotation.label not in category_by_name:
                    raise ValueError(f"label is not in COCO categories: {annotation.label}")
                points = annotation.points
                if annotation.shape_type == ShapeType.KEYPOINT and len(points) < 2:
                    points = [keypoint.point for keypoint in annotation.keypoints if keypoint.visibility > 0]
                rect = rect_from_points(points)
                box_width, box_height = rect.width(), rect.height()
                item = {
                    "id": next_annotation_id, "image_id": image_id,
                    "category_id": category_by_name[annotation.label],
                    "bbox": [rect.left(), rect.top(), box_width, box_height],
                    "area": box_width * box_height, "iscrowd": 0, "segmentation": [],
                }
                if annotation.shape_type == ShapeType.POLYGON:
                    parts = annotation.polygon_parts or [annotation.points]
                    item["segmentation"] = [
                        [coordinate for point in part for coordinate in (point.x(), point.y())]
                        for part in parts
                    ]
                    item["area"] = sum(self._polygon_area(part) for part in parts)
                if annotation.keypoints:
                    item["keypoints"] = [coordinate for keypoint in annotation.keypoints for coordinate in (keypoint.point.x(), keypoint.point.y(), keypoint.visibility)]
                    item["num_keypoints"] = sum(keypoint.visibility > 0 for keypoint in annotation.keypoints)
                    category = next(category for category in document["categories"] if category["id"] == category_by_name[annotation.label])
                    if not category.get("keypoints"):
                        category["keypoints"] = [keypoint.name for keypoint in annotation.keypoints]
                        category["skeleton"] = [[start + 1, end + 1] for start, end in COCO_PERSON_SKELETON] if len(annotation.keypoints) == 17 else []
                document["annotations"].append(item)
                next_annotation_id += 1
        directory.mkdir(parents=True, exist_ok=True)
        store = CocoAnnotationStore(directory)
        store.replace_document(document)
        path = directory / "annotations.json"
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".annotations-", suffix=".tmp", delete=False) as temporary:
            temporary.write(json.dumps(document, ensure_ascii=False, indent=2))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)

    @staticmethod
    def _polygon_area(points: list[QPointF]) -> float:
        if len(points) < 3:
            return 0.0
        return abs(sum(points[index].x() * points[(index + 1) % len(points)].y() - points[(index + 1) % len(points)].x() * points[index].y() for index in range(len(points))) / 2.0)

    def _load_voc(self, image_path: Path, directory: Path, presets: list[LabelPreset], index: DatasetAnnotationIndex | None = None) -> list[Annotation]:
        xml_path = directory / f"{image_path.stem}.xml"
        if not xml_path.exists() and index is not None:
            matches = index.files_by_stem.get(image_path.stem, [])
            xml_path = matches[0] if matches else xml_path
        elif not xml_path.exists() and directory.exists():
            matches = list(directory.rglob(f"{image_path.stem}.xml"))
            xml_path = matches[0] if matches else xml_path
        if not xml_path.exists():
            return []
        root = ET.parse(xml_path).getroot()
        _, by_id = self._preset_maps(presets)
        by_name = {preset.name: preset for preset in presets}
        annotations: list[Annotation] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                raise ValueError(f"missing object name in {xml_path.name}")
            preset = by_name.get(name)
            color = label_color(name)
            box = obj.find("bndbox")
            if box is None:
                raise ValueError(f"missing bndbox for {name}")
            points = [
                QPointF(float(box.findtext("xmin", "0")), float(box.findtext("ymin", "0"))),
                QPointF(float(box.findtext("xmax", "0")), float(box.findtext("ymax", "0"))),
            ]
            annotations.append(Annotation(ShapeType.RECTANGLE, name, points, color=color))
        return annotations

    def _save_voc(
        self,
        image_path: Path,
        annotations: list[Annotation],
        directory: Path,
        presets: list[LabelPreset],
    ) -> None:
        with Image.open(image_path) as image:
            width, height = image.size
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = image_path.parent.name
        ET.SubElement(root, "filename").text = image_path.name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(len(image_path.suffix))
        for annotation in annotations:
            rect = rect_from_points(annotation.points)
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = annotation.label
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(round(rect.left()))
            ET.SubElement(box, "ymin").text = str(round(rect.top()))
            ET.SubElement(box, "xmax").text = str(round(rect.right()))
            ET.SubElement(box, "ymax").text = str(round(rect.bottom()))
        ET.ElementTree(root).write(directory / f"{image_path.stem}.xml", encoding="utf-8", xml_declaration=True)

    def _load_yolo(self, image_path: Path, directory: Path, presets: list[LabelPreset], image_size: tuple[int, int] | None = None, index: DatasetAnnotationIndex | None = None) -> list[Annotation]:
        txt_path = directory / f"{image_path.stem}.txt"
        if not txt_path.exists() and index is not None:
            matches = index.files_by_stem.get(image_path.stem, [])
            txt_path = matches[0] if matches else txt_path
        elif not txt_path.exists() and directory.exists():
            matches = list(directory.rglob(f"{image_path.stem}.txt"))
            txt_path = matches[0] if matches else txt_path
        if not txt_path.exists():
            return []
        if image_size is None:
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            width, height = image_size
        _, by_id = self._preset_maps(presets)
        annotations: list[Annotation] = []
        for line_number, raw in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) != 5:
                raise ValueError(f"{txt_path.name}:{line_number}: expected 5 values")
            class_id = int(parts[0])
            preset = by_id.get(class_id)
            if preset is None:
                raise ValueError(f"{txt_path.name}:{line_number}: unknown class id {class_id}")
            values = tuple(float(item) for item in parts[1:5])
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{txt_path.name}:{line_number}: normalized value out of range")
            rect = yolo_to_rect(values, width, height)
            annotations.append(
                Annotation(
                    ShapeType.RECTANGLE,
                    preset.name,
                    [rect.topLeft(), rect.bottomRight()],
                    color=label_color(preset.name),
                )
            )
        return annotations

    def _load_yolo_pose(self, image_path: Path, directory: Path, presets: list[LabelPreset], image_size: tuple[int, int] | None = None, index: DatasetAnnotationIndex | None = None) -> list[Annotation]:
        txt_path = directory / f"{image_path.stem}.txt"
        if not txt_path.exists() and index is not None:
            matches = index.files_by_stem.get(image_path.stem, [])
            txt_path = matches[0] if matches else txt_path
        if not txt_path.exists() and directory.exists():
            matches = list(directory.rglob(f"{image_path.stem}.txt"))
            txt_path = matches[0] if matches else txt_path
        if not txt_path.exists():
            return []
        if image_size is None:
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            width, height = image_size
        _, by_id = self._preset_maps(presets)
        expected_keypoints = self._yolo_pose_keypoint_count(directory)
        annotations: list[Annotation] = []
        for line_number, raw in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = raw.split()
            if not parts:
                continue
            if (len(parts) - 5) % 3 != 0 or len(parts) < 8:
                raise ValueError(f"{txt_path.name}:{line_number}: invalid YOLO Pose keypoint row")
            row_keypoints = (len(parts) - 5) // 3
            if expected_keypoints is not None and row_keypoints != expected_keypoints:
                raise ValueError(
                    f"{txt_path.name}:{line_number}: expected {expected_keypoints} keypoints, got {row_keypoints}"
                )
            class_id = int(parts[0])
            preset = by_id.get(class_id)
            if preset is None:
                raise ValueError(f"{txt_path.name}:{line_number}: unknown class id {class_id}")
            values = [float(item) for item in parts[1:]]
            bbox_values = values[:4]
            if any(value < 0.0 or value > 1.0 for value in bbox_values):
                raise ValueError(f"{txt_path.name}:{line_number}: normalized bbox out of range")
            rect = yolo_to_rect(tuple(bbox_values), width, height)
            keypoints = []
            for index in range(4, len(values), 3):
                x, y, visibility = values[index:index + 3]
                if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                    raise ValueError(f"{txt_path.name}:{line_number}: normalized keypoint out of range")
                keypoints.append(Keypoint(
                    f"keypoint_{len(keypoints)}",
                    QPointF(x * width, y * height),
                    int(round(visibility)),
                ))
            annotations.append(Annotation(
                ShapeType.KEYPOINT, preset.name,
                [rect.topLeft(), rect.bottomRight()],
                color=label_color(preset.name), keypoints=keypoints,
                schema_name="YOLO Pose",
            ))
        return annotations

    @staticmethod
    def _yolo_pose_keypoint_count(directory: Path) -> int | None:
        candidates = []
        current = Path(directory)
        parents = list(current.parents)[:3]
        for parent in (current, *parents):
            candidates.extend((parent / "data.yaml", parent / "data.yml"))
        for path in candidates:
            if not path.is_file():
                continue
            match = re.search(r"(?m)^\s*kpt_shape\s*:\s*\[\s*(\d+)\s*,\s*([23])\s*\]", path.read_text(encoding="utf-8", errors="ignore"))
            if match:
                if int(match.group(2)) != 3:
                    raise ValueError("YOLO Pose kpt_shape must use [count, 3]")
                return int(match.group(1))
        return None

    def _load_yolo_segmentation(self, image_path: Path, directory: Path, presets: list[LabelPreset], image_size: tuple[int, int] | None = None, index: DatasetAnnotationIndex | None = None) -> list[Annotation]:
        txt_path = directory / f"{image_path.stem}.txt"
        if not txt_path.exists() and index is not None:
            matches = index.files_by_stem.get(image_path.stem, [])
            txt_path = matches[0] if matches else txt_path
        if not txt_path.exists() and directory.exists():
            matches = list(directory.rglob(f"{image_path.stem}.txt"))
            txt_path = matches[0] if matches else txt_path
        if not txt_path.exists():
            return []
        if image_size is None:
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            width, height = image_size
        _, by_id = self._preset_maps(presets)
        annotations: list[Annotation] = []
        for line_number, raw in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = raw.split()
            if len(parts) < 7 or len(parts[1:]) % 2:
                raise ValueError(f"{txt_path.name}:{line_number}: invalid YOLO Segmentation row")
            class_id = int(parts[0]); preset = by_id.get(class_id)
            if preset is None:
                raise ValueError(f"{txt_path.name}:{line_number}: unknown class id {class_id}")
            values = [float(item) for item in parts[1:]]
            if any(value < 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{txt_path.name}:{line_number}: normalized polygon value out of range")
            points = [QPointF(values[index] * width, values[index + 1] * height) for index in range(0, len(values), 2)]
            annotations.append(Annotation(ShapeType.POLYGON, preset.name, points, color=label_color(preset.name)))
        return annotations

    def _save_yolo(
        self,
        image_path: Path,
        annotations: list[Annotation],
        directory: Path,
        presets: list[LabelPreset],
    ) -> None:
        with Image.open(image_path) as image:
            width, height = image.size
        by_name, _ = self._preset_maps(presets)
        lines: list[str] = []
        for annotation in annotations:
            if annotation.label not in by_name:
                raise ValueError(f"label is not in presets: {annotation.label}")
            rect = rect_from_points(annotation.points)
            center_x, center_y, box_width, box_height = rect_to_yolo(rect, width, height)
            lines.append(
                f"{by_name[annotation.label]} {center_x:.6f} {center_y:.6f} "
                f"{box_width:.6f} {box_height:.6f}"
            )
        (directory / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    def _save_yolo_pose(self, image_path: Path, annotations: list[Annotation], directory: Path, presets: list[LabelPreset], settings: ProjectSettings) -> None:
        with Image.open(image_path) as image:
            width, height = image.size
        by_name, _ = self._preset_maps(presets)
        lines: list[str] = []
        for annotation in annotations:
            if annotation.label not in by_name:
                raise ValueError(f"label is not in presets: {annotation.label}")
            if annotation.shape_type != ShapeType.KEYPOINT:
                raise ValueError("YOLO Pose requires keypoint annotations")
            if len(annotation.points) < 2:
                raise ValueError("YOLO Pose requires an outer bounding box")
            rect = rect_to_yolo(QRectF(annotation.points[0], annotation.points[-1]).normalized(), width, height)
            values = [str(by_name[annotation.label]), *(f"{value:.6f}" for value in rect)]
            for keypoint in annotation.keypoints:
                values.extend((
                    f"{keypoint.point.x() / width:.6f}",
                    f"{keypoint.point.y() / height:.6f}",
                    str(keypoint.visibility),
                ))
            lines.append(" ".join(values))
        (directory / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _save_yolo_segmentation(self, image_path: Path, annotations: list[Annotation], directory: Path, presets: list[LabelPreset]) -> None:
        with Image.open(image_path) as image:
            width, height = image.size
        by_name, _ = self._preset_maps(presets)
        lines: list[str] = []
        for annotation in annotations:
            if annotation.label not in by_name:
                raise ValueError(f"label is not in presets: {annotation.label}")
            if annotation.shape_type in {ShapeType.RECTANGLE, ShapeType.SQUARE}:
                rect = rect_from_points(annotation.points)
                points = [rect.topLeft(), QPointF(rect.right(), rect.top()), rect.bottomRight(), QPointF(rect.left(), rect.bottom())]
            elif annotation.shape_type == ShapeType.POLYGON:
                points = annotation.points
            else:
                raise ValueError("YOLO Segmentation does not support keypoints")
            if len(points) < 3:
                raise ValueError("YOLO Segmentation requires at least three polygon points")
            values = [str(by_name[annotation.label])]
            values.extend(f"{coordinate:.6f}" for point in points for coordinate in (point.x() / width, point.y() / height))
            lines.append(" ".join(values))
        (directory / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
