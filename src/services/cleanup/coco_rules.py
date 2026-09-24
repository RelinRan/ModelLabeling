"""COCO validation.

For COCO the JSON *is* the dataset, so referential integrity is not a nicety:
an ``image_id`` that points at nothing, a ``category_id`` that was never
declared, or an ``area`` that disagrees with the box will make ``pycocotools``
mis-evaluate without ever raising.

The old cleanup only ever pruned JSON records for images *it* had just
deleted, so a record whose file disappeared outside the app stayed forever.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.models.annotation import LabelPreset
from src.services.annotation_service import AnnotationService
from src.services.coco_store import CocoAnnotationStore

from .common_rules import ImageProbe, box_iou
from .report import Issue, issue


def validate_coco(
    detected,
    *,
    presets: list[LabelPreset] | None = None,
    probes: list[ImageProbe] | None = None,
    min_box_size: int = 20,
    near_duplicate_iou: float = 0.95,
) -> list[Issue]:
    annotation_dir = detected.annotation_dir
    json_path = _json_path(annotation_dir)
    if json_path is None:
        return [issue("coco.json.missing", annotation_dir)]
    try:
        document = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [issue("coco.json.unreadable", json_path, error=str(exc))]
    if not isinstance(document, dict):
        return [issue("coco.json.unreadable", json_path, error="root is not an object")]

    issues: list[Issue] = []
    images = [item for item in document.get("images", []) if isinstance(item, dict)]
    annotations = [item for item in document.get("annotations", []) if isinstance(item, dict)]
    categories = [item for item in document.get("categories", []) if isinstance(item, dict)]

    issues.extend(_duplicate_ids(images, "coco.image.duplicate_id", json_path, "id"))
    issues.extend(_duplicate_ids(annotations, "coco.annotation.duplicate_id", json_path, "id"))

    image_by_id = _by_id(images)
    category_ids = {_as_int(item.get("id")) for item in categories}
    annotation_counts = _annotation_counts(annotations)

    issues.extend(_dangling_and_categories(
        annotations, image_by_id, category_ids, json_path,
    ))
    issues.extend(_boxes(annotations, image_by_id, json_path, min_box_size, near_duplicate_iou))
    issues.extend(_image_records(images, annotation_counts, detected, json_path, probes))
    issues.extend(_unused_categories(categories, _category_counts(annotations), json_path))
    issues.extend(_divergence(annotation_dir, document, json_path))
    return issues


def coco_document_path(annotation_dir: Path) -> Path | None:
    return _json_path(annotation_dir)


def _json_path(annotation_dir: Path) -> Path | None:
    """The COCO JSON this dataset is read from.

    ``AnnotationService`` names it; a hand-made dataset may use any name, so
    fall back to the first JSON that actually looks like COCO.
    """
    candidate = AnnotationService._coco_json_path(annotation_dir)
    if candidate is not None and candidate.is_file():
        return candidate
    if not annotation_dir.is_dir():
        return None
    for path in sorted(annotation_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict) and {"images", "annotations", "categories"}.issubset(document):
            return path
    return None


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _by_id(rows: list[dict]) -> dict[int, dict]:
    result: dict[int, dict] = {}
    for row in rows:
        identifier = _as_int(row.get("id"))
        if identifier is not None:
            result.setdefault(identifier, row)
    return result


def _duplicate_ids(rows: list[dict], rule: str, target: Path, key: str) -> list[Issue]:
    counts: dict[int, int] = {}
    for row in rows:
        identifier = _as_int(row.get(key))
        if identifier is None:
            continue
        counts[identifier] = counts.get(identifier, 0) + 1
    return [
        issue(rule, target, id=identifier, count=count)
        for identifier, count in sorted(counts.items())
        if count > 1
    ]


def _annotation_counts(annotations: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in annotations:
        image_id = _as_int(row.get("image_id"))
        if image_id is None:
            continue
        counts[image_id] = counts.get(image_id, 0) + 1
    return counts


def _category_counts(annotations: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in annotations:
        category_id = _as_int(row.get("category_id"))
        if category_id is None:
            continue
        counts[category_id] = counts.get(category_id, 0) + 1
    return counts


def _dangling_and_categories(annotations, image_by_id, category_ids, target) -> list[Issue]:
    issues: list[Issue] = []
    for row in annotations:
        identifier = _as_int(row.get("id"))
        image_id = _as_int(row.get("image_id"))
        if image_id is None or image_id not in image_by_id:
            issues.append(issue(
                "coco.annotation.dangling_image", target, id=identifier, image_id=image_id,
            ))
        category_id = _as_int(row.get("category_id"))
        if category_ids and category_id not in category_ids:
            issues.append(issue(
                "coco.annotation.unknown_category", target,
                id=identifier, category_id=category_id,
            ))
    return issues


def _boxes(annotations, image_by_id, target, min_box_size, near_duplicate_iou) -> list[Issue]:
    issues: list[Issue] = []
    grouped: dict[int, list[tuple[int, int, tuple[float, float, float, float]]]] = {}
    for row in annotations:
        identifier = _as_int(row.get("id"))
        bbox = row.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            issues.append(issue("coco.annotation.bad_bbox", target, id=identifier, bbox=bbox))
            continue
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError):
            issues.append(issue("coco.annotation.bad_bbox", target, id=identifier, bbox=bbox))
            continue
        if width <= 0 or height <= 0:
            issues.append(issue("coco.annotation.bad_bbox", target, id=identifier, bbox=bbox))
            continue

        # area must be w*h unless the record carries its own mask area.
        has_segmentation = bool(row.get("segmentation"))
        area = row.get("area")
        if not has_segmentation and area is not None and not _is_crowd(row):
            try:
                if abs(float(area) - width * height) > 0.5:
                    issues.append(issue(
                        "coco.annotation.area_mismatch", target,
                        id=identifier, area=area, expected=width * height,
                    ))
            except (TypeError, ValueError):
                issues.append(issue(
                    "coco.annotation.area_mismatch", target,
                    id=identifier, area=area, expected=width * height,
                ))

        image_id = _as_int(row.get("image_id"))
        record = image_by_id.get(image_id) if image_id is not None else None
        if record is None:
            continue
        image_width = _as_int(record.get("width"))
        image_height = _as_int(record.get("height"))
        box = (x, y, x + width, y + height)
        if image_width and image_height and (
            x < 0 or y < 0 or x + width > image_width or y + height > image_height
        ):
            issues.append(issue(
                "coco.annotation.out_of_bounds", target,
                id=identifier, image_width=image_width, image_height=image_height,
            ))
        if min(width, height) < min_box_size:
            issues.append(issue(
                "coco.annotation.small", target, id=identifier,
                width=round(width, 1), height=round(height, 1), threshold=min_box_size,
            ))
        category_id = _as_int(row.get("category_id"))
        if image_id is not None and category_id is not None:
            grouped.setdefault(image_id, []).append((identifier, category_id, box))

    issues.extend(_duplicate_boxes(grouped, target, near_duplicate_iou))
    return issues


def _is_crowd(row: dict) -> bool:
    try:
        return int(row.get("iscrowd", 0)) == 1
    except (TypeError, ValueError):
        return False


def _duplicate_boxes(grouped, target, near_duplicate_iou) -> list[Issue]:
    issues: list[Issue] = []
    for _image_id, entries in grouped.items():
        exact: dict[tuple, list[int]] = {}
        for identifier, category_id, box in entries:
            quantized = (category_id, *(round(value, 3) for value in box))
            exact.setdefault(quantized, []).append(identifier)
        for identifiers in exact.values():
            if len(identifiers) > 1:
                # Every member, not just the first pair: the repair keeps one
                # and drops the rest, so it needs the whole group.
                issues.append(issue(
                    "coco.annotation.duplicate", target,
                    id=identifiers[0], other_id=identifiers[1], ids=identifiers,
                ))
        for index, (id_a, category_a, box_a) in enumerate(entries):
            for id_b, category_b, box_b in entries[index + 1:]:
                same_box = all(round(a, 3) == round(b, 3) for a, b in zip(box_a, box_b))
                if category_a != category_b:
                    # Two labels on one box is ambiguous, but not a duplicate:
                    # dropping either would be a guess, so it needs a human.
                    if same_box:
                        issues.append(issue(
                            "coco.annotation.geometry_conflict", target,
                            id=id_a, other_id=id_b,
                        ))
                    continue
                if same_box:
                    continue  # already reported as an exact duplicate
                overlap = box_iou(box_a, box_b)
                if overlap >= near_duplicate_iou:
                    issues.append(issue(
                        "coco.annotation.near_duplicate", target,
                        id=id_a, other_id=id_b, iou=f"{overlap:.3f}",
                    ))
    return issues


def _image_records(images, annotation_counts, detected, target, probes) -> list[Issue]:
    """Records versus disk, in both directions."""
    issues: list[Issue] = []
    disk_keys = {
        probe.relative.as_posix().casefold(): probe for probe in probes or []
    }
    basenames: dict[str, int] = {}
    for probe in probes or []:
        basenames[probe.path.name.casefold()] = basenames.get(probe.path.name.casefold(), 0) + 1

    registered: set[str] = set()
    for row in images:
        identifier = _as_int(row.get("id"))
        file_name = str(row.get("file_name") or "")
        key = file_name.replace("\\", "/").removeprefix("./").casefold()
        registered.add(key)
        if key not in disk_keys and not _basename_only_match(key, basenames, disk_keys):
            issues.append(issue(
                "coco.image_record.missing_file", target, id=identifier, file_name=file_name,
            ))
        if identifier is not None and not annotation_counts.get(identifier):
            issues.append(issue("coco.image_record.no_annotation", target, id=identifier))

    for key, probe in sorted(disk_keys.items()):
        if key in registered:
            continue
        # A legacy record that stores only the basename still names this image.
        if _basename_only_match(key, basenames, registered):
            continue
        issues.append(issue("coco.image.on_disk_unregistered", probe.path))
    return issues


def _basename_only_match(key: str, basenames: dict[str, int], candidates) -> bool:
    """Legacy flat records store a basename; that is safe only when unique."""
    if "/" in key:
        return False
    name = Path(key).name
    if basenames.get(name, 0) != 1:
        return False
    return any(Path(candidate).name == name for candidate in candidates)


def _unused_categories(categories, category_counts, target) -> list[Issue]:
    return [
        issue("coco.category.unused", target, id=_as_int(row.get("id")), name=row.get("name"))
        for row in categories
        if _as_int(row.get("id")) is not None and not category_counts.get(_as_int(row.get("id")))
    ]


def _divergence(annotation_dir: Path, document: dict, target: Path) -> list[Issue]:
    """The sqlite store and the exported JSON must describe the same dataset.

    The store is what the app reads and the JSON is what trainers read, so a
    divergence means one of the two audiences sees a different dataset.

    The store is only touched when its file already exists: constructing
    ``CocoAnnotationStore`` creates the directory and the database, and a
    scan must not write anything.
    """
    if not (annotation_dir / ".model_labeling.sqlite3").is_file():
        return []
    try:
        store = CocoAnnotationStore(annotation_dir)
        if not store.is_initialized():
            return []
        stored = store.read_document()
    except (OSError, ValueError):
        return []
    if not stored.get("images"):
        return []
    differences = []
    for key in ("images", "annotations", "categories"):
        stored_count = len(stored.get(key, []))
        file_count = len(document.get(key, []))
        if stored_count != file_count:
            differences.append(f"{key}: DB={stored_count} JSON={file_count}")
    if not differences:
        return []
    return [issue("coco.db.json_divergence", target, detail="; ".join(differences))]
