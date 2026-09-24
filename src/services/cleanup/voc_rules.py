"""Pascal VOC validation.

Two things here differ on purpose from the old scan:

* objects are read with ``findall("object")`` -- direct children only, the
  same way ``AnnotationService._load_voc`` reads them. The old scan used
  ``iter("object")``, which also matches nested elements, so the two could
  disagree about whether a file has any object at all.
* the declared ``<size>`` is checked against the real JPEG. The loader
  ignores it, but ``voc_eval``, mmdetection and most converters trust it, so
  a stale value is a defect even though this app never notices.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.models.annotation import LabelPreset

from .common_rules import ImageProbe, box_iou
from .report import Issue, issue

_FIELDS = ("xmin", "ymin", "xmax", "ymax")


def validate_voc(
    probes: list[ImageProbe],
    annotation_dir: Path,
    *,
    presets: list[LabelPreset] | None = None,
    min_box_size: int = 20,
    near_duplicate_iou: float = 0.95,
) -> list[Issue]:
    known_labels = {preset.name for preset in presets or []}
    issues: list[Issue] = []
    for probe in probes:
        issues.extend(_validate_one(
            probe, annotation_dir, known_labels, min_box_size, near_duplicate_iou,
        ))
    issues.extend(_orphans(probes, annotation_dir))
    return issues


def _orphans(probes: list[ImageProbe], annotation_dir: Path) -> list[Issue]:
    """XML files with no image left.

    ``.xml`` is not a format anyone keeps notes in, so -- unlike the YOLO
    ``.txt`` case -- every candidate here is an annotation by definition.
    """
    if not annotation_dir.is_dir():
        return []
    image_keys = {probe.key for probe in probes}
    issues: list[Issue] = []
    for path in sorted(annotation_dir.rglob("*.xml"), key=lambda item: item.as_posix().casefold()):
        try:
            key = path.relative_to(annotation_dir).with_suffix("").as_posix().casefold()
        except ValueError:
            continue
        if key not in image_keys:
            issues.append(issue("voc.orphan", path))
    return issues


def _validate_one(
    probe: ImageProbe,
    annotation_dir: Path,
    known_labels: set[str],
    min_box_size: int,
    near_duplicate_iou: float,
) -> list[Issue]:
    issues: list[Issue] = []
    xml_path = probe.annotation_path(annotation_dir, ".xml")
    if not xml_path.is_file():
        return [issue("voc.annotation.missing", probe.path)]

    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        return [issue("voc.annotation.unreadable", probe.path, error=str(exc))]

    objects = root.findall("object")
    if not objects:
        issues.append(issue("voc.annotation.empty", probe.path))

    boxes: list[tuple[str, float, float, float, float]] = []
    for index, obj in enumerate(objects, start=1):
        issues.extend(_validate_object(obj, index, probe, known_labels, min_box_size, boxes))

    issues.extend(_validate_header(root, probe, xml_path))
    issues.extend(_compare_boxes(probe, boxes, near_duplicate_iou))
    return issues


def _validate_object(obj, index: int, probe: ImageProbe, known_labels: set[str],
                     min_box_size: int, boxes: list) -> list[Issue]:
    issues: list[Issue] = []
    name = (obj.findtext("name") or "").strip()
    if not name:
        # _load_voc raises on this, so the whole file is unusable until fixed.
        issues.append(issue("voc.object.blank_name", probe.path, index=index))
    elif known_labels and name not in known_labels:
        issues.append(issue("voc.object.unknown_label", probe.path, label=name))

    box = obj.find("bndbox")
    if box is None:
        return issues + [issue("voc.box.missing", probe.path, index=index)]

    values: dict[str, float] = {}
    malformed = False
    for field in _FIELDS:
        raw = box.findtext(field)
        try:
            values[field] = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            issues.append(issue(
                "voc.box.bad_number", probe.path,
                index=index, field=field, value=raw if raw is not None else "<缺失>",
            ))
            malformed = True
    if malformed:
        return issues

    if any(value != int(value) for value in values.values()):
        issues.append(issue("voc.box.not_integral", probe.path, index=index))

    width = values["xmax"] - values["xmin"]
    height = values["ymax"] - values["ymin"]
    if width <= 0 or height <= 0:
        issues.append(issue(
            "voc.box.degenerate", probe.path, index=index, width=width, height=height,
        ))
        return issues

    if probe.ok and (
        values["xmin"] < 0 or values["ymin"] < 0
        or values["xmax"] > probe.width or values["ymax"] > probe.height
    ):
        issues.append(issue(
            "voc.box.out_of_bounds", probe.path, index=index,
            image_width=probe.width, image_height=probe.height,
            **{field: _number(values[field]) for field in _FIELDS},
        ))

    # min() rather than "both": a 400x3 sliver is as unusable as a 12x12 box,
    # and the "both below" reading would let it through.
    if min(width, height) < min_box_size:
        issues.append(issue(
            "voc.box.small", probe.path, index=index,
            width=_number(width), height=_number(height), threshold=min_box_size,
        ))

    # The object's 1-based index rides along so a finding can name the exact
    # <object> to act on: a fixer must not have to re-parse the file and guess.
    boxes.append((
        index, name,
        values["xmin"], values["ymin"], values["xmax"], values["ymax"],
    ))
    return issues


def _validate_header(root, probe: ImageProbe, xml_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    size = root.find("size")
    if size is None:
        issues.append(issue("voc.size.missing", probe.path))
    elif probe.ok:
        try:
            declared = (int(size.findtext("width")), int(size.findtext("height")))
        except (TypeError, ValueError):
            issues.append(issue("voc.size.missing", probe.path))
        else:
            if declared != probe.size:
                issues.append(issue(
                    "voc.size.mismatch", probe.path,
                    declared_width=declared[0], declared_height=declared[1],
                    image_width=probe.width, image_height=probe.height,
                ))
        depth = size.findtext("depth")
        if depth is not None:
            try:
                if int(depth) != probe.bands:
                    issues.append(issue(
                        "voc.depth.mismatch", probe.path,
                        declared_depth=int(depth), actual_depth=probe.bands,
                    ))
            except ValueError:
                pass

    # <filename> is what a converter uses to locate the image, so it goes
    # stale the moment the image is renamed outside the app.
    declared_name = (root.findtext("filename") or "").strip()
    if declared_name and declared_name != probe.path.name:
        issues.append(issue(
            "voc.filename.mismatch", probe.path,
            declared=declared_name, actual=probe.path.name,
        ))
    return issues


def _compare_boxes(probe: ImageProbe, boxes: list, near_duplicate_iou: float) -> list[Issue]:
    """Same-label duplicates, cross-label geometry clashes, near-duplicates."""
    issues: list[Issue] = []
    exact: dict[tuple, list[int]] = {}
    for index, name, x1, y1, x2, y2 in boxes:
        exact.setdefault((name, x1, y1, x2, y2), []).append(index)
    for (name, *_rest), members in exact.items():
        if len(members) > 1:
            issues.append(issue(
                "voc.box.duplicate", probe.path,
                count=len(members), label=name, indices=members,
            ))

    geometry: dict[tuple, list[tuple[int, str]]] = {}
    for index, name, x1, y1, x2, y2 in boxes:
        geometry.setdefault((x1, y1, x2, y2), []).append((index, name))
    for members in geometry.values():
        labels = {name for _index, name in members}
        if len(labels) > 1:
            issues.append(issue(
                "voc.box.geometry_conflict", probe.path,
                labels=", ".join(sorted(labels)),
                indices=[index for index, _name in members],
            ))

    for position, first in enumerate(boxes):
        for second in boxes[position + 1:]:
            if first[1] != second[1]:
                continue
            if first[2:] == second[2:]:
                continue  # already reported as an exact duplicate
            overlap = box_iou(first[2:], second[2:])
            if overlap >= near_duplicate_iou:
                issues.append(issue(
                    "voc.box.near_duplicate", probe.path,
                    label=first[1], iou=f"{overlap:.3f}",
                    first=_box_text(first), second=_box_text(second),
                    first_index=first[0], second_index=second[0],
                ))
    return issues


def _box_text(box) -> str:
    return f"[{_number(box[2])},{_number(box[3])},{_number(box[4])},{_number(box[5])}]"


def _number(value: float) -> str:
    """Render 200.0 as 200 so the report matches what is in the XML."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
