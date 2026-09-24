"""YOLO validation.

The class table is the part worth being careful about. A row's meaning is
carried entirely by its integer class id, so if ``classes.txt`` and the
project presets disagree, every label in the dataset is silently wrong --
nothing crashes, nothing looks broken, and the model learns the wrong names.
The old scan could not see this at all: it only asked whether a file was
non-empty.

``yolo.unrecognized`` exists because of a real data-loss bug in the old
orphan pass, which deleted every ``*.txt`` in the label directory whose stem
did not match an image. A ``notes.txt`` or ``README.txt`` kept next to the
labels was deleted as an "orphan annotation"; only ``classes.txt`` was spared.
"""
from __future__ import annotations

from pathlib import Path

from src.models.annotation import LabelPreset

from ..yolo_metadata import find_yolo_yaml, load_yolo_metadata, yolo_class_names
from .common_rules import ImageProbe, box_iou
from .report import Issue, issue

#: Files that live beside the labels but are not annotations.
_NON_ANNOTATION_NAMES = {"classes.txt", "train.txt", "val.txt", "test.txt"}

_TASK_ARITY = {
    "yolo_detection": 5,
    "yolo_obb": 9,
}


def validate_yolo(
    probes: list[ImageProbe],
    detected,
    *,
    presets: list[LabelPreset] | None = None,
    min_box_size: int = 20,
    near_duplicate_iou: float = 0.95,
) -> list[Issue]:
    classes, issues = _resolve_classes(detected, presets)
    used: set[int] = set()
    for probe in probes:
        issues.extend(_validate_one(
            probe, detected, classes, used, min_box_size, near_duplicate_iou,
        ))
    issues.extend(_unused_classes(classes, used, detected.annotation_dir))
    issues.extend(_annotation_directory(probes, detected, classes))
    issues.extend(_stale_caches(detected.annotation_dir))
    issues.extend(_data_yaml_paths(detected.root))
    return issues


def _resolve_classes(detected, presets: list[LabelPreset] | None) -> tuple[list[str], list[Issue]]:
    """The class table rows are resolved against, plus any mismatch report.

    A table on disk wins, because it is what the dataset ships with and what
    another tool would read. The project presets are the fallback: the app
    writes YOLO rows using the preset class ids and never writes a
    ``classes.txt``, so for a dataset built here the presets *are* the table
    -- falling back to them is what makes the class-id check possible at all.
    """
    issues: list[Issue] = []
    project = [preset.name for preset in presets or []]
    disk, disk_issues = _read_disk_classes(detected)
    issues.extend(disk_issues)

    if disk:
        if project and disk != project:
            issues.append(issue(
                "yolo.classes.preset_mismatch", detected.annotation_dir,
                disk=", ".join(disk), project=", ".join(project),
            ))
        return disk, issues
    if project:
        # Usable, but the dataset does not describe itself: hand it to
        # another tool and the class names are gone.
        issues.append(issue("yolo.classes.missing", detected.annotation_dir))
        return project, issues
    issues.append(issue("yolo.classes.missing", detected.annotation_dir))
    return [], issues


def _read_disk_classes(detected) -> tuple[list[str], list[Issue]]:
    for candidate in (detected.annotation_dir / "classes.txt", detected.root / "classes.txt"):
        if not candidate.is_file():
            continue
        try:
            return [
                line.strip()
                for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ], []
        except OSError as exc:
            return [], [issue("yolo.annotation.unreadable", candidate, error=str(exc))]
    try:
        return yolo_class_names(detected.root), []
    except (ValueError, OSError):
        return [], []


def _validate_one(
    probe: ImageProbe,
    detected,
    classes: list[str],
    used: set[int],
    min_box_size: int,
    near_duplicate_iou: float,
) -> list[Issue]:
    issues: list[Issue] = []
    txt_path = probe.annotation_path(detected.annotation_dir, ".txt")
    if not txt_path.is_file():
        # Distinct from an empty file: this may be a lost annotation rather
        # than a deliberate negative sample, so it stays a REVIEW finding.
        return [issue("yolo.annotation.missing", probe.path)]

    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [issue("yolo.annotation.unreadable", probe.path, error=str(exc))]

    rows: list[tuple[int, tuple[float, float, float, float]]] = []
    rows_with_line: list[tuple[int, int, tuple[float, float, float, float]]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        issues.extend(_validate_row(
            raw, line_number, probe, detected, classes, used,
            min_box_size, rows, rows_with_line,
        ))

    if not rows and not issues:
        issues.append(issue("yolo.annotation.empty", probe.path))
    issues.extend(_compare_rows(probe, rows_with_line, near_duplicate_iou))
    return issues


def _validate_row(raw, line_number, probe, detected, classes, used,
                  min_box_size, rows, rows_with_line) -> list[Issue]:
    parts = raw.split()
    task = detected.task_name or "yolo_detection"
    expected = _TASK_ARITY.get(task)

    if expected is not None and len(parts) != expected:
        return [issue(
            "yolo.row.bad_arity", probe.path,
            line=line_number, count=len(parts), expected=expected,
        )]
    if expected is None and not _matches_task_shape(len(parts), task):
        return [issue(
            "yolo.row.bad_arity", probe.path,
            line=line_number, count=len(parts), expected=_describe_shape(task),
        )]
    if len(parts) < 5:
        return [issue(
            "yolo.row.bad_arity", probe.path,
            line=line_number, count=len(parts), expected=5,
        )]

    try:
        class_id = int(parts[0])
    except ValueError:
        return [issue("yolo.row.bad_number", probe.path, line=line_number, value=parts[0])]

    issues: list[Issue] = []
    if class_id < 0 or (classes and class_id >= len(classes)):
        issues.append(issue(
            "yolo.row.class_unknown", probe.path,
            line=line_number, class_id=class_id, total=len(classes),
        ))
    used.add(class_id)

    values: list[float] = []
    for raw_value in parts[1:]:
        try:
            values.append(float(raw_value))
        except ValueError:
            issues.append(issue(
                "yolo.row.bad_number", probe.path, line=line_number, value=raw_value,
            ))
            return issues
    if any(value < -_EPSILON or value > 1 + _EPSILON for value in values):
        issues.append(issue("yolo.row.out_of_range", probe.path, line=line_number))
        return issues

    box = tuple(values[:4])  # centre_x, centre_y, width, height
    if box[2] <= 0 or box[3] <= 0:
        issues.append(issue(
            "yolo.row.degenerate", probe.path,
            line=line_number, width=f"{box[2]:.6f}", height=f"{box[3]:.6f}",
        ))
        return issues

    if probe.ok:
        pixel_width = box[2] * probe.width
        pixel_height = box[3] * probe.height
        if min(pixel_width, pixel_height) < min_box_size:
            issues.append(issue(
                "yolo.row.small", probe.path, line=line_number,
                width=f"{pixel_width:.1f}", height=f"{pixel_height:.1f}",
                threshold=min_box_size,
            ))
        rows.append((class_id, _pixel_box(box, probe)))
        rows_with_line.append((line_number, class_id, _pixel_box(box, probe)))
    return issues


_EPSILON = 1e-6


def _matches_task_shape(count: int, task: str) -> bool:
    if task == "yolo_pose":
        return count >= 8 and (count - 5) % 3 == 0
    if task == "yolo_segmentation":
        return count >= 7 and (count - 5) % 2 == 0
    return count >= 5


def _describe_shape(task: str) -> str:
    return {"yolo_pose": "5+3k", "yolo_segmentation": "5+2k"}.get(task, ">=5")


def _pixel_box(box, probe: ImageProbe) -> tuple[float, float, float, float]:
    """``(cx, cy, w, h)`` normalized -> ``(xmin, ymin, xmax, ymax)`` in pixels."""
    centre_x, centre_y, width, height = box
    return (
        (centre_x - width / 2) * probe.width,
        (centre_y - height / 2) * probe.height,
        (centre_x + width / 2) * probe.width,
        (centre_y + height / 2) * probe.height,
    )


def _compare_rows(probe: ImageProbe, rows, near_duplicate_iou: float) -> list[Issue]:
    """Duplicates are compared on the values, not the text.

    Two rows written from boxes 0.4px apart round to the same six decimals,
    so comparing the parsed floats is what actually finds what is in the file.
    """
    issues: list[Issue] = []
    if not rows:
        return issues

    exact: dict[tuple, list[int]] = {}
    for line_number, class_id, box in rows:
        quantized = (class_id, *(round(value, 6) for value in box))
        exact.setdefault(quantized, []).append(line_number)
    for (class_id, *_rest), lines in exact.items():
        if len(lines) > 1:
            issues.append(issue(
                "yolo.row.duplicate", probe.path,
                count=len(lines), class_id=class_id, lines=lines,
            ))

    for index, (line_a, class_a, box_a) in enumerate(rows):
        for line_b, class_b, box_b in rows[index + 1:]:
            if class_a != class_b:
                continue
            if (round(box_a[0], 6), round(box_a[1], 6), round(box_a[2], 6), round(box_a[3], 6)) == \
               (round(box_b[0], 6), round(box_b[1], 6), round(box_b[2], 6), round(box_b[3], 6)):
                continue  # already reported as an exact duplicate
            overlap = box_iou(box_a, box_b)
            if overlap >= near_duplicate_iou:
                issues.append(issue(
                    "yolo.row.near_duplicate", probe.path,
                    first=line_a, second=line_b, iou=f"{overlap:.3f}",
                ))
    return issues


def _unused_classes(classes: list[str], used: set[int], annotation_dir: Path) -> list[Issue]:
    if not classes or not used:
        return []
    # A class id outside the table is already reported per row; it must not
    # make the table look complete here.
    unused = [
        classes[index] for index in range(len(classes)) if index not in used
    ]
    if not unused:
        return []
    return [issue("yolo.classes.unused", annotation_dir / "classes.txt", names=", ".join(unused))]


def _annotation_directory(probes: list[ImageProbe], detected, classes: list[str]) -> list[Issue]:
    """Split the label directory into orphans and files that are not labels.

    This is the check that must not delete a ``notes.txt``: a candidate is
    only an orphan when every non-blank line actually looks like an
    annotation row.
    """
    issues: list[Issue] = []
    annotation_dir = detected.annotation_dir
    if not annotation_dir.is_dir():
        return issues
    image_keys = {probe.key for probe in probes}
    for path in sorted(annotation_dir.rglob("*.txt"), key=lambda item: item.as_posix().casefold()):
        if path.name.casefold() in _NON_ANNOTATION_NAMES:
            continue
        try:
            key = path.relative_to(annotation_dir).with_suffix("").as_posix().casefold()
        except ValueError:
            continue
        if key in image_keys:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            issues.append(issue("yolo.annotation.unreadable", path, error=str(exc)))
            continue
        if _looks_like_annotations(text):
            issues.append(issue("yolo.orphan", path))
        else:
            issues.append(issue("yolo.unrecognized", path))
    return issues


def _looks_like_annotations(text: str) -> bool:
    """True when every non-blank line is a class id plus at least one number."""
    saw_line = False
    for raw in text.splitlines():
        if not raw.strip():
            continue
        saw_line = True
        parts = raw.split()
        if len(parts) < 2:
            return False
        try:
            int(parts[0])
            for value in parts[1:]:
                float(value)
        except ValueError:
            return False
    return saw_line


def _stale_caches(annotation_dir: Path) -> list[Issue]:
    """Ultralytics caches label arrays beside the labels.

    They survive a cleanup that deletes images or rewrites rows, and a
    trainer that trusts a stale cache reads the labels the dataset no longer
    has.
    """
    if not annotation_dir.is_dir():
        return []
    return [
        issue("yolo.cache.present", path, name=path.name)
        for path in sorted(annotation_dir.rglob("*.cache"))
    ]


def _data_yaml_paths(root: Path) -> list[Issue]:
    """Only a *declared* path that does not exist is a defect.

    Absence of data.yaml is not reported: the app does not write one for
    detection datasets, so flagging it would fire on nearly every dataset
    built here and drown the findings that matter.
    """
    yaml_path = find_yolo_yaml(root)
    if yaml_path is None:
        return []
    try:
        document = load_yolo_metadata(root)
    except (ValueError, OSError):
        return []
    issues: list[Issue] = []
    base = yaml_path.parent
    declared_root = document.get("path")
    if isinstance(declared_root, str) and declared_root.strip():
        base = base / declared_root.strip()
    for key in ("train", "val", "test"):
        value = document.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if (base / value.strip()).exists():
            continue
        issues.append(issue(
            "yolo.data_yaml.path_missing", yaml_path, key=key, value=value.strip(),
        ))
    return issues
