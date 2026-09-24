"""Read-only dataset scan.

The runner never writes. It probes every image once, hands the probes to the
format's validator, and returns a :class:`~.report.Plan` describing what it
found. Everything that changes data is a separate, opt-in step -- which is
what makes this safe to run on a real dataset at any time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.models.annotation import LabelPreset
from src.services.dataset_detector import DatasetDetector

from . import coco_rules, common_rules, voc_rules, yolo_rules
from .common_rules import ImageProbe, collect_images, probe_image
from .report import Issue, Plan, issue

#: A box is "too small" when its shorter side falls below this many pixels.
#: Deliberately a parameter, not a constant: it is an annotation policy, not
#: a property of the data, and the report says which value was used.
DEFAULT_MIN_BOX_SIZE = 20

#: Same-label boxes overlapping at least this much are reported as
#: near-duplicates. Boxes further apart than this can be genuinely distinct
#: objects, so the finding is a REVIEW rather than a repair.
DEFAULT_NEAR_DUPLICATE_IOU = 0.95


def scan_dataset(
    root: Path,
    *,
    presets: list[LabelPreset] | None = None,
    detected=None,
    min_box_size: int = DEFAULT_MIN_BOX_SIZE,
    near_duplicate_iou: float = DEFAULT_NEAR_DUPLICATE_IOU,
    verify_pixels: bool = True,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Plan:
    """Validate a dataset and describe every defect found.

    Args:
        root: dataset directory.
        presets: project label presets, used to spot label/class mismatches.
        detected: an already-detected layout, to skip re-detection.
        min_box_size: shorter-side threshold for the "tiny box" finding.
        near_duplicate_iou: overlap above which same-label boxes are reported.
        verify_pixels: decode every image. Truncated files keep a valid
            header, so this is the only way to find them -- but it reads
            every byte, and a large dataset will feel it. Off is faster and
            still catches files PIL cannot open at all.
        progress: called as ``(done, total)`` while images are probed.
        is_cancelled: polled between images; a True return stops the probe
            pass and returns what has been checked so far.
    """
    root = Path(root)
    detected = detected or DatasetDetector.detect(root)
    probes = _probe_images(detected.image_dir, verify_pixels, progress, is_cancelled)

    plan = Plan(root=root, format_name=detected.format_name, total_images=len(probes))
    keywords = {
        "min_box_size": min_box_size,
        "near_duplicate_iou": near_duplicate_iou,
        "presets": presets,
    }

    plan.extend(common_rules.stem_conflicts(probes))
    plan.extend(_unreadable(probes))

    if detected.format_name == "voc":
        plan.extend(voc_rules.validate_voc(probes, detected.annotation_dir, **keywords))
    elif detected.format_name == "yolo":
        plan.extend(yolo_rules.validate_yolo(probes, detected, **keywords))
    elif detected.format_name == "coco":
        plan.extend(coco_rules.validate_coco(
            detected, probes=probes,
            min_box_size=min_box_size,
            near_duplicate_iou=near_duplicate_iou,
            presets=presets,
        ))

    plan.issues = common_rules.dedupe_issues(plan.issues)
    return plan


def _unreadable(probes: list[ImageProbe]) -> list[Issue]:
    return [
        issue("image.unreadable", probe.path, error=probe.error)
        for probe in probes
        if not probe.ok
    ]


def _probe_images(
    image_dir: Path,
    verify_pixels: bool,
    progress: Callable[[int, int], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> list[ImageProbe]:
    """Probe every image, reporting progress in batch.

    An image that cannot even be opened is still probed -- the dimension
    checks downstream simply skip it, and the report keeps the file in its
    totals instead of silently dropping it.
    """
    paths = collect_images(image_dir)
    total = len(paths)
    probes: list[ImageProbe] = []
    for index, path in enumerate(paths, start=1):
        probes.append(probe_image(path, image_dir) if verify_pixels
                      else _probe_header_only(path, image_dir))
        if progress is not None and (index % 5 == 0 or index == total):
            progress(index, total)
        if is_cancelled is not None and is_cancelled():
            break
    return probes


def _probe_header_only(path: Path, image_dir: Path) -> ImageProbe:
    """Dimensions without decoding the pixels."""
    from PIL import Image

    from src.utils.pixels import channel_count

    try:
        relative = path.relative_to(image_dir)
    except ValueError:
        relative = Path(path.name)
    try:
        with Image.open(path) as image:
            return ImageProbe(path, relative, image.size, channel_count(image))
    except Exception as exc:  # PIL raises OSError, SyntaxError and its own types
        return ImageProbe(path, relative, None, 0, f"{type(exc).__name__}: {exc}")
