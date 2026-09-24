"""Checks that apply to every format.

The one that matters most here is :func:`probe_image`. The old scan never
opened an image, but every writer in ``AnnotationService`` calls
``Image.open`` for the dimensions -- so a truncated JPEG passes a scan and
then breaks the save. Reading the pixels is the only way to catch it: a
truncated file keeps a valid header, so its size is readable and only a full
decode fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from src.utils.pixels import channel_count

from ..image_service import SUPPORTED_IMAGE_EXTENSIONS
from .report import Issue, issue


@dataclass(frozen=True)
class ImageProbe:
    """What the dataset actually says about one image."""

    path: Path
    relative: Path
    size: tuple[int, int] | None = None
    bands: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.size is not None

    @property
    def key(self) -> str:
        """The annotation-file key: relative path minus suffix, casefolded."""
        return self.relative.with_suffix("").as_posix().casefold()

    @property
    def width(self) -> int:
        return self.size[0] if self.size else 0

    @property
    def height(self) -> int:
        return self.size[1] if self.size else 0

    def annotation_path(self, annotation_dir: Path, suffix: str) -> Path:
        return annotation_dir / self.relative.with_suffix(suffix)


def collect_images(image_dir: Path) -> list[Path]:
    """Every image below ``image_dir``, in a stable order."""
    if not image_dir.is_dir():
        return []
    return sorted(
        (path for path in image_dir.rglob("*")
         if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix().casefold(),
    )


def probe_image(path: Path, image_dir: Path) -> ImageProbe:
    """Read an image's real dimensions by decoding it.

    ``Image.open`` alone is not enough: it reads the header lazily, so the
    truncated files this check exists for look fine until ``load()`` runs.
    """
    try:
        relative = path.relative_to(image_dir)
    except ValueError:
        relative = Path(path.name)
    try:
        with Image.open(path) as image:
            size = image.size
            bands = channel_count(image)
            image.load()
    except Exception as exc:  # PIL raises OSError, SyntaxError and its own types
        return ImageProbe(path, relative, None, 0, f"{type(exc).__name__}: {exc}")
    return ImageProbe(path, relative, size, bands)


def stem_conflicts(probes: list[ImageProbe]) -> list[Issue]:
    """Images that would share one annotation file.

    ``foo.jpg`` and ``foo.png`` in the same directory both mirror onto
    ``foo.txt`` / ``foo.xml``, so opening one shows the other's boxes and
    saving one overwrites the other. Silent, and the old scan never saw it
    because it only ever looked at one image at a time.
    """
    grouped: dict[str, list[ImageProbe]] = {}
    for probe in probes:
        grouped.setdefault(probe.key, []).append(probe)
    conflicts: list[Issue] = []
    for key, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        for member in members:
            others = ", ".join(
                other.path.name for other in members if other.path != member.path
            )
            conflicts.append(issue("image.stem_conflict", member.path, others=others))
    return conflicts


def box_iou(first: tuple[float, float, float, float],
            second: tuple[float, float, float, float]) -> float:
    """Intersection over union of two ``(xmin, ymin, xmax, ymax)`` boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def dedupe_issues(values: list[Issue]) -> list[Issue]:
    """Drop repeats while keeping first-seen order.

    The marker includes the issue's data, not just its rule and target: two
    different boxes below the size threshold in one file are two findings,
    and collapsing them would hide one of the boxes the user has to deal with.
    """
    seen: set[tuple] = set()
    result: list[Issue] = []
    for value in values:
        marker = (value.rule, value.target, tuple(sorted(
            (key, str(item)) for key, item in value.data.items()
        )))
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
