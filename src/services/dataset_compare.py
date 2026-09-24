"""Compare a YOLO ONNX model's output against a dataset's own labels.

The dataset is read strictly read-only. Ground truth comes from the annotation
files themselves rather than through ``AnnotationService``: that loader is built
for interactive editing and caches COCO documents into a ``coco.db`` beside the
JSON, which a comparison has no business writing into a dataset it is only
looking at.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from src.models.annotation import LabelPreset, label_color
from .dataset_detector import DatasetDetector
from .onnx_service import YoloOnnxDetector
from .yolo_metadata import yolo_class_names

#: Token standing in for "no label at all" on either side of the arrow, so a
#: mismatch line reads the same whatever the format's idea of "empty" is.
NO_LABEL = "None"

#: A score is a reading aid; the raw float decides a match. Two decimals is
#: what the report format asks for.
SCORE_DIGITS = 2

#: How many per-image failures are kept. A dataset can be corrupt in one way
#: across thousands of files, and the count is what matters then, not a copy of
#: the same message per file.
MAX_ERRORS = 50

#: Inference workers. One ONNX session is shared: ``run`` releases the GIL, so
#: threads parallelise as well as processes here. Measured on voc-test (6
#: physical cores, 70292 images, letterbox 640): 1 worker 108 ms/image, 2
#: workers 92, 3 workers 100, 4 workers 102, 6 workers 113. The one setting
#: that really hurts is ORT's own intra-op thread count turned *down* -- 1
#: thread costs 290 ms/image -- so the session keeps its default and three
#: workers is where the curve is already flat.
DEFAULT_WORKERS = 3

#: A worker window a few times the pool size keeps every thread fed without
#: queueing the whole dataset as futures, which is what makes cancelling one
#: take effect within a few images rather than a few thousand.
WINDOW_FACTOR = 3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FORMAT_LABELS = {"voc": "Pascal VOC", "yolo": "YOLO", "coco": "COCO"}


@dataclass(frozen=True)
class ClassTally:
    """How often one ground-truth class was involved in a disagreement."""

    images: int = 0
    mismatched: int = 0

    @property
    def rate(self) -> float:
        return self.mismatched / self.images if self.images else 0.0


@dataclass(frozen=True)
class CompareCase:
    """One image's verdict.

    ``expected`` is sorted, ``predicted`` is ordered by descending score, and
    both are de-duplicated: two boxes carrying the same label are one label as
    far as "do the labels agree" is concerned.
    """

    name: str
    expected: tuple[str, ...]
    predicted: tuple[str, ...]
    scores: tuple[float, ...]

    @property
    def matched(self) -> bool:
        return set(self.expected) == set(self.predicted)

    def line(self) -> str:
        expected = "+".join(self.expected) if self.expected else NO_LABEL
        if not self.predicted:
            return f"{self.name} {expected} -> {NO_LABEL}"
        predicted = "+".join(self.predicted)
        scores = "+".join(f"{score:.{SCORE_DIGITS}f}" for score in self.scores)
        return f"{self.name} {expected} -> {predicted} {scores}"


@dataclass
class CompareReport:
    output_path: Path
    model_path: Path
    dataset_dir: Path
    format_name: str = ""
    threshold: float = 0.5
    generated_at: str = ""
    total: int = 0
    matched: int = 0
    mismatched: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    error_count: int = 0
    per_class: dict[str, ClassTally] = field(default_factory=dict)
    cancelled: bool = False

    @property
    def compared(self) -> int:
        return self.matched + self.mismatched

    @property
    def mismatch_rate(self) -> float:
        return self.mismatched / self.compared if self.compared else 0.0

    @property
    def display_format(self) -> str:
        return FORMAT_LABELS.get(self.format_name, self.format_name)


def report_path_for(dataset_dir: Path) -> Path:
    """``<name>_compare.txt`` beside the dataset, as the tool has always written it."""
    dataset_dir = Path(dataset_dir)
    return dataset_dir.parent / f"{dataset_dir.name}_compare.txt"


# ---------------------------- ground truth ----------------------------


def _strip_extension(key: str) -> str:
    return key.rsplit(".", 1)[0] if "." in key else key


def _label_index(annotation_root: Path, suffix: str) -> dict[str, list[str]]:
    """Map every annotation file's identity to the labels it declares.

    Keyed twice -- by the path relative to ``annotation_root`` and by the bare
    stem -- so an image can be matched through a mirrored layout first and a
    flat one second. A stem that appears twice is not a key: two files claiming
    the same image make the pairing a guess, and a comparison that guesses is
    worse than one that reports nothing.
    """
    by_relative: dict[str, list[str]] = {}
    stems: dict[str, list[str]] = {}
    for path in sorted(annotation_root.rglob(f"*{suffix}")):
        relative = path.relative_to(annotation_root).with_suffix("").as_posix()
        labels = _labels_from(path, suffix)
        by_relative[relative] = labels
        stems.setdefault(path.stem, []).append(relative)
    result = dict(by_relative)
    for stem, keys in stems.items():
        if len(keys) == 1:
            result.setdefault(f"\0{stem}", by_relative[keys[0]])
    return result


def _labels_from(path: Path, suffix: str) -> list[str]:
    if suffix == ".xml":
        root = ET.parse(path).getroot()
        return [
            name
            for name in ((node.findtext("name") or "").strip() for node in root.iter("object"))
            if name
        ]
    labels = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        # Only the class column is needed. The rest of the row is geometry,
        # whose arity differs by task -- and a pose or segmentation label file
        # names the same class as a detection one does.
        if parts:
            labels.append(parts[0])
    return labels


def _coco_labels(document: dict) -> dict[str, list[str]]:
    """Return the COCO labels keyed by relative file name and by bare stem."""
    categories = {
        item.get("id"): str(item.get("name", f"class_{item.get('id')}"))
        for item in document.get("categories", [])
    }
    names = {
        item.get("id"): str(item.get("file_name", "")).replace("\\", "/")
        for item in document.get("images", [])
    }
    by_image: dict[int, list[str]] = {image_id: [] for image_id in names}
    for item in document.get("annotations", []):
        labels = by_image.get(item.get("image_id"))
        if labels is None:
            continue
        name = categories.get(item.get("category_id"))
        if name:
            labels.append(name)
    by_relative: dict[str, list[str]] = {
        names[image_id]: labels for image_id, labels in by_image.items() if names[image_id]
    }
    stems: dict[str, list[str]] = {}
    for relative in by_relative:
        stems.setdefault(Path(relative).stem, []).append(relative)
    result = dict(by_relative)
    for stem, keys in stems.items():
        if len(keys) == 1:
            result.setdefault(f"\0{stem}", by_relative[keys[0]])
    return result


def _yolo_class_names(root: Path) -> list[str]:
    """The dataset's own class table: classes.txt first, data.yaml second.

    This is the same precedence the editor uses, and it matters here: naming a
    ground-truth box by the *model's* class list would hide exactly the kind of
    class-set disagreement the comparison exists to find.
    """
    for candidate in (root / "classes.txt", root.parent / "classes.txt"):
        if candidate.is_file():
            names = [
                line.strip()
                for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if names:
                return names
    try:
        return [name for name in yolo_class_names(root) if name]
    except (OSError, ValueError):
        return []


class GroundTruth:
    """The dataset's labels, keyed so an image path finds its own."""

    def __init__(self, detected) -> None:
        self.format_name = detected.format_name
        self.image_root = Path(detected.image_dir)
        self._by_relative: dict[str, list[str]] = {}
        self._class_names: list[str] = []
        self._documents: list[Path] = []
        if self.format_name == "coco":
            self._documents = self._coco_documents(Path(detected.annotation_dir))
            self._by_relative = _coco_labels(
                _read_json(self._documents[0]) if self._documents else {}
            )
        else:
            suffix = ".xml" if self.format_name == "voc" else ".txt"
            self._by_relative = _label_index(Path(detected.annotation_dir), suffix)
            if self.format_name != "voc":
                self._class_names = _yolo_class_names(Path(detected.root))

    @property
    def documents(self) -> list[Path]:
        return self._documents

    @staticmethod
    def _coco_documents(annotation_dir: Path) -> list[Path]:
        if annotation_dir.is_file():
            return [annotation_dir]
        for name in ("annotations.json", "instances.json"):
            candidate = annotation_dir / name
            if candidate.is_file():
                return [candidate]
        return sorted(path for path in annotation_dir.glob("*.json") if path.is_file())

    def labels(self, image: Path) -> list[str]:
        try:
            relative = image.relative_to(self.image_root).with_suffix("").as_posix()
        except ValueError:
            relative = _strip_extension(image.name)
        found = self._by_relative.get(relative)
        if found is None:
            found = self._by_relative.get(f"\0{image.stem}")
        if found is None:
            return []
        if self.format_name != "yolo" or not self._class_names:
            return list(found)
        return [self._yolo_name(token) for token in found]

    def _yolo_name(self, token: str) -> str:
        """YOLO stores a class id; the dataset's table turns it into a name.

        An id the table does not cover is reported as-is rather than dropped.
        The label file does name a class, and a comparison that silently loses
        an out-of-table class would report the image as unlabelled.
        """
        try:
            class_id = int(token)
        except ValueError:
            return token
        if 0 <= class_id < len(self._class_names):
            return self._class_names[class_id]
        return token


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------ inference ------------------------------


def _model_presets(detector: YoloOnnxDetector) -> list[LabelPreset]:
    """The class list the model itself declares.

    A model with no ``names`` metadata cannot be compared: its class column
    would be an index into nothing, and every image would come back unlabelled
    without a single thing being wrong with the dataset.
    """
    names = [name for name in detector.class_names if name]
    if not names:
        raise ValueError("模型未声明类别名称（names 元数据缺失），无法比对标签。")
    return [LabelPreset(name, index, label_color(name)) for index, name in enumerate(names)]


def _predict_labels(
    detector: YoloOnnxDetector,
    presets: list[LabelPreset],
    image_path: Path,
    threshold: float,
    input_size: int,
    nms_threshold: float,
) -> list[tuple[str, float]]:
    """Every label the model finds, with its best score, best first."""
    with Image.open(image_path) as handle:
        annotations = detector.predict(
            handle.convert("RGB"), presets, input_size, threshold, nms_threshold,
            letterbox=True,
        )
    best: dict[str, float] = {}
    for annotation in annotations:
        score = float(annotation.confidence or 0.0)
        if score > best.get(annotation.label, -1.0):
            best[annotation.label] = score
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


# ------------------------------- driver -------------------------------


def _images_under(root: Path) -> list[Path]:
    """Every image, in a stable order so two runs report the same file order."""
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def compare_dataset(
    model_path: Path,
    dataset_dir: Path,
    confidence_threshold: float = 0.5,
    *,
    input_size: int = 640,
    nms_threshold: float = 0.45,
    workers: int = DEFAULT_WORKERS,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> CompareReport:
    """Run the model over ``dataset_dir`` and write the disagreements to a file.

    Returns the report whether or not anything disagreed. A cancelled run
    returns with ``cancelled`` set and writes no file: a partial report reads
    exactly like a clean one, and only one of those is worth keeping.
    """
    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)
    if not model_path.is_file():
        raise ValueError(f"模型文件不存在：{model_path}")
    if not dataset_dir.is_dir():
        raise ValueError(f"数据集目录不存在：{dataset_dir}")
    threshold = max(0.0, min(1.0, float(confidence_threshold)))
    try:
        detected = DatasetDetector.detect(dataset_dir, allow_plain_images=False)
    except ValueError as exc:
        raise ValueError(f"未能识别数据集格式：{dataset_dir}（{exc}）") from exc

    ground_truth = GroundTruth(detected)
    detector = YoloOnnxDetector()
    detector.load(model_path)
    presets = _model_presets(detector)
    images = _images_under(ground_truth.image_root)

    report = CompareReport(
        output_path=report_path_for(detected.root),
        model_path=model_path,
        dataset_dir=detected.root,
        format_name=detected.format_name,
        threshold=threshold,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total=len(images),
    )
    if progress_callback:
        progress_callback(0, report.total)

    cases: list[CompareCase] = []
    tallies: dict[str, list[int]] = {}
    pool = max(1, int(workers))
    processed = 0
    with ThreadPoolExecutor(max_workers=pool, thread_name_prefix="compare") as executor:
        queue = iter(images)
        pending: dict[Future, Path] = {}
        drained = False
        while pending or not drained:
            while not drained and len(pending) < pool * WINDOW_FACTOR:
                if cancel_callback and cancel_callback():
                    report.cancelled = True
                    drained = True
                    break
                try:
                    image = next(queue)
                except StopIteration:
                    drained = True
                    break
                pending[executor.submit(
                    _compare_one, detector, presets, ground_truth, image,
                    threshold, input_size, nms_threshold,
                )] = image
            if not pending:
                break
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                image = pending.pop(future)
                processed += 1
                try:
                    case = future.result()
                except Exception as exc:  # one unreadable image must not end the run
                    report.skipped += 1
                    report.error_count += 1
                    if len(report.errors) < MAX_ERRORS:
                        report.errors.append(f"{image.name}：{exc}")
                    continue
                if case.matched:
                    report.matched += 1
                else:
                    report.mismatched += 1
                    cases.append(case)
                for label in set(case.expected):
                    tally = tallies.setdefault(label, [0, 0])
                    tally[0] += 1
                    if not case.matched:
                        tally[1] += 1
                if progress_callback:
                    progress_callback(processed, report.total)

    report.per_class = {
        label: ClassTally(images, mismatched)
        for label, (images, mismatched) in sorted(
            tallies.items(), key=lambda item: (-item[1][1], item[0]),
        )
    }
    cases.sort(key=lambda case: case.name)
    if not report.cancelled:
        _write_report(report, cases)
    return report


def _compare_one(
    detector: YoloOnnxDetector,
    presets: list[LabelPreset],
    ground_truth: GroundTruth,
    image: Path,
    threshold: float,
    input_size: int,
    nms_threshold: float,
) -> CompareCase:
    predicted = _predict_labels(detector, presets, image, threshold, input_size, nms_threshold)
    return CompareCase(
        name=image.name,
        expected=tuple(sorted(set(ground_truth.labels(image)))),
        predicted=tuple(label for label, _score in predicted),
        scores=tuple(score for _label, score in predicted),
    )


def render_report(report: CompareReport, cases: list[CompareCase]) -> str:
    """The report text. Every line that is not a mismatch starts with ``#``.

    That one rule is the whole contract for a reader: anything else in the file
    is one image that disagreed, in the form
    ``文件名 文件label -> 识别label 识别score``.
    """
    lines = [
        "# 数据对比报告",
        f"# 模型：{report.model_path}",
        f"# 数据集：{report.dataset_dir}",
        f"# 数据格式：{report.display_format}",
        f"# 识别阈值：{report.threshold:g}",
        f"# 生成时间：{report.generated_at}",
        f"# 总图片：{report.total}  一致：{report.matched}  不一致：{report.mismatched}"
        f"  跳过：{report.skipped}  不一致率：{report.mismatch_rate * 100:.2f}%",
    ]
    if report.error_count:
        lines.append(f"# 读取失败：{report.error_count}")
        lines.extend(f"#   {error}" for error in report.errors)
        if report.error_count > len(report.errors):
            lines.append(f"#   ...另有 {report.error_count - len(report.errors)} 个未列出")
    disagreeing = [item for item in report.per_class.items() if item[1].mismatched]
    if disagreeing:
        lines.append("# 各类不一致（不一致 / 该类图片数）：")
        lines.extend(
            f"#   {label}  {tally.mismatched}/{tally.images}  {tally.rate * 100:.2f}%"
            for label, tally in disagreeing
        )
    lines.append(f"# 格式：文件名 文件label -> 识别label 识别score（无 label 记作 {NO_LABEL}）")
    lines.extend(case.line() for case in cases)
    return "\n".join(lines) + "\n"


def _write_report(report: CompareReport, cases: list[CompareCase]) -> None:
    report.output_path.parent.mkdir(parents=True, exist_ok=True)
    # ``utf-8-sig``: this file is opened in Notepad and Excel on Windows, and
    # the file names in it are not all ASCII.
    report.output_path.write_text(render_report(report, cases), encoding="utf-8-sig")
