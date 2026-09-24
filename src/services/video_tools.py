from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.models.annotation import LabelPreset
from .conversion_service import ConversionOptions, ConversionService
from .dataset_detector import DatasetDetector

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg", ".m4v", ".webm"}

def _flat_name(path: Path, root: Path) -> str:
    """Return a filesystem-safe name without retaining source subdirectories."""
    relative = path.relative_to(root).with_suffix("")
    value = "__".join(relative.parts)
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._") or "item"


@dataclass
class SynthesisReport:
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] | None = None


def extract_video_frames(source_dir: Path, output_dir: Path, target_fps: int = 1, progress_callback=None) -> tuple[int, int]:
    import cv2
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise ValueError(f"\u89c6\u9891\u76ee\u5f55\u4e0d\u5b58\u5728\uff1a{source_dir}")
    target_fps = max(1, int(target_fps))
    videos = sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("\u63d0\u5e27\u76ee\u5f55\u5fc5\u987b\u4e3a\u7a7a\uff0c\u63d0\u53d6\u540e\u53ea\u4fdd\u7559\u56fe\u7247\u6587\u4ef6\u3002")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    total = len(videos)
    if progress_callback:
        progress_callback(0, total)
    for video_index, video in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            if progress_callback:
                progress_callback(video_index, total)
            continue
        fps = capture.get(cv2.CAP_PROP_FPS) or float(target_fps)
        stride = max(1, round(fps / target_fps))
        safe_stem = _flat_name(video, source_dir)
        destination = output_dir
        index = 0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                target = destination / f"{safe_stem}_{index:06d}.jpg"
                if cv2.imwrite(str(target), frame):
                    saved += 1
                    index += 1
            frame_index += 1
        capture.release()
        if progress_callback:
            progress_callback(video_index, total)
    return len(videos), saved

def synthesize_dataset(source_dir: Path, output_dir: Path, format_name: str, presets: list[LabelPreset], progress_callback=None) -> SynthesisReport:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise ValueError(f"\u89c6\u9891\u5e27\u76ee\u5f55\u4e0d\u5b58\u5728\uff1a{source_dir}")
    if source_dir.resolve() == output_dir.resolve() or source_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("\u5408\u6210\u76ee\u6807\u4e0d\u80fd\u4e0e\u6e90\u76ee\u5f55\u76f8\u540c\u6216\u4f4d\u4e8e\u6e90\u76ee\u5f55\u5185\u90e8\u3002")
    plain_image_source = True
    source_format = "yolo"
    try:
        detected = DatasetDetector.detect(source_dir, allow_plain_images=False)
        if detected.root.resolve() == source_dir.resolve() and detected.annotation_dir.is_dir():
            source_format = detected.format_name
            plain_image_source = False
    except ValueError:
        pass  # plain extracted frames have no annotations
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("\u5408\u6210\u76ee\u6807\u76ee\u5f55\u5fc5\u987b\u4e3a\u7a7a\uff0c\u4ee5\u907f\u514d\u8986\u76d6\u5df2\u6709\u6570\u636e\u3002")
    output_dir.mkdir(parents=True, exist_ok=True)
    options = ConversionOptions(source_format, source_dir, format_name, output_dir, list(presets), overwrite=False, force_structured_output=True, plain_image_source=plain_image_source, flatten_output=True)
    return ConversionService().convert(options, progress_callback=progress_callback)
