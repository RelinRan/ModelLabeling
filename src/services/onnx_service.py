from __future__ import annotations

from pathlib import Path
from typing import Any
import ast

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, Keypoint, LabelPreset, ShapeType, label_color


class YoloOnnxDetector:
    def __init__(self, session: Any | None = None) -> None:
        self.session = session
        self.input_name: str | None = None
        self.task = "detect"
        self.keypoint_names: list[str] = []
        self.class_names: list[str] = []

    def load(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.task = str(metadata.get("task", "detect")).strip().lower()
        self.class_names = self._metadata_names(metadata.get("names", ""))
        self.keypoint_names = self._metadata_keypoint_names(metadata.get("kpt_names", ""))

    @staticmethod
    def _metadata_names(value: str) -> list[str]:
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, dict):
                return [str(parsed[key]) for key in sorted(parsed)]
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        return []

    @classmethod
    def _metadata_keypoint_names(cls, value: str) -> list[str]:
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, dict):
                parsed = next(iter(parsed.values()), [])
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        return []

    def predict(
        self,
        image: Image.Image,
        presets: list[LabelPreset],
        input_size: int | tuple[int, int] = 640,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
    ) -> list[Annotation]:
        if self.session is None:
            raise RuntimeError("ONNX model is not loaded")
        input_name = self.input_name or self.session.get_inputs()[0].name
        rgb = image.convert("RGB")
        original_width, original_height = rgb.size
        if isinstance(input_size, int):
            input_width = input_height = input_size
        else:
            input_width, input_height = input_size
        # Respect fixed dimensions declared by the ONNX graph. This keeps a
        # model exported at 960x960 compatible with the app's configurable
        # inference size without producing an invalid runtime input tensor.
        graph_shape = self.session.get_inputs()[0].shape if hasattr(self.session.get_inputs()[0], "shape") else None
        if graph_shape and len(graph_shape) >= 4:
            graph_height, graph_width = graph_shape[-2], graph_shape[-1]
            if isinstance(graph_width, int) and graph_width > 0:
                input_width = graph_width
            if isinstance(graph_height, int) and graph_height > 0:
                input_height = graph_height
        input_width = max(1, int(input_width))
        input_height = max(1, int(input_height))
        resized = rgb.resize((input_width, input_height))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]
        output = self.session.run(None, {input_name: array})[0]
        if self.task == "pose":
            return self._predict_pose(output, original_width, original_height, input_width, input_height, presets, confidence_threshold, nms_threshold)
        rows = self._decode_output(np.asarray(output), len(presets))
        candidates = []
        for x1, y1, x2, y2, score, class_id in rows:
            if score < confidence_threshold or not 0 <= class_id < len(presets):
                continue
            candidates.append((x1, y1, x2, y2, score, class_id))
        selected = self._nms(candidates, nms_threshold)
        results: list[Annotation] = []
        for x1, y1, x2, y2, score, class_id in selected:
            preset = presets[class_id]
            # ONNX exports may return either input-pixel coordinates or
            # normalized coordinates. Normalize before mapping to the source image.
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * input_width, x2 * input_width
                y1, y2 = y1 * input_height, y2 * input_height
            sx = original_width / input_width
            sy = original_height / input_height
            left = max(0.0, min(float(original_width), x1 * sx))
            top = max(0.0, min(float(original_height), y1 * sy))
            right = max(0.0, min(float(original_width), x2 * sx))
            bottom = max(0.0, min(float(original_height), y2 * sy))
            if right <= left or bottom <= top:
                continue
            results.append(
                Annotation(
                    ShapeType.RECTANGLE,
                    preset.name,
                    [QPointF(left, top), QPointF(right, bottom)],
                    color=label_color(preset.name),
                    confidence=score,
                    source="onnx",
                )
            )
        return results

    def _predict_pose(
        self,
        output: np.ndarray,
        original_width: int,
        original_height: int,
        input_width: int,
        input_height: int,
        presets: list[LabelPreset],
        confidence_threshold: float,
        nms_threshold: float,
    ) -> list[Annotation]:
        data = np.asarray(output)
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2 or data.shape[1] < 8:
            raise ValueError(f"unsupported YOLO Pose output shape: {data.shape}")
        candidates = []
        keypoint_count = (data.shape[1] - 6) // 3
        names = self.keypoint_names or [f"keypoint_{index}" for index in range(keypoint_count)]
        for row in data:
            x1, y1, x2, y2, score, class_id = map(float, row[:6])
            if score < confidence_threshold or not 0 <= int(class_id) < len(presets):
                continue
            raw_keypoints = []
            for index in range(keypoint_count):
                x, y, visibility = map(float, row[6 + index * 3:9 + index * 3])
                if max(abs(x), abs(y)) <= 1.5:
                    x *= input_width
                    y *= input_height
                visible = 2 if visibility >= 0.5 else 1 if visibility > 0 else 0
                raw_keypoints.append((names[index] if index < len(names) else f"keypoint_{index}", x, y, visible))
            candidates.append((x1, y1, x2, y2, score, int(class_id), raw_keypoints))
        selected = self._nms(candidates, nms_threshold)
        results: list[Annotation] = []
        for x1, y1, x2, y2, score, class_id, raw_keypoints in selected:
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * input_width, x2 * input_width
                y1, y2 = y1 * input_height, y2 * input_height
            sx = original_width / input_width
            sy = original_height / input_height
            left = max(0.0, min(float(original_width), x1 * sx))
            top = max(0.0, min(float(original_height), y1 * sy))
            right = max(0.0, min(float(original_width), x2 * sx))
            bottom = max(0.0, min(float(original_height), y2 * sy))
            keypoints = [
                Keypoint(name, QPointF(max(0.0, min(float(original_width), x * sx)), max(0.0, min(float(original_height), y * sy))), visibility)
                for name, x, y, visibility in raw_keypoints
            ]
            results.append(Annotation(
                ShapeType.KEYPOINT,
                presets[class_id].name,
                [QPointF(left, top), QPointF(right, bottom)],
                color=label_color(presets[class_id].name),
                confidence=score,
                source="onnx",
                keypoints=keypoints,
                schema_name="COCO Person 17" if len(keypoints) == 17 else "YOLO Pose",
            ))
        return results

    @staticmethod
    def _decode_output(output: np.ndarray, class_count: int) -> list[tuple[float, float, float, float, float, int]]:
        data = output
        if data.ndim == 3:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(f"unsupported YOLO output shape: {output.shape}")
        if 6 <= data.shape[0] < data.shape[1] and data.shape[0] <= 128:
            data = data.T
        rows: list[tuple[float, float, float, float, float, int]] = []
        for row in data:
            if len(row) < 6:
                continue
            width = len(row)
            # Some exports include NMS and return [x1, y1, x2, y2, score, class].
            if width == 6 and float(row[5]).is_integer() and 0 <= int(row[5]) < max(class_count, 1):
                x1, y1, x2, y2, score, class_id = map(float, row)
                rows.append((x1, y1, x2, y2, float(score), int(class_id)))
                continue

            x, y, w, h = map(float, row[:4])
            # YOLOv5-style: [cx, cy, w, h, objectness, class scores].
            # YOLOv8-style: [cx, cy, w, h, class scores] (no objectness).
            # Prefer the explicit width when the preset count matches the
            # model. For a model whose class list is larger than the current
            # preset list (a common YOLOv8 case), the extra column is not an
            # objectness score, so use the class-only layout.
            has_objectness = width == 5 + class_count or (width == 6 and class_count <= 1)
            if has_objectness:
                objectness = float(row[4])
                class_scores = row[5:]
                class_id = int(np.argmax(class_scores)) if len(class_scores) else 0
                score = objectness * (float(class_scores[class_id]) if len(class_scores) else 1.0)
            else:
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores)) if len(class_scores) else 0
                score = float(class_scores[class_id]) if len(class_scores) else 1.0
            rows.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2, score, class_id))
        return rows

    @staticmethod
    def _nms(
        boxes: list[tuple[float, float, float, float, float, int]],
        threshold: float,
    ) -> list[tuple[float, float, float, float, float, int]]:
        remaining = sorted(boxes, key=lambda item: item[4], reverse=True)
        selected = []
        while remaining:
            current = remaining.pop(0)
            selected.append(current)
            remaining = [
                item for item in remaining
                if item[5] != current[5] or YoloOnnxDetector._iou(current, item) <= threshold
            ]
        return selected

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[2], b[2])
        bottom = min(a[3], b[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0
