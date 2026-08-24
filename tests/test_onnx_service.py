import numpy as np
from PIL import Image

from src.models.annotation import LabelPreset, ShapeType
from src.services.onnx_service import YoloOnnxDetector


class _Input:
    name = "images"
    shape = [1, 3, 640, 640]


class _Session:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [_Input()]

    def run(self, names, values):
        return [self.output]


def _detector(output):
    detector = YoloOnnxDetector(_Session(output))
    detector.task = "pose"
    detector.class_names = ["person"]
    detector.keypoint_names = ["nose", "eye"]
    return detector


def test_raw_channel_first_pose_output_is_decoded():
    # [cx, cy, w, h, class score, x1, y1, v1, x2, y2, v2]
    row = np.array([320, 320, 200, 100, 0.9, 300, 310, 0.9, 340, 315, 0.2], dtype=np.float32)
    output = row.reshape(1, -1, 1)

    annotations = _detector(output).predict(
        Image.new("RGB", (1280, 720)),
        [LabelPreset("person", 0, "#00e5ff")],
        640, 0.25, 0.45,
    )

    assert len(annotations) == 1
    assert annotations[0].shape_type == ShapeType.KEYPOINT
    assert [item.name for item in annotations[0].keypoints] == ["nose", "eye"]
    assert [item.visibility for item in annotations[0].keypoints] == [2, 1]
    assert annotations[0].points[0].x() == 440.0
    assert annotations[0].points[1].x() == 840.0


def test_end_to_end_pose_output_remains_supported():
    # [x1, y1, x2, y2, score, class, keypoints...]
    row = np.array([100, 200, 300, 400, 0.8, 0, 150, 250, 0.8, 250, 350, 0.7], dtype=np.float32)

    annotations = _detector(row.reshape(1, 1, -1)).predict(
        Image.new("RGB", (640, 640)),
        [LabelPreset("person", 0, "#00e5ff")],
        640, 0.25, 0.45,
    )

    assert len(annotations) == 1
    assert len(annotations[0].keypoints) == 2


def test_raw_segmentation_output_decodes_polygon_mask():
    # One class and one prototype channel: [cx, cy, w, h, score, coefficient].
    detections = np.array([320, 320, 320, 320, 0.95, 10.0], dtype=np.float32).reshape(1, -1, 1)
    prototypes = np.full((1, 1, 160, 160), -10.0, dtype=np.float32)
    prototypes[0, 0, 50:110, 50:110] = 1.0
    session = _Session(detections)
    session.run = lambda names, values: [detections, prototypes]
    detector = YoloOnnxDetector(session)
    detector.task = "segment"
    detector.class_names = ["person"]

    annotations = detector.predict(
        Image.new("RGB", (1280, 720)),
        [LabelPreset("person", 0, "#00e5ff")],
        640, 0.25, 0.45,
    )

    assert len(annotations) == 1
    assert annotations[0].shape_type == ShapeType.POLYGON
    assert len(annotations[0].points) >= 4
    assert all(0 <= point.x() <= 1280 and 0 <= point.y() <= 720 for point in annotations[0].points)


def test_segmentation_requires_prototype_output():
    detector = YoloOnnxDetector(_Session(np.zeros((1, 6, 1), dtype=np.float32)))
    detector.task = "segment"
    detector.class_names = ["person"]

    try:
        detector.predict(
            Image.new("RGB", (640, 640)),
            [LabelPreset("person", 0, "#00e5ff")], 640,
        )
    except ValueError as error:
        assert "prototype" in str(error)
    else:
        raise AssertionError("segmentation without prototypes must fail")
