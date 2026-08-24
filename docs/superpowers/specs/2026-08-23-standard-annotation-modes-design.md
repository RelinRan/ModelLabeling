# Standard Annotation Modes Design

## Goal

Support rectangle, square, polygon, and keypoint annotation while writing only official COCO, YOLO task-specific, and Pascal VOC standard formats.

## Format policy

The application never writes private fields into dataset files and never silently downgrades unsupported geometry.

- COCO supports bbox, segmentation, and keypoints. A square is stored as a bbox with equal width and height.
- YOLO Detection supports bbox only.
- YOLO Segmentation supports bbox and polygon segmentation.
- YOLO Pose supports official pose rows containing an outer bbox and keypoint triplets; a bbox-only object is not a valid YOLO Pose label.
- Pascal VOC supports the official `bndbox` representation only.

The UI and save/conversion services must reject unsupported combinations before writing.

## Internal model

The in-memory model is format-independent. `Annotation` has a `ShapeType`, image-space points, optional bbox, and optional named keypoints. Keypoint schemas define ordered names and skeleton edges. The default schema is COCO Person 17, loaded from pose-model metadata when available.

## Components

- `models/annotation.py`: geometry enum, keypoint value objects, backwards-compatible serialization.
- `services/format_capabilities.py`: official format/task capability registry and validation errors.
- `services/annotation_service.py`: official readers/writers selected by format task.
- `services/onnx_service.py`: detect/pose model metadata detection and output decoding.
- `widgets/canvas_view.py`: format-aware drawing modes and keypoint rendering/editing.
- `services/conversion_service.py`: capability validation before conversion; no lossy fallback.

## Pose model

Ultralytics pose ONNX metadata is read from the model. For `pose.onnx`, output rows are decoded as bbox, confidence, class, and 17 triples of x/y/visibility. Coordinates are mapped back to original image pixels and stored as one keypoint annotation with an outer bbox.

## Compatibility

Existing rectangle/square/polygon project data remains readable. Missing keypoint fields deserialize as empty. Existing dataset formats continue to work under their official capability profile.

## Acceptance criteria

1. All four internal annotation modes can be represented without data loss.
2. Unsupported modes are disabled or rejected for the selected official format.
3. COCO bbox/segmentation/keypoints round-trip correctly.
4. YOLO Detection, Segmentation, and Pose files follow their official layouts.
5. Pascal VOC reads and writes official `bndbox` only.
6. `pose.onnx` produces 17-keypoint annotations.
7. Conversion rejects unsupported geometry instead of silently changing it.
8. Legacy rectangle annotations remain loadable.
