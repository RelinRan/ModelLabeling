# Standard Annotation Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standard-only support for rectangle, square, polygon, and keypoint annotations.

**Architecture:** Keep the existing image/index/task infrastructure. Extend the format-independent annotation model, add a capability registry for official format/task variants, decode Ultralytics YOLO Pose, and validate every write/conversion before output.

**Tech Stack:** Python, PySide6, SQLite, Pillow, NumPy, ONNX Runtime, official COCO/YOLO/VOC layouts.

---

### Task 1: Extend the annotation model

**Files:**
- Modify: `src/models/annotation.py`
- Modify: `src/models/project.py`
- Test: `tests/test_annotation_model.py` (create if test infrastructure is added)

- [ ] Add `ShapeType.KEYPOINT` and a serializable `Keypoint` value object with name, point, and visibility.
- [ ] Add optional keypoints/schema fields to `Annotation` while preserving old dictionaries without those fields.
- [ ] Validate keypoint visibility and allow a keypoint annotation to contain an outer bbox.
- [ ] Run `python -m compileall -q app.py src`.

### Task 2: Add official format capability validation

**Files:**
- Create: `src/services/format_capabilities.py`
- Modify: `src/models/project.py`
- Modify: `src/services/annotation_service.py`
- Modify: `src/services/conversion_service.py`

- [ ] Register COCO, YOLO Detection, YOLO Segmentation, YOLO Pose, and Pascal VOC capabilities.
- [ ] Normalize existing settings values to an explicit task variant without breaking old projects.
- [ ] Add a validation function that reports every unsupported shape before saving or converting.
- [ ] Ensure conversion has no lossy fallback.

### Task 3: Implement official YOLO Pose decoding and writing

**Files:**
- Modify: `src/services/onnx_service.py`
- Modify: `src/services/annotation_service.py`
- Modify: `src/widgets/main_window.py`

- [ ] Read ONNX metadata for task, names, `kpt_shape`, and `kpt_names`.
- [ ] Decode `[1, 300, 57]` pose rows and map coordinates to original pixels.
- [ ] Read/write official YOLO Pose normalized rows and dataset YAML metadata.
- [ ] Use `E:\Dataset\model\onnx\pose.onnx` for a real inference smoke test.

### Task 4: Implement standard COCO segmentation/keypoint I/O

**Files:**
- Modify: `src/services/annotation_service.py`
- Modify: `src/services/dataset_index.py`
- Modify: `src/services/conversion_service.py`

- [ ] Read and write COCO `bbox`, `segmentation`, `keypoints`, `num_keypoints`, category keypoint names, and skeleton.
- [ ] Preserve mixed annotations in one official COCO JSON.
- [ ] Index labels without discarding polygon/keypoint geometry.
- [ ] Validate COCO round trips with synthetic data.

### Task 5: Add canvas keypoint mode and capability-aware UI

**Files:**
- Modify: `src/widgets/canvas_view.py`
- Modify: `src/widgets/main_window.py`
- Create: `src/widgets/keypoint_panel.py`
- Modify: `src/widgets/settings_dialog.py`

- [ ] Add keypoint drawing, selection, dragging, visibility, and skeleton rendering.
- [ ] Keep all coordinates in original image space so zoom does not change saved values.
- [ ] Disable modes unsupported by the active official format/task.
- [ ] Show the active keypoint schema and model-derived schema for pose models.

### Task 6: Conversion and regression verification

**Files:**
- Modify: `src/widgets/conversion_dialog.py`
- Modify: `README.md`
- Test: `tests/` focused smoke tests

- [ ] Add explicit source/target task variants to conversion validation; reject bbox-only objects when the official YOLO Pose target has no keypoint triplets.
- [ ] Test supported conversions and rejection of unsupported geometry.
- [ ] Test old rectangle projects and datasets.
- [ ] Run compile checks, format round trips, pose inference, and Qt smoke tests.
