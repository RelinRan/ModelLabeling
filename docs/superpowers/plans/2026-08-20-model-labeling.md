# ModelLabeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows-first PySide6 desktop annotation tool with manual rectangle/square/polygon labeling, YOLO/VOC persistence and conversion, YOLO ONNX auto-labeling, image search/status tracking, statistics, zooming, autosave, and keyboard shortcuts.

**Architecture:** Keep a single PySide6 application shell with focused model, service, and widget modules. Store annotations in original-image pixel coordinates, render them through a `QGraphicsView` transform, and serialize through format-specific services. Run ONNX inference in a worker thread and communicate with the UI through Qt signals.

**Tech Stack:** Python 3.11+, PySide6, Pillow, numpy, pytest, pytest-qt, optional onnxruntime, optional opencv-python.

---

## File Map

Create the following files:

- `E:\Python\ModelLabeling\app.py`: application entry point and exception hook.
- `E:\Python\ModelLabeling\requirements.txt`: runtime and test dependencies.
- `E:\Python\ModelLabeling\src\__init__.py`
- `E:\Python\ModelLabeling\src\models\__init__.py`
- `E:\Python\ModelLabeling\src\models\annotation.py`: `ShapeType`, `Annotation`, `LabelPreset`.
- `E:\Python\ModelLabeling\src\models\project.py`: project settings, image records, statistics.
- `E:\Python\ModelLabeling\src\utils\geometry.py`: coordinate conversion, bounding boxes, square constraints.
- `E:\Python\ModelLabeling\src\services\annotation_service.py`: VOC/YOLO/internal metadata read/write.
- `E:\Python\ModelLabeling\src\services\image_service.py`: image discovery, metadata, fuzzy search, status.
- `E:\Python\ModelLabeling\src\services\conversion_service.py`: VOC/YOLO batch conversion.
- `E:\Python\ModelLabeling\src\services\project_service.py`: JSON project settings and autosave.
- `E:\Python\ModelLabeling\src\services\onnx_service.py`: YOLO ONNX loading, preprocessing, NMS, result mapping.
- `E:\Python\ModelLabeling\src\widgets\canvas_view.py`: graphics scene, drawing/editing, zoom/pan.
- `E:\Python\ModelLabeling\src\widgets\image_list_panel.py`: thumbnail list, search, status filter.
- `E:\Python\ModelLabeling\src\widgets\preset_panel.py`: label presets and current-label assignment.
- `E:\Python\ModelLabeling\src\widgets\stats_panel.py`: progress and per-label counts.
- `E:\Python\ModelLabeling\src\widgets\settings_dialog.py`: paths, format, visual settings, model settings.
- `E:\Python\ModelLabeling\src\widgets\conversion_dialog.py`: conversion controls and progress.
- `E:\Python\ModelLabeling\src\widgets\main_window.py`: composition, navigation, save flow, shortcuts, worker lifecycle.
- `E:\Python\ModelLabeling\tests\conftest.py`
- `E:\Python\ModelLabeling\tests\test_annotation_model.py`
- `E:\Python\ModelLabeling\tests\test_geometry.py`
- `E:\Python\ModelLabeling\tests\test_annotation_service.py`
- `E:\Python\ModelLabeling\tests\test_conversion_service.py`
- `E:\Python\ModelLabeling\tests\test_image_service.py`
- `E:\Python\ModelLabeling\tests\test_project_service.py`
- `E:\Python\ModelLabeling\tests\test_onnx_service.py`
- `E:\Python\ModelLabeling\tests\test_main_window.py`

## Task 1: Bootstrap the Python Application

**Files:**
- Create: `E:\Python\ModelLabeling\requirements.txt`
- Create: `E:\Python\ModelLabeling\app.py`
- Create: `E:\Python\ModelLabeling\src\__init__.py`
- Create: `E:\Python\ModelLabeling\src\models\__init__.py`
- Create: `E:\Python\ModelLabeling\tests\conftest.py`

- [ ] **Step 1: Add dependency declarations**

Include PySide6, Pillow, numpy, pytest, and pytest-qt. Declare `onnxruntime` and `opencv-python` as optional comments or an optional install line so manual annotation works without them.

- [ ] **Step 2: Add the application entry point**

Implement `main()` to create `QApplication`, set application metadata, install a logging/exception hook, create `MainWindow`, show it, and return `app.exec()`. Import `MainWindow` lazily enough that test collection can still report a missing optional ONNX dependency clearly.

- [ ] **Step 3: Add test fixtures**

Configure pytest-qt and a temporary image fixture generated with Pillow. Run `pytest -q`; expected result is collection success with no tests yet.

## Task 2: Implement Annotation and Project Models

**Files:**
- Create: `E:\Python\ModelLabeling\src\models\annotation.py`
- Create: `E:\Python\ModelLabeling\src\models\project.py`
- Test: `E:\Python\ModelLabeling\tests\test_annotation_model.py`

- [ ] **Step 1: Write failing model tests**

Cover JSON round-tripping, color normalization, empty confidence for manual annotations, polygon point preservation, stable label IDs, and statistics for total images, labeled images, total labels, and per-label counts.

- [ ] **Step 2: Run the focused tests**

Run: `pytest tests/test_annotation_model.py -q`
Expected: FAIL because model classes do not exist.

- [ ] **Step 3: Implement dataclasses and enums**

Define `ShapeType`, `Annotation`, `LabelPreset`, `ImageRecord`, `ProjectSettings`, and `ProjectState`. Use explicit `to_dict()`/`from_dict()` methods, validate non-empty labels and non-negative class IDs, and make statistics a pure method over image records.

- [ ] **Step 4: Run the focused tests**

Run: `pytest tests/test_annotation_model.py -q`
Expected: PASS.

## Task 3: Implement Geometry and Coordinate Invariants

**Files:**
- Create: `E:\Python\ModelLabeling\src\utils\geometry.py`
- Test: `E:\Python\ModelLabeling\tests\test_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

Test `view_to_image`, `image_to_view`, clamping points to image bounds, rectangle normalization, square constraints in all drag directions, polygon bounds, and YOLO normalized coordinate conversion with a one-pixel tolerance.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_geometry.py -q`
Expected: FAIL with missing geometry functions.

- [ ] **Step 3: Implement pure geometry functions**

Provide typed functions:

```python
def normalize_rect(start: QPointF, end: QPointF) -> QRectF: ...
def constrain_square(start: QPointF, end: QPointF) -> QRectF: ...
def image_to_view(point: QPointF, scale: float, offset: QPointF) -> QPointF: ...
def view_to_image(point: QPointF, scale: float, offset: QPointF) -> QPointF: ...
def polygon_bounds(points: list[QPointF]) -> QRectF: ...
def clamp_points(points: list[QPointF], width: int, height: int) -> list[QPointF]: ...
```

Use only original-image coordinates in persisted objects.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_geometry.py -q`
Expected: PASS.

## Task 4: Implement VOC, YOLO, and Internal Metadata Persistence

**Files:**
- Create: `E:\Python\ModelLabeling\src\services\annotation_service.py`
- Test: `E:\Python\ModelLabeling\tests\test_annotation_service.py`

- [ ] **Step 1: Write failing format tests**

Create a temporary image and annotations, save/load VOC and YOLO, assert class names/IDs and boxes survive round-trip, assert malformed files return a structured error, and assert polygon points are written to `annotations.json` while exported formats use the polygon bounding box.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_annotation_service.py -q`
Expected: FAIL because the service is missing.

- [ ] **Step 3: Implement the service**

Expose:

```python
class AnnotationService:
    def load(self, image_path: Path, annotation_dir: Path, settings: ProjectSettings) -> LoadResult: ...
    def save(self, image_path: Path, annotations: list[Annotation], annotation_dir: Path, settings: ProjectSettings) -> SaveResult: ...
    def load_internal_metadata(self, metadata_path: Path) -> dict[str, list[Annotation]]: ...
    def save_internal_metadata(self, metadata_path: Path, data: dict[str, list[Annotation]]) -> None: ...
```

Use `xml.etree.ElementTree` for VOC and line-based numeric parsing for YOLO. Reject invalid class IDs, malformed numbers, out-of-bounds values, and mismatched image dimensions with actionable error messages.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_annotation_service.py -q`
Expected: PASS.

## Task 5: Implement Image Discovery, Search, and Project Settings

**Files:**
- Create: `E:\Python\ModelLabeling\src\services\image_service.py`
- Create: `E:\Python\ModelLabeling\src\services\project_service.py`
- Test: `E:\Python\ModelLabeling\tests\test_image_service.py`
- Test: `E:\Python\ModelLabeling\tests\test_project_service.py`

- [ ] **Step 1: Write failing service tests**

Test supported-image discovery, case-insensitive filename fuzzy search, labeled/unlabeled/error status, image resolution/format/byte-size metadata, JSON settings round-trip, and autosave writing current annotations plus project settings.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_image_service.py tests/test_project_service.py -q`
Expected: FAIL because the services are missing.

- [ ] **Step 3: Implement the services**

`ImageService.scan(directory, annotation_dir, settings)` returns stable, sorted `ImageRecord` objects. `ProjectService` reads/writes UTF-8 JSON atomically through a temporary file and exposes `save_current(...)` used by both manual save and autosave.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_image_service.py tests/test_project_service.py -q`
Expected: PASS.

## Task 6: Implement Dataset Conversion

**Files:**
- Create: `E:\Python\ModelLabeling\src\services\conversion_service.py`
- Test: `E:\Python\ModelLabeling\tests\test_conversion_service.py`

- [ ] **Step 1: Write failing conversion tests**

Cover VOC-to-YOLO and YOLO-to-VOC with class mapping, progress callback values, skip/overwrite behavior, duplicate output handling, and a conversion report containing succeeded/skipped/failed counts.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_conversion_service.py -q`
Expected: FAIL because the conversion service is missing.

- [ ] **Step 3: Implement batch conversion**

Define `ConversionOptions`, `ConversionReport`, and `ConversionService.convert(options, progress_callback, cancel_callback)`. Reuse `AnnotationService` for parsing and writing. Emit progress after each source image and never abort the entire batch for one malformed item.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_conversion_service.py -q`
Expected: PASS.

## Task 7: Implement YOLO ONNX Inference

**Files:**
- Create: `E:\Python\ModelLabeling\src\services\onnx_service.py`
- Test: `E:\Python\ModelLabeling\tests\test_onnx_service.py`

- [ ] **Step 1: Write tests around a fake session**

Inject a fake ONNX session so tests do not require a real model. Verify preprocessing shape, confidence filtering, class mapping, coordinate scaling back to original image pixels, NMS suppression, and clear failure when `onnxruntime` is unavailable.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_onnx_service.py -q`
Expected: FAIL because the ONNX service is missing.

- [ ] **Step 3: Implement the service**

Expose `YoloOnnxDetector` with `load(model_path)`, `predict(image: Image.Image, settings)`, and `close()`. Support common YOLO tensor layouts through an explicit adapter, normalize output to `(x1, y1, x2, y2, score, class_id)`, apply threshold and NMS, and return `Annotation` objects with `source="onnx"`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_onnx_service.py -q`
Expected: PASS.

## Task 8: Build the Graphics Canvas

**Files:**
- Create: `E:\Python\ModelLabeling\src\widgets\canvas_view.py`
- Test: `E:\Python\ModelLabeling\tests\test_main_window.py`

- [ ] **Step 1: Add a minimal widget test**

Create a `CanvasView`, load a test image, set one annotation, zoom in and out, and assert the scene remains populated and the annotation model points remain unchanged.

- [ ] **Step 2: Implement the canvas**

Use `QGraphicsPixmapItem` for the image and custom `QGraphicsItem` subclasses for rectangle/square/polygon overlays. Add drawing mode, selection, moving, polygon closing, delete, zoom centered on cursor, fit-to-window, and spacebar pan. Emit `annotationCreated`, `annotationChanged`, `annotationDeleted`, and `dirtyChanged`.

- [ ] **Step 3: Run the widget test**

Run: `pytest tests/test_main_window.py::test_canvas_zoom_preserves_annotation_coordinates -q`
Expected: PASS.

## Task 9: Build Panels and Main Window

**Files:**
- Create: `E:\Python\ModelLabeling\src\widgets\image_list_panel.py`
- Create: `E:\Python\ModelLabeling\src\widgets\preset_panel.py`
- Create: `E:\Python\ModelLabeling\src\widgets\stats_panel.py`
- Create: `E:\Python\ModelLabeling\src\widgets\settings_dialog.py`
- Create: `E:\Python\ModelLabeling\src\widgets\conversion_dialog.py`
- Create: `E:\Python\ModelLabeling\src\widgets\main_window.py`
- Modify: `E:\Python\ModelLabeling\app.py`
- Test: `E:\Python\ModelLabeling\tests\test_main_window.py`

- [ ] **Step 1: Write UI behavior tests**

Test opening a temporary image directory, selecting an image, assigning a preset label, saving, switching images, filtering by filename/status, and updating statistics. Add a test that unsaved changes prompt when autosave is disabled.

- [ ] **Step 2: Implement panels**

Use standard Qt controls: icon buttons with tooltips for navigation/save/zoom, a search field and status combo box for the image list, color swatches and label rows for presets, and compact numeric/stat controls for progress and per-label counts.

- [ ] **Step 3: Implement `MainWindow` composition**

Wire services and widgets through signals. Maintain `ProjectState`, current image index, dirty state, undo/redo stacks, and a `QStatusBar`. Add the dark technology stylesheet with readable contrast and stable dimensions.

- [ ] **Step 4: Add shortcuts and navigation**

Bind the specified shortcuts through `QAction` objects. Route navigation through one `request_image_change(index)` method so autosave and unsaved-change prompts are applied consistently.

- [ ] **Step 5: Run UI tests**

Run: `pytest tests/test_main_window.py -q`
Expected: PASS.

## Task 10: Add ONNX Worker and Conversion Dialog Integration

**Files:**
- Modify: `E:\Python\ModelLabeling\src\widgets\main_window.py`
- Modify: `E:\Python\ModelLabeling\src\widgets\conversion_dialog.py`
- Modify: `E:\Python\ModelLabeling\src\widgets\settings_dialog.py`
- Test: `E:\Python\ModelLabeling\tests\test_main_window.py`

- [ ] **Step 1: Write worker-state tests**

Use a fake detector and `QSignalSpy` to verify controls are disabled during inference, progress signals update the status bar, partial results remain available after failure, and controls are restored on completion.

- [ ] **Step 2: Implement Qt workers**

Create `AutoLabelWorker(QObject)` and `ConversionWorker(QObject)` with `progress`, `finished`, and `failed` signals. Move workers to `QThread`, keep all UI changes on the main thread, support cancellation, and make partial-result handling explicit.

- [ ] **Step 3: Run focused tests**

Run: `pytest tests/test_main_window.py -q`
Expected: PASS.

## Task 11: Add Styling, Logging, and Packaging Notes

**Files:**
- Modify: `E:\Python\ModelLabeling\src\widgets\main_window.py`
- Create: `E:\Python\ModelLabeling\src\utils\logging_setup.py`
- Modify: `E:\Python\ModelLabeling\requirements.txt`
- Modify: `E:\Python\ModelLabeling\README.md`

- [ ] **Step 1: Implement logging**

Write application logs to a user-local `logs/model_labeling.log`, include timestamps and exception traces, and surface a short error summary in the UI.

- [ ] **Step 2: Finish the dark technology stylesheet**

Use a charcoal background, restrained cyan/green accents, neutral panels, label-specific color swatches, visible focus states, and no low-contrast text. Ensure the image information remains readable at narrow window widths.

- [ ] **Step 3: Document Windows setup**

Document:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Explain the optional ONNX dependencies and expected `classes.txt` format.

## Task 12: Full Verification

**Files:**
- Test: all files under `E:\Python\ModelLabeling\tests`

- [ ] **Step 1: Run all tests**

Run: `pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run the application smoke check**

Run: `python app.py`
Expected: the main window opens without import errors and can be closed cleanly.

- [ ] **Step 3: Verify manual workflow**

Create a temporary image directory, label at least one image with each shape type, save, reload, zoom, navigate, filter, and confirm progress/statistics.

- [ ] **Step 4: Verify conversion**

Convert a small VOC fixture to YOLO and back, inspect generated files, and confirm class IDs and rectangle coordinates remain stable.

- [ ] **Step 5: Verify optional ONNX**

When `onnxruntime` is installed, run the fake-session unit tests and one real YOLO ONNX smoke test. When it is absent, confirm the manual workflow still launches and the model setting reports the optional dependency clearly.

## Self-Review

- Spec coverage: tasks 2-5 cover models, coordinates, persistence, image discovery, project paths, autosave, and statistics; tasks 6-7 cover conversion and YOLO ONNX; tasks 8-10 cover canvas, UI, shortcuts, worker locking, progress, and dialogs; tasks 11-12 cover style, logging, setup, and acceptance.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps are included.
- Type consistency: all services consume `ProjectSettings`, `Annotation`, and `ImageRecord`; canvas persists original-image coordinates; workers communicate through Qt signals.
- Repository state: `E:\Python\ModelLabeling` is currently not a Git repository, so commit commands should be used only after repository initialization.
