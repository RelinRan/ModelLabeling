# ModelLabeling

English | **[简体中文](README.zh-CN.md)**

A Windows-first Python/PySide6 desktop annotation workbench for YOLO / Pascal VOC / COCO datasets. Covers all four official Ultralytics YOLO label-file tasks (detect, segment, pose, OBB), with a large-dataset SQLite index, a workspace-style new-dataset wizard, and ONNX auto labeling.

- Version: v1.0.0
- Author: RelinRan · [GitHub](https://github.com/RelinRan) · relinran@foxmail.com

---

## Contents

1. [Feature Overview](#feature-overview)
2. [Supported Dataset Formats](#supported-dataset-formats)
3. [Annotation Methods & Operations](#annotation-methods--operations)
4. [Quick Start](#quick-start)
5. [User Guide](#user-guide)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [Architecture](#architecture)
8. [Performance & Large Datasets](#performance--large-datasets)
9. [Testing](#testing)
10. [FAQ](#faq)

---

## Feature Overview

| Category | Capability |
| --- | --- |
| Dataset tasks | YOLO detection / segmentation / pose / OBB, Pascal VOC, COCO |
| Annotation methods | Rectangle, square, polygon, rotated box, keypoints (filtered by the current dataset's task) |
| Keypoint types | Predefined point count, point names, and annotation label (Ctrl+K+G); built-in Pose (17 COCO person points, `pose`) and Car (`car`); picking a type on the canvas arms its whole schema |
| Dataset creation | Workspace wizard: workspace + name + options; source images are copied in (existing datasets import their annotations too) |
| Compatibility | Fully unannotated datasets open directly (empty labels/, no Annotations/, plain image folders) |
| Auto labeling | ONNX inference; official YOLO detection/Pose models; runs in background, stoppable |
| Conversion | Batch YOLO / VOC / COCO conversion preserving layout and data.yaml class names |
| Editing | Continuous drawing, undo/redo, move/resize/rotate, keypoint visibility (COCO 0/1/2) |
| Large datasets | SQLite path index + keyset pagination; smooth with tens of thousands of images |
| UI | IDEA-style dark theme, English/Chinese, every menu/dialog/status bar refreshes live |

---

## Supported Dataset Formats

### YOLO (all four official Ultralytics tasks)

Recommended layout:

```
dataset/
├─ images/                images (train/ subfolders supported)
├─ labels/                .txt per image, same name (may be empty)
├─ classes.txt            one class name per line; line number = class_id
└─ data.yaml              optional: task, names, kpt_shape, ...
```

One object per line, depending on the task:

| Task | Row format | Example |
| --- | --- | --- |
| Detection | `class cx cy w h` | `0 0.5125 0.4800 0.2500 0.3600` |
| Segmentation | `class x1 y1 x2 y2 …` (polygon vertices) | `0 0.1 0.1 0.5 0.1 0.3 0.4` |
| Pose | `class cx cy w h px py v …` | `0 0.5 0.5 0.4 0.4 0.4 0.4 2 …` |
| OBB | `class x1 y1 x2 y2 x3 y3 x4 y4` (4 corners) | `0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5` |

- Coordinates are normalized 0–1; keypoint visibility `v`: 0 unlabeled, 1 occluded, 2 visible
- The keypoint count is declared by `kpt_shape: [N, 3]` in data.yaml (editable on the canvas, written back automatically)
- Task detection prefers the `task` field in data.yaml, then infers from the row shape

### Pascal VOC

```
dataset/
├─ JPEGImages/            images
└─ Annotations/           .xml per image, same name
```

The XML contains `filename`, `size`, `object/bndbox` (pixel coordinates). `Annotations/` may be missing; it is created on save.

### COCO

```
dataset/
├─ images/
└─ annotations/
   └─ annotations.json    (instances.json also supported)
```

The JSON contains `images` (id/file_name/width/height), `annotations` (image_id/category_id/bbox as `[x, y, w, h]` pixels), and `categories`. Segmentation and keypoints are supported. A valid document with empty annotation lists opens fine.

---

## Annotation Methods & Operations

Selecting a method **stays armed for continuous drawing** (finish one shape, start the next), matching CVAT/LabelMe conventions.

| Method | Draw | Edit |
| --- | --- | --- |
| Rectangle | Drag with the left button (any direction); hold **Shift** to constrain a square | Drag inside to move; corner handles resize |
| Square | Width and height stay equal while dragging | Same as rectangle |
| Polygon | Left click each vertex; **double click / Enter / right click** closes; Backspace removes the last vertex | Drag vertex handles |
| Rotated box (OBB) | Drag like a rectangle (starts axis aligned) | Drag the green handle to rotate around the center, **Shift snaps to 15°**; drag inside to move |
| Keypoints (Pose) | Click each point; auto-finishes at the type's count; **Tab** skips a point (visibility 0); double click / Enter finishes early | Drag point markers (the box follows); **right click a marker cycles visibility 2→0→1**; the box itself cannot be dragged — its geometry is derived from the points |

The combo at the canvas top-left lists only the methods the current dataset supports (e.g. YOLO detection shows rectangle/square only). In keypoint mode the row continues with the **keypoint type selector** and a live **progress pill** (`point name (i/N)`); the point count is defined by the selected type, not edited on the canvas.

General operations:

- **Ctrl+Z / Ctrl+Y** undo/redo (draw, delete, move, relabel, and the edit dialog are all undoable)
- **Esc** first press drops the half-drawn shape, second exits drawing; **right click** on empty space exits drawing
- **Double click / right click** an annotation opens the edit dialog
- Auto save: the label file is written ~0.3 s after each shape; saving strictly validates the format and rejects anything non-standard
- A tiny accidental drag does not exit the armed state

---

## Quick Start

### Requirements

- Python 3.9+ (64-bit)
- Windows 10/11 (Linux/macOS should work but are not systematically tested)

### Install & Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Optional: `onnxruntime` for auto-label inference (everything else works without it).

### Annotate in three steps

1. **Open** (Ctrl+O) the dataset root — format, folders, and task are detected automatically; or use **New** (Ctrl+N) on a plain image folder
2. Pick an **annotation method** from the canvas top-left combo
3. Draw — auto save is on; the bottom status bar shows loading/statistics/save progress live

---

## User Guide

The in-app **Help → User Guide** contains the full manual (quick start / annotation operations / dataset formats / new-dataset wizard / more features). Highlights:

### New Dataset wizard (Ctrl+N)

Manage datasets like IDE projects: datasets live in a **workspace**; the image source is only material — the real dataset is created at `workspace/<name>/`:

1. Pick a **workspace** (remembered; it can hold many datasets)
2. Enter a **dataset name** (unique per workspace; invalid characters flagged live)
3. Optionally pick a **source** (image folder or dataset folder): a plain folder → images are copied into `images/` (nested folders flattened); an existing dataset → images **and annotations** are imported; leave empty → an empty dataset
4. Choose the target format (six tasks) plus optional class names and keypoint count
5. Click “Create” — it opens automatically when done; the source folder is never modified

### Image navigation & viewing

A/D or arrows to move (safe at both ends); **wheel zoom** anchored at the cursor, Ctrl+± zoom, Ctrl+0 fit; the zoom level persists across images; number keys **1–9** pick labels quickly; filter the list by name/status/label (Ctrl+I+F).

### Startup restore

The **last dataset reopens on launch** (at the last viewed image) by default; switch to an empty start in Settings → General → Startup.

### Label management

Label groups (Ctrl+L+G) maintain a cross-dataset template library (app-level SQLite); new labels are registered into classes.txt / COCO categories on first save.

### Keypoint types (Ctrl+K+G)

Predefine everything about a keypoint schema once, then just pick it while annotating:

- Each type has a **name**, a **point count**, per-point **names** (editable, shown in drawing order), and an **annotation label**
- Built-ins: Pose (17 official COCO person points, label `pose`, protected) and Car (`轮毂/车窗/车灯`, label `car`)
- The canvas top-left selector applies the chosen type instantly; finished annotations automatically use the type's label and color
- A type's label is auto-registered into the default label group and cannot be deleted from label management while a type uses it
- In the edit dialog of a keypoint annotation, the label field is locked to the type's label (no dropdown arrows) — point coordinates and visibility stay editable

### Statistics & conversion

- Statistics (Ctrl+D+S): totals, per-class distribution, progress (computed in the background)
- Conversion (Ctrl+D+C): batch YOLO/VOC/COCO

### Auto labeling (Ctrl+A+L)

Select a YOLO ONNX model in Settings first; official YOLO detection and Pose models are supported (classes and keypoints decoded from model metadata); runs in the background, stoppable, progress in the status bar.

## Keyboard Shortcuts

| Key | Action | Key | Action |
| --- | --- | --- | --- |
| Ctrl+N | New dataset | Ctrl+O | Open |
| Ctrl+S | Save | Ctrl+Q | Exit |
| A / ↑ | Previous image | D / ↓ | Next image |
| W | Toggle drawing | Esc | Drop current shape / exit drawing |
| Ctrl+Z / Y | Undo / redo | Delete | Delete selected annotation |
| Shift+drag | Constrain square | Enter / double click | Finish polygon/keypoints |
| Backspace | Remove last point while drawing | Right click | Exit drawing / edit annotation |
| Ctrl+0 | Fit canvas | Wheel / Ctrl+± | Zoom (cursor-anchored) |
| 1-9 | Select the bound label | Ctrl+1-9 | Bind the selected label to that key |
| Ctrl+L+G | Label groups | Ctrl+K+G | Keypoint types |
| Ctrl+A+S | Settings | Ctrl+I+F | File filter |
| Ctrl+C+A | Annotation assist | Ctrl+D+S | Statistics |
| Ctrl+D+C | Dataset conversion | Ctrl+A+L | Auto labeling |
| Ctrl+H | History | Tab | Skip keypoint (visibility 0) |

---

## Architecture

```
app.py                     entry point
src/
├─ models/                 data models
│  ├─ annotation.py        Annotation / ShapeType / Keypoint / LabelPreset
│  └─ project.py           ProjectSettings / ProjectState / ImageRecord
├─ services/               service layer (no UI dependencies)
│  ├─ dataset_detector.py  layout & task detection (plain-image fallback, unannotated support)
│  ├─ dataset_initializer.py  workspace-based dataset creation
│  ├─ dataset_index.py     per-dataset SQLite path index (paging/filter/locate)
│  ├─ dataset_session.py   authoritative context for the open dataset
│  ├─ annotation_service.py format adapters & load/save (YOLO×4 / VOC / COCO)
│  ├─ coco_store.py        COCO SQLite working copy (per-image transactions)
│  ├─ label_group_store.py app-level label template library (SQLite)
│  ├─ keypoint_group_store.py app-level keypoint-type library (SQLite)
│  ├─ format_capabilities.py task capability matrix & pre-save validation
│  ├─ format_adapters.py   format dispatch layer
│  ├─ yolo_metadata.py     data.yaml read/write (kpt_shape, ...)
│  ├─ conversion_service.py dataset conversion
│  ├─ onnx_service.py      ONNX inference
│  ├─ workers.py           all background workers (scan/count/stats/per-image/auto/save)
│  └─ operation_coordinator.py background operation mutual exclusion
└─ widgets/                UI layer
   ├─ main_window.py       main window & workflow orchestration
   ├─ canvas_view.py       canvas (five methods, rotation handle, undo stack)
   └─ …                    dialogs and panels
tests/                     unit tests (68)
```

### Key design decisions

- **Four SQLite tiers, one lifecycle each**: per-dataset index (`%LOCALAPPDATA%\ModelLabeling\index\`, named by root-path hash), per-directory COCO working copy (`.model_labeling.sqlite3`), app-level label library (`label_groups.sqlite3`), app-level keypoint-type library (`keypoint_groups.sqlite3`) — fully decoupled
- **Open flow**: detect format → DatasetSession → background incremental index build (size+mtime skips unchanged files) → **first batch is annotatable immediately** (non-blocking) → statistics computed in the background
- **Load on demand**: the list reads index metadata only; per-image annotations load when selected; COCO edits are single-transaction upserts
- **Strictly standard output**: `format_capabilities.validate_annotations` runs before every save (shape legality, keypoint count/schema consistency, OBB four corners); violations are rejected rather than silently degraded
- **Threading**: scan/count/stats/per-image/save each run on their own worker; statistics run on a plain Python thread to avoid a PySide6 QThread heap-corruption issue under a busy GUI; `OperationCoordinator` allows only one heavy operation at a time

---

## Performance & Large Datasets

- Thousands of images open in seconds on first open (index build); later opens hit the cache
- The list uses keyset pagination (no OFFSET degradation) with lazy loading
- Ten-thousand-image directories never block the UI: annotating can start while loading

---

## Testing

```powershell
python -m pytest tests -q
```

68 unit tests cover: format round trips (six tasks), dataset detection (unannotated/plain folders/OBB inference), capability validation, canvas interaction (five methods: draw/rotate/undo/keyboard), the creation wizard, the settings dialog, ONNX result mapping, and operation mutual exclusion.

An end-to-end script `test_annotation_flow.py` (three formats opened × five methods annotated × on-disk verification + unannotated/OBB scenarios) can be run directly.

---

## FAQ

**"Unsupported dataset format" when opening a folder?**
Make sure the folder contains images, or initialize it with the New wizard (Ctrl+N).

**What if a YOLO dataset has no classes.txt?**
It opens and annotates fine; the file is created on first save with the current labels.

**What happens to old annotations after changing the keypoint count?**
The point count comes from the selected keypoint type (Ctrl+K+G); editing a type's count keeps custom point names and truncates/appends as needed. Validation rejects datasets mixing keypoint counts — remove or redraw the old annotations first. For YOLO Pose the count is written to `kpt_shape` in data.yaml.

**Why was a keypoint annotation rejected with "keypoint schema conflicts"?**
COCO allows exactly one keypoint-name list per category. The error lists both schemas; use the same names for that label, or switch the type to a different label.

**A quadrilateral segmentation dataset detected as OBB?**
Nine-column rows are inherently ambiguous (a 4-vertex polygon equals an OBB's corners). Switch the task manually in Settings → Task, or add `task: segment` to data.yaml.

**Is editing a large COCO JSON slow?**
The tool normalizes COCO into a SQLite working copy (`annotations/.model_labeling.sqlite3`); per-image edits are single transactions, and the JSON is exported automatically on exit or dataset switch.
