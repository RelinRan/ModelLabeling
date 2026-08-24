# ModelLabeling Full Rebuild Specification

This document records the confirmed rebuild scope for the ModelLabeling desktop application.

## Scope

- PySide6 desktop annotation workbench with an IDEA-style dark layout.
- Manual rectangle, square, and polygon annotation using original-image coordinates.
- COCO, YOLO, and Pascal VOC loading, saving, detection, and pairwise conversion.
- Background dataset loading, ONNX auto-labeling, and dataset conversion through one task manager.
- Label groups and label presets with add, edit, color, delete, and bilingual UI support.
- Responsive left and right workbench panels, image filtering, history, statistics, settings, and shortcuts.

## Non-negotiable behavior

- Opening a dataset detects its format and paths, shows a task item, disables editing while scanning, closes the task list on completion, and shows exactly one success or failure dialog. Cancellation shows no completion dialog.
- Auto-labeling and conversion run in background threads, show `name current/total percent` task rows, can be stopped, and never report a stopped task as completed or failed afterward.
- Annotation coordinates remain in original image pixels across zoom, fit, resize, moving, and corner resizing.
- A/D and Up/Down navigation are bounds-safe at the first and last image.
- All menus and dialogs have working actions; language changes refresh all visible labels, buttons, task text, and status text.

## Defaults

- Dataset format: YOLO.
- Save mode: automatic.
- Side panel width: 360px.
- Progress color: `#2e436e`.
- Progress height: 8px; radius: 4px.
- Default group: `默认标签` / `Default Labels`.
- Default labels: person, head, hand, foot, leg, knee, clothes, coat, shirt, pants, dress, cap, hat, glasses, bag, shoe, sneaker, boot, car, bus, truck, chair, sofa, bed, desk, lamp, mouse, phone, bottle, vase, clock, mirror, window.

## Menus and shortcuts

File: Open Ctrl+O, History Ctrl+H, Save Ctrl+S, Exit Ctrl+Q.

Edit: Label Groups Ctrl+L+G, Application Settings Ctrl+A+S, Image Filter Ctrl+I+F, Crosshair Ctrl+C+A.

View: Task List Ctrl+T+L, Fit Canvas Ctrl+0, Image Zoom In Ctrl++, Image Zoom Out Ctrl+-, Previous Image A/Up, Next Image D/Down.

Tools: Statistics Ctrl+D+S, Dataset Conversion Ctrl+D+C, Auto Label Ctrl+A+L.

Help: Shortcuts and About Software.

## Persistence

The internal annotation model stores shape type, label, class ID, color, point list, confidence, and source. Format adapters own all file parsing and serialization. Polygon export to formats that only support boxes uses the polygon bounding rectangle.

## UI measurements

- All buttons, inputs, combo boxes, panel backgrounds, and borders use 5px radius.
- Left image items are 30px high, with 5px horizontal inner padding, 0px left outer margin, and 15px right outer margin.
- Settings category items are 95px wide and 30px high.
- Task rows are 40px high.
- Image zoom buttons are icon-only, 30px square, and use `icons/ic_shrink.png`, `icons/ic_zoom.png`, and `icons/ic_fit.png`.
- Stop buttons use `icons/ic_stop.png`, 15px icon size, 20px click area, with no button background or border.
- History delete icons use `icons/ic_delete.png`, 10px icon size, and 20px click area.

## Verification

The rebuild must pass service tests for format round trips, dataset detection, filtering, statistics, history, cancellation, and ONNX result mapping, plus GUI tests for every menu action, loading/task lifecycle, safe navigation, box editing, zoom invariants, language refresh, and responsive panel behavior.
