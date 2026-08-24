# Settings and iOS-Style UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make project configuration and label presets fully editable while refreshing the desktop UI into a polished dark iOS-inspired workspace.

**Architecture:** Extend `ProjectSettings` with persisted enabled annotation shapes. Keep label mutations inside `PresetPanel`, use standard Qt file/folder dialogs in `SettingsDialog`, and apply settings to `CanvasView` through `MainWindow`.

**Tech Stack:** Python 3, PySide6, pytest, pytest-qt.

---

### Task 1: Persist enabled annotation shapes

**Files:**
- Modify: `src/models/project.py`
- Test: `tests/test_project_service.py`

- [ ] Add an `enabled_shapes` list to `ProjectSettings`, normalize it to `ShapeType` values, and include it in JSON serialization.
- [ ] Add a round-trip test for selected shape types.

### Task 2: Manage label presets

**Files:**
- Modify: `src/widgets/preset_panel.py`
- Test: `tests/test_preset_panel.py`

- [ ] Add explicit add, edit, and delete controls with validation and stable class IDs.
- [ ] Add Qt tests for label rename and deletion.

### Task 3: Build native settings controls

**Files:**
- Modify: `src/widgets/settings_dialog.py`
- Modify: `src/widgets/canvas_view.py`

- [ ] Use folder/file browse buttons, combo boxes, and shape checkboxes instead of free-form configuration fields.
- [ ] Restrict drawing modes to enabled shapes.

### Task 4: Refresh main workspace

**Files:**
- Modify: `src/widgets/main_window.py`
- Create: `src/widgets/app_icon.py`
- Test: `tests/test_main_window.py`

- [ ] Add the application icon, compact toolbar layout, iOS-inspired dark styling, and selected-shape controls.
- [ ] Route list selection through a save/discard guard before changing the current image.

### Task 5: Verify

**Files:**
- Test: `tests/`

- [ ] Run pytest, compileall, and an offscreen PySide6 smoke test.
