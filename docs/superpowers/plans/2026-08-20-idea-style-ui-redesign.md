# IDEA-style UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the complete PySide6 presentation layer as a JetBrains IDEA-style dark desktop workbench while preserving annotation behavior.

**Architecture:** Keep the existing service/model/canvas contracts. Add a thin workbench shell in `MainWindow`, centralize visual tokens in one QSS string, and give each existing panel/dialog a stable object name and layout role. UI tests assert structure and visibility; service tests continue to verify data behavior.

**Tech Stack:** Python 3, PySide6, pytest, pytest-qt, existing Qt widgets and services.

---

### Task 1: Add workbench structure regression tests

**Files:**
- Modify: `tests/test_main_window.py`
- Test: existing offscreen Qt fixture

- [ ] **Step 1: Add tests for the IDEA workbench regions**

```python
def test_main_window_has_idea_workbench_regions(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.menuBar() is not None
    assert window.findChild(QWidget, "projectToolWindow") is not None
    assert window.findChild(QWidget, "editorToolWindow") is not None
    assert window.findChild(QWidget, "rightToolWindow") is not None
    assert window.findChild(QWidget, "ideaStatusBar") is not None
```

- [ ] **Step 2: Add tests for menu labels and style tokens**

```python
def test_main_window_exposes_idea_menu_and_palette(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert [menu.title() for menu in window.menuBar().findChildren(QMenu)]
    assert "文件" in [action.text() for action in window.menuBar().actions()]
    assert "#1F2023" in window.styleSheet()
```

- [ ] **Step 3: Run the new tests and verify they fail before implementation**

Run: `$env:PYTHONPATH='.'; pytest tests/test_main_window.py -q`

Expected: FAIL because the named workbench regions and menu bar are not yet present.

### Task 2: Introduce shared IDEA visual tokens

**Files:**
- Create: `src/widgets/theme.py`
- Modify: `src/widgets/main_window.py`

- [ ] **Step 1: Create the shared stylesheet function**

```python
IDEA_PALETTE = {
    "window": "#1F2023",
    "panel": "#25262A",
    "input": "#2B2D31",
    "border": "#3C3F41",
    "accent": "#6C63FF",
}

def idea_stylesheet() -> str:
    return """QMainWindow, QWidget { background: #1F2023; color: #D7DAE0; }
QMenuBar, QToolBar, QStatusBar { background: #25262A; border: 0; }
QToolButton, QPushButton { background: #2B2D31; border: 1px solid #3C3F41;
  border-radius: 4px; padding: 5px 10px; color: #D7DAE0; }
QToolButton:hover, QPushButton:hover { background: #353840; border-color: #6C63FF; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget {
  background: #2B2D31; border: 1px solid #3C3F41; border-radius: 4px;
  padding: 5px; selection-background-color: #5148B8; }
QProgressBar { background: #2B2D31; border: 1px solid #3C3F41; border-radius: 3px; }
QProgressBar::chunk { background: #6C63FF; }
QSplitter::handle { background: #3C3F41; }
QToolTip { background: #25262A; color: #D7DAE0; border: 1px solid #6C63FF; }
"""
```

- [ ] **Step 2: Replace MainWindow's inline dark-tech stylesheet with `idea_stylesheet()` plus panel-specific rules**

- [ ] **Step 3: Run the style and existing GUI tests**

Run: `$env:PYTHONPATH='.'; pytest tests/test_main_window.py tests/test_settings_dialog.py -q`

Expected: PASS.

### Task 3: Rebuild MainWindow as an IDEA workbench shell

**Files:**
- Modify: `src/widgets/main_window.py`
- Modify: `src/widgets/image_list_panel.py`
- Modify: `src/widgets/preset_panel.py`
- Modify: `src/widgets/operations_panel.py`

- [ ] **Step 1: Add the menu bar and project-level actions**

Create File/Edit/View/Tools/Help menus and connect File Open, File Save,
Settings, Statistics, Conversion, Auto Label, and Quit to the existing methods.
Keep `save_action` as the existing QAction attribute so autosave behavior remains
unchanged.

- [ ] **Step 2: Add stable workbench object names**

Set `projectToolWindow`, `editorToolWindow`, and `rightToolWindow` on the three
main regions. Create an `ideaStatusBar` QWidget containing format, size, zoom,
autosave, and message labels, while continuing to update `self.status` for the
existing logic.

- [ ] **Step 3: Recompose the right tool window**

Keep `PresetPanel` and `OperationsPanel` as children but place them in a single
tool-window host with consistent 12px margins and 8px spacing. Preserve all
existing signal connections and public attributes.

- [ ] **Step 4: Add panel headers and compact project list hierarchy**

Give the image list and preset/operations panels named headers, compact margins,
and consistent selection/hover states. Do not change the scan/filter data flow.

- [ ] **Step 5: Run the structure tests**

Run: `$env:PYTHONPATH='.'; pytest tests/test_main_window.py -q`

Expected: PASS.

### Task 4: Restyle and align all dialogs

**Files:**
- Modify: `src/widgets/settings_dialog.py`
- Modify: `src/widgets/stats_dialog.py`
- Modify: `src/widgets/conversion_dialog.py`
- Modify: `src/widgets/stats_panel.py`

- [ ] **Step 1: Apply shared stylesheet and fixed dialog sizing**

Set dialog object names, minimum sizes, margins, and button order. Keep every
existing field and `SettingsDialog.apply()` return contract.

- [ ] **Step 2: Convert SettingsDialog to a two-column settings page**

Add a compact category list on the left with categories for Dataset, Annotation,
Automatic Labeling, and General. Keep the current form widgets in the right
content area and make category selection switch visible form groups without
changing stored values.

- [ ] **Step 3: Restyle statistics and conversion content**

Use flat IDEA panels, clear metric hierarchy, progress bars, and consistent
form/button spacing. Preserve empty dataset messaging and conversion handlers.

- [ ] **Step 4: Run all GUI smoke tests**

Run: `$env:PYTHONPATH='.'; pytest -q`

Expected: all existing and new tests pass.

### Task 5: Bilingual and state synchronization pass

**Files:**
- Modify: `src/widgets/i18n.py`
- Modify: `src/widgets/main_window.py`
- Modify: `src/widgets/operations_panel.py`
- Modify: `src/widgets/settings_dialog.py`

- [ ] **Step 1: Add missing menu, status, and workbench translations**

Use the existing `text(key, language)` mechanism for all user-facing strings
introduced by the redesign. Do not leave new hardcoded menu/status text in the
language-switchable interface.

- [ ] **Step 2: Refresh toolbar, panel, and status labels after settings apply**

Ensure language, autosave visibility, annotation format, and current image info
update immediately after `SettingsDialog` is accepted.

- [ ] **Step 3: Verify bilingual construction**

Run: `$env:PYTHONPATH='.'; pytest tests/test_settings_dialog.py tests/test_main_window.py -q`

Expected: PASS with both `zh_CN` and `en_US` settings.

### Task 6: Dataset integration and final verification

**Files:**
- Modify: `tests/test_main_window.py`
- Modify: `tests/test_project_service.py` only if a discovered layout requires a
  narrowly scoped regression test

- [ ] **Step 1: Add a non-dialog dataset scan integration assertion**

Use `ProjectService.resolve_dataset_paths()` and `ImageService.scan()` against
`E:\Dataset\multiple\yolo-multiple-action - backup`, assert the resolved
`images/train` and `labels/train` paths, assert 1,599 records, and assert that
the first record loads at least one annotation with a known class label.

- [ ] **Step 2: Run the complete verification suite**

Run: `$env:PYTHONPATH='.'; pytest -q; python -m compileall -q app.py src tests`

Expected: all tests pass and compileall exits successfully.

- [ ] **Step 3: Run an offscreen application smoke test**

Run a short Python process that creates `QApplication`, `MainWindow`,
`SettingsDialog`, `StatsDialog`, and `ConversionDialog`, then exits without
showing modal dialogs.

Expected: process exits with code 0 and no Qt construction errors.

- [ ] **Step 4: Report the final UI and dataset verification**

Include changed files, test counts, compile result, and the supplied dataset's
resolved paths and record count.
