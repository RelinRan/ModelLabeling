# ModelLabeling IDEA-style UI redesign

## Goal

Redesign the complete PySide6 desktop interface to feel like a JetBrains IDEA
dark workbench while preserving the existing annotation workflows and service
contracts. This is a presentation and layout change; YOLO/VOC I/O, annotation
geometry, ONNX inference, dataset conversion, shortcuts, and autosave behavior
remain unchanged.

## Workbench layout

- Top menu bar: File, Edit, View, Tools, Help.
- Compact toolbar below the menu: Open, Save when autosave is disabled,
  Settings, Statistics, Conversion, and Auto Label.
- Left Project tool window: image search, status filter, image list, and the
  current item position.
- Center editor area: annotation canvas with the existing zoom, fit, drawing,
  selection, and image information behavior.
- Right tool window: Label Settings followed by Annotation Operations. The
  latter contains navigation, zoom controls, current filename, total progress,
  and labels in the current image.
- Bottom status bar: format, image dimensions, zoom level, autosave state, and
  transient operation messages.

## Visual system

- Dark IDEA-inspired palette: window `#1F2023`, panels `#25262A`, inputs
  `#2B2D31`, borders `#3C3F41`, primary accent `#6C63FF`.
- Use restrained flat surfaces with no decorative gradients or floating blobs.
- Use a consistent 6px panel radius, 4px control radius, and 8/12/16px
  spacing scale.
- Panel titles use compact bold text and a narrow accent rule.
- Buttons use icons and tooltips for frequent tool actions; text remains for
  explicit commands such as Open, Save, Apply, and Cancel.
- Standardize lists, selected rows, splitters, scrollbars, progress bars,
  dialogs, form controls, and disabled states through one application QSS.

## Dialogs

- Annotation Settings: IDEA settings-page layout with a left category list and
  a right form. Keep folder selectors, annotation format, language, one shape
  selector, line width, text size, autosave switch, and ONNX parameters.
- Statistics: standalone tool-window-like dialog with summary metrics,
  progress, and per-label counts. When there is no loaded dataset, show the
  existing explanatory message instead of empty metrics.
- Conversion: use the same form spacing, surfaces, controls, and button order
  as Annotation Settings.
- Dialog text uses the existing bilingual translation mechanism. Key titles,
  labels, buttons, and empty/error messages must refresh consistently when the
  selected language is applied.

## Boundaries and compatibility

- Preserve public widget attributes used by the current tests and MainWindow
  logic, including the canvas, preset panel, operations panel, save action,
  and dialog apply methods.
- Keep image and annotation coordinates in image space; zoom only changes the
  canvas view transform.
- Keep auto-labeling in its worker thread and disable editing controls while it
  runs.
- The selected YOLO dataset root must continue resolving `images/train` with
  `labels/train`, `images/val` with `labels/val`, and the flat `images`/`labels`
  layout.

## Verification

- Run all existing unit and GUI smoke tests with the repository root on
  `PYTHONPATH`.
- Add UI regression checks for the menu bar, workbench areas, autosave-dependent
  Save visibility, dialog titles, and key object names.
- Start the app in offscreen mode and instantiate the main window plus all
  dialogs to catch stylesheet and layout construction errors.
- Exercise the supplied YOLO dataset at
  `E:\Dataset\multiple\yolo-multiple-action - backup` and verify image count,
  annotation loading, label counts, and current-image navigation.

## Explicit non-goals

- No new annotation formats or cloud synchronization.
- No replacement of the existing canvas or model inference implementation.
- No branding/logo addition in the top toolbar.
