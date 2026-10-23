# Project audit - 2026-10-23

## Scope

Reviewed dataset detection and indexing, annotation format validation,
conversion, keyboard routing, background-task shutdown, and test isolation.
Retained the pre-existing YOLO-Seg conversion changes in the working tree.
No real dataset was converted, annotated, or deleted for this audit.

## Fixed findings

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1 | Conversion could write to a source directory or an overlapping directory; overwrite could remove source annotations. | Reject overlapping resolved paths before enumerating or writing files. Reject empty conversion paths in the dialog. |
| P1 | Advertised shortcuts were missing handler branches or were shadowed by canvas editing shortcuts. Holding Ctrl during the second stroke lost commands. | Route shortcut-override events, complete command mappings, ignore modifier-only presses, and test both Ctrl-release and Ctrl-held usage. |
| P2 | Global W handling could interfere with text entry or another dialog. | Restrict commands to the main window and exclude text editors, combo boxes, and numeric editors. |
| P2 | Different image directories could share one index when their supplied dataset root matched. | Identify an index by resolved image directory, annotation directory, and format. Old cache files are retained; a new index is built on the next open. |
| P2 | Opening a plain image folder replaced the selected dataset root with its parent. | Preserve the detected session root during scanning. |
| P2 | Nested Pascal VOC/YOLO annotation categories were missed during conversion. Unknown sparse YOLO IDs were assigned different IDs. | Resolve annotation branches per image and preserve source IDs. |
| P2 | Format validation consumed one-shot iterators before checking multipart or task-specific constraints. | Materialize the input once before validation passes. |
| P2 | Application shutdown did not wait for comparison and image-count threads. | Include both in the existing shutdown wait sequence. |
| P2 | Tests used real user label/keypoint databases and index caches. Empty real settings also caused a false isolation failure. | Redirect databases and indexes into per-test directories and correct the empty-settings assertion. |

## Performance improvements

- Explicit YOLO task metadata bypasses annotation-tree traversal.
- Without explicit task metadata, inference samples at most 20 files and
  the first 100 lines per file instead of collecting every annotation path.
- Statistics still process each image, but publish UI snapshots at most
  once every 50 ms, plus exact initial/final progress. This reduces queued
  UI work and repeated counter sorting for large datasets.

## COCO persistence and cache follow-up

- Save nested images by their path relative to the image root. Only unique
  basenames receive lookup aliases, preventing same-name images from sharing
  annotations.
- Preview reads one image through indexed SQLite queries rather than loading
  the entire COCO document.
- Track the source JSON path, size, and modification time. Refresh clean
  caches after external edits; reject conflicts when local edits are pending.
- Export through a temporary file and atomic replacement while serializing
  database writers, and migrate existing cache schemas in place.
- Reject non-overwrite COCO conversion into nonempty output directories to
  avoid replacing a shared JSON document while leaving older images behind.
- Invalidate cached YOLO annotation labels when the class-name mapping changes.
- Preserve statistics errors as failures and display the error on request.
- Remove the synchronous image-count pass before conversion. The worker
  reports initial totals, and cancellation at that point exits before category
  discovery. Cancellation after processing still leaves the target unchanged.

## Validation

- Latest shortcut/menu/i18n follow-up full suite: **356 passed** in 271.87 seconds.
  Targeted shortcut/settings/conversion regression suite: **52 passed** in 147.55 seconds.
- `python -m compileall -q src tests` and `git diff --check`: passed after the
  follow-up edits.

- COCO/cache follow-up full suite: **356 passed**, 351.33 seconds. Report:
  `tmp/optimization-final-20261023.xml`. Cancellation tests cover both initial
  progress and cancellation after processing a file.
- Latest follow-up full suite: **345 passed**, 277.14 seconds, zero failures,
  errors, or skips. Report: `tmp/yolo-verified-final.xml`.
- Latest `python -m compileall -q src tests` and `git diff --check`: passed.
- Full suite: **284 passed**, 289.32 seconds. No expected-failure shortcuts
  remained in that run.
- After the final chord-timer and plain-folder-root adjustments: **16
  keyboard edge tests passed** and **5 open/edit/persistence tests passed**.
- Service regression subset: **55 passed** before those final UI adjustments.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.
- Full-suite machine-readable output: `tmp/audit-final-20261023.xml`.
- Qt GUI tests used the offscreen platform. This is not a manual Windows
  desktop visual acceptance test or a real ONNX inference/100,000-image soak test.

## Remaining design and verification work

1. **Shortcut prefix conflicts.** Ctrl+V is also the prefix of Ctrl+V+F.
   Canvas paste is therefore deferred until the 1,200 ms chord timeout;
   completing the video command cancels that paste. Prefer independent
   non-overlapping shortcut assignments in a future shortcut redesign.
2. **Conversion publication (resolved in follow-up).** Conversion now writes
   into a staging directory and publishes only after all files succeed.
   Cancellation and conversion errors leave the target unchanged; publication
   failures restore the previous directory. Regression coverage includes
   overwrite cleanup, unrelated-file preservation, output filename collisions,
   and YOLO class/task/Pose metadata compatibility.
3. **Large COCO startup.** Detection parses COCO JSON synchronously, and COCO
   export/checkpoint work can run on the UI thread. Move expensive inspection
   and export work into cancellable workers before claiming bounded UI latency.
4. **Window responsibilities.** MainWindow combines task lifecycle, persistence,
   navigation, keyboard routing, and statistics. Extract controllers gradually,
   retaining the existing workflow tests as regression protection.
5. **Load testing.** Measure time to first usable image, UI latency, memory, and
   repeated dataset switching on a representative 100,000-image dataset.
   The passing synthetic tests do not establish that performance guarantee.
