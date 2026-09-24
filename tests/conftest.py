# -*- coding: utf-8 -*-
"""Shared isolation and synthetic-dataset fixtures.

Two rules this module enforces for every test:

1. A test must never read or write the real application settings. The widgets
   reach the registry through ``QSettings("RelinRan", "ModelLabeling")``, and
   ``MainWindow._maybe_reopen_last_dataset`` reads ``reopen/last_root`` from it
   on ``showEvent`` -- so an unisolated test silently opens whatever dataset
   the developer last used. Both the native settings format and the
   module-level ``QSettings`` names are redirected here, centrally, instead of
   relying on each test file to remember.

2. A test must never touch a real dataset. Build one with the
   ``synthetic_dataset`` fixture, which creates scratch data under ``tmp_path``
   and can inject each defect the cleanup logic is meant to catch.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QSettings

from src.widgets import conversion_dialog, main_window, preset_panel, video_tools_dialogs

# Every widget module that resolves QSettings at call time. Patching the
# module attribute is what actually redirects them, because they bind the name
# with ``from PySide6.QtCore import QSettings``.
_SETTINGS_MODULES = (main_window, preset_panel, conversion_dialog, video_tools_dialogs)

_GUARDED_KEYS = ("reopen/last_root",)


def _real_settings() -> QSettings:
    """The developer's actual settings store, bypassing the test redirect.

    NativeFormat is passed explicitly so the IniFormat redirect below (and the
    per-test monkeypatching) cannot reach this store.
    """
    return QSettings(QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
                     "RelinRan", "ModelLabeling")


def _real_values() -> dict:
    store = _real_settings()
    return {key: str(store.value(key, "") or "") for key in _GUARDED_KEYS}


def _redirect_native_settings(directory) -> None:
    """Point even a direct QSettings(...) call at a throwaway INI file."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(directory))
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.SystemScope, str(directory))


@pytest.fixture(scope="session", autouse=True)
def _isolate_application_settings(tmp_path_factory):
    """Keep the real registry out of reach for the whole session.

    The guarded keys are snapshotted before the redirect and re-checked after,
    so a test that still finds a way to the real store fails loudly instead of
    quietly repointing the developer's app at a temp dataset.
    """
    before = _real_values()
    _redirect_native_settings(tmp_path_factory.mktemp("qsettings"))
    yield
    after = _real_values()
    assert after == before, (
        "a test mutated the real application settings "
        f"({ {k: (before[k], after[k]) for k in before if before[k] != after[k]} })"
    )


@pytest.fixture(autouse=True)
def _module_settings_to_memory(tmp_path, monkeypatch):
    """Route each widget module's QSettings at a per-test INI file."""
    ini = tmp_path / "app-settings.ini"

    def factory(*_args, **_kwargs):
        return QSettings(str(ini), QSettings.Format.IniFormat)

    for module in _SETTINGS_MODULES:
        if hasattr(module, "QSettings"):
            monkeypatch.setattr(module, "QSettings", factory)
    return ini


# --------------------------------------------------------------------------
# the designated test dataset
# --------------------------------------------------------------------------

#: The dataset tests are allowed to use. It is a real 1457-image Pascal VOC
#: dataset, so read-only tests may point straight at it; anything that writes
#: must stage a copy with ``voc_test.stage(...)`` first. No other dataset may
#: be opened or modified by a test.
VOC_TEST_ROOT = Path(r"E:\Dataset\single\voc-action\voc-test")


class VocTestDataset:
    """Read-only access to the designated test dataset, plus safe staging."""

    def __init__(self, root: Path, staging_root: Path) -> None:
        self.root = root
        self._staging_root = staging_root

    @property
    def image_dir(self) -> Path:
        return self.root / "JPEGImages"

    @property
    def annotation_dir(self) -> Path:
        return self.root / "Annotations"

    def images(self) -> list[Path]:
        return sorted(self.image_dir.glob("*.jpg"))

    def annotations(self) -> list[Path]:
        return sorted(self.annotation_dir.glob("*.xml"))

    def fingerprint(self) -> tuple:
        """Cheap change detector: counts, total bytes and newest mtime."""
        entries = [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                   for p in self.images() + self.annotations()]
        return (len(entries), sum(e[1] for e in entries), max((e[2] for e in entries), default=0))

    def stage(self, name: str = "voc-test-copy", limit: int | None = None) -> Path:
        """Copy (a slice of) the dataset into tmp_path for tests that mutate."""
        target = self._staging_root / name
        (target / "JPEGImages").mkdir(parents=True, exist_ok=True)
        (target / "Annotations").mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(self.images()):
            if limit is not None and index >= limit:
                break
            shutil.copy2(image, target / "JPEGImages" / image.name)
            annotation = self.annotation_dir / f"{image.stem}.xml"
            if annotation.is_file():
                shutil.copy2(annotation, target / "Annotations" / annotation.name)
        return target


@contextmanager
def dataset_guard(dataset: VocTestDataset):
    """Fail if anything inside the block writes to the shared dataset."""
    before = dataset.fingerprint()
    yield dataset
    after = dataset.fingerprint()
    assert after == before, (
        f"a test modified the shared test dataset at {dataset.root}; "
        "stage a copy with voc_test.stage() instead"
    )


@pytest.fixture(autouse=True)
def _confine_deletions_to_tmp(tmp_path, monkeypatch):
    """Confine the one irreversible thing cleanup does.

    Comparing mtimes of the developer's dataset would misfire: their own
    ``app.py`` may be running and writing to it during the test session, and a
    suite that cries wolf gets ignored. Watching the deletion primitive
    instead is deterministic -- ``fixer._unlink`` only ever sees calls this
    process made. It is also the only one left: the dialog used to delete
    through its own ``_trash``, but the repair pass owns every delete now.

    Tests may delete freely inside their own tmp_path.
    """
    from src.services.cleanup import fixer

    allowed = tmp_path.resolve()

    def check(path) -> Path:
        target = Path(path).resolve()
        assert target == allowed or allowed in target.parents, (
            f"test tried to delete {target}, which is outside its tmp_path "
            f"{allowed}; use voc_test.stage() or synthetic_dataset instead"
        )
        return target

    original_unlink = fixer._unlink

    def guarded_unlink(path):
        check(path)
        return original_unlink(path)

    monkeypatch.setattr(fixer, "_unlink", guarded_unlink)


@pytest.fixture
def locked_file():
    """Hold a file open the way another program would.

    On Windows an open handle without ``FILE_SHARE_DELETE`` makes the unlink
    fail with a sharing violation, which is the real shape of this failure --
    a viewer or a trainer holding the image -- rather than a stubbed one. The
    file stays readable, so the backup copy beside it still succeeds and the
    test exercises the delete step specifically.
    """
    @contextmanager
    def lock(path):
        handle = open(path, "rb")
        try:
            yield Path(path)
        finally:
            handle.close()

    return lock


@pytest.fixture
def run_cleanup(monkeypatch):
    """Drive the cleanup dialog's repair pass, and wait for it.

    ``_clean`` confirms, then hands the work to a worker thread, so a test
    cannot assert on the files until the event loop has been let run. The
    dialog's own busy flag is the signal rather than the shape of the log --
    the log is what is being asserted on.

    Both popups are answered here so no test has to remember them. Quality
    rules are off by default in the UI, and a test that expects a box-less
    image to be deleted has to ask for that, as a user would.
    """
    monkeypatch.setattr("src.widgets.cleanup_dialog.AppDialog.question",
                        classmethod(lambda cls, *args, **kwargs: True))
    monkeypatch.setattr("src.widgets.cleanup_dialog.AppDialog.information",
                        classmethod(lambda cls, *args, **kwargs: None))

    def run(dialog, *, structural: bool = True, quality: bool = False,
            timeout_s: float = 60.0) -> list[str]:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        dialog.fix_structural.setChecked(structural)
        dialog.fix_quality.setChecked(quality)
        dialog._clean()
        deadline = time.monotonic() + timeout_s
        while dialog._scanning and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()
        assert not dialog._scanning, "the repair pass did not finish in time"
        return dialog.log_view.toPlainText().splitlines()

    return run


@pytest.fixture
def voc_test(tmp_path):
    """The designated test dataset, guarded against in-place modification.

    Tests that only read may use ``voc_test.root``. Tests that write must call
    ``voc_test.stage()`` and work on the returned copy, so repeated runs stay
    reproducible and the shared dataset survives.
    """
    if not VOC_TEST_ROOT.is_dir():
        pytest.skip(f"designated test dataset not present: {VOC_TEST_ROOT}")
    with dataset_guard(VocTestDataset(VOC_TEST_ROOT, tmp_path)) as dataset:
        yield dataset


# --------------------------------------------------------------------------
# synthetic dataset factory
# --------------------------------------------------------------------------

#: Defects a test can ask for. Each maps to what the cleanup pass is supposed
#: to detect; "no_object" and "orphan_annotation" are the two the current
#: dialog already handles.
DEFECTS = (
    "no_object",              # image with no annotation at all
    "small_box",              # w < 20 and h < 20
    "zero_area",              # w > 20 but h == 0: passes the small-box rule
    "duplicate_box",          # byte-identical boxes in one image
    "near_duplicate_box",     # IoU > 0.95 but not identical after rounding
    "label_conflict",         # same geometry, different class name
    "out_of_bounds",          # coordinates outside the image
    "size_mismatch",          # <size> disagrees with the real image
    "wrong_depth",            # <depth>4</depth> on a 3-channel JPEG
    "unreadable_image",       # truncated JPEG
    "orphan_annotation",      # annotation file with no image
    "class_name_variants",    # "obj" and "obj " both present
)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Build a throwaway dataset. Never point a test at a real dataset.

    Args:
        name: directory name under tmp_path.
        fmt: "voc", "yolo" or "coco".
        frames: number of images to generate.
        defects: subset of DEFECTS to inject.

    Returns the dataset root Path.
    """
    def build(name="synthetic", *, fmt="voc", width=640, height=480,
              classes=("obj",), frames=6, defects=()):
        unknown = set(defects) - set(DEFECTS)
        if unknown:
            raise ValueError(f"unknown defects: {sorted(unknown)}")
        if fmt not in ("voc", "yolo", "coco"):
            raise ValueError(f"unsupported format: {fmt}")
        if fmt == "coco":
            coco_only = {"size_mismatch", "wrong_depth", "orphan_annotation", "unreadable_image"}
            if set(defects) & coco_only:
                raise ValueError(f"COCO builder does not model: {sorted(set(defects) & coco_only)}")

        root = tmp_path / name
        if fmt == "voc":
            image_dir, annotation_dir = root / "JPEGImages", root / "Annotations"
            suffix = ".xml"
        elif fmt == "yolo":
            image_dir, annotation_dir = root / "images" / "train", root / "labels" / "train"
            suffix = ".txt"
        else:
            image_dir, annotation_dir = root / "images", root
            suffix = ".json"
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        # One box per frame by default, laid out so frame index is visible in
        # the filenames the way a video extraction would produce them.
        plan = []
        for index in range(frames):
            plan.append({
                "stem": f"VIDEO_000000000000_0_{152 + index:06d}",
                "boxes": [(classes[0], 100 + index * 4, 90, 260 + index * 4, 300)],
            })

        # Each defect gets its own frames, packed from index 0, so any
        # combination fits in the fewest frames and no two can overwrite each
        # other. "no_object" takes the tail.
        cursor = 0

        def claim() -> int:
            nonlocal cursor
            slot = cursor
            cursor += 1
            return slot

        if "unreadable_image" in defects:
            plan[claim()]["unreadable"] = True
        if "small_box" in defects:
            plan[claim()]["boxes"] = [(classes[0], 100, 100, 112, 112)]
        if "zero_area" in defects:
            plan[claim()]["boxes"] = [(classes[0], 100, 200, 300, 200)]
        if "duplicate_box" in defects:
            box = (classes[0], 120, 120, 300, 340)
            plan[claim()]["boxes"] = [box, box]
        if "near_duplicate_box" in defects:
            plan[claim()]["boxes"] = [
                (classes[0], 120, 120, 300, 340), (classes[0], 122, 121, 302, 341),
            ]
        if "label_conflict" in defects:
            plan[claim()]["boxes"] = [
                (classes[0], 150, 150, 350, 380), (classes[-1], 150, 150, 350, 380),
            ]
        if "out_of_bounds" in defects:
            plan[claim()]["boxes"] = [(classes[0], -40, -30, width + 60, height + 45)]
        if "class_name_variants" in defects:
            plan[claim()]["boxes"] = [(classes[0], 100, 90, 260, 300)]
            plan[claim()]["boxes"] = [(classes[0] + " ", 110, 95, 270, 310)]

        no_object_count = max(1, frames // 3) if "no_object" in defects else 0
        if cursor + no_object_count > frames:
            raise ValueError(
                f"defects {sorted(defects)} need at least "
                f"{cursor + no_object_count} frames, got {frames}"
            )
        if "no_object" in defects:
            for index in range(frames - no_object_count, frames):
                plan[index]["boxes"] = []

        declared = (width, height)
        if "size_mismatch" in defects:
            declared = (width // 2, height // 2)

        coco_images, coco_annotations, next_ann_id = [], [], 1
        for index, entry in enumerate(plan):
            image_path = image_dir / f"{entry['stem']}.jpg"
            if entry.get("unreadable"):
                image_path.write_bytes(_truncated_jpeg(width, height))
            else:
                Image.new("RGB", (width, height), (20 + index, 30, 40)).save(image_path)

            if fmt == "coco":
                coco_images.append({"id": index + 1, "file_name": f"{entry['stem']}.jpg",
                                    "width": width, "height": height})
                for name, x1, y1, x2, y2 in entry["boxes"]:
                    coco_annotations.append({
                        "id": next_ann_id, "image_id": index + 1,
                        "category_id": list(classes).index(name) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0,
                    })
                    next_ann_id += 1
                continue

            if not entry["boxes"]:
                continue
            if fmt == "voc":
                (annotation_dir / f"{entry['stem']}.xml").write_text(
                    _voc_xml(entry["stem"], entry["boxes"], *declared,
                             depth=4 if "wrong_depth" in defects else 3),
                    encoding="utf-8",
                )
            else:
                (annotation_dir / f"{entry['stem']}.txt").write_text(
                    "\n".join(_yolo_line(name, box, classes, width, height)
                              for name, *box in entry["boxes"]) + "\n",
                    encoding="utf-8",
                )

        if fmt == "coco":
            (annotation_dir / "annotations.json").write_text(json.dumps({
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(classes)],
            }), encoding="utf-8")

        if "orphan_annotation" in defects:
            orphan = "ORPHAN_000001"
            if fmt == "voc":
                (annotation_dir / f"{orphan}.xml").write_text(
                    _voc_xml(orphan, [(classes[0], 10, 10, 60, 60)], *declared, depth=3),
                    encoding="utf-8")
            else:
                (annotation_dir / f"{orphan}.txt").write_text(
                    _yolo_line(classes[0], (10, 10, 60, 60), classes, width, height) + "\n",
                    encoding="utf-8")
        return root
    return build


def _voc_xml(stem, boxes, width, height, *, depth=3) -> str:
    objects = "".join(
        f"<object><name>{name}</name><bndbox>"
        f"<xmin>{int(x1)}</xmin><ymin>{int(y1)}</ymin>"
        f"<xmax>{int(x2)}</xmax><ymax>{int(y2)}</ymax>"
        f"</bndbox></object>"
        for name, x1, y1, x2, y2 in boxes
    )
    return (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        f"<annotation><folder>JPEGImages</folder><filename>{stem}.jpg</filename>"
        f"<size><width>{width}</width><height>{height}</height><depth>{depth}</depth></size>"
        f"{objects}</annotation>"
    )


def _yolo_line(name, box, classes, width, height) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"{list(classes).index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _truncated_jpeg(width, height) -> bytes:
    """A JPEG whose scan data is cut short.

    The header survives, so ``Image.open`` and ``.size`` still succeed -- only
    a full decode (``load()``/``verify()``) fails. That mirrors the real
    failure mode: the file looks fine until a trainer actually reads pixels.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 10, 10)).save(buffer, format="JPEG")
    payload = buffer.getvalue()
    return payload[: len(payload) // 3]
