import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtWidgets import QApplication
import pytest

import src.widgets.main_window as main_window_module
from src.models.annotation import Annotation, LabelPreset, ShapeType
from src.models.project import ProjectSettings
from src.services.annotation_service import AnnotationService
from src.services.conversion_service import ConversionOptions, ConversionService
from src.services.dataset_detector import DatasetDetector
from src.widgets.cleanup_dialog import CleanupDialog
from src.widgets.main_window import MainWindow


def _wait(app, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _make_nested_yolo(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "classes.txt").write_text("person\ncar\n", encoding="utf-8")

    generated = {
        "train/shared.jpg": ("0 0.25 0.30 0.20 0.20\n", (30, 70, 110)),
        "val/shared.jpg": ("1 0.75 0.50 0.20 0.20\n", (110, 70, 30)),
        "train/edit.jpg": ("0 0.50 0.50 0.30 0.30\n", (60, 120, 80)),
        "val/empty.jpg": ("0 0.50 0.50 0.10 0.10\n", (90, 90, 90)),
    }
    for relative, (label, color) in generated.items():
        image_path = root / "images" / relative
        Image.new("RGB", (200, 140), color).save(image_path, "JPEG")
        (root / "labels" / Path(relative).with_suffix(".txt")).write_text(label, encoding="utf-8")

    # Cleanup must remove this annotation without associating it with either
    # of the two same-named shared.jpg images.
    (root / "labels" / "orphan.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _make_other_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (200, 140), (20, 40, 60)).save(root / "images" / "other.jpg", "JPEG")
    (root / "labels" / "other.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "classes.txt").write_text("other\n", encoding="utf-8")


def _open(window: MainWindow, app: QApplication, root: Path) -> None:
    window._start_open_path(root)
    assert _wait(app, lambda: window.dataset_root is not None and window.dataset_root.resolve() == root.resolve())
    assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
    assert _wait(app, lambda: window.state.current_image is not None)


def _relative(window: MainWindow, path: Path) -> str:
    return path.relative_to(Path(window.settings.image_dir)).as_posix()


def _select(window: MainWindow, app: QApplication, relative: str):
    row = next(
        index for index, item in enumerate(window.image_panel.records)
        if _relative(window, item.path) == relative
    )
    window.image_panel.list.setCurrentRow(row)
    assert _wait(
        app,
        lambda: window.state.current_image is not None
        and _relative(window, window.state.current_image.path) == relative,
    )
    assert _wait(app, lambda: window._annotation_thread is None and window.state.current_image.metadata_loaded)
    return window.state.current_image


def _save_and_wait(window: MainWindow, app: QApplication) -> None:
    window.save_current()
    assert _wait(app, lambda: window._save_thread is None or not window._save_thread.is_alive())
    assert _wait(app, lambda: not window.dirty)


def _add_box(window: MainWindow, label: str, x: float, y: float) -> None:
    window.canvas.push_undo_snapshot()
    annotation = Annotation(
        ShapeType.RECTANGLE,
        label,
        [QPointF(x, y), QPointF(x + 50, y + 40)],
        color="#ffcc00",
    )
    window.canvas.annotations.append(annotation)
    window.canvas._add_annotation_item(annotation)
    window.canvas.annotationCreated.emit(annotation)
    window.canvas.dirtyChanged.emit(True)


def _settings(detected, presets):
    return ProjectSettings(
        image_dir=detected.image_dir,
        annotation_dir=detected.annotation_dir,
        annotation_format=detected.format_name,
        dataset_task=detected.task_name,
        label_presets=presets,
    )


def _snapshot(root: Path, presets: list[LabelPreset]):
    detected = DatasetDetector.detect(root)
    service = AnnotationService()
    settings = _settings(detected, presets)
    result = {}
    for image_path in sorted(
        path for path in detected.image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ):
        loaded = service.load(image_path, detected.annotation_dir, settings)
        assert loaded.error is None
        result[image_path.relative_to(detected.image_dir).as_posix()] = [
            (
                annotation.label,
                tuple(
                    (round(point.x(), 3), round(point.y(), 3))
                    for point in annotation.points
                ),
            )
            for annotation in loaded.annotations
        ]
    return result


def test_generated_complex_chained_edit_filter_cleanup_convert_history_reload(tmp_path, monkeypatch):
    """Exercise a generated dataset through a long, multi-feature GUI workflow."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr("src.widgets.cleanup_dialog.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr("src.widgets.cleanup_dialog.AppDialog.question", classmethod(lambda cls, *a, **k: True))

    source = tmp_path / "source-yolo"
    other = tmp_path / "other-yolo"
    voc = tmp_path / "converted-voc"
    coco = tmp_path / "converted-coco"
    _make_nested_yolo(source)
    _make_other_dataset(other)
    presets = [LabelPreset("person", 0, "#00e5ff"), LabelPreset("car", 1, "#ffcc00")]

    window = MainWindow()
    try:
        _open(window, app, source)
        assert sorted(_relative(window, item.path) for item in window.state.images) == [
            "train/edit.jpg", "train/shared.jpg", "val/empty.jpg", "val/shared.jpg"
        ]

        # Duplicate basename #1: relabel, add, undo, redo, and save.
        _select(window, app, "train/shared.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        _add_box(window, "person", 100, 60)
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        assert window.canvas.undo()
        assert [item.label for item in window.canvas.annotations] == ["car"]
        assert window.canvas.redo()
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        _save_and_wait(window, app)

        # Duplicate basename #2: delete its original box and create a box at
        # a clearly different location. It must never inherit train/shared.
        _select(window, app, "val/shared.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.delete_selected()
        _add_box(window, "person", 20, 80)
        _save_and_wait(window, app)

        _select(window, app, "train/edit.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        _save_and_wait(window, app)

        # Deleting the final box writes an empty annotation file immediately;
        # cleanup should then remove both that image/file and the orphan file.
        _select(window, app, "val/empty.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.delete_selected()
        _save_and_wait(window, app)
        assert (source / "labels" / "val" / "empty.txt").read_text(encoding="utf-8") == ""

        # Chain search + status + label filters, then reset them without
        # losing records from the indexed dataset.
        window.image_panel.search.setText("shared")
        assert _wait(app, lambda: len(window.image_panel.records) == 2)
        window.image_panel.set_label_filter("car")
        window.refresh_image_list()
        assert _wait(app, lambda: len(window.image_panel.records) == 1)
        assert _relative(window, window.image_panel.records[0].path) == "train/shared.jpg"
        window.image_panel.list.setCurrentRow(0)
        assert _wait(app, lambda: _relative(window, window.state.current_image.path) == "train/shared.jpg")
        window.image_panel.set_label_filter("")
        window.image_panel.search.clear()
        window.image_panel.status.setCurrentIndex(window.image_panel.status.findData("unlabeled"))
        assert _wait(app, lambda: len(window.image_panel.records) == 1)
        assert _relative(window, window.image_panel.records[0].path) == "val/empty.jpg"
        window.image_panel.reset_filters()
        assert _wait(app, lambda: len(window.image_panel.records) == 4)
        assert _relative(window, window.state.current_image.path) == "train/shared.jpg"

        before_cleanup = _snapshot(source, presets)
        assert [item[0] for item in before_cleanup["train/shared.jpg"]] == ["car", "person"]
        assert before_cleanup["val/shared.jpg"] == [("person", ((20.0, 80.0), (70.0, 120.0)))]

        dialog = CleanupDialog(window.settings.label_presets, window, str(source), window.settings.language)
        monkeypatch.setattr(dialog, "_trash", lambda path: (path.unlink(missing_ok=True), True)[1])
        dialog._start_scan()
        assert _wait(app, lambda: not dialog._scanning)
        assert sorted(path.relative_to(source).as_posix() for path in dialog._to_delete_images) == ["images/val/empty.jpg"]
        assert sorted(path.relative_to(source).as_posix() for path in dialog._to_delete_annotations) == ["labels/orphan.txt"]
        dialog._clean()
        assert dialog.result() == dialog.DialogCode.Accepted
        window._reload_cleaned_dataset(Path(dialog.cleaned_source))
        assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
        assert _wait(app, lambda: len(window.state.images) == 3)
        dialog.close()

        assert not (source / "images" / "val" / "empty.jpg").exists()
        assert not (source / "labels" / "val" / "empty.txt").exists()
        assert not (source / "labels" / "orphan.txt").exists()
        expected = _snapshot(source, presets)

        # Convert the cleaned source twice and compare every relative image,
        # label, and rectangle after both format boundaries.
        report = ConversionService().convert(ConversionOptions(
            "yolo", source, "voc", voc, presets,
            overwrite=True, source_task="yolo_detection", output_task="voc",
        ))
        assert (report.succeeded, report.failed, report.skipped) == (3, 0, 0)
        assert _snapshot(voc, presets) == expected

        report = ConversionService().convert(ConversionOptions(
            "voc", voc, "coco", coco, presets,
            overwrite=True, source_task="voc", output_task="coco",
        ))
        assert (report.succeeded, report.failed, report.skipped) == (3, 0, 0)
        assert _snapshot(coco, presets) == expected

        # Open a converted dataset, another generated history entry, then the
        # original again. Every open must clear stale filters and stale boxes.
        _open(window, app, coco)
        _select(window, app, "train/shared.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        _select(window, app, "val/shared.jpg")
        assert [item.label for item in window.canvas.annotations] == ["person"]
        assert window.canvas.annotations[0].points[0].x() == pytest.approx(20.0)

        _open(window, app, other)
        _select(window, app, "other.jpg")
        assert [item.label for item in window.canvas.annotations] == ["other"]
        _open(window, app, source)
        assert window.image_panel.search.text() == ""
        assert window.image_panel.selected_status() == "all"
        assert window.image_panel.selected_label() == ""
        assert len(window.state.images) == 3
        _select(window, app, "train/shared.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        _select(window, app, "val/shared.jpg")
        assert [item.label for item in window.canvas.annotations] == ["person"]
        assert window.canvas.annotations[0].points[0].x() == pytest.approx(20.0)

        history = window._history_paths()
        assert str(source.resolve()) in history
        assert str(coco.resolve()) in history
        assert str(other.resolve()) in history
    finally:
        window.close()



def test_generated_rapid_navigation_auto_saves_each_image_without_box_crossover(tmp_path, monkeypatch):
    """Edits followed immediately by navigation must save the image edited, not the next image."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))

    source = tmp_path / "rapid-yolo"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    (source / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
    for index, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
        Image.new("RGB", (200, 140), (40 + index * 30, 80, 120)).save(source / "images" / name, "JPEG")
        (source / "labels" / f"{Path(name).stem}.txt").write_text(
            f"{index % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    presets = [LabelPreset("person", 0, "#00e5ff"), LabelPreset("car", 1, "#ffcc00")]
    window = MainWindow()
    try:
        _open(window, app, source)

        # Do not call save_current: navigate before the 300 ms auto-save timer.
        _select(window, app, "a.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        _add_box(window, "person", 110, 70)
        _select(window, app, "b.jpg")

        # Edit the second image while the first image's save may still be in flight,
        # then immediately navigate again.
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.delete_selected()
        _add_box(window, "person", 15, 20)
        _select(window, app, "c.jpg")

        assert _wait(app, lambda: not window.dirty and (window._save_thread is None or not window._save_thread.is_alive()))
        # Force a true dataset reload rather than trusting in-memory records.
        window._reload_cleaned_dataset(source)
        assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
        _select(window, app, "a.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        assert window.canvas.annotations[1].points[0].x() == pytest.approx(110.0)
        _select(window, app, "b.jpg")
        assert [item.label for item in window.canvas.annotations] == ["person"]
        assert window.canvas.annotations[0].points[0].x() == pytest.approx(15.0)
        _select(window, app, "c.jpg")
        assert [item.label for item in window.canvas.annotations] == ["person"]
        assert window.canvas.annotations[0].points[0].x() == pytest.approx(80.0)
    finally:
        window.close()



def test_generated_dirty_edit_is_saved_before_history_dataset_switch(tmp_path, monkeypatch):
    """Switching datasets before auto-save fires must persist the old dataset only."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))

    source = tmp_path / "switch-source"
    target = tmp_path / "switch-target"
    _make_other_dataset(source)
    _make_other_dataset(target)
    # Give the target a distinct class/box so a stale source save is obvious.
    (target / "classes.txt").write_text("target\n", encoding="utf-8")
    target_original = "0 0.25 0.25 0.1 0.1\n"
    (target / "labels" / "other.txt").write_text(target_original, encoding="utf-8")

    window = MainWindow()
    try:
        _open(window, app, source)
        _select(window, app, "other.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        _add_box(window, "person", 90, 60)
        assert window.dirty

        # Switch immediately, before the delayed auto-save timer can fire.
        _open(window, app, target)
        _select(window, app, "other.jpg")
        assert [item.label for item in window.canvas.annotations] == ["target"]
        assert (target / "labels" / "other.txt").read_text(encoding="utf-8") == target_original

        _open(window, app, source)
        _select(window, app, "other.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        assert window.canvas.annotations[1].points[0].x() == pytest.approx(90.0)
        assert str(source.resolve()) in window._history_paths()
        assert str(target.resolve()) in window._history_paths()
    finally:
        window.close()


def test_generated_paged_dataset_edit_filter_reset_reload_keeps_file_identity(tmp_path, monkeypatch):
    """Exercise the >100-row index page boundary with edit/save/filter/reload chaining."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))

    source = tmp_path / "paged-yolo"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    (source / "classes.txt").write_text("person\ncar\n", encoding="utf-8")
    for index in range(103):
        stem = f"img_{index:04d}"
        Image.new("RGB", (200, 140), (index % 251, 80, 120)).save(source / "images" / f"{stem}.jpg", "JPEG")
        (source / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
        )

    window = MainWindow()
    cache_files = []
    try:
        _open(window, app, source)
        assert window.dataset_total_images == 103
        assert len(window.image_panel.records) == 100
        assert window.image_panel.list_model.canFetchMore()

        window.image_panel.list_model.fetchMore()
        assert _wait(app, lambda: len(window.image_panel.records) == 103), (
            f"records={len(window.image_panel.records)} "
            f"model={len(window.image_panel.list_model.records)} "
            f"total={window.image_panel.list_model._total_count} "
            f"can_fetch={window.image_panel.list_model.canFetchMore()} "
            f"repo_count={window.dataset_index_repository.count() if window.dataset_index_repository else None} "
            f"page={len(window.dataset_index_repository.get_page(100, 100)) if window.dataset_index_repository else None}"
        )
        assert len({record.path for record in window.state.images}) == 103

        _select(window, app, "img_0102.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        _add_box(window, "person", 2, 3)
        _select(window, app, "img_0001.jpg")
        assert _wait(app, lambda: window._save_thread is None or not window._save_thread.is_alive())

        # Filtering to another file and clearing the filter must not use the
        # filtered row index to replace/switch the canonical current image.
        window.image_panel.search.setText("img_0102")
        assert _wait(app, lambda: len(window.image_panel.records) == 1)
        assert _relative(window, window.state.current_image.path) == "img_0001.jpg"
        window.image_panel.search.clear()
        assert _wait(app, lambda: len(window.image_panel.records) == 100)
        assert _relative(window, window.state.current_image.path) == "img_0001.jpg"

        # Reopen from disk, cross the page boundary again, and verify only the
        # edited target owns the two expected boxes.
        window._reload_cleaned_dataset(source)
        assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
        window.image_panel.list_model.fetchMore()
        assert _wait(app, lambda: len(window.image_panel.records) == 103)
        _select(window, app, "img_0102.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
        _select(window, app, "img_0001.jpg")
        assert [item.label for item in window.canvas.annotations] == ["person"]

        if window.dataset_index_repository is not None:
            base = window.dataset_index_repository.path
            cache_files = [base, Path(f"{base}-wal"), Path(f"{base}-shm")]
    finally:
        window.close()
        for path in cache_files:
            path.unlink(missing_ok=True)



def test_generated_close_immediately_flushes_last_unsaved_edit(tmp_path, monkeypatch):
    """Closing inside the autosave debounce window must still persist the edit."""
    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    source = tmp_path / "close-flush"
    _make_other_dataset(source)

    window = MainWindow()
    _open(window, app, source)
    _select(window, app, "other.jpg")
    window.canvas.annotation_items[0].setSelected(True)
    assert window.canvas.update_selected_label("car", "#ffcc00")
    _add_box(window, "person", 90, 60)
    assert window.dirty

    # No processEvents and no wait for the 300 ms timer: close itself is the
    # durability barrier.
    assert window.close()
    assert (source / "labels" / "other.txt").read_text(encoding="utf-8").count("\n") == 2

    reopened = MainWindow()
    try:
        _open(reopened, app, source)
        _select(reopened, app, "other.jpg")
        assert [item.label for item in reopened.canvas.annotations] == ["car", "person"]
    finally:
        reopened.close()


def test_generated_edit_during_inflight_save_is_queued_as_new_snapshot(tmp_path, monkeypatch):
    """A completion callback from save N must never mark newer edit N+1 saved."""
    import threading

    app = QApplication.instance() or QApplication([])
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module,
        "QSettings",
        lambda *args, **kwargs: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.widgets.main_window.AppDialog.information", classmethod(lambda cls, *a, **k: None))
    source = tmp_path / "inflight-save"
    _make_other_dataset(source)

    started = threading.Event()
    release = threading.Event()
    original_run = main_window_module.DatasetAnnotationSaveWorker.run
    calls = {"count": 0}

    def delayed_first_run(worker):
        calls["count"] += 1
        if calls["count"] == 1:
            started.set()
            assert release.wait(10)
        return original_run(worker)

    monkeypatch.setattr(main_window_module.DatasetAnnotationSaveWorker, "run", delayed_first_run)
    window = MainWindow()
    try:
        _open(window, app, source)
        _select(window, app, "other.jpg")
        window.canvas.annotation_items[0].setSelected(True)
        assert window.canvas.update_selected_label("car", "#ffcc00")
        window.save_current()
        assert started.wait(5)

        # Mutate the same image while the first immutable snapshot is writing.
        _add_box(window, "person", 90, 60)
        assert window.dirty
        release.set()
        assert window._flush_pending_annotation_save()
        assert calls["count"] == 2

        window._reload_cleaned_dataset(source)
        assert _wait(app, lambda: window._dataset_scan_completed and window._dataset_thread is None)
        _select(window, app, "other.jpg")
        assert [item.label for item in window.canvas.annotations] == ["car", "person"]
    finally:
        release.set()
        window.close()
