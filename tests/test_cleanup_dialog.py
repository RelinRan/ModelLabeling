import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.widgets.cleanup_dialog import CleanupDialog
from src.widgets.common_dialogs import AppDialog


def _yolo_layout(root) -> None:
    root.mkdir(parents=True)
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "images" / "a.jpg").write_bytes(b"image")
    (root / "images" / "b.jpg").write_bytes(b"image")
    (root / "images" / "c.jpg").write_bytes(b"image")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _wait_for_scan(app, dialog, timeout_ms: int = 10000) -> None:
    elapsed = 0
    while dialog._scanning and elapsed < timeout_ms:
        app.processEvents()
        time.sleep(0.02)
        elapsed += 20
    app.processEvents()


def test_scan_and_cleanup_log_formats(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "yolo"
    _yolo_layout(root)
    # The completion popup is modal; the log itself carries the summary.
    monkeypatch.setattr(AppDialog, "information", classmethod(lambda cls, *a, **k: None))
    dialog = CleanupDialog([], None, str(root))
    dialog.source_path.setText(str(root))
    # Delete straight to unlink: hermetic, no recycle-bin dependency.
    monkeypatch.setattr(dialog, "_trash", lambda path: (path.unlink(missing_ok=True), True)[1])

    dialog._start_scan()
    _wait_for_scan(app, dialog)

    scan_text = dialog.result_label.toPlainText()
    assert scan_text.splitlines()[0] == "[总图片]：3张  [有效标注]：1张  [无标注]：2张"
    assert "[无标注] b.jpg" in scan_text.splitlines()
    assert "[无标注] c.jpg" in scan_text.splitlines()
    assert "[孤立标注] orphan.txt" in scan_text.splitlines()
    assert dialog.confirm_button.isEnabled()

    dialog._clean()
    cleanup_lines = dialog.log_view.toPlainText().splitlines()
    assert cleanup_lines[0] == "[总图片]：1张  [有效标注]：1张  [无标注]：0张"
    assert "[已清理] b.jpg" in cleanup_lines
    assert "[已清理] c.jpg" in cleanup_lines
    assert "[已清理] orphan.txt" in cleanup_lines
    assert cleanup_lines[-1] == "[清理完成]  [总图片]：1张  [有效标注]：1张  [无标注]：0张"
    # Fully annotated files survive; cleaned ones are gone.
    assert (root / "images" / "a.jpg").exists()
    assert not (root / "images" / "b.jpg").exists()
    assert not (root / "labels" / "orphan.txt").exists()

    dialog.close()


def test_scan_progress_line_format(tmp_path):
    app = QApplication.instance() or QApplication([])
    dialog = CleanupDialog([], None, str(tmp_path))
    dialog._on_scan_progress(10, 1000, 1)
    assert dialog.result_label.toPlainText() == "[扫描] 10/1000  1%"
    # Cleanup stays disarmed until a scan finds something to delete.
    assert not dialog.confirm_button.isEnabled()
    dialog.close()


def test_fully_annotated_dataset_reports_nothing_to_clean(tmp_path):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "clean-yolo"
    root.mkdir()
    (root / "images").mkdir()
    (root / "labels").mkdir()
    (root / "images" / "a.jpg").write_bytes(b"image")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    dialog = CleanupDialog([], None, str(root))
    dialog.source_path.setText(str(root))
    dialog._start_scan()
    _wait_for_scan(app, dialog)

    lines = dialog.result_label.toPlainText().splitlines()
    assert lines == ["[总图片]：1张  [有效标注]：1张  [无标注]：0张  [无需清理]"]
    assert not dialog.confirm_button.isEnabled()
    dialog.close()
