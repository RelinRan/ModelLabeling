import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PIL import Image

from src.widgets.cleanup_dialog import CleanupDialog
from src.widgets.common_dialogs import AppDialog


def _yolo_layout(root) -> None:
    """Three real JPEGs, one annotated, plus an orphan label file.

    The images are decodable on purpose: a fixture of placeholder bytes is a
    corrupt dataset, and the validator is right to say so.
    """
    root.mkdir(parents=True)
    (root / "images").mkdir()
    (root / "labels").mkdir()
    for name in ("a", "b", "c"):
        Image.new("RGB", (64, 64), (10, 20, 30)).save(root / "images" / f"{name}.jpg")
    # 0.5 of 64px is 32px, comfortably over the 20px threshold -- a smaller
    # box would be dropped by the quality rules and take its image with it.
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (root / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def _wait_for_scan(app, dialog, timeout_ms: int = 10000) -> None:
    elapsed = 0
    while dialog._scanning and elapsed < timeout_ms:
        app.processEvents()
        time.sleep(0.02)
        elapsed += 20
    app.processEvents()


def test_scan_and_cleanup_log_formats(tmp_path, monkeypatch, run_cleanup):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "yolo"
    _yolo_layout(root)
    # Both popups are modal; the log itself carries the summary.
    monkeypatch.setattr(AppDialog, "information", classmethod(lambda cls, *a, **k: None))
    monkeypatch.setattr(AppDialog, "question", classmethod(lambda cls, *a, **k: True))
    dialog = CleanupDialog([], None, str(root))
    dialog.source_path.setText(str(root))

    dialog._start_scan()
    _wait_for_scan(app, dialog)

    scan_text = dialog.result_label.toPlainText()
    assert scan_text.splitlines()[0] == "[总图片]：3张  [有效标注]：1张  [无标注]：2张"
    assert "[无标注] b.jpg" in scan_text.splitlines()
    assert "[无标注] c.jpg" in scan_text.splitlines()
    assert "[孤立标注] orphan.txt" in scan_text.splitlines()
    assert dialog.confirm_button.isEnabled()

    # An image with no boxes is a quality call, not a broken invariant, so it
    # is off by default and has to be asked for.
    cleanup_lines = run_cleanup(dialog, quality=True)
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
    """Nothing to delete *and* nothing for the validator to report.

    The image has to be a real, decodable one and the class table has to
    exist: otherwise the validator legitimately finds a corrupt image or a
    dataset that does not describe its own classes, and the pane is no longer
    empty for a reason that has nothing to do with this test.
    """
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "clean-yolo"
    root.mkdir()
    (root / "images").mkdir()
    (root / "labels").mkdir()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(root / "images" / "a.jpg")
    (root / "labels" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (root / "labels" / "classes.txt").write_text("obj\n", encoding="utf-8")

    dialog = CleanupDialog([], None, str(root))
    dialog.source_path.setText(str(root))
    dialog._start_scan()
    _wait_for_scan(app, dialog)

    lines = dialog.result_label.toPlainText().splitlines()
    assert lines == ["[总图片]：1张  [有效标注]：1张  [无标注]：0张  [无需清理]"]
    assert not dialog.confirm_button.isEnabled()
    dialog.close()


def _voc_layout(root, depth: int = 3) -> None:
    """A minimal, otherwise perfect VOC dataset with one box per image."""
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    for index in range(2):
        stem = f"frame_{index:04d}"
        Image.new("RGB", (64, 64), (10, 20, 30)).save(root / "JPEGImages" / f"{stem}.jpg")
        (root / "Annotations" / f"{stem}.xml").write_text(
            "<?xml version='1.0' encoding='utf-8'?>\n"
            f"<annotation><folder>JPEGImages</folder><filename>{stem}.jpg</filename>"
            "<size><width>64</width><height>64</height>"
            f"<depth>{depth}</depth></size>"
            "<object><name>obj</name><bndbox>"
            "<xmin>4</xmin><ymin>4</ymin><xmax>40</xmax><ymax>40</ymax>"
            "</bndbox></object></annotation>",
            encoding="utf-8",
        )


def _scan(app, root) -> CleanupDialog:
    dialog = CleanupDialog([], None, str(root))
    dialog.source_path.setText(str(root))
    dialog._start_scan()
    _wait_for_scan(app, dialog)
    return dialog


def test_scan_report_includes_validator_findings(tmp_path):
    """A defect the deletion pass cannot see still reaches the report."""
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "voc-depth"
    _voc_layout(root, depth=4)  # the images are 3-channel RGB

    dialog = _scan(app, root)
    lines = dialog.result_label.toPlainText().splitlines()

    # No "[无需清理]": there is nothing to delete, but there is work to do.
    assert lines[0] == "[总图片]：2张  [有效标注]：2张  [无标注]：0张"
    assert any(line.startswith("[校验] 结构 2") for line in lines), lines
    assert "[通道数不符] 2  (voc.depth.mismatch)" in lines
    assert "    frame_0000.jpg" in lines
    dialog.close()


def test_scan_report_omits_the_validation_block_when_clean(tmp_path):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "voc-clean"
    _voc_layout(root, depth=3)

    dialog = _scan(app, root)
    lines = dialog.result_label.toPlainText().splitlines()
    assert lines == ["[总图片]：2张  [有效标注]：2张  [无标注]：0张  [无需清理]"]
    dialog.close()


def _declared_depth(root, stem: str) -> str:
    import xml.etree.ElementTree as ET

    size = ET.parse(root / "Annotations" / f"{stem}.xml").getroot().find("size")
    return size.findtext("depth")


def test_quality_rules_are_off_by_default(tmp_path):
    """A quality rule encodes annotation policy, so it is the user's call;
    only the repairs with one defensible answer are on to begin with."""
    app = QApplication.instance() or QApplication([])
    dialog = CleanupDialog([], None, str(tmp_path))
    assert dialog.fix_structural.isChecked()
    assert not dialog.fix_quality.isChecked()
    assert "结构修复" in dialog.policy_detail.text()
    dialog.close()


def test_the_structural_toggle_governs_the_structural_repairs(tmp_path, run_cleanup):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "voc-depth"
    _voc_layout(root, depth=4)  # the images are 3-channel RGB
    dialog = _scan(app, root)
    stem = "frame_0000"
    assert _declared_depth(root, stem) == "4"

    # Quality policy only: the file declares the wrong depth and nothing does
    # anything about it.
    lines = run_cleanup(dialog, structural=False, quality=True)
    assert _declared_depth(root, stem) == "4"
    assert not any(line.startswith("[已改写]") for line in lines), lines

    # Structural on: one defensible answer, so it is simply applied.
    lines = run_cleanup(dialog, structural=True, quality=False)
    assert _declared_depth(root, stem) == "3"
    assert "[修复] 通道数不符 2  (voc.depth.mismatch)" in lines
    assert any(line.startswith("[已改写] 2 个标注文件") for line in lines), lines
    # The original files are recoverable, and the log says where from.
    assert any(line.startswith("[备份]") for line in lines), lines
    dialog.close()


def test_min_box_size_control_changes_the_findings(tmp_path):
    """The threshold is a control, not a constant: it decides what counts as
    an unusable box, so changing it must change the report."""
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "voc-small"
    _voc_layout(root, depth=3)  # each box is 36x36 px

    dialog = CleanupDialog([], None, str(root))
    dialog.source_path.setText(str(root))
    # 36x36 boxes are usable at the default 20px ...
    dialog._start_scan()
    _wait_for_scan(app, dialog)
    assert "voc.box.small" not in dialog.result_label.toPlainText()

    # ... but not once the threshold is raised above them.
    dialog.min_box_size.setValue(40)
    dialog._start_scan()
    _wait_for_scan(app, dialog)
    text = dialog.result_label.toPlainText()
    assert "[过小框] 2  (voc.box.small)" in text
    dialog.close()
