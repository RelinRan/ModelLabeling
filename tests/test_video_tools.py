from pathlib import Path

from PIL import Image

from src.models.annotation import LabelPreset
from src.services.video_tools import synthesize_dataset

def test_synthesis_creates_structured_yolo_dataset_from_nested_frames(tmp_path: Path):
    source = tmp_path / "frames" / "camera-a"
    source.mkdir(parents=True)
    Image.new("RGB", (16, 12), "white").save(source / "frame-001.jpg")

    output = tmp_path / "dataset"
    progress = []
    report = synthesize_dataset(
        tmp_path / "frames", output, "yolo", [LabelPreset("person", 0, "#ff0000")],
        progress_callback=lambda current, total: progress.append((current, total)),
    )

    assert report.succeeded == 1
    assert progress[-1] == (1, 1)
    assert (output / "images" / "camera-a__frame-001.jpg").is_file()
    assert (output / "labels" / "camera-a__frame-001.txt").is_file()
    assert (output / "classes.txt").read_text(encoding="utf-8").strip() == "person"

def test_synthesis_rejects_nonempty_output(tmp_path: Path):
    source = tmp_path / "frames"
    source.mkdir()
    Image.new("RGB", (4, 4)).save(source / "one.jpg")
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "keep.txt").write_text("keep")

    try:
        synthesize_dataset(source, output, "voc", [])
    except ValueError as exc:
        assert "\u5fc5\u987b\u4e3a\u7a7a" in str(exc)
    else:
        raise AssertionError("nonempty output directory should be rejected")


def test_video_tool_dialogs_have_plain_bordered_forms_and_default_fps():
    from PySide6.QtWidgets import QApplication, QLabel
    from src.widgets.video_tools_dialogs import VideoFrameDialog, DatasetSynthesisDialog

    app = QApplication.instance() or QApplication([])
    video_dialog = VideoFrameDialog()
    assert video_dialog.fps.text() == "5"
    assert video_dialog.fps.validator().bottom() == 1
    assert video_dialog.fps.validator().top() == 120
    assert not video_dialog.findChildren(QLabel, "sectionCardTitle")
    assert not video_dialog.findChildren(QLabel, "sectionCardDot")
    synthesis_dialog = DatasetSynthesisDialog()
    assert not synthesis_dialog.findChildren(QLabel, "sectionCardTitle")
    assert not synthesis_dialog.findChildren(QLabel, "sectionCardDot")

def test_video_tool_help_is_localized_and_documents_progress():
    from PySide6.QtWidgets import QApplication, QLabel, QPlainTextEdit, QWidget
    from src.widgets.help_dialogs import AboutDialog, ShortcutsDialog, UsageGuideDialog

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.settings = type("Settings", (), {"language": "zh_CN"})()
    shortcuts = ShortcutsDialog(parent)
    shortcut_text = " ".join(label.text() for label in shortcuts.findChildren(QLabel))
    assert "\u89c6\u9891\u63d0\u5e27" in shortcut_text
    assert "\u6570\u636e\u5408\u6210" in shortcut_text
    about = AboutDialog(parent)
    assert about.windowTitle() == "\u5173\u4e8e\u8f6f\u4ef6"
    guide = UsageGuideDialog(parent)
    guide_text = " ".join(page.toPlainText() for page in guide.findChildren(QPlainTextEdit))
    assert "\u72b6\u6001\u680f\u53f3\u4fa7" in guide_text
    assert "Ctrl+V+F" in guide_text and "Ctrl+D+S" in guide_text

def test_usage_guide_opens_at_its_default_size():
    """700x420 is the size the manual is laid out for: the Chinese guide's
    longest line is 564px of the 596px text viewport, so it reads without a
    horizontal scrollbar. Shown rather than merely constructed -- a layout's
    minimum size overrides the requested one at show time."""
    from PySide6.QtWidgets import QApplication, QWidget
    from src.widgets.help_dialogs import UsageGuideDialog

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.settings = type("Settings", (), {"language": "zh_CN"})()
    guide = UsageGuideDialog(parent)
    guide.show()
    app.processEvents()
    try:
        assert (guide.width(), guide.height()) == (700, 420)
    finally:
        guide.close()

def test_video_tool_dialogs_remember_last_directories(monkeypatch, tmp_path: Path):
    from PySide6.QtWidgets import QApplication
    import src.widgets.video_tools_dialogs as dialogs

    app = QApplication.instance() or QApplication([])
    stored = {}

    class FakeSettings:
        def __init__(self, *_args):
            pass

        def value(self, key, default=""):
            return stored.get(key, default)

        def setValue(self, key, value):
            stored[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(dialogs, "QSettings", FakeSettings)
    video_source = tmp_path / "videos"
    video_source.mkdir()
    video_output = tmp_path / "frames"
    synthesis_source = tmp_path / "input-frames"
    synthesis_source.mkdir()
    synthesis_output = tmp_path / "dataset"

    extract = dialogs.VideoFrameDialog()
    extract.video_dir.setText(str(video_source))
    extract.output_dir.setText(str(video_output))
    extract.accept()
    reopened_extract = dialogs.VideoFrameDialog()
    assert reopened_extract.video_dir.text() == str(video_source)
    assert reopened_extract.output_dir.text() == str(video_output)

    synthesis = dialogs.DatasetSynthesisDialog()
    synthesis.source_dir.setText(str(synthesis_source))
    synthesis.output_dir.setText(str(synthesis_output))
    synthesis.accept()
    reopened_synthesis = dialogs.DatasetSynthesisDialog()
    assert reopened_synthesis.source_dir.text() == str(synthesis_source)
    assert reopened_synthesis.output_dir.text() == str(synthesis_output)
