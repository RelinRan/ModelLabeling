"""The image list search field carries the File Filter entry button."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, QSize
from PySide6.QtWidgets import QApplication

import src.widgets.main_window as main_window_module
from src.widgets.image_list_panel import ImageListPanel
from src.widgets.main_window import MainWindow


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_filter_button_lives_inside_the_search_field():
    app = _app()
    panel = ImageListPanel()
    panel.resize(260, 360)
    panel.show()
    app.processEvents()
    try:
        button = panel.search.action_button
        assert button.parent() is panel.search
        assert button.objectName() == "searchActionButton"
        assert button.size() == QSize(20, 20)
        # Same compact visual language as the history row's menu button.
        assert "background: transparent" in button.styleSheet()
        assert "border: none" in button.styleSheet()
        assert "QToolButton#searchActionButton:hover" in button.styleSheet()
        # Right-aligned inside the field and vertically centred.
        assert button.geometry().right() < panel.search.width()
        assert button.geometry().left() > panel.search.width() // 2
        assert abs(button.geometry().center().y() - panel.search.rect().center().y()) <= 2
    finally:
        panel.close()


def test_clicking_the_search_button_requests_the_filter_dialog():
    app = _app()
    panel = ImageListPanel()
    panel.show()
    app.processEvents()
    requested: list[bool] = []
    panel.filterRequested.connect(lambda: requested.append(True))
    try:
        panel.search.action_button.click()
        assert requested == [True]
    finally:
        panel.close()


def test_the_text_below_the_search_field_is_gone():
    panel = ImageListPanel()
    try:
        assert not hasattr(panel, "hint_label")
        assert not hasattr(panel, "set_result_hint")
    finally:
        panel.close()


def test_search_button_tooltip_follows_the_language():
    panel = ImageListPanel()
    try:
        panel.set_language("zh_CN")
        assert panel.search.action_button.toolTip() == "文件筛选"
        panel.set_language("en_US")
        assert panel.search.action_button.toolTip() == "File Filter"
    finally:
        panel.close()


def test_main_window_search_button_opens_the_file_filter_dialog(tmp_path, monkeypatch):
    app = _app()
    settings_ini = tmp_path / "settings.ini"
    monkeypatch.setattr(
        main_window_module, "QSettings",
        lambda *a, **k: QSettings(str(settings_ini), QSettings.Format.IniFormat),
    )
    monkeypatch.setattr("src.services.label_group_store.DEFAULT_LABEL_GROUP_DB", tmp_path / "label_groups.sqlite3")
    monkeypatch.setattr("src.services.keypoint_group_store.DEFAULT_KEYPOINT_GROUP_DB", tmp_path / "keypoint_groups.sqlite3")

    opened: list[tuple[str, str, str]] = []

    class FakeFilterDialog:
        class DialogCode:
            Accepted = 1

        def __init__(self, query, status, label, parent=None) -> None:
            opened.append((query, status, label))

        def exec(self) -> int:
            return 0

        def values(self) -> tuple[str, str, str]:
            return "", "all", ""

    monkeypatch.setattr(main_window_module, "ImageFilterDialog", FakeFilterDialog)
    window = MainWindow()
    try:
        window.image_panel.search.action_button.click()
        assert len(opened) == 1
        app.processEvents()
    finally:
        window.close()