from __future__ import annotations

from pathlib import Path


IDEA_PALETTE = {
    "window": "#2B2D30",
    "panel": "#25272A",
    "input": "#303236",
    "border": "#464A50",
    "accent": "#2E436E",
}


def idea_stylesheet() -> str:
    stylesheet = """
    QMainWindow, QDialog { background: #2B2D30; color: #D7DAE0; }
    QWidget { color: #D7DAE0; }
    QWidget#workbenchCentral { background: #1F2023; }
    QWidget#projectPanel { background: #25272A; border: none; }
    QSplitter { background: #1F2023; }
    QMenuBar { background: #2B2D30; color: #D7DAE0; border-bottom: 1px solid #464A50; padding: 2px 6px; }
    QMenuBar::item { padding: 5px 9px; border-radius: 4px; }
    QMenuBar::item:selected, QMenu::item:selected { background: #3A3D42; color: #FFFFFF; }
    QMenu { background: #303236; border: 1px solid #464A50; padding: 4px; }
    QMenu::item { padding: 6px 24px 6px 10px; }
    QToolBar { background: #25272A; border-bottom: 1px solid #464A50; spacing: 4px; padding: 5px 8px; }
    QToolButton, QPushButton { background: #35383D; border: 1px solid #464A50; border-radius: 5px; padding: 5px 10px; color: #D7DAE0; }
    QToolButton:hover, QPushButton:hover { background: #41454C; border-color: #6A84B8; color: #FFFFFF; }
    QToolButton:pressed, QPushButton:pressed { background: #2E436E; }
    QToolButton:disabled, QPushButton:disabled { color: #737780; border-color: #3A3D42; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #303236; border: 1px solid #464A50; border-radius: 5px; padding: 5px; selection-background-color: #2E436E; }
    QListWidget { background: #25272A; border: 1px solid #464A50; border-radius: 5px; padding: 5px; selection-background-color: #2E436E; }
    QListWidget#settingsCategories, QListWidget#settingsCategories:focus { background: transparent; border: none; padding: 0; outline: 0; }
    QListWidget#settingsCategories::item { height: 30px; padding: 0 8px; margin: 0 0 2px 0; border-radius: 5px; border: none; outline: none; }
    QListWidget#settingsCategories::item:hover { background: #343940; color: #FFFFFF; }
    QListWidget#settingsCategories::item:selected, QListWidget#settingsCategories::item:selected:focus { background: #2E436E; color: #FFFFFF; font-weight: 600; border: none; outline: 0; }
    QComboBox QAbstractItemView { background: #303236; color: #D7DAE0; selection-background-color: #2E436E; }
    QComboBox::drop-down { width: 24px; border: none; background: transparent; }
    QProgressBar { background: #303236; border: 1px solid #464A50; border-radius: 5px; text-align: center; color: #D7DAE0; }
    QProgressBar::chunk { background: #2E436E; border-radius: 5px; }
    QGraphicsView#annotationCanvas { background: #1F2023; border: none; }
    QGraphicsView#annotationCanvas:disabled { background: #10151C; color: #D7DAE0; }
    QListWidget#imageFileList:disabled, QScrollArea:disabled, QComboBox:disabled { background: #25272A; color: #D7DAE0; }
    QSplitter::handle { background: #464A50; width: 8px; }
    QStatusBar { background: #25272A; border-top: 1px solid #464A50; color: #A4A8B0; }
    QStatusBar QLabel { color: #A4A8B0; padding: 0 4px; }
    QStatusBar QLabel#statusTask { color: #B8C7E6; }
    QFrame#statusSeparator { color: #464A50; background: #464A50; max-width: 1px; margin: 4px 5px; }
    QScrollBar:vertical { background: #25272A; width: 10px; margin: 0; }
    QScrollBar::handle:vertical { background: #555A63; min-height: 24px; border-radius: 5px; }
    QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
    QLabel#panelTitle { color: #FFFFFF; font-size: 13px; font-weight: 700; }
    QLabel#imageInfoOverlay { color: #FFFFFF; background: transparent; }
    QLabel#statusMuted { color: #8B9099; }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #6A84B8; }
    QDialog { background: #2B2D30; }
    QDialogButtonBox QPushButton { min-width: 76px; }
    QFrame#labelArea, QFrame#labelActions { background: #25272A; border: 1px solid #464A50; border-radius: 5px; }
    """
    arrow_path = (Path(__file__).resolve().parents[2] / "icons" / "ic_arrow_down.png").as_posix()
    return stylesheet + f"""
    QComboBox::down-arrow {{
        image: url({arrow_path});
        width: 15px;
        height: 15px;
    }}
    QComboBox[singleGroup="true"]::drop-down {{ width: 0px; border: none; }}
    QComboBox[singleGroup="true"]::down-arrow {{ image: none; width: 0px; height: 0px; }}
    """
