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
    """Global dark theme: refined hover/focus states, slim scrollbars,
    styled menus, tooltips, and headers for a tighter, polished surface."""
    stylesheet = """
    QMainWindow, QDialog { background: #2B2D30; color: #D7DAE0; }
    QWidget { color: #D7DAE0; }
    QWidget#workbenchCentral { background: #1F2023; }
    QWidget#projectPanel { background: #25272A; border: none; }
    QSplitter { background: #1F2023; }

    /* Menus: tighter items, accent hover, subtle separators */
    QMenuBar { background: #2B2D30; color: #C9CDD4; border-bottom: 1px solid #3C4046; padding: 1px 6px; }
    QMenuBar::item { padding: 4px 9px; border-radius: 4px; }
    QMenuBar::item:selected { background: #3A3D42; color: #FFFFFF; }
    QMenu { background: #303236; border: 1px solid #4A4E55; border-radius: 6px; padding: 5px; }
    QMenu::item { padding: 5px 24px 5px 12px; border-radius: 4px; }
    QMenu::item:selected { background: #2E436E; color: #FFFFFF; }
    QMenu::separator { height: 1px; background: #3C4046; margin: 4px 8px; }

    QToolBar { background: #25272A; border-bottom: 1px solid #464A50; spacing: 4px; padding: 5px 8px; }

    /* Buttons: crisper hover ring, clear pressed + default states */
    QToolButton, QPushButton { background: #35383D; border: 1px solid #4A4E55; border-radius: 5px; padding: 4px 12px; color: #D7DAE0; }
    QToolButton:hover, QPushButton:hover { background: #3E4249; border-color: #6A84B8; color: #FFFFFF; }
    QToolButton:pressed, QPushButton:pressed { background: #2E436E; border-color: #6A84B8; color: #FFFFFF; }
    QPushButton:focus { border-color: #6A84B8; }
    QPushButton:default { border-color: #6A84B8; color: #FFFFFF; background: #3A4E78; }
    QToolButton:disabled, QPushButton:disabled { color: #737780; border-color: #3A3D42; background: #303236; }

    /* Inputs: hover hint, focused ring */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #2F3135; border: 1px solid #464A50; border-radius: 5px; padding: 4px 6px; selection-background-color: #2E436E; selection-color: #FFFFFF; }
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #565B63; }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #6A84B8; background: #303236; }
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled { color: #737780; }
    QComboBox QAbstractItemView { background: #303236; color: #D7DAE0; selection-background-color: #2E436E; selection-color: #FFFFFF; border: 1px solid #4A4E55; padding: 2px; }
    QComboBox::drop-down { width: 22px; border: none; background: transparent; }

    QListWidget { background: #25272A; border: 1px solid #464A50; border-radius: 5px; padding: 4px; selection-background-color: #2E436E; selection-color: #FFFFFF; }
    QListWidget::item { border-radius: 4px; }
    QListWidget#settingsCategories, QListWidget#settingsCategories:focus { background: transparent; border: none; padding: 0; outline: 0; }
    QListWidget#settingsCategories::item { height: 30px; padding: 0 8px; margin: 0 0 2px 0; border-radius: 5px; border: none; outline: none; }
    QListWidget#settingsCategories::item:hover { background: #343940; color: #FFFFFF; }
    QListWidget#settingsCategories::item:selected, QListWidget#settingsCategories::item:selected:focus { background: #2E436E; color: #FFFFFF; font-weight: 600; border: none; outline: 0; }

    /* Tables and headers (statistics dialogs) */
    QHeaderView::section { background: #2B2D30; color: #AEB3BB; border: none; border-bottom: 1px solid #464A50; padding: 4px 8px; }
    QTableWidget { background: #25272A; alternate-background-color: #282A2E; gridline-color: #3A3D42; border: 1px solid #464A50; border-radius: 5px; }
    QTableWidget::item { padding: 3px 6px; }
    QTableWidget::item:selected { background: #2E436E; color: #FFFFFF; }

    QProgressBar { background: #303236; border: 1px solid #464A50; border-radius: 4px; text-align: center; color: #D7DAE0; }
    QProgressBar::chunk { background: #4C6CA8; border-radius: 3px; }

    /* Slim scrollbars with hover feedback */
    QScrollBar:vertical { background: transparent; width: 9px; margin: 2px 1px; }
    QScrollBar::handle:vertical { background: #4A4E55; min-height: 26px; border-radius: 4px; }
    QScrollBar::handle:vertical:hover { background: #5C626B; }
    QScrollBar::handle:vertical:pressed { background: #6A84B8; }
    QScrollBar:horizontal { background: transparent; height: 9px; margin: 1px 2px; }
    QScrollBar::handle:horizontal { background: #4A4E55; min-width: 26px; border-radius: 4px; }
    QScrollBar::handle:horizontal:hover { background: #5C626B; }
    QScrollBar::handle:horizontal:pressed { background: #6A84B8; }
    QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
    QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

    /* Tooltips match the theme */
    QToolTip { background: #303236; color: #E6E9ED; border: 1px solid #4A4E55; border-radius: 4px; padding: 4px 8px; }

    QTextEdit, QPlainTextEdit { background: #25272A; color: #D7DAE0; border: 1px solid #464A50; border-radius: 5px; selection-background-color: #2E436E; }

    QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; border: 1px solid #565B63; border-radius: 3px; background: #2F3135; }
    QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #4C6CA8; border-color: #6A84B8; }
    QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #6A84B8; }
    QRadioButton::indicator { border-radius: 7px; }

    QGraphicsView#annotationCanvas { background: #1F2023; border: none; }
    QGraphicsView#annotationCanvas:disabled { background: #10151C; color: #D7DAE0; }
    QListWidget#imageFileList:disabled, QScrollArea:disabled, QComboBox:disabled { background: #25272A; color: #D7DAE0; }
    QSplitter::handle { background: #3A3D42; width: 6px; }
    QSplitter::handle:hover { background: #6A84B8; }

    QStatusBar { background: #25272A; border-top: 1px solid #3C4046; color: #A4A8B0; }
    QStatusBar QLabel { color: #A4A8B0; padding: 0 3px; }
    QStatusBar QLabel#statusTask { color: #B8C7E6; }
    QFrame#statusSeparator { color: #3C4046; background: #3C4046; max-width: 1px; margin: 3px 6px; }

    QLabel#panelTitle { color: #FFFFFF; font-size: 13px; font-weight: 700; letter-spacing: 0.3px; }

    /* Section cards: distinct rounded blocks for grouped settings */
    QFrame#sectionCard { background: #282A2F; border: 1px solid #3C4148; border-radius: 8px; }
    QFrame#sectionCard[variant="distribution"] { background: #2F3237; }
    QLabel#sectionCardTitle {
        color: #A9B4C6; font-size: 12px; font-weight: 700; letter-spacing: 1px;
        border: none; background: transparent; padding: 1px 0;
    }
    QLabel#sectionCardDot { background: #5B7FCC; border-radius: 4px; border: none; }
    QLabel#imageInfoOverlay { color: #FFFFFF; background: transparent; }
    QLabel#statusMuted { color: #8B9099; }
    QDialog { background: #2B2D30; }
    QDialogButtonBox QPushButton { min-width: 76px; }
    QFrame#labelArea, QFrame#labelActions { background: #25272A; border: 1px solid #3E4147; border-radius: 6px; }
    """
    arrow_path = (Path(__file__).resolve().parents[2] / "icons" / "ic_arrow_down.png").as_posix()
    return stylesheet + f"""
    QComboBox::down-arrow {{
        image: url({arrow_path});
        width: 14px;
        height: 14px;
    }}
    QComboBox[singleGroup="true"]::drop-down {{ width: 0px; border: none; }}
    QComboBox[singleGroup="true"]::down-arrow {{ image: none; width: 0px; height: 0px; }}
    """
