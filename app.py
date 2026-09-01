from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QFont, QIcon, QFontDatabase
from PySide6.QtWidgets import QApplication

from src.app_paths import resource_path
from src.widgets.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ModelLabeling")
    app.setOrganizationName("ModelLabeling")
    # Prefer a Windows CJK font so Chinese labels and menu text do not depend
    # on platform fallback behavior. Keep the default fallback when none is installed.
    families = set(QFontDatabase.families())
    for family in ("Microsoft YaHei UI", "Microsoft YaHei", "SimSun"):
        if family in families:
            app.setFont(QFont(family, 9))
            break
    icon_path = resource_path("icons/icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
