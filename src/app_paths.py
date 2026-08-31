from __future__ import annotations

"""Resource location that also works inside a PyInstaller bundle.

Read-only resources (icon.png, icons/) live at the project root in a source
checkout. A frozen app unpacks them into sys._MEIPASS, so every lookup must
go through this helper to keep icons working in the shipped binaries.
"""

import sys
from pathlib import Path


def project_root() -> Path:
    """The source tree root, or the bundle root when frozen."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def resource_path(relative: str) -> Path:
    """Absolute path of a bundled resource ('' = the root itself)."""
    return project_root() / relative if relative else project_root()
