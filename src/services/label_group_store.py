from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from src.models.annotation import LabelPreset
from src.models.project import LabelGroup


DEFAULT_LABEL_GROUP_DB = Path.home() / "AppData" / "Local" / "ModelLabeling" / "label_groups.sqlite3"


class LabelGroupStore:
    """Persistent application-level storage for user label templates.

    The database intentionally has no dataset or project foreign key. Label
    groups are reusable editor templates and must survive dataset switching.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_LABEL_GROUP_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS label_groups (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        protected INTEGER NOT NULL DEFAULT 0,
                        sort_order INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS label_presets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        group_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        class_id INTEGER NOT NULL,
                        color TEXT NOT NULL DEFAULT '#00e5ff',
                        enabled INTEGER NOT NULL DEFAULT 1,
                        sort_order INTEGER NOT NULL,
                        FOREIGN KEY(group_id) REFERENCES label_groups(id) ON DELETE CASCADE,
                        UNIQUE(group_id, name)
                    );
                    CREATE INDEX IF NOT EXISTS idx_label_presets_group_order
                        ON label_presets(group_id, sort_order, id);
                    """
                )

    def load_or_initialize(self, fallback: list[LabelGroup]) -> list[LabelGroup]:
        groups = self.load_groups()
        if groups:
            return groups
        self.save_groups(fallback)
        return self.load_groups()

    def load_groups(self) -> list[LabelGroup]:
        with closing(self._connect()) as connection:
            group_rows = connection.execute(
                "SELECT id, name, protected FROM label_groups ORDER BY sort_order, id"
            ).fetchall()
            label_rows = connection.execute(
                "SELECT group_id, name, class_id, color, enabled "
                "FROM label_presets ORDER BY group_id, sort_order, id"
            ).fetchall()

        presets_by_group: dict[int, list[LabelPreset]] = {}
        for group_id, name, class_id, color, enabled in label_rows:
            presets_by_group.setdefault(int(group_id), []).append(
                LabelPreset(name, int(class_id), color or "#00e5ff", bool(enabled))
            )
        return [
            LabelGroup(
                name,
                presets_by_group.get(int(group_id), []),
                bool(protected),
            )
            for group_id, name, protected in group_rows
        ]

    def save_groups(self, groups: list[LabelGroup]) -> None:
        """Replace the template set atomically after a user edit."""
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM label_presets")
                connection.execute("DELETE FROM label_groups")
                for group_order, group in enumerate(groups):
                    cursor = connection.execute(
                        "INSERT INTO label_groups(name, protected, sort_order) VALUES (?, ?, ?)",
                        (group.name, int(group.protected), group_order),
                    )
                    group_id = cursor.lastrowid
                    connection.executemany(
                        "INSERT INTO label_presets(group_id, name, class_id, color, enabled, sort_order) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                group_id,
                                preset.name,
                                preset.class_id,
                                preset.color or "#00e5ff",
                                int(preset.enabled),
                                preset_order,
                            )
                            for preset_order, preset in enumerate(group.presets)
                        ],
                    )
