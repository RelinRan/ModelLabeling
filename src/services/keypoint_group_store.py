from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from src.models.project import KeypointGroup


DEFAULT_KEYPOINT_GROUP_DB = Path.home() / "AppData" / "Local" / "ModelLabeling" / "keypoint_groups.sqlite3"


class KeypointGroupStore:
    """Persistent application-level storage for keypoint name templates.

    Mirrors LabelGroupStore: the database has no dataset or project foreign
    key, so keypoint groups survive dataset switching.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or DEFAULT_KEYPOINT_GROUP_DB)
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
                    CREATE TABLE IF NOT EXISTS keypoint_groups (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        protected INTEGER NOT NULL DEFAULT 0,
                        label TEXT NOT NULL DEFAULT '',
                        sort_order INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS keypoint_names (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        group_id INTEGER NOT NULL,
                        name TEXT NOT NULL,
                        sort_order INTEGER NOT NULL,
                        FOREIGN KEY(group_id) REFERENCES keypoint_groups(id) ON DELETE CASCADE,
                        UNIQUE(group_id, name)
                    );
                    CREATE INDEX IF NOT EXISTS idx_keypoint_names_group_order
                        ON keypoint_names(group_id, sort_order, id);
                    """
                )
                # Databases created before the label column existed.
                columns = {row[1] for row in connection.execute("PRAGMA table_info(keypoint_groups)")}
                if "label" not in columns:
                    connection.execute("ALTER TABLE keypoint_groups ADD COLUMN label TEXT NOT NULL DEFAULT ''")

    def load_or_initialize(self, fallback: list[KeypointGroup]) -> list[KeypointGroup]:
        groups = self.load_groups()
        if groups:
            return groups
        self.save_groups(fallback)
        return self.load_groups()

    def load_groups(self) -> list[KeypointGroup]:
        with closing(self._connect()) as connection:
            group_rows = connection.execute(
                "SELECT id, name, protected, label FROM keypoint_groups ORDER BY sort_order, id"
            ).fetchall()
            name_rows = connection.execute(
                "SELECT group_id, name FROM keypoint_names ORDER BY group_id, sort_order, id"
            ).fetchall()

        names_by_group: dict[int, list[str]] = {}
        for group_id, name in name_rows:
            names_by_group.setdefault(int(group_id), []).append(name)
        return [
            KeypointGroup(name, names_by_group.get(int(group_id), []), bool(protected), (label or "").strip())
            for group_id, name, protected, label in group_rows
        ]

    def save_groups(self, groups: list[KeypointGroup]) -> None:
        """Replace the group set atomically after a user edit."""
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM keypoint_names")
                connection.execute("DELETE FROM keypoint_groups")
                for group_order, group in enumerate(groups):
                    cursor = connection.execute(
                        "INSERT INTO keypoint_groups(name, protected, label, sort_order) VALUES (?, ?, ?, ?)",
                        (group.name, int(group.protected), (group.label or "").strip(), group_order),
                    )
                    group_id = cursor.lastrowid
                    connection.executemany(
                        "INSERT INTO keypoint_names(group_id, name, sort_order) VALUES (?, ?, ?)",
                        [
                            (group_id, name, name_order)
                            for name_order, name in enumerate(group.keypoint_names)
                        ],
                    )
