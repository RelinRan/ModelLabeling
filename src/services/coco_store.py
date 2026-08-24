from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class CocoAnnotationStore:
    """Normalized local cache for COCO editing.

    COCO JSON remains the interchange format. This store is the mutable
    working copy used by the editor, so a single-image edit does not depend
    on the lifetime or integrity of an external JSON file.
    """

    def __init__(self, annotation_dir: Path) -> None:
        self.annotation_dir = Path(annotation_dir)
        self.path = self.annotation_dir / ".model_labeling.sqlite3"
        self.annotation_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS coco_document (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    info TEXT NOT NULL,
                    licenses TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coco_images (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coco_categories (
                    id INTEGER PRIMARY KEY,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS coco_annotations (
                    id INTEGER PRIMARY KEY,
                    image_id INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(image_id) REFERENCES coco_images(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_coco_annotations_image
                    ON coco_annotations(image_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def is_initialized(self) -> bool:
        with self._connect() as connection:
            return connection.execute("SELECT 1 FROM coco_document WHERE id=1").fetchone() is not None

    def replace_document(self, document: dict) -> None:
        images = document.get("images", [])
        categories = document.get("categories", [])
        annotations = document.get("annotations", [])
        with self._connect() as connection:
            connection.execute("DELETE FROM coco_annotations")
            connection.execute("DELETE FROM coco_images")
            connection.execute("DELETE FROM coco_categories")
            connection.execute("DELETE FROM coco_document")
            connection.execute(
                "INSERT INTO coco_document(id,info,licenses) VALUES(1,?,?)",
                (json.dumps(document.get("info", {}), ensure_ascii=False),
                 json.dumps(document.get("licenses", []), ensure_ascii=False)),
            )
            connection.executemany(
                "INSERT INTO coco_images(id,file_name,width,height) VALUES(?,?,?,?)",
                [(int(item["id"]), str(item.get("file_name", "")), int(item.get("width", 0)), int(item.get("height", 0))) for item in images],
            )
            connection.executemany(
                "INSERT INTO coco_categories(id,payload) VALUES(?,?)",
                [(int(item["id"]), json.dumps(item, ensure_ascii=False)) for item in categories],
            )
            connection.executemany(
                "INSERT INTO coco_annotations(id,image_id,payload) VALUES(?,?,?)",
                [(int(item["id"]), int(item["image_id"]), json.dumps(item, ensure_ascii=False)) for item in annotations],
            )

    def read_document(self) -> dict:
        with self._connect() as connection:
            base = connection.execute("SELECT info,licenses FROM coco_document WHERE id=1").fetchone()
            if base is None:
                return {"images": [], "annotations": [], "categories": []}
            images = [dict(id=row[0], file_name=row[1], width=row[2], height=row[3]) for row in connection.execute("SELECT id,file_name,width,height FROM coco_images ORDER BY id")]
            categories = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM coco_categories ORDER BY id")]
            annotations = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM coco_annotations ORDER BY id")]
        return {
            "info": json.loads(base[0]), "licenses": json.loads(base[1]),
            "images": images, "annotations": annotations, "categories": categories,
        }

