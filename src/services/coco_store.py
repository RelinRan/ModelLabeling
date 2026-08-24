from __future__ import annotations

import json
import os
import sqlite3
import tempfile
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
                    licenses TEXT NOT NULL,
                    dirty INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS coco_images (
                    id INTEGER PRIMARY KEY,
                    file_name TEXT NOT NULL UNIQUE,
                    basename TEXT NOT NULL DEFAULT '',
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
            columns = {row[1] for row in connection.execute("PRAGMA table_info(coco_document)")}
            if "dirty" not in columns:
                connection.execute("ALTER TABLE coco_document ADD COLUMN dirty INTEGER NOT NULL DEFAULT 0")
            image_columns = {row[1] for row in connection.execute("PRAGMA table_info(coco_images)")}
            if "basename" not in image_columns:
                connection.execute("ALTER TABLE coco_images ADD COLUMN basename TEXT NOT NULL DEFAULT ''")
                connection.executemany(
                    "UPDATE coco_images SET basename=? WHERE id=?",
                    [(Path(str(file_name)).name, int(image_id)) for image_id, file_name in connection.execute("SELECT id,file_name FROM coco_images")],
                )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_coco_images_basename ON coco_images(basename)")

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
                "INSERT INTO coco_document(id,info,licenses,dirty) VALUES(1,?,?,0)",
                (json.dumps(document.get("info", {}), ensure_ascii=False),
                 json.dumps(document.get("licenses", []), ensure_ascii=False)),
            )
            connection.executemany(
                "INSERT INTO coco_images(id,file_name,basename,width,height) VALUES(?,?,?,?,?)",
                [(int(item["id"]), str(item.get("file_name", "")), Path(str(item.get("file_name", ""))).name, int(item.get("width", 0)), int(item.get("height", 0))) for item in images],
            )
            connection.executemany(
                "INSERT INTO coco_categories(id,payload) VALUES(?,?)",
                [(int(item["id"]), json.dumps(item, ensure_ascii=False)) for item in categories],
            )
            connection.executemany(
                "INSERT INTO coco_annotations(id,image_id,payload) VALUES(?,?,?)",
                [(int(item["id"]), int(item["image_id"]), json.dumps(item, ensure_ascii=False)) for item in annotations],
            )

    def upsert_image(self, file_name: str, width: int, height: int, categories: list[dict], annotations: list[dict]) -> None:
        """Persist one image and its annotations in one SQLite transaction."""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO coco_document(id,info,licenses,dirty) VALUES(1,'{}','[]',0)"
            )
            image_row = connection.execute(
                "SELECT id,file_name FROM coco_images WHERE basename=? ORDER BY id LIMIT 1",
                (Path(file_name).name,),
            ).fetchone()
            if image_row is None:
                image_id = int(connection.execute("SELECT COALESCE(MAX(id),0)+1 FROM coco_images").fetchone()[0])
                connection.execute(
                    "INSERT INTO coco_images(id,file_name,basename,width,height) VALUES(?,?,?,?,?)",
                    (image_id, file_name, Path(file_name).name, int(width), int(height)),
                )
            else:
                image_id = int(image_row[0])
                connection.execute(
                    "UPDATE coco_images SET width=?,height=? WHERE id=?",
                    (int(width), int(height), image_id),
                )

            existing_categories = {
                str(payload.get("name", "")): (int(category_id), payload)
                for category_id, raw in connection.execute("SELECT id,payload FROM coco_categories")
                for payload in [json.loads(raw)]
            }
            next_category_id = int(connection.execute("SELECT COALESCE(MAX(id),0)+1 FROM coco_categories").fetchone()[0])
            for category in categories:
                name = str(category.get("name", ""))
                if not name:
                    continue
                existing = existing_categories.get(name)
                category_id = existing[0] if existing else next_category_id
                if existing is None:
                    next_category_id += 1
                payload = dict(existing[1]) if existing else {"id": category_id, "name": name, "supercategory": "object"}
                payload.update({key: value for key, value in category.items() if key != "id"})
                payload["id"] = category_id
                connection.execute(
                    "INSERT INTO coco_categories(id,payload) VALUES(?,?) "
                    "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                    (category_id, json.dumps(payload, ensure_ascii=False)),
                )
                existing_categories[name] = (category_id, payload)

            connection.execute("DELETE FROM coco_annotations WHERE image_id=?", (image_id,))
            next_annotation_id = int(connection.execute("SELECT COALESCE(MAX(id),0)+1 FROM coco_annotations").fetchone()[0])
            rows = []
            for annotation in annotations:
                payload = dict(annotation)
                category_name = str(payload.pop("category_name"))
                payload.update({
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": existing_categories[category_name][0],
                })
                rows.append((next_annotation_id, image_id, json.dumps(payload, ensure_ascii=False)))
                next_annotation_id += 1
            connection.executemany(
                "INSERT INTO coco_annotations(id,image_id,payload) VALUES(?,?,?)",
                rows,
            )
            connection.execute("UPDATE coco_document SET dirty=1 WHERE id=1")

    def is_dirty(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT dirty FROM coco_document WHERE id=1").fetchone()
            return bool(row and row[0])

    def export_json(self, path: Path | None = None) -> Path:
        """Atomically export the working database to official COCO JSON."""
        target = Path(path or (self.annotation_dir / "annotations.json"))
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self.read_document()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.stem}-", suffix=".tmp", delete=False,
        ) as temporary:
            temporary.write(json.dumps(document, ensure_ascii=False, indent=2))
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
        with self._connect() as connection:
            connection.execute("UPDATE coco_document SET dirty=0 WHERE id=1")
        return target

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
