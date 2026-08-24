from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class IndexedImage:
    id: int
    path: Path
    relative_path: str
    file_name: str
    file_size: int
    mtime_ns: int
    width: int = 0
    height: int = 0
    annotation_path: Path | None = None
    annotation_status: str = "unknown"
    annotation_labels: tuple[str, ...] = ()
    annotation_size: int = 0
    annotation_mtime_ns: int = 0


class DatasetIndexRepository:
    """Persistent path index for large datasets.

    The index deliberately stores file metadata only. Image pixels and
    annotations are loaded on demand when the user selects an image.
    """

    def __init__(self, dataset_root: Path, image_dir: Path, annotation_dir: Path, annotation_format: str) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.image_dir = Path(image_dir).resolve()
        self.annotation_dir = Path(annotation_dir).resolve()
        self.annotation_format = annotation_format.lower()
        cache_dir = Path.home() / "AppData" / "Local" / "ModelLabeling" / "index"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(str(self.dataset_root).casefold().encode("utf-8")).hexdigest()
        self.path = cache_dir / f"{key}.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    annotation_path TEXT,
                    annotation_size INTEGER NOT NULL DEFAULT 0,
                    annotation_mtime_ns INTEGER NOT NULL DEFAULT 0,
                    annotation_status TEXT NOT NULL DEFAULT 'unknown',
                    sort_key TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS image_labels (
                    image_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    PRIMARY KEY(image_id, label),
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_images_sort ON images(sort_key, id);
                CREATE INDEX IF NOT EXISTS idx_images_name ON images(file_name);
                CREATE INDEX IF NOT EXISTS idx_images_annotation_status ON images(annotation_status);
                CREATE INDEX IF NOT EXISTS idx_image_labels_label ON image_labels(label);
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
            if "annotation_size" not in columns:
                connection.execute("ALTER TABLE images ADD COLUMN annotation_size INTEGER NOT NULL DEFAULT 0")
            if "annotation_mtime_ns" not in columns:
                connection.execute("ALTER TABLE images ADD COLUMN annotation_mtime_ns INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                "INSERT INTO dataset_meta(key, value) VALUES('format', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (self.annotation_format,),
            )

    @staticmethod
    def _filter_clause(query: str = "", status: str = "all", label: str = "") -> tuple[str, list[object]]:
        conditions: list[str] = []
        parameters: list[object] = []
        normalized = query.casefold().strip()
        if normalized:
            conditions.append("file_name LIKE ? COLLATE NOCASE")
            parameters.append(f"%{normalized}%")
        if status == "labeled":
            conditions.append("annotation_status = ?")
            parameters.append("present")
        elif status == "unlabeled":
            conditions.append("annotation_status = ?")
            parameters.append("missing")
        normalized_label = label.casefold().strip()
        if normalized_label:
            conditions.append(
                "EXISTS (SELECT 1 FROM image_labels il "
                "WHERE il.image_id = images.id AND il.label LIKE ? COLLATE NOCASE)"
            )
            parameters.append(f"%{normalized_label}%")
        return (f" WHERE {' AND '.join(conditions)}" if conditions else "", parameters)

    def count(self, query: str = "", status: str = "all", label: str = "") -> int:
        where, parameters = self._filter_clause(query, status, label)
        with self._connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM images{where}", parameters).fetchone()[0])

    def is_complete(self) -> bool:
        with self._connect() as connection:
            value = connection.execute("SELECT value FROM dataset_meta WHERE key='complete'").fetchone()
            return bool(value and value[0] == "1")

    def set_complete(self, value: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO dataset_meta(key,value) VALUES('complete',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ("1" if value else "0",),
            )

    def annotation_signature(self) -> str:
        entries = []
        if self.annotation_dir.is_file():
            candidates = [self.annotation_dir]
        else:
            candidates = (Path(root) / name for root, _dirs, files in os.walk(self.annotation_dir) for name in files)
        for path in candidates:
            if self.annotation_format == "voc" and path.suffix.lower() != ".xml":
                continue
            if self.annotation_format == "yolo" and path.suffix.lower() != ".txt":
                continue
            if self.annotation_format == "coco" and path.suffix.lower() != ".json":
                continue
            try:
                stat = path.stat()
                entries.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            except OSError:
                continue
        entries.sort()
        return hashlib.sha1(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest()

    def load_statistics(self, signature: str) -> dict | None:
        with self._connect() as connection:
            rows = dict(connection.execute("SELECT key,value FROM dataset_meta WHERE key IN ('stats_signature','stats_payload')"))
        if rows.get("stats_signature") != signature or not rows.get("stats_payload"):
            return None
        try:
            return json.loads(rows["stats_payload"])
        except (TypeError, json.JSONDecodeError):
            return None

    def save_statistics(self, signature: str, snapshot: dict) -> None:
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO dataset_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [("stats_signature", signature), ("stats_payload", payload)],
            )

    def get_page(self, offset: int, limit: int, query: str = "", status: str = "all", label: str = "") -> list[IndexedImage]:
        where, parameters = self._filter_clause(query, status, label)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,path,relative_path,file_name,file_size,mtime_ns,width,height,annotation_path,annotation_status,annotation_size,annotation_mtime_ns "
                f"FROM images{where} ORDER BY sort_key,id LIMIT ? OFFSET ?",
                [*parameters, int(limit), int(offset)],
            )
            return [self._row_to_image(row) for row in rows]

    def get_page_after(self, sort_key: str | None, limit: int, query: str = "", status: str = "all", label: str = "", after_id: int = 0) -> list[IndexedImage]:
        """Keyset pagination that stays fast on very large datasets."""
        where, parameters = self._filter_clause(query, status, label)
        if sort_key is not None:
            condition = "(sort_key > ? OR (sort_key = ? AND id > ?))"
            where = f"{where} AND {condition}" if where else f" WHERE {condition}"
            parameters.extend((sort_key, sort_key, int(after_id)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,path,relative_path,file_name,file_size,mtime_ns,width,height,annotation_path,annotation_status,annotation_size,annotation_mtime_ns "
                f"FROM images{where} ORDER BY sort_key,id LIMIT ?",
                [*parameters, int(limit)],
            )
            return [self._row_to_image(row) for row in rows]

    def iter_pages(self, page_size: int = 500, query: str = "", status: str = "all", label: str = ""):
        """Yield indexed images without OFFSET degradation."""
        last_key: str | None = None
        last_id = 0
        while True:
            page = self.get_page_after(last_key, page_size, query, status, label, last_id)
            if not page:
                return
            yield page
            last_key = page[-1].file_name.casefold()
            last_id = page[-1].id

    def position(self, path: Path) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sort_key FROM images WHERE path=?", (str(Path(path).resolve()),)
            ).fetchone()
            if row is None:
                return -1
            return int(connection.execute(
                "SELECT COUNT(*) FROM images WHERE sort_key < ?", (row[0],)
            ).fetchone()[0])

    def upsert_batch(self, batch: Iterable[IndexedImage]) -> None:
        batch = list(batch)
        values = [
            (item.id or None, str(item.path), item.relative_path, item.file_name, item.file_size,
             item.mtime_ns, item.width, item.height, str(item.annotation_path) if item.annotation_path else None,
            item.annotation_status, item.file_name.casefold(), item.annotation_size, item.annotation_mtime_ns)
            for item in batch
        ]
        if not values:
            return
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO images(id,path,relative_path,file_name,file_size,mtime_ns,width,height,annotation_path,annotation_status,sort_key,annotation_size,annotation_mtime_ns) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "relative_path=excluded.relative_path,file_name=excluded.file_name,file_size=excluded.file_size,mtime_ns=excluded.mtime_ns,"
                "annotation_path=excluded.annotation_path,annotation_status=excluded.annotation_status,"
                "annotation_size=excluded.annotation_size,annotation_mtime_ns=excluded.annotation_mtime_ns",
                values,
            )
            for item in batch:
                image_id = connection.execute("SELECT id FROM images WHERE path=?", (str(item.path),)).fetchone()[0]
                connection.execute("DELETE FROM image_labels WHERE image_id=?", (image_id,))
                connection.executemany(
                    "INSERT OR IGNORE INTO image_labels(image_id,label) VALUES(?,?)",
                    [(image_id, label) for label in item.annotation_labels if label],
                )

    def update_metadata(self, image_id: int, width: int, height: int, annotation_status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE images SET width=?,height=?,annotation_status=? WHERE id=?",
                (int(width), int(height), annotation_status, int(image_id)),
            )

    def update_annotation(self, image_path: Path, annotation_path: Path | None, labels: Iterable[str]) -> None:
        """Synchronize one successful editor save with the persistent index."""
        annotation_size = annotation_mtime_ns = 0
        if annotation_path is not None and annotation_path.exists():
            stat = annotation_path.stat()
            annotation_size, annotation_mtime_ns = stat.st_size, stat.st_mtime_ns
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM images WHERE path=?", (str(Path(image_path).resolve()),)
            ).fetchone()
            if row is None:
                return
            image_id = int(row[0])
            connection.execute(
                "UPDATE images SET annotation_path=?,annotation_status=?,annotation_size=?,annotation_mtime_ns=? WHERE id=?",
                (str(annotation_path) if annotation_path else None,
                 "present" if annotation_path is not None else "missing",
                 annotation_size, annotation_mtime_ns, image_id),
            )
            connection.execute("DELETE FROM image_labels WHERE image_id=?", (image_id,))
            connection.executemany(
                "INSERT OR IGNORE INTO image_labels(image_id,label) VALUES(?,?)",
                [(image_id, str(label)) for label in labels if str(label)],
            )

    def prune_missing(self, cancel_callback=None) -> None:
        """Remove stale rows after a completed directory scan."""
        with self._connect() as connection:
            paths = [row[0] for row in connection.execute("SELECT path FROM images")]
            stale = []
            for value in paths:
                if cancel_callback and cancel_callback():
                    return
                if not Path(value).is_file():
                    stale.append((value,))
            if stale:
                connection.executemany("DELETE FROM images WHERE path=?", stale)

    def scan_paths(self, cancel_callback=None, batch_size: int = 500, label_names: list[str] | None = None):
        coco_labels = self._load_coco_labels() if self.annotation_format == "coco" else {}
        cached = self._cached_annotation_rows()
        batch: list[IndexedImage] = []
        for root, _dirs, files in os.walk(self.image_dir):
            for name in files:
                if cancel_callback and cancel_callback():
                    return
                path = Path(root) / name
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                relative = path.relative_to(self.image_dir)
                annotation_path = self._annotation_path(path, relative)
                annotation_size = annotation_mtime_ns = 0
                if annotation_path is not None and annotation_path.exists():
                    try:
                        annotation_stat = annotation_path.stat()
                        annotation_size, annotation_mtime_ns = annotation_stat.st_size, annotation_stat.st_mtime_ns
                    except OSError:
                        pass
                old = cached.get(str(path.resolve()))
                if old and old[0] == stat.st_size and old[1] == stat.st_mtime_ns and old[2] == annotation_size and old[3] == annotation_mtime_ns:
                    labels = old[4]
                else:
                    labels = self._read_annotation_labels(path, relative, label_names or [], coco_labels)
                batch.append(IndexedImage(
                    0, path, str(relative), path.name, stat.st_size, stat.st_mtime_ns,
                    annotation_path=annotation_path,
                    annotation_status="present" if annotation_path and annotation_path.exists() else "missing",
                    annotation_labels=tuple(labels), annotation_size=annotation_size,
                    annotation_mtime_ns=annotation_mtime_ns,
                ))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def _cached_annotation_rows(self) -> dict[str, tuple[int, int, int, int, tuple[str, ...]]]:
        """Read cached signatures and labels once for an incremental scan."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,path,file_size,mtime_ns,annotation_size,annotation_mtime_ns FROM images"
            ).fetchall()
            labels = {}
            for image_id, label in connection.execute("SELECT image_id,label FROM image_labels"):
                labels.setdefault(int(image_id), []).append(str(label))
        return {
            str(path): (int(file_size), int(mtime_ns), int(annotation_size), int(annotation_mtime_ns), tuple(labels.get(int(image_id), [])))
            for image_id, path, file_size, mtime_ns, annotation_size, annotation_mtime_ns in rows
        }

    def _read_annotation_labels(self, image_path: Path, relative: Path, label_names: list[str], coco_labels: dict[str, set[str]]) -> set[str]:
        if self.annotation_format == "coco":
            return coco_labels.get(image_path.name, set()) | coco_labels.get(relative.as_posix(), set())
        path = self._annotation_path(image_path, relative)
        if not path or not path.exists():
            return set()
        try:
            if self.annotation_format == "voc":
                root = ET.parse(path).getroot()
                return {name.strip() for name in (node.findtext("name", "") for node in root.findall("object")) if name.strip()}
            labels = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if parts and parts[0].isdigit():
                    index = int(parts[0])
                    if 0 <= index < len(label_names):
                        labels.add(label_names[index])
            return labels
        except (OSError, ValueError, ET.ParseError):
            return set()

    def _load_coco_labels(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        try:
            path = self.annotation_dir if self.annotation_dir.is_file() else next(self.annotation_dir.glob("*.json"))
            document = json.loads(path.read_text(encoding="utf-8"))
            categories = {item.get("id"): str(item.get("name", "")) for item in document.get("categories", [])}
            names = {item.get("id"): str(item.get("file_name", "")) for item in document.get("images", [])}
            for item in document.get("annotations", []):
                name = categories.get(item.get("category_id"), "")
                image_name = names.get(item.get("image_id"), "")
                if name and image_name:
                    result.setdefault(image_name, set()).add(name)
                    result.setdefault(Path(image_name).name, set()).add(name)
        except (OSError, StopIteration, json.JSONDecodeError):
            pass
        return result

    def _annotation_path(self, image_path: Path, relative: Path) -> Path | None:
        if self.annotation_format == "coco":
            return self.annotation_dir
        suffix = ".xml" if self.annotation_format == "voc" else ".txt"
        return self.annotation_dir / relative.with_suffix(suffix)

    @staticmethod
    def _row_to_image(row) -> IndexedImage:
        return IndexedImage(
            id=int(row[0]), path=Path(row[1]), relative_path=row[2], file_name=row[3],
            file_size=int(row[4]), mtime_ns=int(row[5]), width=int(row[6]), height=int(row[7]),
            annotation_path=Path(row[8]) if row[8] else None, annotation_status=row[9],
            annotation_size=int(row[10]), annotation_mtime_ns=int(row[11]),
        )
