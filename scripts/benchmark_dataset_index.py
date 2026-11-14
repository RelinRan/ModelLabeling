from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.dataset_index import DatasetIndexRepository
from src.services import dataset_index


def create_dataset(root: Path, count: int, batch_size: int) -> tuple[Path, Path]:
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for start in range(0, count, batch_size):
        image_batch = []
        label_batch = []
        for index in range(start, min(start + batch_size, count)):
            image_batch.append(images / f"image_{index:08d}.jpg")
            label_batch.append(labels / f"image_{index:08d}.txt")
        for path in image_batch:
            path.touch()
        for path in label_batch:
            path.touch()
    return images, labels


def scan(repository: DatasetIndexRepository, batch_size: int) -> tuple[int, float, float, float]:
    started = time.perf_counter()
    scanned = 0
    scan_finished = started
    for batch in repository.scan_paths(batch_size=batch_size):
        if not scanned:
            scan_finished = time.perf_counter()
        repository.upsert_batch(batch)
        scanned += len(batch)
    if not scanned:
        scan_finished = time.perf_counter()
    repository.prune_missing()
    repository.set_complete(True)
    finished = time.perf_counter()
    return scanned, scan_finished - started, finished - scan_finished, finished - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark first and cached image-index scans.")
    parser.add_argument("--counts", nargs="+", type=int, default=[10_000, 50_000, 100_000])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[100, 500, 1000, 2500])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="model-labeling-index-bench-") as temporary:
        base = Path(temporary)
        dataset_index.DEFAULT_INDEX_CACHE = base / "index-cache"
        for batch_size in args.batch_sizes:
            for count in args.counts:
                root = base / f"{batch_size}-{count}"
                images, labels = create_dataset(root, count, batch_size)
                repository = DatasetIndexRepository(root, images, labels, "yolo")
                first_count, first_walk_seconds, first_write_seconds, first_seconds = scan(repository, batch_size)
                cached_repository = DatasetIndexRepository(root, images, labels, "yolo")
                cached_count, cached_walk_seconds, cached_write_seconds, cached_seconds = scan(cached_repository, batch_size)
                results.append({
                    "batch_size": batch_size,
                    "images": count,
                    "first_scan_images": first_count,
                    "first_scan_seconds": round(first_seconds, 3),
                    "first_enumeration_seconds": round(first_walk_seconds, 3),
                    "first_index_write_seconds": round(first_write_seconds, 3),
                    "cached_scan_images": cached_count,
                    "cached_scan_seconds": round(cached_seconds, 3),
                    "cached_enumeration_seconds": round(cached_walk_seconds, 3),
                    "cached_index_write_seconds": round(cached_write_seconds, 3),
                })
    output = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
