from pathlib import Path

from PySide6.QtCore import QPointF

from src.models.annotation import Annotation, ShapeType
from src.models.project import ImageRecord
from src.services.image_service import ImageService


def _record(name: str, label: str) -> ImageRecord:
    return ImageRecord(
        path=Path(name), width=100, height=100, file_format="JPEG", file_size=1,
        annotations=[Annotation(ShapeType.RECTANGLE, label, [QPointF(1, 1), QPointF(10, 10)])],
        status="labeled",
    )


def test_label_filter_requires_exact_case_insensitive_name():
    records = [_record("car.jpg", "car"), _record("cart.jpg", "cart"), _record("race.jpg", "racecar")]

    assert [item.path.name for item in ImageService.filter_records(records, label="car")] == ["car.jpg"]
    assert [item.path.name for item in ImageService.filter_records(records, label="CAR")] == ["car.jpg"]
