from __future__ import annotations

from math import hypot

from PySide6.QtCore import QPointF, QRectF


def normalize_rect(start: QPointF, end: QPointF) -> QRectF:
    return QRectF(start, end).normalized()


def constrain_square(start: QPointF, end: QPointF) -> QRectF:
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    size = max(abs(dx), abs(dy))
    return QRectF(
        start.x(),
        start.y(),
        size if dx >= 0 else -size,
        size if dy >= 0 else -size,
    ).normalized()


def image_to_view(point: QPointF, scale: float, offset: QPointF) -> QPointF:
    return QPointF(point.x() * scale + offset.x(), point.y() * scale + offset.y())


def view_to_image(point: QPointF, scale: float, offset: QPointF) -> QPointF:
    if scale == 0:
        raise ValueError("scale must not be zero")
    return QPointF((point.x() - offset.x()) / scale, (point.y() - offset.y()) / scale)


def polygon_bounds(points: list[QPointF]) -> QRectF:
    if not points:
        return QRectF()
    xs = [point.x() for point in points]
    ys = [point.y() for point in points]
    return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def clamp_points(points: list[QPointF], width: int, height: int) -> list[QPointF]:
    return [
        QPointF(
            min(max(point.x(), 0.0), max(0, width - 1)),
            min(max(point.y(), 0.0), max(0, height - 1)),
        )
        for point in points
    ]


def rect_to_yolo(rect: QRectF, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    normalized = rect.normalized()
    return (
        (normalized.center().x()) / image_width,
        (normalized.center().y()) / image_height,
        normalized.width() / image_width,
        normalized.height() / image_height,
    )


def yolo_to_rect(
    values: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> QRectF:
    center_x, center_y, width, height = values
    return QRectF(
        (center_x - width / 2) * image_width,
        (center_y - height / 2) * image_height,
        width * image_width,
        height * image_height,
    ).normalized()


def rect_from_points(points: list[QPointF]) -> QRectF:
    return polygon_bounds(points)
