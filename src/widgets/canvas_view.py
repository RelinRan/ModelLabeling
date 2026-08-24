from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF, QIcon
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QLabel, QToolButton, QHBoxLayout, QWidget

from src.models.annotation import Annotation, Keypoint, ShapeType
from src.models.keypoint import COCO_PERSON_KEYPOINTS
from src.utils.geometry import constrain_square, normalize_rect, polygon_bounds


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, index: int, parent: QGraphicsItem) -> None:
        super().__init__(-4, -4, 8, 8, parent)
        self.index = index
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setBrush(QColor("#00d9ff"))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        cursor = Qt.CursorShape.SizeFDiagCursor if index in {0, 2} else Qt.CursorShape.SizeBDiagCursor
        self.setCursor(cursor)
        self.setZValue(20)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor("#ffffff"))
        self.setPen(QPen(QColor("#00d9ff"), 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor("#00d9ff"))
        self.setPen(QPen(QColor("#ffffff"), 1))
        super().hoverLeaveEvent(event)


class KeypointMarker(QGraphicsEllipseItem):
    def __init__(self, index: int, parent: QGraphicsItem) -> None:
        super().__init__(-5, -5, 10, 10, parent)
        self.keypoint_index = index
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setBrush(QColor("#ffcc00"))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(21)


class PolygonVertexMarker(QGraphicsEllipseItem):
    """Small, transform-invariant handles used to edit polygon vertices."""

    def __init__(self, index: int, parent: QGraphicsItem) -> None:
        super().__init__(-4, -4, 8, 8, parent)
        self.vertex_index = index
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setBrush(QColor("#00d9ff"))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(20)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor("#ffffff"))
        self.setPen(QPen(QColor("#00d9ff"), 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor("#00d9ff"))
        self.setPen(QPen(QColor("#ffffff"), 1))
        super().hoverLeaveEvent(event)


class AnnotationItem(QGraphicsPolygonItem):
    def __init__(self, annotation: Annotation, text_size: int = 14, line_width: int = 2) -> None:
        super().__init__()
        self.annotation = annotation
        self.line_width = line_width
        self.text_item = QGraphicsSimpleTextItem(annotation.label, self)
        self.text_item.setBrush(QColor(annotation.color))
        self.text_item.setFont(QFont("Segoe UI", text_size))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.drag_corner: int | None = None
        self.handles = [ResizeHandle(index, self) for index in range(4)]
        self.keypoint_markers: list[KeypointMarker] = []
        self.polygon_markers: list[PolygonVertexMarker] = []
        self.skeleton_lines: list[QGraphicsLineItem] = []
        self.refresh()

    def refresh(self) -> None:
        self.text_item.setText(self.annotation.label)
        for marker in self.keypoint_markers:
            if marker.scene():
                marker.scene().removeItem(marker)
        self.keypoint_markers = []
        for marker in self.polygon_markers:
            if marker.scene():
                marker.scene().removeItem(marker)
        self.polygon_markers = []
        for line in self.skeleton_lines:
            if line.scene():
                line.scene().removeItem(line)
        self.skeleton_lines = []
        if self.annotation.shape_type in {ShapeType.RECTANGLE, ShapeType.SQUARE}:
            rect = QRectF(self.annotation.points[0], self.annotation.points[-1]).normalized()
            polygon = QPolygonF(rect)
            self.text_item.setPos(rect.topLeft() + QPointF(5, 5))
            corners = (rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft())
            for handle, corner in zip(self.handles, corners):
                handle.setPos(corner)
                handle.setVisible(self.isSelected())
        elif self.annotation.shape_type == ShapeType.KEYPOINT:
            points = [item.point for item in self.annotation.keypoints]
            rect = QRectF(self.annotation.points[0], self.annotation.points[-1]).normalized() if self.annotation.points else polygon_bounds(points)
            polygon = QPolygonF(rect)
            self.text_item.setPos(rect.topLeft() + QPointF(5, 5))
            for index, keypoint in enumerate(self.annotation.keypoints):
                marker = KeypointMarker(index, self)
                marker.setPos(keypoint.point)
                marker.setVisible(self.isSelected() or keypoint.visibility > 0)
                self.keypoint_markers.append(marker)
            corners = (rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft())
            for handle, corner in zip(self.handles, corners):
                handle.setPos(corner)
                handle.setVisible(self.isSelected() and not rect.isNull())
        else:
            polygon = QPolygonF(self.annotation.points)
            self.text_item.setPos(polygon_bounds(self.annotation.points).topLeft() + QPointF(5, 5))
            for handle in self.handles:
                handle.hide()
            for index, point in enumerate(self.annotation.points):
                marker = PolygonVertexMarker(index, self)
                marker.setPos(point)
                marker.setVisible(self.isSelected())
                self.polygon_markers.append(marker)
        self.setPolygon(polygon)
        color = QColor(self.annotation.color)
        self.setPen(QPen(color, self.line_width))
        fill = QColor(color)
        fill.setAlpha(72 if self.isSelected() else 0)
        self.setBrush(fill)
        if self.annotation.shape_type == ShapeType.KEYPOINT and len(self.annotation.keypoints) >= 17:
            from src.models.keypoint import COCO_PERSON_SKELETON
            points = [item.point for item in self.annotation.keypoints]
            skeleton_pen = QPen(color, max(1, self.line_width - 1))
            skeleton_pen.setStyle(Qt.PenStyle.SolidLine)
            for start, end in COCO_PERSON_SKELETON:
                if start >= len(points) or end >= len(points):
                    continue
                line = QGraphicsLineItem(points[start].x(), points[start].y(), points[end].x(), points[end].y(), self)
                line.setPen(skeleton_pen)
                line.setZValue(19)
                line.setVisible(self.isSelected() or any(self.annotation.keypoints[index].visibility > 0 for index in (start, end)))
                self.skeleton_lines.append(line)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.refresh()
        return super().itemChange(change, value)


class CanvasView(QGraphicsView):
    annotationCreated = Signal(object)
    annotationChanged = Signal()
    annotationDeleted = Signal()
    annotationSelected = Signal(object)
    annotationEditRequested = Signal(object)
    dirtyChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("annotationCanvas")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # QGraphicsView can briefly use the disabled palette while dataset
        # loading disables interaction. Keep both the frame and viewport
        # opaque so that transition never exposes the platform white default.
        self.setStyleSheet("QGraphicsView#annotationCanvas { background-color: #10151c; border: none; }")
        self.viewport().setStyleSheet("background-color: #10151c; border: none;")
        self.viewport().setAutoFillBackground(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#10151c"))
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._last_widget_size = self.size()
        self.image_item: QGraphicsPixmapItem | None = None
        self.annotation_items: list[AnnotationItem] = []
        self.annotations: list[Annotation] = []
        self.mode = ShapeType.RECTANGLE
        self.current_label = "object"
        self.current_color = "#00e5ff"
        self.line_width = 2
        self.text_size = 14
        self.crosshair_line_width = 2
        self.crosshair_color = "#ffea00"
        self.drawing = False
        self.draw_enabled = False
        self.start = QPointF()
        self.preview: QGraphicsPolygonItem | None = None
        self.polygon_points: list[QPointF] = []
        self.pan_mode = False
        self.enabled_shapes = set(ShapeType)
        self.drag_item: AnnotationItem | None = None
        self.drag_mode = ""
        self.drag_start = QPointF()
        self.drag_points: list[QPointF] = []
        self.drag_keypoint_index: int | None = None
        self.drag_vertex_index: int | None = None
        self.drag_changed = False
        self.pending_keypoints: list[Keypoint] = []
        self.keypoint_schema: list[str] = list(COCO_PERSON_KEYPOINTS)
        self.crosshair_horizontal: QGraphicsLineItem | None = None
        self.crosshair_vertical: QGraphicsLineItem | None = None
        self.image_info = QLabel(self)
        self.image_info.setObjectName("imageInfoOverlay")
        self.image_info.hide()
        self.annotation_labels = QLabel(self)
        self.annotation_labels.hide()
        self.zoom_tools = QWidget(self)
        self.zoom_tools.setObjectName("zoomTools")
        zoom_layout = QHBoxLayout(self.zoom_tools)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(4)
        icon_root = self._icon_root()
        for name, filename, handler, tip in (("zoomIn", "ic_zoom.png", self.zoom_in, "放大"), ("zoomOut", "ic_shrink.png", self.zoom_out, "缩小"), ("fit", "ic_fit.png", self.fit_image, "适应")):
            button = QToolButton(self.zoom_tools)
            button.setObjectName(name)
            icon_path = icon_root / filename
            button.setIcon(QIcon(str(icon_path)) if icon_path.exists() else QIcon())
            button.setIconSize(QSize(15, 15))
            button.setToolTip(tip)
            button.clicked.connect(handler)
            button.setFixedSize(30, 30)
            zoom_layout.addWidget(button)
        self.zoom_tools.adjustSize()
        self.zoom_tools.show()
        self.zoom_tools.raise_()
        self.zoom_tools.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.zoom_tools.setStyleSheet(
            "QWidget#zoomTools { background: transparent; border: none; }"
            "QToolButton#zoomIn, QToolButton#zoomOut, QToolButton#fit { "
            "background: transparent; border: none; padding: 0; margin: 0; }"
            "QToolButton#zoomIn:hover, QToolButton#zoomOut:hover, QToolButton#fit:hover { "
            "background: #3A3D42; border-radius: 5px; }"
        )
        self.scene.selectionChanged.connect(self._selection_changed)
        self.fit_scale = 1.0

    @staticmethod
    def _icon_root() -> Path:
        project_root = Path(__file__).resolve().parents[2]
        for folder in ("icons", "icon"):
            candidate = project_root / folder
            if candidate.is_dir():
                return candidate
        return project_root / "icons"

    @property
    def selected_annotation(self) -> Annotation | None:
        selected = [item for item in self.annotation_items if item.isSelected()]
        return selected[0].annotation if selected else None

    @property
    def selected_annotations(self) -> list[Annotation]:
        return [item.annotation for item in self.annotation_items if item.isSelected()]

    def _selection_changed(self) -> None:
        for item in self.annotation_items:
            item.refresh()
        self.annotationSelected.emit(self.selected_annotation)

    def _update_label_overlay(self) -> None:
        self.zoom_tools.move(self.width() - self.zoom_tools.width() - 12, self.height() - self.zoom_tools.height() - 12)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.image_info.move(14, 14)
        self._update_label_overlay()
        self.zoom_tools.show()
        self.zoom_tools.raise_()
        widget_size = self.size()
        externally_resized = widget_size != self._last_widget_size
        self._last_widget_size = widget_size
        if externally_resized and self.image_item and not self.drawing and not self.drag_item:
            self.fit_image()

    def load_image(self, image: QImage, annotations: list[Annotation]) -> None:
        blocker = QSignalBlocker(self.scene)
        self.annotation_items.clear()
        self.image_item = None
        self.crosshair_horizontal = None
        self.crosshair_vertical = None
        self.scene.clear()
        del blocker
        self.annotations = annotations
        self.image_item = self.scene.addPixmap(QPixmap.fromImage(image))
        self.image_item.setZValue(-1)
        for annotation in annotations:
            self._add_annotation_item(annotation)
        self.setSceneRect(self.image_item.boundingRect())
        self._create_crosshair()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_scale = max(self.transform().m11(), 0.0001)
        self._update_label_overlay()
        self.zoom_tools.show()
        self.zoom_tools.raise_()
        self.annotationSelected.emit(None)
        self.dirtyChanged.emit(False)

    def set_image_info(self, name: str, position: int, total: int, file_format: str = "", file_size: int = 0) -> None:
        self.image_info.hide()

    def set_mode(self, mode: ShapeType) -> None:
        if mode in self.enabled_shapes:
            self.mode = mode
            self.cancel_drawing()

    def set_enabled_shapes(self, shapes) -> None:
        self.enabled_shapes = set(shapes)
        if self.mode not in self.enabled_shapes:
            self.mode = next(iter(self.enabled_shapes), ShapeType.RECTANGLE)

    def set_keypoint_schema(self, names: list[str]) -> None:
        self.keypoint_schema = [str(name).strip() for name in names if str(name).strip()]

    def set_current_label(self, label: str, color: str | None = None) -> None:
        self.current_label = label
        if color is not None:
            self.current_color = color

    def set_visual_settings(self, line_width: int, text_size: int) -> None:
        self.line_width, self.text_size = line_width, text_size

    def set_crosshair_settings(self, line_width: int, color: str) -> None:
        self.crosshair_line_width = max(1, min(12, int(line_width)))
        self.crosshair_color = color
        pen = QPen(QColor(self.crosshair_color), self.crosshair_line_width, Qt.PenStyle.SolidLine)
        for line in (self.crosshair_horizontal, self.crosshair_vertical):
            if line:
                line.setPen(pen)

    def zoom_in(self) -> None:
        self._set_zoom_scale(min(5.0, self.transform().m11() * 1.2))
    def zoom_out(self) -> None:
        self._set_zoom_scale(max(0.1, self.transform().m11() / 1.2))
    def _set_zoom_scale(self, target: float) -> None:
        current = self.transform().m11()
        if current > 0 and abs(target - current) > 1e-9:
            factor = target / current
            self.scale(factor, factor)
    def fit_image(self) -> None:
        if self.image_item:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.fit_scale = max(self.transform().m11(), 0.0001)

    def cancel_drawing(self) -> None:
        self.drawing = False
        self.polygon_points.clear()
        self.pending_keypoints.clear()
        if self.preview:
            self.scene.removeItem(self.preview)
            self.preview = None

    def _create_crosshair(self) -> None:
        pen = QPen(QColor(self.crosshair_color), self.crosshair_line_width, Qt.PenStyle.SolidLine)
        self.crosshair_horizontal = self.scene.addLine(0, 0, 0, 0, pen)
        self.crosshair_vertical = self.scene.addLine(0, 0, 0, 0, pen)
        for line in (self.crosshair_horizontal, self.crosshair_vertical):
            line.setZValue(50)
            line.hide()

    def _update_crosshair(self, point: QPointF) -> None:
        if not self.draw_enabled or not self.image_item:
            self._hide_crosshair()
            return
        bounds = self.image_item.boundingRect()
        if not bounds.contains(point):
            self._hide_crosshair()
            return
        if not self.crosshair_horizontal or not self.crosshair_vertical:
            self._create_crosshair()
        self.crosshair_horizontal.setLine(bounds.left(), point.y(), bounds.right(), point.y())
        self.crosshair_vertical.setLine(point.x(), bounds.top(), point.x(), bounds.bottom())
        self.crosshair_horizontal.show()
        self.crosshair_vertical.show()

    def _hide_crosshair(self) -> None:
        for line in (self.crosshair_horizontal, self.crosshair_vertical):
            if line:
                line.hide()

    def _disable_draw_mode(self) -> None:
        self.draw_enabled = False
        self.cancel_drawing()
        self._hide_crosshair()
        self.unsetCursor()

    def _update_box_preview(self, rect: QRectF) -> None:
        """Update one temporary item instead of rebuilding the scene per move."""
        if self.preview is None:
            self.preview = QGraphicsPolygonItem()
            self.preview.setPen(QPen(QColor(self.current_color), self.line_width, Qt.PenStyle.DashLine))
            self.preview.setBrush(Qt.BrushStyle.NoBrush)
            self.preview.setZValue(1)
            self.scene.addItem(self.preview)
        self.preview.setPolygon(QPolygonF(rect))

    def _update_polygon_preview(self, current: QPointF | None = None) -> None:
        """Show the committed vertices and the segment under the cursor."""
        points = list(self.polygon_points)
        if current is not None and (not points or (points[-1] - current).manhattanLength() > 0.01):
            points.append(current)
        if len(points) < 2:
            if self.preview:
                self.preview.hide()
            return
        if self.preview is None:
            self.preview = QGraphicsPolygonItem()
            self.preview.setPen(QPen(QColor(self.current_color), self.line_width, Qt.PenStyle.DashLine))
            self.preview.setBrush(Qt.BrushStyle.NoBrush)
            self.preview.setZValue(1)
            self.scene.addItem(self.preview)
        self.preview.setPolygon(QPolygonF(points))
        self.preview.show()

    def _enable_draw_mode(self) -> None:
        self.draw_enabled = True
        # The settings dialog/menu can retain keyboard focus. W must make the
        # next mouse gesture belong to the canvas, otherwise the first click
        # may be delivered to another child and cancel the draw workflow.
        self.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _add_annotation_item(self, annotation: Annotation) -> None:
        item = AnnotationItem(annotation, self.text_size, self.line_width)
        self.scene.addItem(item)
        self.annotation_items.append(item)

    def _item_at_event(self, event) -> AnnotationItem | None:
        item = self.itemAt(event.position().toPoint())
        while item and not isinstance(item, AnnotationItem):
            item = item.parentItem()
        return item if isinstance(item, AnnotationItem) else None

    def _select_item(self, item: AnnotationItem) -> None:
        self.scene.clearSelection()
        item.setSelected(True)

    def _corner_index(self, item: AnnotationItem, point: QPointF) -> int | None:
        if item.annotation.shape_type not in {ShapeType.RECTANGLE, ShapeType.SQUARE, ShapeType.KEYPOINT}:
            return None
        if len(item.annotation.points) < 2:
            return None
        rect = QRectF(item.annotation.points[0], item.annotation.points[-1]).normalized()
        corners = (rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft())
        return next((i for i, corner in enumerate(corners) if (corner - point).manhattanLength() <= 10), None)

    def _keypoint_index(self, item: AnnotationItem, point: QPointF) -> int | None:
        if item.annotation.shape_type != ShapeType.KEYPOINT:
            return None
        return next((index for index, keypoint in enumerate(item.annotation.keypoints) if (keypoint.point - point).manhattanLength() <= 10), None)

    def _polygon_vertex_index(self, item: AnnotationItem, point: QPointF) -> int | None:
        if item.annotation.shape_type != ShapeType.POLYGON:
            return None
        return next((index for index, vertex in enumerate(item.annotation.points) if (vertex - point).manhattanLength() <= 10), None)

    def _begin_box_drag(self, item: AnnotationItem, point: QPointF) -> None:
        self._select_item(item)
        self.drag_item = item
        self.drag_start = point
        self.drag_points = list(item.annotation.points)
        self.drag_keypoint_index = self._keypoint_index(item, point)
        vertex_index = self._polygon_vertex_index(item, point)
        item.drag_corner = self._corner_index(item, point)
        self.drag_mode = (
            "keypoint" if self.drag_keypoint_index is not None
            else "vertex" if vertex_index is not None
            else "resize" if item.drag_corner is not None else "move"
        )
        self.drag_vertex_index = vertex_index
        self.drag_changed = False

    def _drag_box(self, point: QPointF) -> None:
        if not self.drag_item:
            return
        annotation = self.drag_item.annotation
        dx, dy = point.x() - self.drag_start.x(), point.y() - self.drag_start.y()
        if self.drag_mode == "keypoint" and self.drag_keypoint_index is not None:
            keypoint = annotation.keypoints[self.drag_keypoint_index]
            keypoint.point = point
        elif self.drag_mode == "vertex" and self.drag_vertex_index is not None:
            updated = list(annotation.points)
            updated[self.drag_vertex_index] = point
            annotation.points = updated
        elif self.drag_mode == "move":
            annotation.points = [QPointF(p.x() + dx, p.y() + dy) for p in self.drag_points]
            if annotation.shape_type == ShapeType.KEYPOINT:
                annotation.keypoints = [Keypoint(item.name, QPointF(item.point.x() + dx, item.point.y() + dy), item.visibility) for item in annotation.keypoints]
        else:
            rect = QRectF(self.drag_points[0], self.drag_points[-1]).normalized()
            if annotation.shape_type == ShapeType.SQUARE:
                opposite = (rect.bottomRight() if self.drag_item.drag_corner == 0 else rect.bottomLeft() if self.drag_item.drag_corner == 1 else rect.topLeft() if self.drag_item.drag_corner == 2 else rect.topRight())
                rect = constrain_square(opposite, point)
            elif self.drag_item.drag_corner == 0: rect.setTopLeft(point)
            elif self.drag_item.drag_corner == 1: rect.setTopRight(point)
            elif self.drag_item.drag_corner == 2: rect.setBottomRight(point)
            else: rect.setBottomLeft(point)
            annotation.points = [rect.topLeft(), rect.bottomRight()]
        self.drag_item.refresh()
        self.drag_changed = True

    def update_selected_label(self, label: str, color: str | None = None) -> bool:
        annotations = self.selected_annotations
        if not annotations: return False
        for annotation in annotations:
            annotation.label = label
            if color is not None:
                annotation.color = color
        for item in self.annotation_items:
            if item.annotation in annotations: item.refresh()
        self.annotationChanged.emit(); self.dirtyChanged.emit(True)
        return True

    def refresh_annotations(self) -> None:
        """Refresh scene items after an editor changed annotation fields."""
        for item in self.annotation_items:
            item.refresh()
        self.annotationChanged.emit()
        self.dirtyChanged.emit(True)

    def delete_selected(self) -> bool:
        selected = [item for item in self.annotation_items if item.isSelected()]
        for item in selected:
            if item.annotation in self.annotations: self.annotations.remove(item.annotation)
            self.scene.removeItem(item); self.annotation_items.remove(item)
        if selected: self.annotationDeleted.emit(); self.dirtyChanged.emit(True)
        return bool(selected)

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        # Any canvas click clears the independent label-list selection.
        self.annotationSelected.emit(None)
        item = self._item_at_event(event)
        if event.button() == Qt.MouseButton.RightButton and item:
            self._disable_draw_mode(); self._select_item(item); self.annotationEditRequested.emit(item.annotation); return
        if event.button() == Qt.MouseButton.RightButton:
            self._disable_draw_mode(); self.scene.clearSelection(); return
        if event.button() == Qt.MouseButton.MiddleButton or self.pan_mode:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag); super().mousePressEvent(event); return
        if event.button() == Qt.MouseButton.LeftButton and item:
            self._disable_draw_mode()
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                item.setSelected(not item.isSelected())
                return
            self._begin_box_drag(item, self.mapToScene(event.position().toPoint())); return
        if event.button() == Qt.MouseButton.LeftButton and not item:
            self.scene.clearSelection()
        if event.button() != Qt.MouseButton.LeftButton or not self.image_item or not self.draw_enabled:
            super().mousePressEvent(event); return
        point = self.mapToScene(event.position().toPoint())
        if self.mode == ShapeType.POLYGON:
            self.polygon_points.append(point); self.drawing = True
            self._update_polygon_preview()
            return
        if self.mode == ShapeType.KEYPOINT:
            name = self.keypoint_schema[len(self.pending_keypoints)] if len(self.pending_keypoints) < len(self.keypoint_schema) else f"keypoint_{len(self.pending_keypoints)}"
            self.pending_keypoints.append(Keypoint(name, point, 2)); self.drawing = True
            if self.keypoint_schema and len(self.pending_keypoints) >= len(self.keypoint_schema):
                self._finish_keypoint_annotation()
            return
        self.start = point; self.drawing = True

    def mouseMoveEvent(self, event) -> None:
        scene_point = self.mapToScene(event.position().toPoint())
        self._update_crosshair(scene_point)
        if self.drag_item and self.drag_mode:
            self._drag_box(scene_point); return
        if self.drawing and self.mode == ShapeType.POLYGON:
            self._update_polygon_preview(scene_point)
            return
        if self.drawing and self.mode != ShapeType.POLYGON:
            current = scene_point
            rect = constrain_square(self.start, current) if self.mode == ShapeType.SQUARE else normalize_rect(self.start, current)
            self._update_box_preview(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.drag_item:
            changed = self.drag_changed
            self.drag_item = None; self.drag_mode = ""; self.drag_points = []; self.drag_vertex_index = None; self.drag_changed = False
            if changed:
                self.annotationChanged.emit()
                self.dirtyChanged.emit(True)
            return
        if event.button() == Qt.MouseButton.MiddleButton or self.pan_mode:
            super().mouseReleaseEvent(event); self.setDragMode(QGraphicsView.DragMode.RubberBandDrag); return
        if self.drawing and self.mode == ShapeType.KEYPOINT and event.button() == Qt.MouseButton.LeftButton:
            return
        if self.drawing and self.mode != ShapeType.POLYGON and event.button() == Qt.MouseButton.LeftButton:
            current = self.mapToScene(event.position().toPoint()); rect = constrain_square(self.start, current) if self.mode == ShapeType.SQUARE else normalize_rect(self.start, current)
            if rect.width() >= 3 and rect.height() >= 3:
                annotation = Annotation(self.mode, self.current_label, [rect.topLeft(), rect.bottomRight()], self.current_color); self.annotations.append(annotation); self._add_annotation_item(annotation); self.annotationCreated.emit(annotation); self.dirtyChanged.emit(True)
            self._disable_draw_mode(); return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self._item_at_event(event)
        if item: self._select_item(item); self.annotationEditRequested.emit(item.annotation); return
        if self.mode == ShapeType.POLYGON and len(self.polygon_points) >= 3:
            points = list(self.polygon_points)
            if len(points) >= 2 and (points[-1] - points[-2]).manhattanLength() <= 10:
                points.pop()
            if len(points) >= 3:
                annotation = Annotation(ShapeType.POLYGON, self.current_label, points, self.current_color)
                self.annotations.append(annotation); self._add_annotation_item(annotation); self.annotationCreated.emit(annotation); self.dirtyChanged.emit(True)
            self._disable_draw_mode(); return
        if self.mode == ShapeType.KEYPOINT and self.pending_keypoints:
            self._finish_keypoint_annotation(); return
        super().mouseDoubleClickEvent(event)

    def _finish_keypoint_annotation(self) -> None:
        if not self.pending_keypoints:
            return
        points = [item.point for item in self.pending_keypoints if item.visibility > 0]
        bbox = polygon_bounds(points) if points else QRectF()
        annotation = Annotation(
            ShapeType.KEYPOINT,
            self.current_label,
            [bbox.topLeft(), bbox.bottomRight()] if not bbox.isNull() else [],
            self.current_color,
            keypoints=list(self.pending_keypoints),
            schema_name="Custom Keypoints",
        )
        self.annotations.append(annotation)
        self._add_annotation_item(annotation)
        self.annotationCreated.emit(annotation)
        self.dirtyChanged.emit(True)
        self._disable_draw_mode()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape: self._disable_draw_mode()
        elif event.key() == Qt.Key.Key_W:
            if self.draw_enabled:
                self._disable_draw_mode()
            else:
                self._enable_draw_mode()
        elif event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}: self.delete_selected()
        elif event.key() in {Qt.Key.Key_Plus, Qt.Key.Key_Equal} and event.modifiers() & Qt.KeyboardModifier.ControlModifier: self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus and event.modifiers() & Qt.KeyboardModifier.ControlModifier: self.zoom_out()
        elif event.key() == Qt.Key.Key_0 and event.modifiers() & Qt.KeyboardModifier.ControlModifier: self.fit_image()
        else: super().keyPressEvent(event)
