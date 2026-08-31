from __future__ import annotations

import math

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QIntValidator, QPainter, QPen, QPixmap, QPolygonF, QIcon
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QComboBox, QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QLabel, QLineEdit, QToolButton, QHBoxLayout, QWidget

from src.models.annotation import Annotation, Keypoint, ShapeType, label_color
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
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def hoverEnterEvent(self, event) -> None:
        self.setRect(-6.5, -6.5, 13, 13)
        self.setPen(QPen(QColor("#7ee787"), 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setRect(-5, -5, 10, 10)
        self.setPen(QPen(QColor("#ffffff"), 1))
        super().hoverLeaveEvent(event)

    def set_visibility_style(self, visibility: int) -> None:
        """Mainstream convention: keep every keypoint clickable and show its
        COCO/YOLO visibility state by fill (2 solid, 1 translucent, 0 hollow)."""
        if int(visibility) >= 2:
            self.setBrush(QColor("#ffcc00"))
            self.setPen(QPen(QColor("#ffffff"), 1))
        elif int(visibility) == 1:
            self.setBrush(QColor(255, 204, 0, 110))
            self.setPen(QPen(QColor("#ffffff"), 1))
        else:
            self.setBrush(QColor(255, 204, 0, 0))
            self.setPen(QPen(QColor("#8a8f98"), 1))


class PolygonVertexMarker(QGraphicsEllipseItem):
    """Small, transform-invariant handles used to edit polygon vertices."""

    def __init__(self, index: int, parent: QGraphicsItem, part_index: int = 0) -> None:
        super().__init__(-4, -4, 8, 8, parent)
        self.vertex_index = index
        self.part_index = part_index
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


class RotationHandle(QGraphicsEllipseItem):
    """Transform-invariant handle used to rotate an OBB around its center."""

    def __init__(self, parent: QGraphicsItem) -> None:
        super().__init__(-5, -5, 10, 10, parent)
        self.setPen(QPen(QColor("#ffffff"), 1))
        self.setBrush(QColor("#7ee787"))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(22)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor("#ffffff"))
        self.setPen(QPen(QColor("#7ee787"), 2))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor("#7ee787"))
        self.setPen(QPen(QColor("#ffffff"), 1))
        super().hoverLeaveEvent(event)


class AnnotationItem(QGraphicsPolygonItem):
    def __init__(self, annotation: Annotation, text_size: int = 14, line_width: int = 2) -> None:
        super().__init__()
        self.annotation = annotation
        self.line_width = line_width
        self.text_item = QGraphicsSimpleTextItem(annotation.label, self)
        self.setToolTip(self._build_tooltip())
        self.text_item.setBrush(QColor(annotation.color))
        self.text_item.setFont(QFont("Segoe UI", text_size))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.drag_corner: int | None = None
        self.handles = [ResizeHandle(index, self) for index in range(4)]
        self.keypoint_markers: list[KeypointMarker] = []
        self.polygon_markers: list[PolygonVertexMarker] = []
        self._edge_midpoint_hints: list[QGraphicsEllipseItem] = []
        self.polygon_part_items: list[QGraphicsPolygonItem] = []
        self.skeleton_lines: list[QGraphicsLineItem] = []
        self.rotation_handle: RotationHandle | None = None
        self.refresh()

    def _build_tooltip(self):
        a = self.annotation
        parts = [a.label, str(a.shape_type.value)]
        if a.shape_type == ShapeType.POLYGON:
            pts = a.polygon_parts[0] if a.polygon_parts else a.points
            parts.append("%d 个顶点" % len(pts))
        elif a.shape_type == ShapeType.KEYPOINT:
            parts.append("%d 个点位" % len(a.keypoints))
        return chr(10).join(parts)

    def refresh(self) -> None:
        self.text_item.setText(self.annotation.label)
        # Keep the label text and the box outline on the same color whenever
        # the annotation color changes (e.g. relabeling from the right panel).
        self.text_item.setBrush(QColor(self.annotation.color))
        for marker in self.keypoint_markers:
            if marker.scene():
                marker.scene().removeItem(marker)
        self.keypoint_markers = []
        for marker in self.polygon_markers:
            if marker.scene():
                marker.scene().removeItem(marker)
        self.polygon_markers = []
        for hint in self._edge_midpoint_hints:
            if hint.scene():
                hint.scene().removeItem(hint)
        self._edge_midpoint_hints = []
        for part_item in self.polygon_part_items:
            if part_item.scene():
                part_item.scene().removeItem(part_item)
        self.polygon_part_items = []
        for line in self.skeleton_lines:
            if line.scene():
                line.scene().removeItem(line)
        self.skeleton_lines = []
        if self.rotation_handle is not None:
            if self.rotation_handle.scene():
                self.rotation_handle.scene().removeItem(self.rotation_handle)
            self.rotation_handle = None
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
                vis = {0: "未标注", 1: "遮挡", 2: "可见"}.get(keypoint.visibility, "?")
                marker.setToolTip(keypoint.name + "  " + vis + " (右键切换可见性 / 拖动调位置)")
                marker.set_visibility_style(keypoint.visibility)
                marker.setVisible(True)
                self.keypoint_markers.append(marker)
            corners = (rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft())
            for handle, corner in zip(self.handles, corners):
                handle.setPos(corner)
                handle.setVisible(self.isSelected() and not rect.isNull())
        elif self.annotation.shape_type == ShapeType.OBB:
            corners = list(self.annotation.points[:4])
            polygon = QPolygonF(corners)
            bounds = polygon_bounds(corners)
            self.text_item.setPos(bounds.topLeft() + QPointF(5, 5))
            for handle in self.handles:
                handle.hide()
            if not self.rotation_handle:
                self.rotation_handle = RotationHandle(self)
            # Park the rotation handle just outside the first edge's midpoint,
            # along the direction pointing away from the box center.
            center = QPointF(
                sum(point.x() for point in corners) / len(corners),
                sum(point.y() for point in corners) / len(corners),
            )
            mid = QPointF((corners[0].x() + corners[1].x()) / 2, (corners[0].y() + corners[1].y()) / 2)
            direction = QPointF(mid.x() - center.x(), mid.y() - center.y())
            length = math.hypot(direction.x(), direction.y()) or 1.0
            self.rotation_handle.setPos(
                QPointF(mid.x() + direction.x() / length * 24, mid.y() + direction.y() / length * 24)
            )
            self.rotation_handle.setVisible(self.isSelected())
        else:
            parts = self.annotation.polygon_parts or [self.annotation.points]
            polygon = QPolygonF(parts[0])
            all_points = [point for part in parts for point in part]
            self.text_item.setPos(polygon_bounds(all_points).topLeft() + QPointF(5, 5))
            for handle in self.handles:
                handle.hide()
            for part_index, part in enumerate(parts):
                if part_index:
                    part_item = QGraphicsPolygonItem(QPolygonF(part), self)
                    part_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self.polygon_part_items.append(part_item)
                for index, point in enumerate(part):
                    marker = PolygonVertexMarker(index, self, part_index)
                    marker.setPos(point)
                    marker.setVisible(self.isSelected())
                    marker.setToolTip("顶点 " + str(index + 1) + " (拖动调位置 / 右键删除)")
                    self.polygon_markers.append(marker)
                if self.isSelected() and len(part) >= 3:
                    for k in range(len(part)):
                        a = part[k]
                        b = part[(k + 1) % len(part)]
                        mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
                        hint = QGraphicsEllipseItem(-3, -3, 6, 6, self)
                        hint.setPos(mid)
                        hint.setPen(QPen(QColor("#7ee787"), 1))
                        hint.setBrush(QColor(126, 231, 135, 60))
                        hint.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                        hint.setZValue(21)
                        hint.setCursor(Qt.CursorShape.PointingHandCursor)
                        self._edge_midpoint_hints.append(hint)
        self.setPolygon(polygon)
        color = QColor(self.annotation.color)
        self.setPen(QPen(color, self.line_width))
        fill = QColor(color)
        fill.setAlpha(72 if self.isSelected() else 0)
        self.setBrush(fill)
        for part_item in self.polygon_part_items:
            part_item.setPen(QPen(color, self.line_width))
            part_item.setBrush(fill)
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


class _KeypointCountBox(QWidget):
    """A plain number edit with a fixed unit label; typing digits is all it takes."""

    valueChanged = Signal(int)

    def __init__(self, parent, unit: str) -> None:
        super().__init__(parent)
        self.setObjectName("keypointCountBox")
        self.setFixedHeight(26)
        # Plain QWidget ignores stylesheet backgrounds unless this is set.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # The border lives on the container so the unit reads as part of the
        # box while the editor stays the only editable area.
        # Python subclass names are invisible to Qt's stylesheet engine;
        # select by objectName instead so the border actually renders.
        self.setStyleSheet(
            "#keypointCountBox { background: rgba(43, 45, 48, 215); "
            "border: 1px solid #4A4E55; border-radius: 5px; }"
            "#keypointCountBox:focus-within { border-color: #6A84B8; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 8, 0)
        layout.setSpacing(3)
        self._editor = QLineEdit()
        self._editor.setFixedWidth(30)
        self._editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._editor.setValidator(QIntValidator(1, 135, self))
        self._editor.setStyleSheet(
            "QLineEdit { background: transparent; color: #FFFFFF; border: none; "
            "padding: 0; font-weight: 600; }"
        )
        self._unit = QLabel(unit)
        self._unit.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._unit.setStyleSheet("color: #9aa0a8; border: none; background: transparent; font-size: 12px;")
        layout.addWidget(self._editor)
        layout.addWidget(self._unit)
        self._value = 17
        self._editor.setText("17")
        self._editor.textChanged.connect(self._on_text)

    def _on_text(self, text: str) -> None:
        digits = "".join(ch for ch in text if ch.isdigit())
        if digits != text:
            self._editor.setText(digits)
            return
        if not digits:
            return
        value = max(1, min(135, int(digits)))
        self._value = value
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        value = max(1, min(135, int(value)))
        self._value = value
        self._editor.setText(str(value))

    def set_unit(self, unit: str) -> None:
        self._unit.setText(unit)


class CanvasView(QGraphicsView):
    annotationCreated = Signal(object)
    annotationChanged = Signal()
    annotationDeleted = Signal()
    annotationSelected = Signal(object)
    annotationEditRequested = Signal(object)
    dirtyChanged = Signal(bool)
    methodRequested = Signal(object)
    keypointCountChanged = Signal(int)
    keypointGroupSelected = Signal(str)

    # Display order of the annotation methods shown in the canvas selector.
    METHOD_ORDER = (ShapeType.RECTANGLE, ShapeType.SQUARE, ShapeType.POLYGON, ShapeType.OBB, ShapeType.KEYPOINT)

    @staticmethod
    def method_label(shape: ShapeType, language: str = "zh_CN") -> str:
        names = {
            ShapeType.RECTANGLE: ("矩形", "Rectangle"),
            ShapeType.SQUARE: ("正方形", "Square"),
            ShapeType.POLYGON: ("多边形", "Polygon"),
            ShapeType.OBB: ("旋转框", "Rotated Box"),
            ShapeType.KEYPOINT: ("关键点位", "Keypoints"),
        }
        zh, en = names.get(shape, (shape.value, shape.value))
        return en if language == "en_US" else zh

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
        self._undo_stack: list[list[Annotation]] = []
        self._redo_stack: list[list[Annotation]] = []
        self._drag_undo_pending: list[Annotation] | None = None
        self.mode = ShapeType.RECTANGLE
        self.current_label = "object"
        self.current_color = "#00e5ff"
        self.line_width = 2
        self.text_size = 14
        self.crosshair_line_width = 1
        self.crosshair_color = "#000000"
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
        self.drag_vertex_index: tuple[int, int] | None = None
        self.drag_polygon_parts: list[list[QPointF]] = []
        self.drag_changed = False
        self.drag_center = QPointF()
        self.drag_start_angle = 0.0
        self.pending_keypoints: list[Keypoint] = []
        self._hints_shown: set[ShapeType] = set()
        self._hint_label = None
        self._hint_timer = None
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
        self._kept_view: tuple[float, float, float] | None = None
        # Annotation-method selector: only methods supported by the current
        # dataset task are listed; picking one arms continuous drawing.
        self.language = "zh_CN"
        self._method_syncing = False
        self.method_combo = QComboBox(self)
        self.method_combo.setObjectName("methodCombo")
        self.method_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.method_combo.setFixedHeight(26)
        self.method_combo.setToolTip("标注方式，选择后可连续标注，Esc 退出绘制" if self.language == "zh_CN" else "Annotation method; stays armed until Esc")
        self.method_combo.setStyleSheet(
            "QComboBox#methodCombo { background: rgba(43, 45, 48, 215); color: #FFFFFF; "
            "border: 1px solid #4A4E55; border-radius: 5px; padding: 2px 9px 2px 8px; } "
            "QComboBox#methodCombo::drop-down { border: none; width: 16px; } "
            "QComboBox#methodCombo QAbstractItemView { background: #2B2D30; color: #D7DAE0; "
            "selection-background-color: #2e436e; selection-color: #FFFFFF; border: 1px solid #464A50; }"
        )
        self.method_combo.currentIndexChanged.connect(self._method_combo_changed)
        # Width follows the widest item (e.g. 旋转框/关键点), not a stale
        # first-show hint.
        self.method_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        # Compact keypoint-count stepper shown next to the method selector
        # while the keypoint method is active. The dataset's kpt_shape is the
        # authoritative source; this control lets the user change it quickly.
        self._count_syncing = False
        self.keypoint_count_box = _KeypointCountBox(self, "点" if self.language == "zh_CN" else "pts")
        self.keypoint_count_box.setToolTip(
            "关键点位数量，修改后写入 data.yaml 的 kpt_shape" if self.language == "zh_CN"
            else "Keypoint count; synced to kpt_shape in data.yaml"
        )
        self.keypoint_count_box.valueChanged.connect(self._keypoint_count_changed)
        self.keypoint_count_box.hide()
        # Keypoint-group selector: choosing a template arms its point names
        # for the next keypoint annotation (see set_keypoint_groups).
        self._group_syncing = False
        self.keypoint_groups: list[tuple[str, list[str]]] = []
        self.keypoint_group_box = QComboBox(self)
        self.keypoint_group_box.setObjectName("keypointGroupBox")
        self.keypoint_group_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.keypoint_group_box.setFixedHeight(26)
        # The selector must fit its widest type name ("姿态 · 17点"),
        # including after the user edits types in the dialog.
        self.keypoint_group_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.keypoint_group_box.setToolTip(
            "点位类型，选择后按类型点位名称标注" if self.language == "zh_CN"
            else "Keypoint type; the selected template names the points you draw"
        )
        self.keypoint_group_box.setStyleSheet(
            "QComboBox#keypointGroupBox { background: rgba(43, 45, 48, 215); color: #FFFFFF; "
            "border: 1px solid #4A4E55; border-radius: 5px; padding: 2px 9px 2px 8px; } "
            "QComboBox#keypointGroupBox::drop-down { border: none; width: 16px; } "
            "QComboBox#keypointGroupBox QAbstractItemView { background: #2B2D30; color: #D7DAE0; "
            "selection-background-color: #2e436e; selection-color: #FFFFFF; border: 1px solid #464A50; }"
        )
        self.keypoint_group_box.currentIndexChanged.connect(self._keypoint_group_changed)
        self.keypoint_group_box.hide()
        # Qt swallows Tab (focus navigation) and Ctrl+A (scene select-all)
        # before keyPressEvent; QShortcut intercepts them.
        self._tab_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Tab), self)
        self._tab_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._tab_shortcut.activated.connect(self._skip_current_keypoint)
        self._tab_shortcut.setEnabled(False)
        self._sa_shortcut = QShortcut(QKeySequence("Ctrl+A"), self)
        self._sa_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._sa_shortcut.activated.connect(self.select_all)
        self._sc_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self._sc_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._sc_shortcut.activated.connect(self.copy_selected)
        self._sv_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        self._sv_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._sv_shortcut.activated.connect(self.paste_clipboard)

        self._update_method_combo()

    @staticmethod
    def _icon_root() -> Path:
        from src.app_paths import resource_path
        for folder in ("icons", "icon"):
            candidate = resource_path(folder)
            if candidate.is_dir():
                return candidate
        return resource_path("icons")

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
        self._position_method_widgets(self.mode == ShapeType.KEYPOINT)
        self._update_label_overlay()
        self.zoom_tools.show()
        self.zoom_tools.raise_()
        widget_size = self.size()
        externally_resized = widget_size != self._last_widget_size
        self._last_widget_size = widget_size
        if externally_resized and self.image_item and not self.drawing and not self.drag_item:
            self.fit_image()

    def load_image(self, image: QImage, annotations: list[Annotation]) -> None:
        self._capture_view_state()
        self.cancel_drawing()  # switching images drops any half-drawn shape
        self._hide_keypoint_overlay()
        self._hide_polygon_count_overlay()
        blocker = QSignalBlocker(self.scene)
        self.annotation_items.clear()
        self.image_item = None
        self.crosshair_horizontal = None
        self.crosshair_vertical = None
        self.scene.clear()
        del blocker
        # History is per image: restoring another image's annotations here
        # would autosave them over this image's files.
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._drag_undo_pending = None
        self.annotations = annotations
        self.image_item = self.scene.addPixmap(QPixmap.fromImage(image))
        self.image_item.setZValue(-1)
        for annotation in annotations:
            self._add_annotation_item(annotation)
        self.setSceneRect(self.image_item.boundingRect())
        self._create_crosshair()
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_scale = max(self.transform().m11(), 0.0001)
        self._restore_view_state()
        self._update_label_overlay()
        self.zoom_tools.show()
        self.zoom_tools.raise_()
        self.annotationSelected.emit(None)
        self.dirtyChanged.emit(False)

    def set_image_info(self, name: str, position: int, total: int, file_format: str = "", file_size: int = 0) -> None:
        self.image_info.hide()

    def set_mode(self, mode: ShapeType) -> None:
        if mode in self.enabled_shapes:
            if self.mode == ShapeType.KEYPOINT and mode != ShapeType.KEYPOINT:
                self._hide_keypoint_overlay()
            self.mode = mode
            self.cancel_drawing()
            self._update_method_combo()

    def set_enabled_shapes(self, shapes) -> None:
        self.enabled_shapes = set(shapes)
        if self.mode not in self.enabled_shapes:
            self.mode = next(iter(self.enabled_shapes), ShapeType.RECTANGLE)
        self._update_method_combo()

    def set_language(self, language: str) -> None:
        self.language = "en_US" if language == "en_US" else "zh_CN"
        english = self.language == "en_US"
        self.method_combo.setToolTip(
            "Annotation method; stays armed until Esc" if english
            else "标注方式，选择后可连续标注，Esc 退出绘制"
        )
        self.keypoint_count_box.setToolTip(
            "Keypoint count; synced to kpt_shape in data.yaml" if english
            else "关键点位数量，修改后写入 data.yaml 的 kpt_shape"
        )
        self.keypoint_count_box.set_unit("pts" if english else "点")
        self._update_method_combo()

    def set_keypoint_count(self, count: int, names: list[str] | None = None) -> None:
        """Apply a keypoint count: 17 keeps the official COCO person names,
        anything else uses compact generated names unless explicit ones are
        provided (e.g. kpt_names from data.yaml)."""
        count = max(1, min(135, int(count)))
        if names and len(names) == count:
            schema = [str(name).strip() or f"kpt_{index + 1}" for index, name in enumerate(names)]
        elif count == len(COCO_PERSON_KEYPOINTS):
            schema = list(COCO_PERSON_KEYPOINTS)
        else:
            schema = [f"kpt_{index + 1}" for index in range(count)]
        self.set_keypoint_schema(schema)
        self._count_syncing = True
        self.keypoint_count_box.setValue(count)
        self._count_syncing = False

    def set_keypoint_groups(self, groups, current_name: str | None = None) -> None:
        """Refresh the keypoint-group selector; apply the current group's
        point names to the pending schema."""
        # (name, point names, predefined annotation label)
        self.keypoint_groups = [
            (group.name, list(group.keypoint_names), getattr(group, "label", "") or "")
            for group in groups
        ]
        self._group_syncing = True
        self.keypoint_group_box.clear()
        english = self.language == "en_US"
        for name, names, label in self.keypoint_groups:
            display = f"{name} · {len(names)}{' pts' if english else '点'}" if names else name
            self.keypoint_group_box.addItem(display, name)
        if self.keypoint_groups:
            index = self.keypoint_group_box.findData(current_name or "")
            if index < 0:
                index = 0
            self.keypoint_group_box.setCurrentIndex(index)
        self._group_syncing = False
        if self.keypoint_groups:
            self._apply_keypoint_group(self.keypoint_groups[self.keypoint_group_box.currentIndex()])
        self._position_method_widgets(self.mode == ShapeType.KEYPOINT)

    def active_keypoint_group(self) -> str | None:
        """Name of the group currently applied to keypoint drawing, if any."""
        index = self.keypoint_group_box.currentIndex()
        if 0 <= index < len(self.keypoint_groups):
            return self.keypoint_groups[index][0]
        return None

    def apply_keypoint_group_by_name(self, name: str) -> None:
        """Re-apply a named group (e.g. after a dataset schema reset)."""
        index = self.keypoint_group_box.findData(name)
        if index >= 0:
            self.keypoint_group_box.setCurrentIndex(index)
            self._apply_keypoint_group(self.keypoint_groups[index])

    def _keypoint_group_changed(self) -> None:
        if self._group_syncing:
            return
        index = self.keypoint_group_box.currentIndex()
        if 0 <= index < len(self.keypoint_groups):
            self._apply_keypoint_group(self.keypoint_groups[index])

    def _apply_keypoint_group(self, group: tuple[str, list[str], str]) -> None:
        name, names, _label = group
        if not names:
            return
        self.set_keypoint_count(len(names), names)
        self.pending_keypoints.clear()
        self.keypointGroupSelected.emit(name)
        self.keypointCountChanged.emit(len(names))
        self._update_keypoint_overlay()

    def _keypoint_count_changed(self, value: int) -> None:
        if self._count_syncing:
            return
        self.set_keypoint_count(value)
        self.pending_keypoints.clear()
        self.keypointCountChanged.emit(value)

    def _update_method_combo(self) -> None:
        """List only the methods the current dataset task supports."""
        available = [shape for shape in self.METHOD_ORDER if shape in self.enabled_shapes]
        self._method_syncing = True
        self.method_combo.clear()
        for shape in available:
            self.method_combo.addItem(self.method_label(shape, self.language), shape)
        index = self.method_combo.findData(self.mode)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        self.method_combo.setVisible(bool(available))
        self.method_combo.adjustSize()
        self._method_syncing = False
        self._position_method_widgets(self.mode == ShapeType.KEYPOINT)

    def _position_method_widgets(self, show_count: bool) -> None:
        self.method_combo.move(14, 14)
        # The point count is defined by the selected keypoint type (edited in
        # the 点位类型 dialog), so the canvas only offers the type selector.
        self.keypoint_count_box.hide()
        self.keypoint_group_box.setVisible(show_count and bool(self.keypoint_groups))
        if show_count:
            self.keypoint_group_box.adjustSize()
            self.keypoint_group_box.move(14 + self.method_combo.width() + 8, 14)
        # The in-drawing progress pill sits in the same row as the method and
        # type controls, so it re-anchors whenever they move.
        overlay = getattr(self, "_kp_overlay", None)
        if overlay is not None and overlay.isVisible():
            overlay.move(self._method_row_end_x(), 14)
        poly_overlay = getattr(self, "_poly_overlay", None)
        if poly_overlay is not None and poly_overlay.isVisible():
            poly_overlay.move(self._method_row_end_x(), 14)

    def _method_row_end_x(self) -> int:
        """X just past the last visible control in the top-left row."""
        x = 14 + self.method_combo.width()
        if self.keypoint_group_box.isVisible():
            self.keypoint_group_box.adjustSize()
            x += 8 + self.keypoint_group_box.width()
        return x + 8

    def _method_combo_changed(self) -> None:
        if self._method_syncing:
            return
        shape = self.method_combo.currentData()
        if shape is None:
            return
        self.set_mode(shape)
        self._enable_draw_mode()
        self.methodRequested.emit(shape)

    def set_keypoint_schema(self, names: list[str]) -> None:
        parsed = [str(name).strip() for name in names if str(name).strip()]
        self.keypoint_schema = parsed or list(COCO_PERSON_KEYPOINTS)

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
            # An explicit fit clears the remembered zoom for the next image.
            self._kept_view = None

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().keyReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        """Wheel zooms toward the cursor, the standard image-view gesture."""
        if not self.image_item or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom_at_view_pos(event.position().toPoint(), factor)

    def _zoom_at_view_pos(self, view_pos, factor: float) -> None:
        current = self.transform().m11()
        target = max(0.05, min(20.0, current * factor))
        if current <= 0 or abs(target - current) < 1e-9:
            return
        scene_pos = self.mapToScene(view_pos)
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.scale(target / current, target / current)
        self.setTransformationAnchor(anchor)
        shifted = self.mapFromScene(scene_pos)
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(shifted.x() - view_pos.x()))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(shifted.y() - view_pos.y()))

    def _capture_view_state(self) -> None:
        """Remember zoom level and center before the image is replaced."""
        if self.image_item is None or self.fit_scale <= 0:
            return
        rect = self.sceneRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return
        center = self.mapToScene(self.viewport().rect().center())
        self._kept_view = (
            self.transform().m11() / self.fit_scale,
            center.x() / rect.width(),
            center.y() / rect.height(),
        )

    def _restore_view_state(self) -> None:
        """Re-apply the remembered zoom so detail work survives image switches."""
        if not self._kept_view or not self.image_item or self.fit_scale <= 0:
            return
        ratio, cx, cy = self._kept_view
        if abs(ratio - 1.0) < 1e-3:
            return
        target = max(0.05, min(20.0, self.fit_scale * ratio))
        current = self.transform().m11()
        if current > 0 and abs(target - current) > 1e-9:
            self.scale(target / current, target / current)
        rect = self.sceneRect()
        self.centerOn(QPointF(rect.width() * cx, rect.height() * cy))

    def cancel_drawing(self) -> None:
        self.drawing = False
        self.polygon_points.clear()
        self.pending_keypoints.clear()
        self._clear_polygon_drawing_dots()
        # Scene-level drawing aids must go too: orphan preview dots sit above
        # (z=22) the keypoint markers and would block every later click on them.
        self._clear_keypoint_drawing_preview()
        self._hide_polygon_count_overlay()
        self._hide_keypoint_overlay()
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
        self._tab_shortcut.setEnabled(False)
        self._hide_keypoint_overlay()
        self._hide_polygon_count_overlay()
        self.cancel_drawing()
        self._hide_crosshair()
        self.unsetCursor()

    def _rearm_draw(self) -> None:
        """Stay armed after a completed shape so the next one can start at once.

        cancel_drawing clears the finished shape's live aids (dots, previews,
        overlays); _enable_draw_mode then re-shows the fresh progress overlay.

        This is the conventional behavior of mainstream tools (CVAT, LabelMe):
        selecting an annotation method stays active until the user exits
        with Esc, right click, or the W toggle.
        """
        self.cancel_drawing()
        self._hide_crosshair()
        if self.draw_enabled:
            self._enable_draw_mode()

    def _snapshot_annotations(self) -> list[Annotation]:
        return [Annotation.from_dict(item.to_dict()) for item in self.annotations]

    def push_undo_snapshot(self) -> None:
        """Capture the current annotations so the next change can be undone."""
        self._undo_stack.append(self._snapshot_annotations())
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot_annotations())
        self._restore_annotations(self._undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot_annotations())
        self._restore_annotations(self._redo_stack.pop())
        return True

    def _restore_annotations(self, snapshot: list[Annotation]) -> None:
        self.cancel_drawing()
        # Copy again so consecutive undo/redo steps never alias each other.
        self.annotations = [Annotation.from_dict(item.to_dict()) for item in snapshot]
        self._rebuild_annotation_items()
        self.annotationChanged.emit()
        self.dirtyChanged.emit(True)

    def _rebuild_annotation_items(self) -> None:
        for item in self.annotation_items:
            if item.scene():
                self.scene.removeItem(item)
        self.annotation_items = []
        for annotation in self.annotations:
            self._add_annotation_item(annotation)
        self.scene.clearSelection()

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
        """Show committed vertices as dots plus the segment under the cursor."""
        points = list(self.polygon_points)
        # Live vertex dots: the user sees exactly which points are committed.
        self._refresh_polygon_drawing_dots()
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
        if self.mode == ShapeType.KEYPOINT:
            self._update_keypoint_overlay()
        self._tab_shortcut.setEnabled(self.mode == ShapeType.KEYPOINT)
        self._show_first_use_hint()
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

    def _polygon_vertex_index(self, item: AnnotationItem, point: QPointF) -> tuple[int, int] | None:
        if item.annotation.shape_type != ShapeType.POLYGON:
            return None
        parts = item.annotation.polygon_parts or [item.annotation.points]
        return next(
            ((part_index, index) for part_index, part in enumerate(parts)
             for index, vertex in enumerate(part) if (vertex - point).manhattanLength() <= 10),
            None,
        )

    def _begin_box_drag(self, item: AnnotationItem, point: QPointF) -> None:
        self._select_item(item)
        self.drag_keypoint_index = self._keypoint_index(item, point)
        vertex_index = self._polygon_vertex_index(item, point)
        item.drag_corner = self._corner_index(item, point)
        self.drag_mode = (
            "keypoint" if self.drag_keypoint_index is not None
            else "vertex" if vertex_index is not None
            else "resize" if item.drag_corner is not None else "move"
        )
        if self.drag_mode == "move" and item.annotation.shape_type == ShapeType.KEYPOINT:
            # A keypoint annotation's box is derived from its points, so
            # dragging the box body must not move it; drag the point markers
            # instead. Selection from the press is kept.
            self.drag_mode = ""
            self.drag_item = None
            self.drag_keypoint_index = None
            item.drag_corner = None
            return
        self.drag_item = item
        self.drag_start = point
        self.drag_points = list(item.annotation.points)
        self.drag_polygon_parts = [list(part) for part in item.annotation.polygon_parts]
        self.drag_vertex_index = vertex_index
        self.drag_changed = False
        # Capture the pre-drag state; it only becomes an undo step if the
        # gesture actually changes geometry on release.
        self._drag_undo_pending = self._snapshot_annotations()

    def _rotation_handle_hit(self, item: AnnotationItem, point: QPointF) -> bool:
        handle = item.rotation_handle
        return handle is not None and handle.isVisible() and (handle.pos() - point).manhattanLength() <= 12

    def _begin_rotation_drag(self, item: AnnotationItem, point: QPointF) -> None:
        """Start rotating an OBB around its center from the rotation handle."""
        self._select_item(item)
        self.drag_item = item
        self.drag_start = point
        self.drag_points = list(item.annotation.points)
        self.drag_polygon_parts = []
        self.drag_keypoint_index = None
        self.drag_vertex_index = None
        item.drag_corner = None
        self.drag_mode = "rotate"
        corners = item.annotation.points
        self.drag_center = QPointF(
            sum(p.x() for p in corners) / len(corners),
            sum(p.y() for p in corners) / len(corners),
        )
        self.drag_start_angle = math.atan2(
            point.y() - self.drag_center.y(), point.x() - self.drag_center.x()
        )
        self.drag_changed = False
        self._drag_undo_pending = self._snapshot_annotations()

    def _drag_box(self, point: QPointF) -> None:
        if not self.drag_item:
            return
        annotation = self.drag_item.annotation
        dx, dy = point.x() - self.drag_start.x(), point.y() - self.drag_start.y()
        if self.drag_mode == "rotate":
            center = self.drag_center
            angle = math.atan2(point.y() - center.y(), point.x() - center.x()) - self.drag_start_angle
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                step = math.pi / 12  # 15-degree snapping, the common convention
                angle = round(angle / step) * step
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            annotation.points = [
                QPointF(
                    center.x() + (p.x() - center.x()) * cos_a - (p.y() - center.y()) * sin_a,
                    center.y() + (p.x() - center.x()) * sin_a + (p.y() - center.y()) * cos_a,
                )
                for p in self.drag_points
            ]
        elif self.drag_mode == "keypoint" and self.drag_keypoint_index is not None:
            keypoint = annotation.keypoints[self.drag_keypoint_index]
            keypoint.point = point
            # Keep the outer bbox wrapping every visible keypoint so the
            # saved YOLO/COCO bbox never drifts away from the points.
            visible = [kp.point for kp in annotation.keypoints if kp.visibility > 0]
            if visible:
                bounds = polygon_bounds(visible)
                annotation.points = [bounds.topLeft(), bounds.bottomRight()]
        elif self.drag_mode == "vertex" and self.drag_vertex_index is not None:
            part_index, vertex_index = self.drag_vertex_index
            parts = [list(part) for part in (annotation.polygon_parts or [annotation.points])]
            parts[part_index][vertex_index] = point
            annotation.polygon_parts = parts
            annotation.points = list(parts[0])
        elif self.drag_mode == "move":
            if annotation.shape_type == ShapeType.POLYGON:
                source_parts = self.drag_polygon_parts or [self.drag_points]
                annotation.polygon_parts = [
                    [QPointF(p.x() + dx, p.y() + dy) for p in part]
                    for part in source_parts
                ]
                annotation.points = list(annotation.polygon_parts[0])
            else:
                annotation.points = [QPointF(p.x() + dx, p.y() + dy) for p in self.drag_points]
            if annotation.shape_type == ShapeType.KEYPOINT:
                annotation.keypoints = [Keypoint(item.name, QPointF(item.point.x() + dx, item.point.y() + dy), item.visibility) for item in annotation.keypoints]
        elif self.drag_mode == "resize" and annotation.shape_type == ShapeType.KEYPOINT:
            # Scaling the bbox also scales the keypoints so the pose stays
            # in sync with its enclosing box.
            old_rect = QRectF(self.drag_points[0], self.drag_points[-1]).normalized()
            new_rect = QRectF(annotation.points[0], annotation.points[-1]).normalized()
            if old_rect.width() > 0 and old_rect.height() > 0 and not new_rect.isNull():
                sx = new_rect.width() / old_rect.width()
                sy = new_rect.height() / old_rect.height()
                for kp in annotation.keypoints:
                    kp.point = QPointF(
                        new_rect.left() + (kp.point.x() - old_rect.left()) * sx,
                        new_rect.top() + (kp.point.y() - old_rect.top()) * sy,
                    )
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
        self.push_undo_snapshot()
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

    _clipboard = None

    def copy_selected(self) -> bool:
        if self.draw_enabled:
            return False
        selected = [item for item in self.annotation_items if item.isSelected()]
        if not selected:
            return False
        CanvasView._clipboard = [item.annotation for item in selected]
        return True

    def paste_clipboard(self) -> int:
        if self.draw_enabled or not CanvasView._clipboard:
            return 0
        self.push_undo_snapshot()
        pasted = 0
        for annotation in CanvasView._clipboard:
            import copy
            clone = Annotation.from_dict(annotation.to_dict())
            # Offset by 20px so the paste is visible and selectable.
            offset = QPointF(20, 20)
            if clone.shape_type in {ShapeType.RECTANGLE, ShapeType.SQUARE, ShapeType.OBB}:
                clone.points = [QPointF(p.x() + offset.x(), p.y() + offset.y()) for p in clone.points]
            elif clone.shape_type == ShapeType.POLYGON:
                parts = clone.polygon_parts or [clone.points]
                parts = [[QPointF(p.x() + offset.x(), p.y() + offset.y()) for p in part] for part in parts]
                clone.polygon_parts = parts
                clone.points = list(parts[0])
            elif clone.shape_type == ShapeType.KEYPOINT:
                clone.keypoints = [Keypoint(kp.name, QPointF(kp.point.x() + offset.x(), kp.point.y() + offset.y()), kp.visibility) for kp in clone.keypoints]
                if clone.points:
                    clone.points = [QPointF(p.x() + offset.x(), p.y() + offset.y()) for p in clone.points]
            self.annotations.append(clone)
            self._add_annotation_item(clone)
            self.annotationCreated.emit(clone)
            pasted += 1
        if pasted:
            self.dirtyChanged.emit(True)
        return pasted

    def select_all(self) -> None:
        if self.draw_enabled:
            return
        for item in self.annotation_items:
            item.setSelected(True)

    def delete_selected(self) -> bool:
        selected = [item for item in self.annotation_items if item.isSelected()]
        if not selected:
            return False
        self.push_undo_snapshot()
        for item in selected:
            if item.annotation in self.annotations: self.annotations.remove(item.annotation)
            self.scene.removeItem(item); self.annotation_items.remove(item)
        self.annotationDeleted.emit(); self.dirtyChanged.emit(True)
        return True

    def mousePressEvent(self, event) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        # Any canvas click clears the independent label-list selection.
        self.annotationSelected.emit(None)
        if event.button() == Qt.MouseButton.RightButton and self._finish_polygon_on_right_click():
            return
        if event.button() == Qt.MouseButton.RightButton and self.draw_enabled and self.mode == ShapeType.KEYPOINT and self.pending_keypoints:
            # Right-click during drawing skips the current keypoint (visibility=0).
            center = self.mapToScene(self.viewport().rect().center())
            kp_name = self.keypoint_schema[len(self.pending_keypoints)] if len(self.pending_keypoints) < len(self.keypoint_schema) else "keypoint_%d" % len(self.pending_keypoints)
            self.pending_keypoints.append(Keypoint(kp_name, center, 0))
            self._update_keypoint_overlay()
            if self.keypoint_schema and len(self.pending_keypoints) >= len(self.keypoint_schema):
                self._finish_keypoint_annotation()
            return
        if event.button() == Qt.MouseButton.RightButton and self._cycle_keypoint_visibility_at(event):
                return
        item = self._item_at_event(event)
        if event.button() == Qt.MouseButton.RightButton and item:
            if self._try_delete_polygon_vertex(item, self.mapToScene(event.position().toPoint())):
                return
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
            scene_point = self.mapToScene(event.position().toPoint())
            if item.annotation.shape_type == ShapeType.OBB and self._rotation_handle_hit(item, scene_point):
                self._begin_rotation_drag(item, scene_point); return
            self._begin_box_drag(item, scene_point); return
        if event.button() == Qt.MouseButton.LeftButton and not item:
            self.scene.clearSelection()
        if event.button() != Qt.MouseButton.LeftButton or not self.image_item or not self.draw_enabled:
            super().mousePressEvent(event); return
        point = self.mapToScene(event.position().toPoint())
        if self.image_item is not None and not self.image_item.boundingRect().contains(point):
            # Clicks outside the image are ignored while drawing: they would
            # otherwise create out-of-range coordinates that break YOLO/COCO.
            super().mousePressEvent(event)
            return
        if self.mode == ShapeType.POLYGON:
            # Snap-to-close: clicking near the first vertex closes the polygon
            # instead of adding a duplicate point (CVAT/LabelMe convention).
            if len(self.polygon_points) >= 3:
                snap_radius = 14.0 / max(self.transform().m11(), 0.001)
                if (point - self.polygon_points[0]).manhattanLength() <= snap_radius:
                    self._commit_polygon_now()
                    return
            self.polygon_points.append(point); self.drawing = True
            self._update_polygon_preview()
            self._update_polygon_count_overlay()
            return
        if self.mode == ShapeType.KEYPOINT:
            # Drag a nearby pending keypoint to reposition it before finishing.
            threshold = 14.0 / max(self.transform().m11(), 0.001)
            for idx, kp in enumerate(self.pending_keypoints):
                if (kp.point - point).manhattanLength() <= threshold:
                    self._pending_kp_drag = idx
                    self._pending_kp_drag_origin = QPointF(kp.point)
                    return
            self._pending_kp_drag = None
            name = self.keypoint_schema[len(self.pending_keypoints)] if len(self.pending_keypoints) < len(self.keypoint_schema) else f"keypoint_{len(self.pending_keypoints)}"
            self.pending_keypoints.append(Keypoint(name, point, 2)); self.drawing = True
            self._update_keypoint_overlay()
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
            self._update_first_vertex_glow(scene_point)
            self._update_polygon_count_overlay()
            return
        if self.drawing and self.mode == ShapeType.KEYPOINT:
            if getattr(self, "_pending_kp_drag", None) is not None and self._pending_kp_drag < len(self.pending_keypoints):
                self.pending_keypoints[self._pending_kp_drag].point = scene_point
            self._refresh_keypoint_drawing_preview()
        if self.drawing and self.mode != ShapeType.POLYGON:
            current = scene_point
            square = self.mode == ShapeType.SQUARE or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            rect = constrain_square(self.start, current) if square else normalize_rect(self.start, current)
            self._update_box_preview(rect)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.drag_item:
            changed = self.drag_changed
            self.drag_item = None; self.drag_mode = ""; self.drag_points = []; self.drag_polygon_parts = []; self.drag_vertex_index = None; self.drag_changed = False
            if changed:
                if self._drag_undo_pending is not None:
                    self._undo_stack.append(self._drag_undo_pending)
                    if len(self._undo_stack) > 100:
                        self._undo_stack.pop(0)
                    self._redo_stack.clear()
                self.annotationChanged.emit()
                self.dirtyChanged.emit(True)
            self._drag_undo_pending = None
            return
        if event.button() == Qt.MouseButton.MiddleButton or self.pan_mode:
            super().mouseReleaseEvent(event); self.setDragMode(QGraphicsView.DragMode.RubberBandDrag); return
        if self.drawing and self.mode == ShapeType.KEYPOINT and event.button() == Qt.MouseButton.LeftButton:
            self._pending_kp_drag = None
            return
        if self.drawing and self.mode != ShapeType.POLYGON and event.button() == Qt.MouseButton.LeftButton:
            current = self.mapToScene(event.position().toPoint())
            square = self.mode == ShapeType.SQUARE or bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            rect = constrain_square(self.start, current) if square else normalize_rect(self.start, current)
            if rect.width() >= 3 and rect.height() >= 3:
                self.push_undo_snapshot()
                if self.mode == ShapeType.OBB:
                    corners = [
                        rect.topLeft(), QPointF(rect.right(), rect.top()),
                        rect.bottomRight(), QPointF(rect.left(), rect.bottom()),
                    ]
                    annotation = Annotation(self.mode, self.current_label, corners, self.current_color)
                else:
                    annotation = Annotation(self.mode, self.current_label, [rect.topLeft(), rect.bottomRight()], self.current_color)
                self.annotations.append(annotation); self._add_annotation_item(annotation); self.annotationCreated.emit(annotation); self.dirtyChanged.emit(True)
                self._rearm_draw(); return
            # A tiny drag is treated as an accidental click, not a shape; the
            # method stays armed so drawing can continue.
            self._rearm_draw(); return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        item = self._item_at_event(event)
        if item:
            if self._try_insert_polygon_vertex(item, self.mapToScene(event.position().toPoint())):
                return
            self._select_item(item); self.annotationEditRequested.emit(item.annotation); return
        if self.mode == ShapeType.POLYGON and len(self.polygon_points) >= 3:
            points = list(self.polygon_points)
            if len(points) >= 2 and (points[-1] - points[-2]).manhattanLength() <= 10:
                points.pop()
            if len(points) >= 3:
                self.push_undo_snapshot()
                annotation = Annotation(ShapeType.POLYGON, self.current_label, points, self.current_color)
                self.annotations.append(annotation); self._add_annotation_item(annotation); self.annotationCreated.emit(annotation); self.dirtyChanged.emit(True)
            self._rearm_draw(); return
        if self.mode == ShapeType.KEYPOINT and self.pending_keypoints:
            self._finish_keypoint_annotation(); return
        super().mouseDoubleClickEvent(event)

    @staticmethod
    def _point_to_segment_distance(point, a, b):
        import math
        dx, dy = b.x() - a.x(), b.y() - a.y()
        if dx == 0 and dy == 0:
            return math.hypot(point.x() - a.x(), point.y() - a.y())
        t = max(0.0, min(1.0, ((point.x() - a.x()) * dx + (point.y() - a.y()) * dy) / (dx * dx + dy * dy)))
        return math.hypot(point.x() - (a.x() + t * dx), point.y() - (a.y() + t * dy))

    def _try_insert_polygon_vertex(self, item, point):
        if item.annotation.shape_type not in {ShapeType.POLYGON, ShapeType.OBB}:
            return False
        parts = item.annotation.polygon_parts or [item.annotation.points]
        if len(parts[0]) < 3:
            return False
        threshold = 12.0 / max(self.transform().m11(), 0.001)
        for k in range(len(parts[0])):
            p1 = parts[0][k]
            p2 = parts[0][(k + 1) % len(parts[0])]
            if self._point_to_segment_distance(point, p1, p2) <= threshold:
                self.push_undo_snapshot()
                parts[0].insert(k + 1, QPointF(point))
                item.annotation.points = list(parts[0])
                item.annotation.polygon_parts = [list(p) for p in parts]
                item.refresh()
                self.annotationChanged.emit()
                self.dirtyChanged.emit(True)
                return True
        return False

    def _try_delete_polygon_vertex(self, item, point):
        if item.annotation.shape_type not in {ShapeType.POLYGON, ShapeType.OBB}:
            return False
        parts = item.annotation.polygon_parts or [item.annotation.points]
        threshold = 12.0 / max(self.transform().m11(), 0.001)
        for part in parts:
            if len(part) <= 3:
                continue
            for vi, vertex in enumerate(part):
                if (vertex - point).manhattanLength() <= threshold:
                    self.push_undo_snapshot()
                    part.pop(vi)
                    item.annotation.points = list(parts[0])
                    item.annotation.polygon_parts = [list(p) for p in parts]
                    item.refresh()
                    self.annotationChanged.emit()
                    self.dirtyChanged.emit(True)
                    return True
        return False

    def _update_keypoint_overlay(self):
        if not (self.draw_enabled and self.mode == ShapeType.KEYPOINT):
            self._hide_keypoint_overlay()
            return
        total = len(self.keypoint_schema)
        done = len(self.pending_keypoints)
        next_name = self.keypoint_schema[done] if done < len(self.keypoint_schema) else "keypoint_%d" % done
        label_text = "%s  (%d/%d)" % (next_name, done + 1, total) if total else next_name
        if getattr(self, "_kp_overlay", None) is None:
            self._kp_overlay = QLabel(self)
            self._kp_overlay.setObjectName("keypointOverlay")
            self._kp_overlay.setStyleSheet(
                "QLabel { background: rgba(43,45,48,200); color: #7ee787; "
                "border: 1px solid #4A4E55; border-radius: 5px; "
                "padding: 0 10px; font-size: 12px; font-weight: 600; }"
            )
            self._kp_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._kp_overlay.setText(label_text)
        self._kp_overlay.setFixedHeight(26)
        self._kp_overlay.adjustSize()
        self._kp_overlay.move(self._method_row_end_x(), 14)
        self._kp_overlay.show()
        self._kp_overlay.raise_()

    def _refresh_polygon_drawing_dots(self):
        """Small persistent dots at each committed polygon vertex."""
        self._clear_polygon_drawing_dots()
        if not self.drawing or self.mode != ShapeType.POLYGON:
            return
        if not hasattr(self, "_polygon_dots"):
            self._polygon_dots = []
        for i, point in enumerate(self.polygon_points):
            dot = QGraphicsEllipseItem(-3.5, -3.5, 7, 7)
            dot.setPos(point)
            dot.setPen(QPen(QColor("#ffffff"), 1))
            dot.setBrush(QColor(self.current_color))
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            dot.setZValue(22)
            self.scene.addItem(dot)
            self._polygon_dots.append(dot)

    def _clear_polygon_drawing_dots(self):
        for dot in getattr(self, "_polygon_dots", []):
            if dot.scene():
                dot.scene().removeItem(dot)
        self._polygon_dots = []

    def _refresh_keypoint_drawing_preview(self):
        """Solid dots at committed keypoints, plus skeleton lines."""
        self._clear_keypoint_drawing_preview()
        if not self.drawing or self.mode != ShapeType.KEYPOINT:
            return
        if not hasattr(self, "_kp_drawing_items"):
            self._kp_drawing_items = []
        color = QColor(self.current_color)
        for kp in self.pending_keypoints:
            dot = QGraphicsEllipseItem(-4, -4, 8, 8)
            dot.setPos(kp.point)
            if kp.visibility > 0:
                dot.setPen(QPen(QColor("#ffffff"), 1))
                dot.setBrush(color)
            else:
                dot.setPen(QPen(QColor("#8a8f98"), 1))
                dot.setBrush(QColor(0, 0, 0, 0))
            dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            dot.setZValue(22)
            self.scene.addItem(dot)
            self._kp_drawing_items.append(dot)
        from src.models.keypoint import COCO_PERSON_SKELETON
        if len(self.pending_keypoints) >= 17:
            pen = QPen(color, 1)
            for start, end in COCO_PERSON_SKELETON:
                if start < len(self.pending_keypoints) and end < len(self.pending_keypoints):
                    a = self.pending_keypoints[start].point
                    b = self.pending_keypoints[end].point
                    line = QGraphicsLineItem(a.x(), a.y(), b.x(), b.y())
                    line.setPen(pen)
                    line.setZValue(19)
                    self.scene.addItem(line)
                    self._kp_drawing_items.append(line)

    def _clear_keypoint_drawing_preview(self):
        for item in getattr(self, "_kp_drawing_items", []):
            if item.scene():
                item.scene().removeItem(item)
        self._kp_drawing_items = []

    def _commit_polygon_now(self):
        """Close the polygon at the first vertex (snap-to-close)."""
        if len(self.polygon_points) < 3:
            return
        points = list(self.polygon_points)
        self.push_undo_snapshot()
        annotation = Annotation(ShapeType.POLYGON, self.current_label, points, self.current_color)
        self.annotations.append(annotation)
        self._add_annotation_item(annotation)
        self.annotationCreated.emit(annotation)
        self.dirtyChanged.emit(True)
        self._rearm_draw()

    def _update_first_vertex_glow(self, cursor: QPointF):
        """Enlarge + brighten the first vertex dot when the cursor
        is close enough to trigger snap-to-close."""
        snap_radius = 14.0 / max(self.transform().m11(), 0.001)
        near = len(self.polygon_points) >= 3 and (cursor - self.polygon_points[0]).manhattanLength() <= snap_radius
        dots = getattr(self, "_polygon_dots", [])
        if not dots:
            return
        first = dots[0]
        if near:
            first.setRect(-5.5, -5.5, 11, 11)
            first.setBrush(QColor("#7ee787"))
            first.setPen(QPen(QColor("#ffffff"), 2))
        else:
            first.setRect(-3.5, -3.5, 7, 7)
            first.setBrush(QColor(self.current_color))
            first.setPen(QPen(QColor("#ffffff"), 1))

    def _update_polygon_count_overlay(self):
        count = len(self.polygon_points)
        if count == 0:
            self._hide_polygon_count_overlay()
            return
        snap_hint = '  (点击起点闭合)' if count >= 3 else ''
        label_text = '%d 个顶点%s' % (count, snap_hint)
        if getattr(self, '_poly_overlay', None) is None:
            self._poly_overlay = QLabel(self)
            self._poly_overlay.setObjectName('polygonOverlay')
            self._poly_overlay.setStyleSheet(
                'QLabel { background: rgba(43,45,48,200); color: #7ee787; '
                'border: 1px solid #4A4E55; border-radius: 5px; '
                'padding: 0 10px; font-size: 12px; font-weight: 600; }'
            )
            self._poly_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._poly_overlay.setText(label_text)
        self._poly_overlay.setFixedHeight(26)
        self._poly_overlay.adjustSize()
        self._poly_overlay.move(self._method_row_end_x(), 14)
        self._poly_overlay.show()
        self._poly_overlay.raise_()

    def _hide_polygon_count_overlay(self):
        if getattr(self, '_poly_overlay', None) is not None:
            self._poly_overlay.hide()

    _HINT_TEXTS = {
        ShapeType.RECTANGLE: "拖拽绘制矩形 · 按住 Shift 画正方形 · Esc 退出",
        ShapeType.SQUARE: "拖拽绘制正方形 · Esc 退出",
        ShapeType.POLYGON: "左键加点 · 双击/Enter/点起点闭合 · Backspace 撤点 · Esc 取消",
        ShapeType.OBB: "拖拽绘制旋转框 · 选中后拖绿色手柄旋转 · Esc 退出",
        ShapeType.KEYPOINT: "左键加点 · 点满自动完成 · Tab 跳过 · Backspace 撤点 · Esc 取消",
    }

    def _show_first_use_hint(self):
        """First time a method is used this session, show its操作
        hints for 3 seconds so new annotators can learn the gestures."""
        if self.mode in self._hints_shown:
            return
        self._hints_shown.add(self.mode)
        text = self._HINT_TEXTS.get(self.mode)
        if not text:
            return
        if self._hint_label is None:
            self._hint_label = QLabel(self)
            self._hint_label.setObjectName("drawHint")
            self._hint_label.setStyleSheet(
                "QLabel { background: rgba(43,45,48,220); color: #E6E9ED; "
                "border: 1px solid #4A4E55; border-radius: 6px; "
                "padding: 6px 14px; font-size: 13px; font-weight: 600; }"
            )
            self._hint_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._hint_label.setText(text)
        self._hint_label.adjustSize()
        # Below the method/count/group/progress row: y=14 is reserved for the
        # controls, and the centered hint would collide with them.
        self._hint_label.move((self.width() - self._hint_label.width()) // 2, 56)
        self._hint_label.show()
        self._hint_label.raise_()
        if self._hint_timer is not None:
            self._hint_timer.stop()
        self._hint_timer = QTimer(self)
        self._hint_timer.setSingleShot(True)
        self._hint_timer.timeout.connect(self._hint_label.hide)
        self._hint_timer.start(3000)

    def _hide_keypoint_overlay(self):
        if getattr(self, "_kp_overlay", None) is not None:
            self._kp_overlay.hide()

    def _finish_polygon_annotation(self) -> None:
        """Close the polygon from keyboard (Enter) without the dup-vertex pop."""
        if len(self.polygon_points) < 3:
            return
        points = list(self.polygon_points)
        self.push_undo_snapshot()
        annotation = Annotation(ShapeType.POLYGON, self.current_label, points, self.current_color)
        self.annotations.append(annotation)
        self._add_annotation_item(annotation)
        self.annotationCreated.emit(annotation)
        self.dirtyChanged.emit(True)
        self._rearm_draw()

    def active_keypoint_label(self) -> str | None:
        """The selected type's predefined annotation label, if it has one."""
        index = self.keypoint_group_box.currentIndex()
        if 0 <= index < len(self.keypoint_groups):
            return self.keypoint_groups[index][2] or None
        return None

    def _finish_keypoint_annotation(self) -> None:
        if not self.pending_keypoints:
            return
        points = [item.point for item in self.pending_keypoints if item.visibility > 0]
        bbox = polygon_bounds(points) if points else QRectF()
        self.push_undo_snapshot()
        # A keypoint type may predefine the annotation label; it wins over
        # whatever label the preset panel last armed.
        group_label = self.active_keypoint_label()
        label = group_label or self.current_label
        color = label_color(group_label) if group_label else self.current_color
        annotation = Annotation(
            ShapeType.KEYPOINT,
            label,
            [bbox.topLeft(), bbox.bottomRight()] if not bbox.isNull() else [],
            color,
            keypoints=list(self.pending_keypoints),
            schema_name="Custom Keypoints",
        )
        self.annotations.append(annotation)
        self._add_annotation_item(annotation)
        self.annotationCreated.emit(annotation)
        self.dirtyChanged.emit(True)
        self._rearm_draw()

    def _finish_polygon_on_right_click(self) -> bool:
        """Right click closes the polygon being drawn (LabelMe convention)."""
        if not (self.draw_enabled and self.mode == ShapeType.POLYGON and len(self.polygon_points) >= 3):
            return False
        points = list(self.polygon_points)
        if (points[-1] - points[-2]).manhattanLength() <= 10:
            points.pop()
        if len(points) >= 3:
            self.push_undo_snapshot()
            annotation = Annotation(ShapeType.POLYGON, self.current_label, points, self.current_color)
            self.annotations.append(annotation)
            self._add_annotation_item(annotation)
            self.annotationCreated.emit(annotation)
            self.dirtyChanged.emit(True)
        self._rearm_draw()
        return True

    def _cycle_keypoint_visibility_at(self, event) -> bool:
        """Right click on a keypoint marker cycles visibility 0 -> 1 -> 2.

        The three states are the COCO/YOLO keypoint visibility convention,
        so saved datasets stay standard-compliant.
        """
        hit = self.itemAt(event.position().toPoint())
        if not isinstance(hit, KeypointMarker):
            return False
        item = hit.parentItem()
        while item is not None and not isinstance(item, AnnotationItem):
            item = item.parentItem()
        if not isinstance(item, AnnotationItem):
            return False
        keypoint = item.annotation.keypoints[hit.keypoint_index]
        self.push_undo_snapshot()
        keypoint.visibility = (int(keypoint.visibility) + 1) % 3
        item.refresh()
        self.annotationChanged.emit()
        self.dirtyChanged.emit(True)
        return True

    def _skip_current_keypoint(self):
        """Skip the current keypoint: mark it invisible (vis=0)."""
        if not (self.draw_enabled and self.mode == ShapeType.KEYPOINT):
            return
        center = self.mapToScene(self.viewport().rect().center())
        kp_name = self.keypoint_schema[len(self.pending_keypoints)] if len(self.pending_keypoints) < len(self.keypoint_schema) else "keypoint_%d" % len(self.pending_keypoints)
        self.pending_keypoints.append(Keypoint(kp_name, center, 0))
        self._update_keypoint_overlay()
        self._refresh_keypoint_drawing_preview()
        if self.keypoint_schema and len(self.pending_keypoints) >= len(self.keypoint_schema):
            self._finish_keypoint_annotation()

    def leaveEvent(self, event) -> None:
        self._hide_crosshair()
        if not self.drawing:
            self._clear_polygon_drawing_dots()
            self._clear_keypoint_drawing_preview()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space and self.draw_enabled:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        elif event.key() == Qt.Key.Key_Escape:
            in_progress = self.drawing or self.polygon_points or self.pending_keypoints
            if self.draw_enabled and in_progress:
                # First Esc drops the half-drawn shape and keeps the method
                # armed; a second Esc exits drawing entirely.
                self.cancel_drawing()
                self._hide_crosshair()
            else:
                self._disable_draw_mode()
        elif event.key() == Qt.Key.Key_W:
            if self.draw_enabled:
                self._disable_draw_mode()
            else:
                self._enable_draw_mode()
        elif event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter} and self.draw_enabled:
            if self.mode == ShapeType.POLYGON:
                self._finish_polygon_annotation()
            elif self.mode == ShapeType.KEYPOINT and self.pending_keypoints:
                self._finish_keypoint_annotation()
        elif event.key() == Qt.Key.Key_Backspace and self.draw_enabled:
            # While drawing, Backspace removes the last placed vertex/point
            # instead of deleting a selected annotation.
            if self.mode == ShapeType.POLYGON and self.polygon_points:
                self.polygon_points.pop()
                self._update_polygon_preview()
            elif self.mode == ShapeType.KEYPOINT and self.pending_keypoints:
                self.pending_keypoints.pop()
        elif event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}: self.delete_selected()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier and self.draw_enabled and self.mode == ShapeType.POLYGON and self.polygon_points:
            self.polygon_points.pop()
            self._update_polygon_preview()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
        elif event.key() == Qt.Key.Key_Y and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.redo()
        elif event.key() in {Qt.Key.Key_Plus, Qt.Key.Key_Equal} and event.modifiers() & Qt.KeyboardModifier.ControlModifier: self.zoom_in()
        elif event.key() == Qt.Key.Key_Minus and event.modifiers() & Qt.KeyboardModifier.ControlModifier: self.zoom_out()
        elif event.key() == Qt.Key.Key_0 and event.modifiers() & Qt.KeyboardModifier.ControlModifier: self.fit_image()
        else: super().keyPressEvent(event)
