from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap


def application_icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("#101722"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#00d9ff"))
    painter.setPen(QColor("#7cf4ff"))
    painter.drawRoundedRect(QRectF(8, 8, 48, 48), 14, 14)
    painter.setBrush(QColor("#101722"))
    painter.drawRoundedRect(QRectF(20, 20, 24, 24), 6, 6)
    painter.end()
    return QIcon(pixmap)
