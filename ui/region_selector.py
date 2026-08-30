from __future__ import annotations

import math

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


def scale_region_to_capture(
    region: tuple[int, int, int, int],
    logical_size: tuple[int, int],
    capture_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert a Qt logical-pixel region to MSS physical pixels."""
    left, top, width, height = [int(value) for value in region]
    logical_width, logical_height = [max(1, int(value)) for value in logical_size]
    capture_width, capture_height = [max(1, int(value)) for value in capture_size]
    scale_x = capture_width / float(logical_width)
    scale_y = capture_height / float(logical_height)

    physical_left = max(0, int(math.floor(left * scale_x)))
    physical_top = max(0, int(math.floor(top * scale_y)))
    physical_right = min(capture_width, int(math.ceil((left + width) * scale_x)))
    physical_bottom = min(capture_height, int(math.ceil((top + height) * scale_y)))
    return (
        physical_left,
        physical_top,
        max(1, physical_right - physical_left),
        max(1, physical_bottom - physical_top),
    )


class RegionSelectOverlay(QWidget):
    region_selected = pyqtSignal(tuple)
    cancelled = pyqtSignal()

    def __init__(
        self,
        logical_geometry: tuple[int, int, int, int],
        capture_size: tuple[int, int] | None = None,
    ):
        super().__init__()
        left, top, width, height = logical_geometry
        self._logical_size = (max(1, int(width)), max(1, int(height)))
        self._capture_size = capture_size or self._logical_size
        self.setGeometry(left, top, width, height)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)

        self._dragging = False
        self._start = QPoint()
        self._current = QPoint()

    def show_for_selection(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start = event.position().toPoint()
            self._current = self._start
            self.update()
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()
            self.close()
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._current = event.position().toPoint()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._current = event.position().toPoint()
            rect = QRect(self._start, self._current).normalized()
            if rect.width() >= 8 and rect.height() >= 8:
                logical_region = (rect.left(), rect.top(), rect.width(), rect.height())
                self.region_selected.emit(
                    scale_region_to_capture(
                        logical_region,
                        self._logical_size,
                        self._capture_size,
                    )
                )
            else:
                self.cancelled.emit()
            self.close()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))

        if self._dragging:
            rect = QRect(self._start, self._current).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            pen = QPen(QColor(0, 220, 255, 220), 2)
            painter.setPen(pen)
            painter.drawRect(rect)
