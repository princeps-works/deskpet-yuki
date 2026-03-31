from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class RegionSelectOverlay(QWidget):
    region_selected = pyqtSignal(tuple)
    cancelled = pyqtSignal()

    def __init__(self, monitor_geometry: tuple[int, int, int, int]):
        super().__init__()
        left, top, width, height = monitor_geometry
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
                self.region_selected.emit((rect.left(), rect.top(), rect.width(), rect.height()))
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
