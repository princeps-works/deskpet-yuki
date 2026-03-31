from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class ScanScheduler(QObject):
    tick = pyqtSignal()

    def __init__(self, interval_sec: int) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, interval_sec) * 1000)
        self._timer.timeout.connect(self.tick.emit)

    def start(self) -> None:
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
