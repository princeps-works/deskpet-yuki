from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel


class SpeechBubble(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setStyleSheet(
            "background: rgba(20, 20, 20, 180);"
            "color: white;"
            "border-radius: 8px;"
            "padding: 8px;"
        )
        self.hide()

    def say(self, text: str):
        self.setText(text)
        if self.maximumWidth() > 0:
            self.resize(self.maximumWidth(), 1)
        self.adjustSize()
        self.show()
