from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor

QtWebEngineCore = importlib.import_module("PyQt6.QtWebEngineCore")
QtWebEngineWidgets = importlib.import_module("PyQt6.QtWebEngineWidgets")
QWebEngineSettings = QtWebEngineCore.QWebEngineSettings
QWebEngineViewBase = QtWebEngineWidgets.QWebEngineView


class Live2DView(QWebEngineViewBase):
    def __init__(
        self,
        model_json_path: Path,
        *,
        follow_cursor: bool = True,
        model_scale: float = 1.0,
        idle_group: str = "Idle",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._model_json_path = model_json_path
        self._follow_cursor = follow_cursor
        self._model_scale = model_scale
        self._idle_group = idle_group
        self._last_pointer_x = 0.5
        self._last_pointer_y = 0.5
        self._last_pointer_sent_at = 0.0
        self._pointer_interval_sec = 0.08
        self._pointer_delta_threshold = 0.02
        self._scan_busy = False

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

        page = self.page()
        page.setBackgroundColor(QColor(0, 0, 0, 0))

        web_settings = page.settings()
        web_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        web_settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        web_settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        web_settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)

        self.loadFinished.connect(self._on_html_ready)

        html_path = Path(__file__).resolve().parent / "live2d_web" / "index.html"
        self.load(QUrl.fromLocalFile(str(html_path)))

    def _run_js(self, script: str) -> None:
        self.page().runJavaScript(script)

    def _on_html_ready(self, ok: bool) -> None:
        if not ok:
            print("[LIVE2D] web runtime load failed")
            return

        model_url = QUrl.fromLocalFile(str(self._model_json_path)).toString()
        options = {
            "followCursor": self._follow_cursor,
            "modelScale": self._model_scale,
            "idleGroup": self._idle_group,
        }
        payload = json.dumps(options, ensure_ascii=False)
        self._run_js(f"window.petLive2d && window.petLive2d.loadModel({json.dumps(model_url)}, {payload});")

    def set_pointer(self, normalized_x: float, normalized_y: float) -> None:
        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))

        dx = abs(x - self._last_pointer_x)
        dy = abs(y - self._last_pointer_y)
        now = time.monotonic()
        if dx < self._pointer_delta_threshold and dy < self._pointer_delta_threshold:
            if (now - self._last_pointer_sent_at) < self._pointer_interval_sec:
                return

        self._last_pointer_x = x
        self._last_pointer_y = y
        self._last_pointer_sent_at = now
        self._run_js(f"window.petLive2d && window.petLive2d.setPointer({x}, {y});")

    def set_scan_busy(self, busy: bool) -> None:
        self._scan_busy = bool(busy)
        if self._scan_busy:
            self._pointer_interval_sec = 0.20
            self._pointer_delta_threshold = 0.05
        else:
            self._pointer_interval_sec = 0.08
            self._pointer_delta_threshold = 0.02

    def tap_at(self, normalized_x: float, normalized_y: float) -> None:
        x = max(0.0, min(1.0, float(normalized_x)))
        y = max(0.0, min(1.0, float(normalized_y)))
        self._run_js(f"window.petLive2d && window.petLive2d.tapAt({x}, {y});")

    def set_follow_cursor(self, enabled: bool) -> None:
        self._run_js(f"window.petLive2d && window.petLive2d.setFollowCursor({str(bool(enabled)).lower()});")

    def play_motion(self, group: str) -> None:
        self._run_js(f"window.petLive2d && window.petLive2d.playMotion({json.dumps(group)});")

    def set_expression(self, expression_name: str) -> None:
        self._run_js(
            f"window.petLive2d && window.petLive2d.setExpression({json.dumps(expression_name)});"
        )
