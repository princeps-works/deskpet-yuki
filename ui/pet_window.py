from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QEvent, QPoint, QRect, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QCursor, QGuiApplication, QPixmap
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from desktop_pet.config.settings import Settings
from desktop_pet.ui.bubble import SpeechBubble

try:
    from desktop_pet.ui.live2d_view import Live2DView
except Exception:
    Live2DView = None


class DesktopPet(QWidget):
    open_chat_requested = pyqtSignal()
    auto_comment_requested = pyqtSignal()
    auto_scan_toggled = pyqtSignal(bool)
    quit_requested = pyqtSignal()
    select_scan_region_requested = pyqtSignal()
    clear_scan_region_requested = pyqtSignal()
    tts_mute_toggled = pyqtSignal(bool)
    gaze_follow_toggled = pyqtSignal(bool)
    tutor_mode_toggled = pyqtSignal(bool)

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.live2d_py_min_w = 180
        self.live2d_py_min_h = 220
        self.live2d_py_max_w = 2000
        self.live2d_py_max_h = 2600
        self.live2d_py_mode = bool(settings.enable_live2d_py)
        self.live2d_py_model_w = max(self.live2d_py_min_w, min(self.live2d_py_max_w, int(settings.live2d_py_window_width)))
        self.live2d_py_model_h = max(self.live2d_py_min_h, min(self.live2d_py_max_h, int(settings.live2d_py_window_height)))
        self.live2d_py_aspect = self.live2d_py_model_w / max(1, self.live2d_py_model_h)
        self.live2d_ui_gap = 6
        self.live2d_side_panel_w = 88
        self.live2d_side_gap = 8
        self.drag_position = QPoint()
        self.dragging = False
        self.drag_moved = False
        self.drag_start_global = QPoint()
        self.drag_start_window_pos = QPoint()
        self.resizing = False
        self.resize_start_pos = QPoint()
        self.resize_start_w = 0
        self.resize_start_h = 0
        self.resize_margin = 16
        self.aspect_ratio = 1.0
        self.default_scale = 0.4
        self.auto_scan_enabled = True
        self.tts_muted = False
        self.gaze_follow_enabled = bool(settings.live2d_follow_cursor)
        self.tutor_mode_enabled = bool(settings.enable_tutor_persona)
        self.scan_busy = False
        self.live2d_view = None
        self.image_label = None
        self.control_bar = None
        self.control_border = None
        self.drag_overlay = None
        self.btn_chat = None
        self.btn_comment = None
        self.btn_toggle_scan = None
        self.btn_select_region = None
        self.btn_clear_region = None
        self.btn_tts = None
        self.btn_gaze = None
        self.btn_tutor = None
        self.btn_quit = None
        self._bar_dragging = False
        self._ui_anchor_until = 0.0
        self._controls_visible = True
        self._controls_hide_timer = QTimer(self)
        self._controls_hide_timer.setSingleShot(True)
        self._controls_hide_timer.timeout.connect(self._hide_controls_if_idle)
        self._controls_proximity_timer = QTimer(self)
        self._controls_proximity_timer.setInterval(150)
        self._controls_proximity_timer.timeout.connect(self._check_controls_proximity)
        self._controls_style_visible = (
            "background-color: rgba(64, 156, 255, 3);"
            "border: 1px solid rgba(64,156,255,8);"
            "border-radius: 10px;"
        )
        self._controls_style_hidden = (
            "background: rgba(0, 0, 0, 0);"
            "border: 0px solid rgba(0,0,0,0);"
            "border-radius: 10px;"
        )
        self.init_ui()

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.setMinimumSize(100, 100)

        if self.live2d_py_mode:
            self._init_live2d_py_host_window()
        else:
            live2d_ready = self._init_live2d_renderer()
            if not live2d_ready:
                self._init_image_renderer()
            self._build_basic_controls()

        if self.image_label is not None:
            self.image_label.setGeometry(self.rect())
        if self.live2d_view is not None:
            self.live2d_view.setGeometry(self.rect())

        self.bubble = SpeechBubble(self)
        self.bubble.setMaximumWidth(260)
        self._bubble_hide_timer = QTimer(self)
        self._bubble_hide_timer.setSingleShot(True)
        self._bubble_hide_timer.timeout.connect(self.bubble.hide)

        self.resize_handle = QLabel("◢", self)
        self.resize_handle.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.resize_handle.setStyleSheet(
            "color: rgba(255, 255, 255, 220);"
            "font-size: 16px;"
            "background: rgba(0, 0, 0, 90);"
            "padding: 0px 2px 1px 2px;"
            "border-radius: 4px;"
        )
        self.resize_handle.adjustSize()
        self._update_resize_handle_pos()
        if self.live2d_py_mode:
            self.resize_handle.hide()
        self._move_to_bottom_right()

        self._gaze_timer = QTimer(self)
        self._gaze_timer.setInterval(80)
        self._gaze_timer.timeout.connect(self._push_cursor_gaze)
        self._gaze_timer.start()

        if self.live2d_py_mode:
            self._controls_proximity_timer.start()

    def _init_live2d_py_host_window(self) -> None:
        # live2d-py renders in an external process window; host overlays model for drag/menu controls.
        host_w, host_h = self._calc_live2d_py_ui_size()
        self.aspect_ratio = host_w / max(1, host_h)
        self.resize(host_w, host_h)
        self._build_live2d_py_controls()

    def _calc_live2d_py_ui_size(self) -> tuple[int, int]:
        ui_w = max(240, self.live2d_py_model_w + self.live2d_side_gap + self.live2d_side_panel_w)
        ui_h = max(220, self.live2d_py_model_h)
        return ui_w, ui_h

    def _apply_live2d_py_model_size(self, model_w: int, model_h: int) -> None:
        self.live2d_py_model_w = max(self.live2d_py_min_w, min(self.live2d_py_max_w, int(model_w)))
        self.live2d_py_model_h = max(self.live2d_py_min_h, min(self.live2d_py_max_h, int(model_h)))
        self.live2d_py_aspect = self.live2d_py_model_w / max(1, self.live2d_py_model_h)
        ui_w, ui_h = self._calc_live2d_py_ui_size()
        self.resize(ui_w, ui_h)
        self.aspect_ratio = ui_w / max(1, ui_h)

    def _build_live2d_py_controls(self) -> None:
        self.drag_overlay = QWidget(self)
        self.drag_overlay.installEventFilter(self)
        self.drag_overlay.setMouseTracking(True)
        self.drag_overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.drag_overlay.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.drag_overlay.setStyleSheet(
            "background: rgba(64,156,255,3);"
            "border: 0px solid rgba(0,0,0,0);"
        )

        self.control_border = QWidget(self)
        self.control_border.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.control_border.setStyleSheet(
            "background: rgba(64,156,255,3);"
            "border: 1px solid rgba(64,156,255,8);"
            "border-radius: 10px;"
        )

        self.control_bar = QWidget(self)
        self.control_bar.installEventFilter(self)
        self.control_bar.setMouseTracking(True)
        self.control_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.control_bar.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.control_bar.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.control_bar.setAutoFillBackground(False)
        self.control_bar.setStyleSheet(self._controls_style_visible)
        row = QVBoxLayout(self.control_bar)
        row.setContentsMargins(6, 8, 6, 8)
        row.setSpacing(6)

        def make_btn(text: str) -> QPushButton:
            btn = QPushButton(text, self.control_bar)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton {"
                " background: rgba(58, 132, 255, 165);"
                " color: white;"
                " border: none;"
                " border-radius: 8px;"
                " padding: 5px 9px;"
                "}"
                "QPushButton:hover { background: rgba(58, 132, 255, 210); }"
            )
            return btn

        self.btn_chat = make_btn("聊天")
        self.btn_comment = make_btn("评论")
        self.btn_toggle_scan = make_btn("暂停扫描")
        self.btn_select_region = make_btn("选区域")
        self.btn_clear_region = make_btn("清区域")
        self.btn_tts = make_btn("静音")
        self.btn_gaze = make_btn("关闭跟随")
        self.btn_tutor = make_btn("教学关")
        self.btn_quit = make_btn("退出")

        row.addWidget(self.btn_chat)
        row.addWidget(self.btn_comment)
        row.addWidget(self.btn_toggle_scan)
        row.addWidget(self.btn_select_region)
        row.addWidget(self.btn_clear_region)
        row.addWidget(self.btn_tts)
        row.addWidget(self.btn_gaze)
        row.addWidget(self.btn_tutor)
        row.addWidget(self.btn_quit)
        row.addStretch(1)

        self.btn_chat.clicked.connect(self.open_chat_requested.emit)
        self.btn_comment.clicked.connect(self.auto_comment_requested.emit)
        self.btn_toggle_scan.clicked.connect(lambda: self.auto_scan_toggled.emit(not self.auto_scan_enabled))
        self.btn_select_region.clicked.connect(self.select_scan_region_requested.emit)
        self.btn_clear_region.clicked.connect(self.clear_scan_region_requested.emit)
        self.btn_tts.clicked.connect(lambda: self.tts_mute_toggled.emit(not self.tts_muted))
        self.btn_gaze.clicked.connect(lambda: self.gaze_follow_toggled.emit(not self.gaze_follow_enabled))
        self.btn_tutor.clicked.connect(lambda: self.tutor_mode_toggled.emit(not self.tutor_mode_enabled))
        self.btn_quit.clicked.connect(self.quit_requested.emit)
        self.set_gaze_follow_enabled(self.gaze_follow_enabled)
        self.set_tutor_mode_enabled(self.tutor_mode_enabled)

        # Hover-to-show behavior: hidden by default, shown when mouse approaches.
        self._set_controls_visible(False)

    def _build_basic_controls(self) -> None:
        # Keep core interaction UI available even when all Live2D backends are disabled.
        self.control_bar = QWidget(self)
        self.control_bar.setMouseTracking(True)
        self.control_bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.control_bar.setStyleSheet(self._controls_style_visible)
        row = QVBoxLayout(self.control_bar)
        row.setContentsMargins(6, 8, 6, 8)
        row.setSpacing(6)

        def make_btn(text: str) -> QPushButton:
            btn = QPushButton(text, self.control_bar)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton {"
                " background: rgba(58, 132, 255, 185);"
                " color: white;"
                " border: none;"
                " border-radius: 8px;"
                " padding: 5px 9px;"
                "}"
                "QPushButton:hover { background: rgba(58, 132, 255, 220); }"
            )
            return btn

        self.btn_chat = make_btn("聊天")
        self.btn_comment = make_btn("评论")
        self.btn_toggle_scan = make_btn("暂停扫描")
        self.btn_select_region = make_btn("选区域")
        self.btn_clear_region = make_btn("清区域")
        self.btn_tts = make_btn("静音")
        self.btn_gaze = make_btn("关闭跟随")
        self.btn_tutor = make_btn("教学关")
        self.btn_quit = make_btn("退出")

        row.addWidget(self.btn_chat)
        row.addWidget(self.btn_comment)
        row.addWidget(self.btn_toggle_scan)
        row.addWidget(self.btn_select_region)
        row.addWidget(self.btn_clear_region)
        row.addWidget(self.btn_tts)
        row.addWidget(self.btn_gaze)
        row.addWidget(self.btn_tutor)
        row.addWidget(self.btn_quit)
        row.addStretch(1)

        self.btn_chat.clicked.connect(self.open_chat_requested.emit)
        self.btn_comment.clicked.connect(self.auto_comment_requested.emit)
        self.btn_toggle_scan.clicked.connect(lambda: self.auto_scan_toggled.emit(not self.auto_scan_enabled))
        self.btn_select_region.clicked.connect(self.select_scan_region_requested.emit)
        self.btn_clear_region.clicked.connect(self.clear_scan_region_requested.emit)
        self.btn_tts.clicked.connect(lambda: self.tts_mute_toggled.emit(not self.tts_muted))
        self.btn_gaze.clicked.connect(lambda: self.gaze_follow_toggled.emit(not self.gaze_follow_enabled))
        self.btn_tutor.clicked.connect(lambda: self.tutor_mode_toggled.emit(not self.tutor_mode_enabled))
        self.btn_quit.clicked.connect(self.quit_requested.emit)
        self.set_gaze_follow_enabled(self.gaze_follow_enabled)
        self.set_tutor_mode_enabled(self.tutor_mode_enabled)
        self._set_controls_visible(True)

    def eventFilter(self, obj, event):
        if obj in {self.control_bar, self.drag_overlay} and self.live2d_py_mode:
            if event.type() in {QEvent.Type.Enter, QEvent.Type.MouseMove}:
                self._set_controls_visible(True)

            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.dragging = True
                self._bar_dragging = True
                self.drag_moved = False
                self.drag_start_global = event.globalPosition().toPoint()
                self.drag_start_window_pos = self.pos()
                self._ui_anchor_until = time.monotonic() + 0.8
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return True

            if (
                event.type() == QEvent.Type.MouseMove
                and self._bar_dragging
                and (event.buttons() & Qt.MouseButton.LeftButton)
            ):
                if (event.globalPosition().toPoint() - self.drag_start_global).manhattanLength() > 3:
                    self.drag_moved = True
                self._move_by_drag_delta(event.globalPosition().toPoint())
                self._ui_anchor_until = time.monotonic() + 0.8
                event.accept()
                return True

            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._bar_dragging:
                    self.dragging = False
                    self._bar_dragging = False
                    self._ui_anchor_until = time.monotonic() + 0.8
                    self.setCursor(Qt.CursorShape.ArrowCursor)
                    event.accept()
                    return True

        return super().eventFilter(obj, event)

    def _set_controls_visible(self, visible: bool) -> None:
        if self.control_bar is None:
            return
        self._controls_visible = bool(visible)
        if self.control_border is not None:
            self.control_border.setVisible(self._controls_visible)
        if self.control_bar is not None:
            self.control_bar.setVisible(self._controls_visible)
        if self.btn_chat is not None:
            self.btn_chat.setVisible(self._controls_visible)
        if self.btn_comment is not None:
            self.btn_comment.setVisible(self._controls_visible)
        if self.btn_toggle_scan is not None:
            self.btn_toggle_scan.setVisible(self._controls_visible)
        if self.btn_select_region is not None:
            self.btn_select_region.setVisible(self._controls_visible)
        if self.btn_clear_region is not None:
            self.btn_clear_region.setVisible(self._controls_visible)
        if self.btn_tts is not None:
            self.btn_tts.setVisible(self._controls_visible)
        if self.btn_gaze is not None:
            self.btn_gaze.setVisible(self._controls_visible)
        if self.btn_quit is not None:
            self.btn_quit.setVisible(self._controls_visible)
        self.control_bar.setStyleSheet(
            self._controls_style_visible if self._controls_visible else self._controls_style_hidden
        )
        if self._controls_visible:
            self._controls_hide_timer.stop()

    def _hide_controls_if_idle(self) -> None:
        if not self.live2d_py_mode:
            return
        if self.dragging or self.resizing or self._bar_dragging:
            self._controls_hide_timer.start(900)
            return
        self._set_controls_visible(False)

    def _check_controls_proximity(self) -> None:
        if not self.live2d_py_mode or self.control_bar is None:
            return
        if self.dragging or self.resizing or self._bar_dragging:
            self._set_controls_visible(True)
            self._controls_hide_timer.start(900)
            return

        cursor = QCursor.pos()
        ui_rect = QRect(self.x() - 12, self.y() - 12, self.width() + 24, self.height() + 24)
        mx, my, mw, mh = self.get_live2d_py_target_geometry()
        model_rect = QRect(mx - 28, my - 28, mw + 56, mh + 56)

        if ui_rect.contains(cursor) or model_rect.contains(cursor):
            self._set_controls_visible(True)
            self._controls_hide_timer.start(900)

    def _init_live2d_renderer(self) -> bool:
        model_path = Path(self.settings.live2d_model_json).expanduser()
        if not self.settings.enable_live2d:
            return False
        if Live2DView is None:
            print("[LIVE2D] PyQt6-WebEngine 未安装，回退到静态立绘")
            return False
        if not model_path.exists():
            print(f"[LIVE2D] model3.json not found: {model_path}")
            return False

        try:
            self.live2d_view = Live2DView(
                model_json_path=model_path,
                follow_cursor=self.settings.live2d_follow_cursor,
                model_scale=self.settings.live2d_model_scale,
                idle_group=self.settings.live2d_idle_group,
                parent=self,
            )
            self.live2d_view.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.aspect_ratio = 0.78
            init_h = max(self.minimumHeight(), 520)
            init_w = max(self.minimumWidth(), int(round(init_h * self.aspect_ratio)))
            self.resize(init_w, init_h)
            print(f"[LIVE2D] model loaded from: {model_path}")
            return True
        except Exception as exc:
            print(f"[LIVE2D] init failed, fallback to image: {exc}")
            self.live2d_view = None
            return False

    def _init_image_renderer(self) -> None:
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)

        img_path = self.settings.pet_image_path
        pixmap = QPixmap(str(img_path))
        print(f"[INFO] image path: {img_path}")

        if pixmap.isNull():
            self.image_label.setText("pet.png 加载失败")
            self.image_label.setStyleSheet(
                "color: white; background: rgba(0, 0, 0, 160); padding: 8px;"
            )
            self.image_label.adjustSize()
            self.resize(self.image_label.size())
            if self.height() > 0:
                self.aspect_ratio = self.width() / self.height()
        else:
            self.image_label.setPixmap(pixmap)
            self.aspect_ratio = pixmap.width() / pixmap.height()
            init_w = max(self.minimumWidth(), int(round(pixmap.width() * self.default_scale)))
            init_h = max(self.minimumHeight(), int(round(init_w / self.aspect_ratio)))
            self.resize(init_w, init_h)

    def _to_normalized_pos(self, pos: QPoint) -> tuple[float, float]:
        w = max(1, self.width())
        h = max(1, self.height())
        nx = max(0.0, min(1.0, pos.x() / w))
        ny = max(0.0, min(1.0, pos.y() / h))
        return nx, ny

    def _move_by_drag_delta(self, global_pos: QPoint) -> None:
        delta = global_pos - self.drag_start_global
        self.move(self.drag_start_window_pos + delta)

    def _push_cursor_gaze(self) -> None:
        if self.live2d_view is None or not self.gaze_follow_enabled:
            return
        if self.dragging or self.resizing:
            return

        # During scan pipeline busy periods, reduce JS update pressure.
        if self.scan_busy and (self._gaze_timer.interval() != 180):
            self._gaze_timer.setInterval(180)
        elif (not self.scan_busy) and (self._gaze_timer.interval() != 80):
            self._gaze_timer.setInterval(80)

        global_pos = QCursor.pos()
        local_pos = self.mapFromGlobal(global_pos)
        nx, ny = self._to_normalized_pos(local_pos)
        self.live2d_view.set_pointer(nx, ny)

    def _in_resize_zone(self, pos: QPoint) -> bool:
        return (
            pos.x() >= self.width() - self.resize_margin
            and pos.y() >= self.height() - self.resize_margin
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._in_resize_zone(event.position().toPoint()):
                self.resizing = True
                self.resize_start_pos = event.globalPosition().toPoint()
                if self.live2d_py_mode:
                    self.resize_start_w = self.live2d_py_model_w
                    self.resize_start_h = self.live2d_py_model_h
                else:
                    self.resize_start_w = self.width()
                    self.resize_start_h = self.height()
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                if self.live2d_py_mode:
                    self.grabMouse(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.dragging = True
                self.drag_moved = False
                self.drag_start_global = event.globalPosition().toPoint()
                self.drag_start_window_pos = self.pos()
                if self.live2d_py_mode:
                    self.grabMouse(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.live2d_py_mode:
            self._set_controls_visible(True)
            self._controls_hide_timer.start(1200)

        if self.live2d_py_mode and self.resizing and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            new_model_w = max(180, self.resize_start_w + delta.x())
            new_model_h = max(220, int(round(new_model_w / self.live2d_py_aspect)))
            self._apply_live2d_py_model_size(new_model_w, new_model_h)
            event.accept()
            return

        if self.live2d_py_mode and self.dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            self._move_by_drag_delta(event.globalPosition().toPoint())
            event.accept()
            return

        if self.resizing and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            if abs(delta.x()) >= abs(delta.y() * self.aspect_ratio):
                new_w = max(self.minimumWidth(), self.resize_start_w + delta.x())
                new_h = max(self.minimumHeight(), int(round(new_w / self.aspect_ratio)))
                new_w = int(round(new_h * self.aspect_ratio))
            else:
                new_h = max(self.minimumHeight(), self.resize_start_h + delta.y())
                new_w = max(self.minimumWidth(), int(round(new_h * self.aspect_ratio)))
                new_h = int(round(new_w / self.aspect_ratio))
            self.resize(new_w, new_h)
            event.accept()
            return

        if self.dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            if (event.globalPosition().toPoint() - self.drag_start_global).manhattanLength() > 4:
                self.drag_moved = True
            self._move_by_drag_delta(event.globalPosition().toPoint())
            event.accept()
            return

        if self._in_resize_zone(event.position().toPoint()):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        moved_by_position = (self.pos() - self.drag_start_window_pos).manhattanLength() > 2
        if moved_by_position:
            self.drag_moved = True

        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.live2d_view is not None
            and not self.resizing
            and not self.drag_moved
        ):
            nx, ny = self._to_normalized_pos(event.position().toPoint())
            self.live2d_view.tap_at(nx, ny)

        self.dragging = False
        self.drag_moved = False
        self.resizing = False
        if self.live2d_py_mode and self.mouseGrabber() is self:
            self.releaseMouse()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        if self.image_label is not None:
            self.image_label.setGeometry(self.rect())
        if self.live2d_view is not None:
            self.live2d_view.setGeometry(self.rect())
        if hasattr(self, "resize_handle"):
            self._update_resize_handle_pos()
        if hasattr(self, "bubble"):
            self._update_bubble_pos()
        if self.live2d_py_mode and self.control_bar is not None:
            side_w = self.live2d_side_panel_w
            bar_h = min(self.height() - 12, 340)
            bar_x = min(
                max(0, self.width() - side_w - 6),
                self.live2d_py_model_w + self.live2d_side_gap,
            )
            bar_y = max(6, (self.height() - bar_h) // 2)
            self.control_bar.setGeometry(bar_x, bar_y, side_w, bar_h)
        if (not self.live2d_py_mode) and self.control_bar is not None:
            side_w = 96
            bar_h = min(self.height() - 12, 340)
            bar_x = max(6, self.width() - side_w - 6)
            bar_y = max(6, (self.height() - bar_h) // 2)
            self.control_bar.setGeometry(bar_x, bar_y, side_w, bar_h)
            self.control_bar.raise_()
        if self.live2d_py_mode and self.control_border is not None:
            self.control_border.setGeometry(0, 0, self.live2d_py_model_w, self.live2d_py_model_h)
            self.control_border.raise_()
        if self.live2d_py_mode and self.drag_overlay is not None:
            self.drag_overlay.setGeometry(0, 0, self.live2d_py_model_w, self.live2d_py_model_h)
            self.drag_overlay.raise_()
        if self.live2d_py_mode and self.control_bar is not None:
            self.control_bar.raise_()
        super().resizeEvent(event)

    def _update_resize_handle_pos(self):
        margin = 2
        x = self.width() - self.resize_handle.width() - margin
        y = self.height() - self.resize_handle.height() - margin
        self.resize_handle.move(max(0, x), max(0, y))

    def _update_bubble_pos(self):
        margin = 8
        x = max(margin, (self.width() - self.bubble.width()) // 2)
        y = margin + (4 if self.live2d_py_mode else 0)
        self.bubble.move(x, y)

    def show_comment_bubble(self, text: str, duration_ms: int = 7000):
        self.bubble.say(text)
        self.bubble.adjustSize()
        self._update_bubble_pos()
        self._bubble_hide_timer.start(max(500, duration_ms))

    def set_auto_scan_enabled(self, enabled: bool):
        self.auto_scan_enabled = enabled
        if self.btn_toggle_scan is not None:
            self.btn_toggle_scan.setText("暂停扫描" if enabled else "开始扫描")

    def set_tts_muted(self, muted: bool):
        self.tts_muted = muted
        if self.btn_tts is not None:
            self.btn_tts.setText("取消静音" if muted else "静音")

    def set_gaze_follow_enabled(self, enabled: bool):
        self.gaze_follow_enabled = bool(enabled)
        if self.btn_gaze is not None:
            self.btn_gaze.setText("关闭跟随" if self.gaze_follow_enabled else "开启跟随")

    def set_tutor_mode_enabled(self, enabled: bool):
        self.tutor_mode_enabled = bool(enabled)
        if self.btn_tutor is not None:
            self.btn_tutor.setText("教学开" if self.tutor_mode_enabled else "教学关")

    def set_scan_busy(self, busy: bool):
        self.scan_busy = bool(busy)
        if self.live2d_view is not None:
            self.live2d_view.set_scan_busy(self.scan_busy)

    def _move_to_bottom_right(self):
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        margin = 20
        if self.live2d_py_mode:
            host_w, _host_h = self._calc_live2d_py_ui_size()
            x = available.right() - host_w - margin
            y = available.bottom() - self.live2d_py_model_h - margin
        else:
            x = available.right() - self.width() - margin
            y = available.bottom() - self.height() - margin
        self.move(max(available.left(), x), max(available.top(), y))

    def get_live2d_py_target_geometry(self) -> tuple[int, int, int, int]:
        x = self.x()
        y = self.y()
        return x, y, self.live2d_py_model_w, self.live2d_py_model_h

    def is_live2d_py_interacting(self) -> bool:
        return bool(self.live2d_py_mode and (self.dragging or self.resizing))

    def should_anchor_model_to_ui(self) -> bool:
        if not self.live2d_py_mode:
            return False
        return time.monotonic() < self._ui_anchor_until

    def sync_from_live2d_py_rect(self, x: int, y: int, w: int, h: int) -> None:
        if not self.live2d_py_mode:
            return
        self._apply_live2d_py_model_size(w, h)
        self.move(int(x), int(y))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def enterEvent(self, event):
        if self.live2d_py_mode:
            self._set_controls_visible(True)
            self._controls_hide_timer.start(1200)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.live2d_py_mode:
            self._controls_hide_timer.start(600)
        super().leaveEvent(event)

    def contextMenuEvent(self, event):
        event.accept()
