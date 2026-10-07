from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from desktop_pet.config.settings import Settings


class VnSidebar(QWidget):
    """Side panel for visual-novel tuning and plot memory, shown on hover.

    Every control emits a signal instead of mutating settings, because
    ``Settings`` is a frozen snapshot loaded once at startup: runtime changes
    have to be pushed into ``state``/resolver objects by the owner.
    """

    capture_mode_changed = pyqtSignal(str)
    vision_quota_changed = pyqtSignal(int)
    vision_change_threshold_changed = pyqtSignal(float)
    text_ratio_changed = pyqtSignal(float)
    window_margin_changed = pyqtSignal(int)
    pin_target_requested = pyqtSignal()
    unpin_target_requested = pyqtSignal()
    select_vision_region_requested = pyqtSignal()
    reset_vision_region_requested = pyqtSignal()
    select_ocr_region_requested = pyqtSignal()
    reset_ocr_region_requested = pyqtSignal()
    probe_ocr_requested = pyqtSignal()
    select_story_requested = pyqtSignal()
    add_memory_requested = pyqtSignal(str)
    reload_memory_requested = pyqtSignal()
    reset_story_requested = pyqtSignal()
    view_memory_requested = pyqtSignal()
    select_scan_region_requested = pyqtSignal()
    clear_scan_region_requested = pyqtSignal()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("视觉小说侧栏")
        self.setFixedWidth(268)
        # Needed so leaveEvent fires and the panel can close without waiting for
        # the owner's hover poll.
        self.setMouseTracking(True)
        self.on_hover_changed = None
        self._build(settings)

    # -- construction -------------------------------------------------------

    def _build(self, settings: Settings) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        panel = QFrame(self)
        panel.setObjectName("vnPanel")
        panel.setStyleSheet(
            "#vnPanel {"
            " background: rgba(16, 20, 28, 226);"
            " border: 1px solid rgba(255,255,255,34);"
            " border-radius: 10px;"
            "}"
            "QLabel { color: #dce5f2; }"
            "QLabel#vnTitle { color: #9dc4ff; font-weight: bold; }"
            "QLabel#vnHint { color: #8b97a8; }"
            "QPushButton {"
            " background: rgba(58, 132, 255, 170);"
            " color: white; border: none; border-radius: 6px; padding: 4px 8px;"
            "}"
            "QPushButton:hover { background: rgba(58, 132, 255, 215); }"
            "QComboBox, QSpinBox, QLineEdit {"
            " background: rgba(10, 12, 16, 200);"
            " color: #e8eef8;"
            " border: 1px solid rgba(255,255,255,30);"
            " border-radius: 5px;"
            " padding: 2px 4px;"
            "}"
        )
        self.pages = QStackedWidget(self)
        outer.addWidget(self.pages)
        home = QWidget(self)
        home_layout = QVBoxLayout(home)
        home_layout.setContentsMargins(10, 10, 10, 10)
        self.scan_button = QPushButton("扫描设置", home)
        self.memory_button = QPushButton("记忆设置", home)
        self.story_button = QPushButton("剧情缓存", home)
        for button in (self.scan_button, self.memory_button, self.story_button):
            home_layout.addWidget(button)
        home.setStyleSheet(panel.styleSheet().replace("#vnPanel", "QWidget"))
        self.pages.addWidget(home)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        self.pages.addWidget(scroll)
        self.scan_button.clicked.connect(lambda: self._show_page(1))
        self.memory_button.clicked.connect(lambda: self._show_page(2))
        self.story_button.clicked.connect(self.select_story_requested.emit)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("视觉小说", panel)
        title.setObjectName("vnTitle")
        layout.addWidget(title)

        back = QPushButton("返回", panel)
        back.clicked.connect(lambda: self._show_page(0))
        layout.addWidget(back)

        status_label = QLabel("画面理解", panel)
        status_label.setObjectName("vnTitle")
        layout.addWidget(status_label)
        self.vision_status = QLabel("-", panel)
        self.vision_status.setObjectName("vnHint")
        self.vision_status.setWordWrap(True)
        layout.addWidget(self.vision_status)

        ocr_label = QLabel("文本识别（OCR）", panel)
        ocr_label.setObjectName("vnTitle")
        layout.addWidget(ocr_label)

        self.target_status = QLabel("-", panel)
        self.target_status.setObjectName("vnHint")
        self.target_status.setWordWrap(True)
        layout.addWidget(self.target_status)

        target_row = QHBoxLayout()
        pin_btn = QPushButton("锁定此窗口", panel)
        pin_btn.setToolTip(
            "把截图固定到当前最上层的窗口上。默认每次都会重新挑最上层窗口，"
            "浏览器或工具窗挡住游戏时就会抓错。"
        )
        pin_btn.clicked.connect(self.pin_target_requested.emit)
        unpin_btn = QPushButton("自动", panel)
        unpin_btn.setToolTip("取消锁定，改回自动挑最上层窗口。")
        unpin_btn.clicked.connect(self.unpin_target_requested.emit)
        target_row.addWidget(pin_btn)
        target_row.addWidget(unpin_btn)
        layout.addLayout(target_row)

        self.ocr_status = QLabel("-", panel)
        self.ocr_status.setObjectName("vnHint")
        self.ocr_status.setWordWrap(True)
        layout.addWidget(self.ocr_status)

        ocr_row = QHBoxLayout()
        ocr_region_btn = QPushButton("框选文本区", panel)
        ocr_region_btn.setToolTip(
            "框住游戏对话框的范围。设好之后 OCR 只识别这块区域，识别率会明显提高。"
        )
        ocr_region_btn.clicked.connect(self.select_ocr_region_requested.emit)
        ocr_reset_btn = QPushButton("自动", panel)
        ocr_reset_btn.setToolTip("清除文本区，改回自动猜测（取窗口下方一定比例）。")
        ocr_reset_btn.clicked.connect(self.reset_ocr_region_requested.emit)
        ocr_row.addWidget(ocr_region_btn)
        ocr_row.addWidget(ocr_reset_btn)
        layout.addLayout(ocr_row)

        probe_btn = QPushButton("试读一次文本", panel)
        probe_btn.setToolTip("立刻按当前文本区识别一次，并把结果打印到控制台，方便确认框选对不对。")
        probe_btn.clicked.connect(self.probe_ocr_requested.emit)
        layout.addWidget(probe_btn)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.capture_mode = QComboBox(panel)
        self.capture_mode.addItem("跟随游戏窗口", "window")
        self.capture_mode.addItem("手动框选画面", "manual")
        index = self.capture_mode.findData(str(settings.vision_capture_mode))
        self.capture_mode.setCurrentIndex(index if index >= 0 else 0)
        self.capture_mode.setToolTip(
            "跟随游戏窗口：直接读取游戏窗口自身的内容（PrintWindow），"
            "不会被桌宠或其它窗口遮挡，也不会把桌宠拍进去。首选。\n"
            "手动框选画面：在屏幕上框出游戏画面范围。用于独占全屏、"
            "PrintWindow 拿不到内容的游戏。"
        )
        self.capture_mode.currentIndexChanged.connect(self._on_capture_mode)
        form.addRow(QLabel("截图来源", panel), self.capture_mode)

        self.vision_quota = QSpinBox(panel)
        self.vision_quota.setRange(1, 60)
        self.vision_quota.setSuffix(" 张/分")
        self.vision_quota.setValue(int(round(float(settings.vision_shots_per_minute))))
        self.vision_quota.setToolTip(
            "画面理解的取样配额。按「张/分钟」补充：3 表示平均每 20 秒一张，"
            "但画面出现大变化时可以立刻用掉已有配额，不需要等时钟。"
        )
        self.vision_quota.valueChanged.connect(self.vision_quota_changed.emit)
        form.addRow(QLabel("画面理解配额", panel), self.vision_quota)

        self.change_threshold = QComboBox(panel)
        for label, value in (
            ("灵敏 0.05", 0.05),
            ("标准 0.10", 0.10),
            ("保守 0.20", 0.20),
            ("只在剧变 0.35", 0.35),
        ):
            self.change_threshold.addItem(label, value)
        current = min(
            range(self.change_threshold.count()),
            key=lambda i: abs(float(self.change_threshold.itemData(i)) - float(settings.vision_change_threshold)),
        )
        self.change_threshold.setCurrentIndex(current)
        self.change_threshold.currentIndexChanged.connect(self._on_threshold)
        form.addRow(QLabel("触发敏感度", panel), self.change_threshold)

        self.text_ratio = QSpinBox(panel)
        self.text_ratio.setRange(20, 95)
        self.text_ratio.setSuffix(" %")
        self.text_ratio.setToolTip("没有框选文本区时，自动从窗口的这个比例往下取作为文本区。")
        self.text_ratio.setValue(int(round(float(settings.visual_novel_text_ratio) * 100)))
        self.text_ratio.valueChanged.connect(self._on_text_ratio)
        form.addRow(QLabel("自动文本区起点", panel), self.text_ratio)

        self.window_margin = QSpinBox(panel)
        self.window_margin.setRange(-200, 400)
        self.window_margin.setSuffix(" px")
        self.window_margin.setValue(int(settings.vision_window_margin_px))
        self.window_margin.valueChanged.connect(self.window_margin_changed.emit)
        form.addRow(QLabel("窗口外扩", panel), self.window_margin)

        layout.addLayout(form)

        hint = QLabel(
            "鼠标移到视觉小说按钮上显示菜单；"
            "鼠标离开工具栏和面板后会自动收起。",
            panel,
        )
        hint.setObjectName("vnHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        vision_region_row = QHBoxLayout()
        vision_region_btn = QPushButton("框选画面区域", panel)
        vision_region_btn.setToolTip(
            "配合「手动框选画面」使用：框出游戏画面范围，视觉和 OCR 都在这块区域内工作。"
        )
        vision_region_btn.clicked.connect(self.select_vision_region_requested.emit)
        vision_region_reset = QPushButton("自动", panel)
        vision_region_reset.setToolTip("清除画面区域。")
        vision_region_reset.clicked.connect(self.reset_vision_region_requested.emit)
        vision_region_row.addWidget(vision_region_btn)
        vision_region_row.addWidget(vision_region_reset)
        layout.addLayout(vision_region_row)

        scan_region_row = QHBoxLayout()
        select_scan = QPushButton("选区域", panel)
        select_scan.clicked.connect(self.select_scan_region_requested.emit)
        clear_scan = QPushButton("清区域", panel)
        clear_scan.clicked.connect(self.clear_scan_region_requested.emit)
        scan_region_row.addWidget(select_scan)
        scan_region_row.addWidget(clear_scan)
        layout.addLayout(scan_region_row)

        memory_panel = QFrame(self)
        memory_panel.setObjectName("vnPanel")
        memory_panel.setStyleSheet(panel.styleSheet())
        self.pages.addWidget(memory_panel)
        panel = memory_panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        back = QPushButton("返回", panel)
        back.clicked.connect(lambda: self._show_page(0))
        layout.addWidget(back)
        story_row = QHBoxLayout()
        self.story_label = QLabel("剧情缓存：-", panel)
        self.story_label.setWordWrap(True)
        story_row.addWidget(self.story_label, 1)
        layout.addLayout(story_row)

        memory_title = QLabel("剧情记忆", panel)
        memory_title.setObjectName("vnTitle")
        layout.addWidget(memory_title)
        self.memory_status = QLabel("-", panel)
        self.memory_status.setObjectName("vnHint")
        self.memory_status.setWordWrap(True)
        layout.addWidget(self.memory_status)

        self.memory_input = QLineEdit(panel)
        self.memory_input.setPlaceholderText("手动补充一条剧情记忆…")
        self.memory_input.returnPressed.connect(self._on_add_memory)
        layout.addWidget(self.memory_input)

        memory_row = QHBoxLayout()
        add_btn = QPushButton("新增记忆", panel)
        add_btn.clicked.connect(self._on_add_memory)
        reload_btn = QPushButton("重新载入", panel)
        reload_btn.clicked.connect(self.reload_memory_requested.emit)
        memory_row.addWidget(add_btn)
        memory_row.addWidget(reload_btn)
        layout.addLayout(memory_row)

        view_memory = QPushButton("查看详细记忆文件", panel)
        view_memory.clicked.connect(self.view_memory_requested.emit)
        layout.addWidget(view_memory)
        reset_story = QPushButton("清空记忆缓存", panel)
        reset_story.clicked.connect(self.reset_story_requested.emit)
        layout.addWidget(reset_story)
        self._show_page(0)

    # -- slots --------------------------------------------------------------

    def _show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        height = 600 if index == 1 else (340 if index == 2 else 132)
        screen = self.screen()
        if screen is not None:
            height = min(height, screen.availableGeometry().height() - 20)
        self.setFixedHeight(height)
        if self.isVisible():
            self.show_near(*self._anchor)

    def hideEvent(self, event):
        self._show_page(0)
        super().hideEvent(event)

    def _on_capture_mode(self, _index: int) -> None:
        self.capture_mode_changed.emit(str(self.capture_mode.currentData()))

    def _on_threshold(self, _index: int) -> None:
        self.vision_change_threshold_changed.emit(float(self.change_threshold.currentData()))

    def _on_text_ratio(self, value: int) -> None:
        self.text_ratio_changed.emit(float(value) / 100.0)

    def _on_add_memory(self) -> None:
        text = self.memory_input.text().strip()
        if not text:
            return
        self.memory_input.clear()
        self.add_memory_requested.emit(text)

    # -- state display ------------------------------------------------------

    def set_story_name(self, name: str) -> None:
        self.story_label.setText(f"剧情缓存：{name or '-'}")

    def set_vision_status(self, text: str) -> None:
        self.vision_status.setText(text or "-")

    def set_ocr_status(self, text: str) -> None:
        self.ocr_status.setText(text or "-")

    def set_target_status(self, text: str) -> None:
        self.target_status.setText(text or "-")

    def set_memory_status(self, text: str) -> None:
        self.memory_status.setText(text or "-")

    def show_near(self, anchor_x: int, anchor_y: int, anchor_w: int, anchor_h: int) -> None:
        """Place the panel to the right of the pet window (or left if no room)."""
        self._anchor = (anchor_x, anchor_y, anchor_w, anchor_h)
        width = self.width()
        height = self.height()
        screen = QGuiApplication.screenAt(QPoint(anchor_x, anchor_y)) or self.screen()
        available = screen.availableGeometry() if screen is not None else None
        x = anchor_x + anchor_w + 6
        if available is not None and x + width > available.right():
            x = max(available.left(), anchor_x - width - 6)
        y = anchor_y
        if available is not None:
            y = max(available.top(), min(y, available.bottom() - height))
        self.move(int(x), int(y))
        if not self.isVisible():
            self.show()

    def enterEvent(self, event):
        if callable(self.on_hover_changed):
            self.on_hover_changed(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        if callable(self.on_hover_changed):
            self.on_hover_changed(False)
        super().leaveEvent(event)
