from __future__ import annotations

import io
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from PyQt6.QtCore import QBuffer, QEvent, QIODevice, QTimer, Qt
from PyQt6.QtGui import QCloseEvent, QGuiApplication, QImage
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

from desktop_pet.llm.dialog_manager import DialogManager
from desktop_pet.vision.ocr import extract_text


class ChatPanel(QWidget):
    def __init__(self, dialog_manager: DialogManager, on_pet_reply: Optional[Callable[[str], None]] = None):
        super().__init__()
        self.dialog_manager = dialog_manager
        self.on_pet_reply = on_pet_reply
        self.disable_auto_archive_on_close = False
        self.overlay_mode = False
        self._reply_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat-reply")
        self._pending_future: Future | None = None
        self._pending_task_type: str | None = None
        self._uploaded_image_path: str = ""
        self._uploaded_image_ocr_text: str = ""
        self.setWindowTitle("桌宠聊天")
        self.resize(420, 520)

        self.history = QTextEdit(self)
        self.history.setReadOnly(True)

        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText("输入想和桌宠说的话...")
        self.input_line.returnPressed.connect(self.on_send)

        self.send_btn = QPushButton("发送", self)
        self.send_btn.clicked.connect(self.on_send)

        self.upload_btn = QPushButton("上传图片", self)
        self.upload_btn.clicked.connect(self.on_upload_image)

        self.paste_btn = QPushButton("粘贴图片", self)
        self.paste_btn.clicked.connect(self.on_paste_image)

        self.status_label = QLabel("", self)
        self.status_label.setStyleSheet("color: #999;")

        self.ocr_preview = QTextEdit(self)
        self.ocr_preview.setReadOnly(True)
        self.ocr_preview.setPlaceholderText("OCR预览会显示在这里...")
        self.ocr_preview.setMaximumHeight(120)

        self.clear_ocr_btn = QPushButton("清空OCR", self)
        self.clear_ocr_btn.clicked.connect(self._clear_ocr_context)

        self.comment_btn = QPushButton("基于屏幕评论", self)
        self.comment_btn.clicked.connect(self.request_screen_comment)

        self.view_memory_btn = QPushButton("查看日记", self)
        self.view_memory_btn.clicked.connect(self.on_view_diary)

        self.end_chat_btn = QPushButton("结束聊天", self)
        self.end_chat_btn.clicked.connect(self.on_end_chat)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_line, 1)
        input_row.addWidget(self.paste_btn)
        input_row.addWidget(self.upload_btn)
        input_row.addWidget(self.send_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("对话记录"))
        layout.addWidget(self.history, 1)
        layout.addWidget(QLabel("OCR预览"))
        layout.addWidget(self.ocr_preview)
        layout.addLayout(input_row)
        layout.addWidget(self.clear_ocr_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.comment_btn)
        layout.addWidget(self.view_memory_btn)
        layout.addWidget(self.end_chat_btn)

        self.reply_poll_timer = QTimer(self)
        self.reply_poll_timer.setInterval(100)
        self.reply_poll_timer.timeout.connect(self._poll_pending_reply)
        self.input_line.installEventFilter(self)

    def enable_live2d_overlay_mode(self):
        self.overlay_mode = True
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.resize(460, 700)
        self.setStyleSheet(
            "QWidget {"
            " background: rgba(15, 18, 24, 205);"
            " color: #e8eef8;"
            " border: 1px solid rgba(255,255,255,38);"
            " border-radius: 10px;"
            "}"
            "QTextEdit, QLineEdit {"
            " background: rgba(10, 12, 16, 185);"
            " color: #e8eef8;"
            " border: 1px solid rgba(255,255,255,28);"
            " border-radius: 8px;"
            " padding: 6px;"
            "}"
            "QPushButton {"
            " background: rgba(67, 150, 255, 150);"
            " color: white;"
            " border: none;"
            " border-radius: 8px;"
            " padding: 6px 8px;"
            "}"
            "QPushButton:hover { background: rgba(67, 150, 255, 195); }"
        )

    def dock_to_rect(self, x: int, y: int, w: int, h: int):
        if not self.overlay_mode:
            return

        gap = 10
        target_w = max(300, min(460, int(round(w * 0.70))))
        target_h = max(300, min(620, int(round(h * 0.62))))
        if self.width() != target_w or self.height() != target_h:
            self.resize(target_w, target_h)
        desired_x = x + w + gap
        desired_y = y + min(40, max(0, h // 6))
        self.move(desired_x, desired_y)

    def append_message(self, role: str, text: str):
        self.history.append(f"{role}: {text}")

    def on_send(self):
        user_text = self.input_line.text().strip()
        if not user_text:
            return
        if self._pending_future is not None:
            self.append_message("系统", "妹妹还在思考上一条，稍等一下哦。")
            return
        self.input_line.clear()
        self.append_message("你", user_text)
        ocr_context = ""
        if self._uploaded_image_ocr_text:
            ocr_context = (
                "用户上传了一张图片，OCR识别内容如下：\n"
                f"{self._uploaded_image_ocr_text[:1200]}"
            )
        self._start_async_reply(user_text, ocr_context)

    def _update_ocr_preview(self, text: str):
        preview = text.strip()
        if not preview:
            self.ocr_preview.clear()
            return
        if len(preview) > 1000:
            preview = preview[:1000] + "\n...（内容较长，已截断预览）"
        self.ocr_preview.setPlainText(preview)

    def _clear_ocr_context(self):
        self._uploaded_image_path = ""
        self._uploaded_image_ocr_text = ""
        self._update_ocr_preview("")
        self.append_message("系统", "已清空OCR上下文。")

    def request_screen_comment(self):
        demo_context = "检测到你正在编辑代码，界面较简洁。"
        prompt = f"请你根据这段屏幕摘要做一句可爱点评：{demo_context}"
        if self._pending_future is not None:
            self.append_message("系统", "妹妹还在思考上一条，稍等一下哦。")
            return
        self._start_async_reply(prompt)

    def on_upload_image(self):
        if self._pending_future is not None:
            self.append_message("系统", "当前有任务在进行，请稍后再上传。")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not file_path:
            return

        self._uploaded_image_path = file_path
        self._start_async_ocr(file_path)

    def on_paste_image(self):
        if self._pending_future is not None:
            self.append_message("系统", "当前有任务在进行，请稍后再粘贴。")
            return

        clipboard = QGuiApplication.clipboard()
        mime_data = clipboard.mimeData()
        if mime_data is None or not mime_data.hasImage():
            self.append_message("系统", "剪贴板里没有图片，可直接 Ctrl+V 粘贴文字。")
            return

        image = clipboard.image()
        if image.isNull():
            self.append_message("系统", "读取剪贴板图片失败。")
            return

        try:
            pil_image = self._qimage_to_pil(image)
        except Exception as exc:
            self.append_message("系统", f"剪贴板图片转换失败：{exc}")
            return

        self._uploaded_image_path = "[clipboard]"
        self._start_async_ocr_image(pil_image)

    def _qimage_to_pil(self, image: QImage) -> Image.Image:
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        try:
            if not image.save(buffer, "PNG"):
                raise RuntimeError("QImage保存失败")
            data = bytes(buffer.data())
        finally:
            buffer.close()

        with Image.open(io.BytesIO(data)) as decoded:
            return decoded.convert("RGB")

    def _ocr_image_file(self, file_path: str) -> str:
        with Image.open(file_path) as image:
            ocr_text = extract_text(image)
        return ocr_text.strip()

    def _ocr_pil_image(self, image: Image.Image) -> str:
        ocr_text = extract_text(image)
        return ocr_text.strip()

    def _start_async_reply(self, prompt: str, ocr_context: str = ""):
        self.status_label.setText("妹妹思考中...")
        self.send_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.comment_btn.setEnabled(False)
        self._pending_task_type = "reply"
        self._pending_future = self._reply_executor.submit(
            self.dialog_manager.reply,
            prompt,
            ocr_context,
        )
        self.reply_poll_timer.start()

    def _start_async_ocr(self, file_path: str):
        self.status_label.setText("图片识别中...")
        self.send_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.comment_btn.setEnabled(False)
        self._pending_task_type = "ocr"
        self._pending_future = self._reply_executor.submit(self._ocr_image_file, file_path)
        self.reply_poll_timer.start()

    def _start_async_ocr_image(self, image: Image.Image):
        self.status_label.setText("图片识别中...")
        self.send_btn.setEnabled(False)
        self.paste_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.comment_btn.setEnabled(False)
        self._pending_task_type = "ocr"
        self._pending_future = self._reply_executor.submit(self._ocr_pil_image, image)
        self.reply_poll_timer.start()

    def _poll_pending_reply(self):
        future = self._pending_future
        if future is None:
            self.reply_poll_timer.stop()
            return
        if not future.done():
            return

        task_type = self._pending_task_type
        self._pending_future = None
        self._pending_task_type = None
        self.reply_poll_timer.stop()
        self.status_label.setText("")
        self.send_btn.setEnabled(True)
        self.paste_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.comment_btn.setEnabled(True)

        try:
            result = future.result()
        except Exception as exc:
            result = f"[错误] {exc}"

        if task_type == "ocr":
            text = str(result).strip()
            if text.startswith("[错误]"):
                self._uploaded_image_ocr_text = ""
                self._update_ocr_preview("")
                self.append_message("系统", f"图片识别失败：{text}")
            elif not text:
                self._uploaded_image_ocr_text = ""
                self._update_ocr_preview("")
                self.append_message("系统", "图片识别完成，但未识别到文字。")
            else:
                self._uploaded_image_ocr_text = text
                self._update_ocr_preview(text)
                self.append_message("系统", "图片识别完成，发送消息时会结合图片内容回复。")
            return

        reply = str(result)

        self.append_message("桌宠", reply)
        if self.on_pet_reply is not None:
            self.on_pet_reply(reply)

    def start_new_chat(self):
        memory_count = self.dialog_manager.start_new_chat()
        self.history.clear()
        if memory_count > 0:
            self.append_message("系统", f"已开启新聊天，已加载{memory_count}条长期记忆。")
        else:
            self.append_message("系统", "已开启新聊天。")

    def on_end_chat(self):
        summary = self.dialog_manager.end_current_chat()
        if summary:
            self.append_message("系统", f"已归档本次聊天要点：{summary}")
        else:
            self.append_message("系统", "本次聊天无可归档内容。")
        self.hide()

    def set_disable_auto_archive_on_close(self, disabled: bool):
        self.disable_auto_archive_on_close = disabled

    def closeEvent(self, event: QCloseEvent):
        self.reply_poll_timer.stop()
        if not self.disable_auto_archive_on_close:
            self.dialog_manager.end_current_chat()
        event.accept()

    def on_view_diary(self):
        entries = self.dialog_manager.list_long_memory(limit=10)
        if not entries:
            self.append_message("系统", "暂无长期记忆日记。")
            return

        self.append_message("系统", "以下为最近长期记忆日记：")
        for idx, item in enumerate(reversed(entries), start=1):
            ts = item.get("timestamp", "")
            summary = item.get("summary", "")
            if ts:
                self.append_message("日记", f"{idx}. [{ts}] {summary}")
            else:
                self.append_message("日记", f"{idx}. {summary}")

    def show_and_focus(self):
        self.show()
        self.raise_()
        self.activateWindow()
        self.input_line.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def eventFilter(self, obj, event):
        if obj is self.input_line and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_V and bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                clipboard = QGuiApplication.clipboard()
                mime_data = clipboard.mimeData()
                if mime_data is not None and mime_data.hasImage():
                    self.on_paste_image()
                    return True
        return super().eventFilter(obj, event)
