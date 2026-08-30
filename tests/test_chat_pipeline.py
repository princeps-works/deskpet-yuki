import threading

from PIL import Image

from desktop_pet.ui.chat_panel import ChatPanel


class _FakeDialog:
    def __init__(self, web_started: threading.Event) -> None:
        self.web_started = web_started
        self.last_reply_kwargs = {}
        self.last_context = ""

    def prepare_web_search(self, _prompt: str):
        self.web_started.set()
        return "联网参考", "confidence=0.90"

    def reply(self, _prompt: str, context: str, **kwargs):
        self.last_context = context
        self.last_reply_kwargs = kwargs
        return "完成"


class _FakePanel:
    pass


def test_screen_image_and_web_context_are_prepared_in_parallel():
    web_started = threading.Event()
    dialog = _FakeDialog(web_started)
    panel = _FakePanel()
    panel.dialog_manager = dialog
    panel.web_search_enabled = True
    panel.screen_context_max_chars = 1600

    def _screen_provider(_prompt: str) -> str:
        if not web_started.wait(timeout=1.0):
            raise RuntimeError("web prefetch did not start while screen context was running")
        return "屏幕视觉摘要"

    panel.screen_context_provider = _screen_provider
    panel.uploaded_image_context_provider = lambda _image, _prompt, _ocr: "上传图片OCR与Vision摘要"
    uploaded = Image.new("RGB", (32, 32), "white")

    result = ChatPanel._reply_with_optional_screen(
        panel,
        "比较图片与屏幕",
        "",
        True,
        uploaded,
        "图片文字",
    )

    assert result == ("完成", "", True, True, "")
    assert "屏幕视觉摘要" in dialog.last_context
    assert "上传图片OCR与Vision摘要" in dialog.last_context
    assert dialog.last_reply_kwargs["prepared_web_search"] == ("联网参考", "confidence=0.90")
    assert dialog.last_reply_kwargs["extra_context_kind"] == "visual"
