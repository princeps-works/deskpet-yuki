from types import SimpleNamespace

from PIL import Image

from desktop_pet.main import _build_ocr_first_context
from desktop_pet.vision.ocr import OCRLine, OCRResult


class _FakeVisionClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def describe_image(self, image, user_text: str, system_prompt: str) -> str:
        self.prompts.append(user_text)
        return "界面中央有预览区域，右侧按钮处于选中状态。"


def _settings():
    return SimpleNamespace(
        enable_multimodal_vision=True,
        enable_mm_screen_comment=True,
        enable_mm_compat_mode=True,
        enable_ocr_first_routing=True,
        ocr_only_min_chars=120,
        ocr_only_min_confidence=0.75,
        ocr_hybrid_min_chars=30,
        ocr_context_max_chars=1200,
        mm_timeout_sec=2.0,
        mm_image_max_edge=640,
        mm_failure_threshold=3,
        mm_cooldown_sec=60,
        mm_auto_min_interval_sec=300,
    )


def _ocr(text: str, confidence: float = 0.95) -> OCRResult:
    return OCRResult(
        lines=(OCRLine(text=text, confidence=confidence, box=(0.1, 0.1, 0.9, 0.2)),),
        text=text,
        average_confidence=confidence,
        text_coverage=0.08,
        text_hash="stable-hash",
    )


def test_ocr_only_route_does_not_call_vision():
    client = _FakeVisionClient()
    result = _build_ocr_first_context(
        client,
        Image.new("RGB", (800, 600), "white"),
        _ocr("代码编辑器中的普通文本内容。" * 15),
        _settings(),
        {},
        respect_vision_interval=True,
    )
    assert result["vision_route"] == "ocr_only"
    assert result["mode"] == "ocr"
    assert not client.prompts


def test_hybrid_route_sends_compact_ocr_to_vision():
    client = _FakeVisionClient()
    ocr_text = "项目预览 图表数据 当前选中按钮 " * 3
    result = _build_ocr_first_context(
        client,
        Image.new("RGB", (800, 600), "white"),
        _ocr(ocr_text),
        _settings(),
        {},
        respect_vision_interval=True,
        focus_query="右侧按钮为什么不能点击？",
    )
    assert result["vision_route"] == "vision+ocr"
    assert result["mode"] == "vision+ocr"
    assert client.prompts
    assert "本地OCR" in client.prompts[0]
    assert "项目预览" in client.prompts[0]
    assert "右侧按钮为什么不能点击" in client.prompts[0]


def test_manual_chat_can_enable_vision_independently_from_auto_comment():
    client = _FakeVisionClient()
    settings = _settings()
    settings.enable_mm_screen_comment = False
    result = _build_ocr_first_context(
        client,
        Image.new("RGB", (800, 600), "white"),
        _ocr("播放"),
        settings,
        {},
        respect_vision_interval=False,
        vision_enabled=True,
    )
    assert result["mode"] == "vision+ocr"
    assert client.prompts
