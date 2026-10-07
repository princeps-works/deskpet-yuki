from types import SimpleNamespace

from PIL import Image

from desktop_pet.main import _build_ocr_first_context
from desktop_pet.vision.ocr import OCRLine, OCRResult, filter_visual_novel_ocr_result


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


def _multi_line_ocr(lines, text_hash="frame") -> OCRResult:
    text = "\n".join(line.text for line in lines)
    return OCRResult(
        lines=tuple(lines),
        text=text,
        average_confidence=sum(line.confidence for line in lines) / len(lines),
        text_coverage=0.1,
        text_hash=text_hash,
    )


def test_vn_line_filter_removes_obvious_artwork_noise_without_delaying_dialogue():
    result = _multi_line_ocr(
        [
            OCRLine("郊区线的电车正以稳定的速度行驶着。", 0.99, (0.08, 0.62, 0.85, 0.72)),
            OCRLine("烨 厂", 0.58, (0.72, 0.18, 0.81, 0.24)),
            OCRLine("XIXI 夕堂書店 不", 0.81, (0.55, 0.12, 0.88, 0.20)),
        ]
    )
    filtered, note = filter_visual_novel_ocr_result(result, {})
    assert filtered.text == "郊区线的电车正以稳定的速度行驶着。"
    assert "short_weak=1" in note
    assert "mixed_label=1" in note


def test_vn_line_filter_keeps_a_high_confidence_short_character_name():
    result = _multi_line_ocr([OCRLine("萤", 0.93, (0.08, 0.48, 0.16, 0.56))])
    filtered, note = filter_visual_novel_ocr_result(result, {})
    assert filtered.text == "萤"
    assert not note


def test_vn_line_filter_keeps_a_short_dialogue_without_waiting_for_more_frames():
    result = _multi_line_ocr([OCRLine("好吧", 0.68, (0.08, 0.62, 0.18, 0.70))])
    filtered, note = filter_visual_novel_ocr_result(result, {})
    assert filtered.text == "好吧"
    assert not note


def test_vn_line_filter_suppresses_static_text_outside_learned_dialogue_lane():
    tracking = {}
    bootstrap = _multi_line_ocr(
        [OCRLine("电车正以稳定的速度向前行驶着。", 0.98, (0.08, 0.62, 0.86, 0.72))],
        "bootstrap",
    )
    filter_visual_novel_ocr_result(bootstrap, tracking)

    background = OCRLine("这是背景中始终没有变化的招牌文字", 0.94, (0.55, 0.10, 0.92, 0.18))
    for index in range(2):
        frame = _multi_line_ocr(
            [
                OCRLine(f"这是第{index + 1}句正常的剧情台词。", 0.97, (0.08, 0.62, 0.86, 0.72)),
                background,
            ],
            f"changed-{index}",
        )
        filtered, _note = filter_visual_novel_ocr_result(frame, tracking)
        assert background.text in filtered.text

    third = _multi_line_ocr(
        [
            OCRLine("这是第三句正常的剧情台词。", 0.97, (0.08, 0.62, 0.86, 0.72)),
            background,
        ],
        "changed-3",
    )
    filtered, note = filter_visual_novel_ocr_result(third, tracking)
    assert "第三句" in filtered.text
    assert background.text not in filtered.text
    assert "stable_background=1" in note
