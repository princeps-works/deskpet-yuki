from desktop_pet.vision.ocr import OCRLine, OCRResult
from desktop_pet.vision.router import decide_vision_route


def _ocr_result(text: str, confidence: float) -> OCRResult:
    line = OCRLine(text=text, confidence=confidence, box=(0.1, 0.2, 0.9, 0.3))
    return OCRResult(
        lines=(line,),
        text=text,
        average_confidence=confidence,
        text_coverage=0.08,
        text_hash="hash" if text else "",
    )


def test_text_rich_screen_uses_ocr_only():
    result = _ocr_result("正在编辑代码并查看终端输出。" * 12, 0.92)
    decision = decide_vision_route(result)
    assert decision.route == "ocr_only"


def test_partial_text_screen_uses_hybrid_route():
    result = _ocr_result("正在查看项目预览和图表数据" * 3, 0.90)
    decision = decide_vision_route(result)
    assert decision.route == "vision+ocr"


def test_sparse_text_screen_uses_vision_only():
    result = _ocr_result("播放", 0.95)
    decision = decide_vision_route(result)
    assert decision.route == "vision_only"


def test_prompt_text_keeps_coarse_position_and_limit():
    result = _ocr_result("TypeError in main.py line 128", 0.95)
    prompt_text = result.to_prompt_text(max_chars=80)
    assert prompt_text.startswith("[顶部]")
    assert len(prompt_text) <= 80
