from __future__ import annotations

from dataclasses import dataclass

from desktop_pet.vision.ocr import OCRResult


VISUAL_CUE_KEYWORDS = (
    "图片",
    "图像",
    "图表",
    "视频",
    "人物",
    "地图",
    "画布",
    "预览",
    "颜色",
    "弹窗",
    "选中",
    "禁用",
    "按钮状态",
    "拖拽",
    "布局",
)


@dataclass(frozen=True)
class VisionRouteDecision:
    route: str
    reason: str


def decide_vision_route(
    ocr_result: OCRResult,
    *,
    ocr_only_min_chars: int = 120,
    ocr_only_min_confidence: float = 0.75,
    hybrid_min_chars: int = 30,
) -> VisionRouteDecision:
    """Choose the cheapest route that still preserves useful screen context."""
    chars = ocr_result.char_count
    confidence = float(ocr_result.average_confidence)
    text = ocr_result.text.casefold()
    has_visual_cue = any(keyword.casefold() in text for keyword in VISUAL_CUE_KEYWORDS)

    if chars <= 0:
        return VisionRouteDecision("vision_only", "ocr_empty")

    if (
        chars >= max(1, int(ocr_only_min_chars))
        and confidence >= max(0.0, min(1.0, float(ocr_only_min_confidence)))
        and not has_visual_cue
    ):
        return VisionRouteDecision("ocr_only", "text_rich_high_confidence")

    if chars >= max(1, int(hybrid_min_chars)):
        reason = "visual_cue" if has_visual_cue else "partial_text"
        return VisionRouteDecision("vision+ocr", reason)

    return VisionRouteDecision("vision_only", "ocr_sparse")
