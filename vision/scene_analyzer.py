from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneSummary:
    summary: str
    confidence: float
    should_comment: bool


def analyze_scene(text: str) -> SceneSummary:
    cleaned = " ".join(text.split())
    if not cleaned:
        return SceneSummary(summary="未识别到明显文本", confidence=0.0, should_comment=False)

    # 文本过短或信息量过低时不触发自动评论。
    if len(cleaned) < 8:
        return SceneSummary(summary=f"屏幕文本较少: {cleaned}", confidence=0.2, should_comment=False)

    # Keep larger source content so downstream comment generation can stay grounded on long OCR captures.
    summary = cleaned[:1200]
    return SceneSummary(
        summary=f"屏幕内容摘要: {summary}",
        confidence=0.75,
        should_comment=True,
    )
