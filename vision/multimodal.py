from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from PIL import Image

from desktop_pet.config.prompts import SYSTEM_VISION_PROMPT
from desktop_pet.llm.client import LLMClient


@dataclass
class VisionCompatResult:
    summary: str
    reason: str
    elapsed_ms: int


def _resize_image_for_vision(image: Image.Image, max_edge: int) -> Image.Image:
    edge = max(320, int(max_edge))
    width, height = image.size
    longest = max(width, height)
    if longest <= edge:
        return image

    ratio = edge / float(longest)
    target_w = max(1, int(round(width * ratio)))
    target_h = max(1, int(round(height * ratio)))
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)


def describe_screen_image_compat(
    llm_client: LLMClient,
    image: Image.Image,
    *,
    timeout_sec: float,
    max_edge: int,
    ocr_context: str = "",
    route: str = "vision_only",
    focus_query: str = "",
    content_kind: str = "screen",
) -> VisionCompatResult:
    compact_ocr = str(ocr_context or "").strip()
    compact_query = " ".join(str(focus_query or "").split())[:300]
    focus_instruction = ""
    if compact_query:
        focus_instruction = (
            f"用户本轮问题：{compact_query}\n"
            "请优先提取回答该问题所需的可见事实；不要直接代替聊天模型回答。\n"
        )
    uploaded_image = str(content_kind or "").strip().lower() in {"image", "upload", "uploaded_image"}
    if route == "vision+ocr" and compact_ocr:
        if uploaded_image:
            prompt = focus_instruction + (
                "下面已经提供这张用户上传图片的本地OCR结果，不要重新抄写全部文字。"
                "请补充OCR无法提供的视觉信息：人物或物体数量、外观特征、动作、关系、场景、构图、"
                "图表或界面含义。不要根据画风擅自确认具体角色身份。"
                "输出简短中文，不超过220字；不要输出解释或Markdown。\n\n"
                f"本地OCR：\n{compact_ocr}"
            )
        else:
            prompt = focus_instruction + (
                "下面已经提供本地OCR结果，不要重新抄写全部文字。"
                "请只补充OCR难以提供的视觉信息：应用或页面类型、主要区域关系、"
                "按钮/图标/弹窗/选中/禁用状态、图片/人物/视频/图表含义，以及用户可能的操作。"
                "输出简短中文，不超过180字；不要输出解释或Markdown。\n\n"
                f"本地OCR（含粗略位置）：\n{compact_ocr}"
            )
    else:
        if uploaded_image:
            prompt = focus_instruction + (
                "请分析这张用户上传的图片。输出简短中文，不超过220字。"
                "优先包含：人物或物体数量、外观特征、动作、相互关系、场景、构图及明显文字含义。"
                "不要根据画风擅自确认具体角色身份。"
            )
        else:
            prompt = focus_instruction + (
                "请分析这张屏幕截图。输出简短中文，不超过180字。"
                "优先包含：页面类型、主要区域、非文字视觉内容、界面状态和用户可能进行的操作。"
                "只保留评论所需的关键信息，不要逐字抄写整张屏幕。"
            )
    prepared = _resize_image_for_vision(image, max_edge=max_edge)
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-compat")
    future = executor.submit(
        llm_client.describe_image,
        prepared,
        prompt,
        SYSTEM_VISION_PROMPT,
    )

    try:
        summary = future.result(timeout=max(0.5, float(timeout_sec))).strip()
    except FutureTimeoutError:
        try:
            future.cancel()
        except Exception:
            pass
        executor.shutdown(wait=False, cancel_futures=True)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return VisionCompatResult(summary="", reason="timeout", elapsed_ms=elapsed_ms)
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return VisionCompatResult(summary="", reason="error", elapsed_ms=elapsed_ms)
    else:
        executor.shutdown(wait=False)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not summary:
        return VisionCompatResult(summary="", reason="empty", elapsed_ms=elapsed_ms)
    return VisionCompatResult(summary=summary, reason="ok", elapsed_ms=elapsed_ms)


def describe_screen_image(
    llm_client: LLMClient,
    image: Image.Image,
    *,
    ocr_context: str = "",
    route: str = "vision_only",
    focus_query: str = "",
) -> str:
    result = describe_screen_image_compat(
        llm_client,
        image,
        timeout_sec=5.0,
        max_edge=1280,
        ocr_context=ocr_context,
        route=route,
        focus_query=focus_query,
    )
    return result.summary.strip()
