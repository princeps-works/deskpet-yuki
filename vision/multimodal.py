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
    if str(content_kind or "").strip().lower() == "visual_novel":
        prompt = (
            "请描述这张视觉小说截图，只输出JSON对象，不要Markdown，总计不超过320字。"
            '格式：{"appearance_environment":"人物可见外观、人数、背景场景、CG或立绘类型",'
            '"moment":"此刻可见的动作与表情；没有则留空",'
            '"speaker_name":"姓名栏清晰可读时逐字抄写，否则空","speaker_object":"姓名栏旁头像与某对象外观明显相同时填对象id，否则空",'
            '"objects":[{"id":"A","type":"human/animal/other/unknown","position":"左侧/右侧等位置","appearance":"该对象可见外观"}]}。'
            "人物和动物分别列入objects，最多6个；id只在本次截图有效，不填写推测的角色姓名。"
            "对白头像可能是同一立绘的重复显示，明显相同时不重复计为新人物，不确定就注明。"
            "说话人只能根据姓名栏与头像对应，不根据嘴型或台词提到的人名猜测；姓名识别与OCR不同也如实抄写供比较。"
            "appearance_environment不要包含表情、动作、台词、心理推测或剧情判断。"
            "moment只描述可见表现，不推断心理、动机或具体人物身份。"
            "不记录存档、读取等游戏按钮，不重抄台词；场景只按可见背景判断，不根据台词猜地点。"
            f"\n本地OCR仅用于区分对白与界面：\n{compact_ocr}"
        )
    prepared = _resize_image_for_vision(image, max_edge=max_edge)
    started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-compat")
    vision_model = str(getattr(getattr(llm_client, "settings", None), "vision_model_name", ""))
    request_options = {"disable_thinking": True} if content_kind == "visual_novel" and vision_model.lower().startswith("deepseek") else {}
    future = executor.submit(
        llm_client.describe_image,
        prepared,
        prompt,
        SYSTEM_VISION_PROMPT,
        **request_options,
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
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        status = getattr(exc, "status_code", None)
        reason = f"error:{type(exc).__name__}" + (f":status={status}" if isinstance(status, int) else "")
        return VisionCompatResult(summary="", reason=reason, elapsed_ms=elapsed_ms)
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
