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
) -> VisionCompatResult:
    prompt = "请用2-3句中文描述这张屏幕截图，突出用户当前在做什么。"
    prepared = _resize_image_for_vision(image, max_edge=max_edge)
    started = time.perf_counter()

    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-compat") as executor:
            future = executor.submit(
                llm_client.describe_image,
                prepared,
                prompt,
                SYSTEM_VISION_PROMPT,
            )
            summary = future.result(timeout=max(0.5, float(timeout_sec))).strip()
    except FutureTimeoutError:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return VisionCompatResult(summary="", reason="timeout", elapsed_ms=elapsed_ms)
    except Exception:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return VisionCompatResult(summary="", reason="error", elapsed_ms=elapsed_ms)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not summary:
        return VisionCompatResult(summary="", reason="empty", elapsed_ms=elapsed_ms)
    return VisionCompatResult(summary=summary, reason="ok", elapsed_ms=elapsed_ms)


def describe_screen_image(llm_client: LLMClient, image: Image.Image) -> str:
    result = describe_screen_image_compat(
        llm_client,
        image,
        timeout_sec=5.0,
        max_edge=1280,
    )
    return result.summary.strip()
