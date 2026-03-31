from __future__ import annotations

import importlib
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)
_engine = None
_engine_init_error = ""
_last_init_try_ts = 0.0
_retry_interval_sec = 10.0


def _ensure_engine() -> None:
    global _engine, _engine_init_error, _last_init_try_ts
    if _engine is not None:
        return

    now = time.monotonic()
    if now - _last_init_try_ts < _retry_interval_sec:
        return
    _last_init_try_ts = now
    try:
        mod = importlib.import_module("rapidocr_onnxruntime")
        RapidOCR = getattr(mod, "RapidOCR")
    except Exception as exc:  # pragma: no cover
        _engine_init_error = f"rapidocr_onnxruntime import failed: {exc}"
        return

    try:
        _engine = RapidOCR()
        _engine_init_error = ""
    except Exception as exc:  # pragma: no cover
        _engine_init_error = f"RapidOCR init failed: {exc}"
        _engine = None


def get_ocr_runtime_status() -> tuple[bool, str]:
    _ensure_engine()
    if _engine is not None:
        return True, "ok"
    if _engine_init_error:
        return False, _engine_init_error
    return False, "engine unavailable"


def warmup_ocr_engine() -> tuple[bool, str]:
    """Initialize OCR engine eagerly and return current status."""
    _ensure_engine()
    return get_ocr_runtime_status()


def _extract_text_from_item(item: Any) -> str:
    if item is None:
        return ""

    # Common format: [box, text, score]
    if isinstance(item, (list, tuple)):
        if len(item) >= 2:
            value = item[1]
            if isinstance(value, (list, tuple)) and value:
                return str(value[0]).strip()
            return str(value).strip()
        if item:
            return str(item[0]).strip()
        return ""

    if isinstance(item, dict):
        for key in ("text", "txt", "label"):
            if key in item and item[key]:
                return str(item[key]).strip()
        return ""

    return str(item).strip()


def extract_text(image: Image.Image) -> str:
    _ensure_engine()
    if _engine is None:
        return ""

    img_np = np.array(image)
    try:
        result, _ = _engine(img_np)
    except Exception as exc:  # pragma: no cover
        _logger.warning("OCR failed: %s", exc)
        return ""

    if result is None:
        return ""

    lines: list[str] = []
    if isinstance(result, (list, tuple)):
        for item in result:
            text = _extract_text_from_item(item)
            if text:
                lines.append(text)
    else:
        text = _extract_text_from_item(result)
        if text:
            lines.append(text)

    return "\n".join([x for x in lines if x])
