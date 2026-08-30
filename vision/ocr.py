from __future__ import annotations

import importlib
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)
_engine = None
_engine_init_error = ""
_last_init_try_ts = 0.0
_retry_interval_sec = 10.0


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class OCRResult:
    lines: tuple[OCRLine, ...]
    text: str
    average_confidence: float
    text_coverage: float
    text_hash: str

    @property
    def char_count(self) -> int:
        return len(re.sub(r"\s+", "", self.text))

    def to_prompt_text(self, max_chars: int = 1200) -> str:
        """Return compact OCR with coarse position hints for multimodal fusion."""
        limit = max(80, int(max_chars))
        rows: list[str] = []
        seen: set[str] = set()
        for line in self.lines:
            cleaned = " ".join(line.text.split())
            normalized = cleaned.casefold()
            if not cleaned or normalized in seen:
                continue
            seen.add(normalized)
            if line.box is None:
                row = cleaned
            else:
                _left, top, _right, bottom = line.box
                center_y = (top + bottom) / 2.0
                # Boxes are normalized to 0..1 when the result is built.
                if center_y < 0.34:
                    position = "顶部"
                elif center_y > 0.66:
                    position = "底部"
                else:
                    position = "中部"
                row = f"[{position}] {cleaned}"
            next_size = len("\n".join(rows + [row]))
            if next_size > limit:
                break
            rows.append(row)
        return "\n".join(rows)[:limit]


def _empty_result() -> OCRResult:
    return OCRResult(lines=(), text="", average_confidence=0.0, text_coverage=0.0, text_hash="")


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
        kwargs: dict[str, Any] = {}
        # Explicitly pass RapidOCR thread knobs; this is more reliable than generic OMP env vars.
        threads = int(os.getenv("OCR_CPU_THREADS", "0") or 0)
        if threads > 0:
            kwargs["intra_op_num_threads"] = max(1, threads)
            kwargs["inter_op_num_threads"] = 1

        use_dml = os.getenv("OCR_USE_DML", "false").lower() in {"1", "true", "yes", "on"}
        use_cuda = os.getenv("OCR_USE_CUDA", "false").lower() in {"1", "true", "yes", "on"}
        if use_dml:
            kwargs["use_dml"] = True
        if use_cuda:
            kwargs["use_cuda"] = True

        _engine = RapidOCR(**kwargs)
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


def _extract_confidence_from_item(item: Any) -> float:
    value: Any = None
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        value = item[2]
    elif (
        isinstance(item, (list, tuple))
        and len(item) >= 2
        and isinstance(item[1], (list, tuple))
        and len(item[1]) >= 2
    ):
        value = item[1][1]
    elif isinstance(item, dict):
        for key in ("score", "confidence", "conf"):
            if key in item:
                value = item[key]
                break
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _extract_box_from_item(
    item: Any,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    raw_box: Any = None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        raw_box = item[0]
    elif isinstance(item, dict):
        raw_box = item.get("box") or item.get("bbox") or item.get("points")

    points: list[tuple[float, float]] = []
    if isinstance(raw_box, (list, tuple)):
        if len(raw_box) == 4 and all(isinstance(v, (int, float)) for v in raw_box):
            left, top, right, bottom = [float(v) for v in raw_box]
            points = [(left, top), (right, bottom)]
        else:
            for point in raw_box:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        points.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        continue
    if not points:
        return None

    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = max(0.0, min(1.0, min(xs) / float(width)))
    top = max(0.0, min(1.0, min(ys) / float(height)))
    right = max(0.0, min(1.0, max(xs) / float(width)))
    bottom = max(0.0, min(1.0, max(ys) / float(height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _build_ocr_result(items: list[Any], image_size: tuple[int, int]) -> OCRResult:
    lines: list[OCRLine] = []
    coverage = 0.0
    for item in items:
        text = _extract_text_from_item(item)
        if not text:
            continue
        box = _extract_box_from_item(item, image_size)
        confidence = _extract_confidence_from_item(item)
        lines.append(OCRLine(text=text, confidence=confidence, box=box))
        if box is not None:
            left, top, right, bottom = box
            coverage += max(0.0, right - left) * max(0.0, bottom - top)

    text = "\n".join(line.text for line in lines if line.text)
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""
    confidences = [line.confidence for line in lines if line.confidence > 0.0]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return OCRResult(
        lines=tuple(lines),
        text=text,
        average_confidence=average_confidence,
        text_coverage=min(1.0, coverage),
        text_hash=text_hash,
    )


def extract_ocr_result(image: Image.Image) -> OCRResult:
    _ensure_engine()
    if _engine is None:
        return _empty_result()

    img_np = np.array(image)
    try:
        result, _ = _engine(img_np)
    except Exception as exc:  # pragma: no cover
        _logger.warning("OCR failed: %s", exc)
        return _empty_result()

    if result is None:
        return _empty_result()
    items = list(result) if isinstance(result, (list, tuple)) else [result]
    return _build_ocr_result(items, image.size)


def extract_text(image: Image.Image) -> str:
    return extract_ocr_result(image).text
