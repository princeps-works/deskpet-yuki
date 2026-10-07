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


def _result_from_lines(lines: list[OCRLine]) -> OCRResult:
    text = "\n".join(line.text for line in lines if line.text)
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""
    confidences = [line.confidence for line in lines if line.confidence > 0.0]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    coverage = 0.0
    for line in lines:
        if line.box is None:
            continue
        left, top, right, bottom = line.box
        coverage += max(0.0, right - left) * max(0.0, bottom - top)
    return OCRResult(
        lines=tuple(lines),
        text=text,
        average_confidence=average_confidence,
        text_coverage=min(1.0, coverage),
        text_hash=text_hash,
    )


def filter_visual_novel_ocr_result(
    result: OCRResult,
    tracking: dict[str, Any],
) -> tuple[OCRResult, str]:
    """Cheap, conservative filtering for text read through VN artwork.

    Strong sentence-like lines are never delayed. Only weak short fragments and
    short mixed-script labels are rejected immediately. A line outside the
    learned dialogue lane is also suppressed after it stays at the same place
    across three otherwise-changing OCR frames, which catches background signs
    without treating a slowly-read dialogue line as static artwork.
    """
    if not result.lines:
        return result, ""

    frame = int(tracking.get("frame", 0) or 0) + 1
    tracking["frame"] = frame
    samples = tracking.setdefault("dialogue_samples", [])
    tracks = tracking.setdefault("line_tracks", {})
    highest_confidence = max((float(line.confidence) for line in result.lines), default=0.0)
    frame_signature = result.text_hash or f"frame-{frame}"
    kept: list[OCRLine] = []
    rejected: dict[str, int] = {}
    learning_candidates: list[tuple[int, float, tuple[float, float, float]]] = []

    def geometry(line: OCRLine) -> tuple[float, float, float] | None:
        if line.box is None:
            return None
        left, top, _right, bottom = line.box
        return ((top + bottom) / 2.0, max(0.01, bottom - top), left)

    def in_dialogue_lane(item: tuple[float, float, float] | None) -> bool:
        if item is None or not samples:
            return False
        center_y, height, left = item
        return any(
            abs(center_y - sample_y) <= max(0.07, (height + sample_height) * 1.4)
            and abs(left - sample_left) <= 0.22
            for sample_y, sample_height, sample_left in samples
        )

    for line in result.lines:
        compact = re.sub(r"\s+", "", str(line.text or ""))
        if not compact:
            continue
        cjk = sum(
            "\u3400" <= char <= "\u9fff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
            for char in compact
        )
        ascii_letters = sum(char.isascii() and char.isalpha() for char in compact)
        has_dialogue_punctuation = any(char in "。！？!?「」『』…，," for char in compact)
        confidence = float(line.confidence or 0.0)
        item_geometry = geometry(line)
        known_lane = in_dialogue_lane(item_geometry)
        sentence_like = cjk >= 6 or has_dialogue_punctuation
        name_like = 2 <= cjk <= 6 and ascii_letters == 0 and confidence >= 0.78
        clean_short_cjk = (
            cjk == len(compact)
            and not re.search(r"\S\s+\S", str(line.text or ""))
            and 1 <= cjk <= 6
        )
        short_weak = (
            len(compact) <= 4
            and confidence < 0.72
            and not has_dialogue_punctuation
            and not clean_short_cjk
        )
        mixed_label = (
            ascii_letters >= 2
            and cjk <= 6
            and len(compact) <= 18
            and not has_dialogue_punctuation
        )
        relative_weak = (
            confidence > 0.0
            and confidence < max(0.30, highest_confidence - 0.35)
            and not sentence_like
            and not name_like
            and not clean_short_cjk
        )

        track_key = compact.casefold()
        track = tracks.get(track_key)
        if not isinstance(track, dict):
            track = {"signatures": [], "geometry": item_geometry, "last_frame": frame}
            tracks[track_key] = track
        previous_geometry = track.get("geometry")
        if item_geometry is not None and previous_geometry is not None:
            if abs(item_geometry[0] - previous_geometry[0]) > 0.08:
                track["signatures"] = []
        signatures = track.setdefault("signatures", [])
        if frame_signature not in signatures:
            signatures.append(frame_signature)
            del signatures[:-4]
        track["geometry"] = item_geometry
        track["last_frame"] = frame
        stable_background = len(signatures) >= 3 and not name_like

        reason = ""
        if short_weak:
            reason = "short_weak"
        elif mixed_label and not known_lane:
            reason = "mixed_label"
        elif relative_weak and not known_lane:
            reason = "relative_weak"
        elif stable_background:
            reason = "stable_background"
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        kept.append(line)
        if (
            item_geometry is not None
            and sentence_like
            and confidence >= 0.72
            and ascii_letters <= cjk
            and (not samples or known_lane)
        ):
            learning_candidates.append((cjk, confidence, item_geometry))

    for _cjk, _confidence, item_geometry in sorted(learning_candidates, reverse=True)[:1]:
        samples.append(item_geometry)
    del samples[:-24]
    for key in list(tracks):
        if frame - int(tracks[key].get("last_frame", frame) or frame) > 20:
            del tracks[key]

    if not rejected:
        return result, ""
    note = ",".join(f"{name}={count}" for name, count in sorted(rejected.items()))
    return _result_from_lines(kept), note


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
        # Measured: leaving this at 0 makes a dialogue read take ~4.5s instead of
        # ~1.5s, so default to a sane thread count rather than inheriting
        # whatever the engine picks.
        threads = int(os.getenv("OCR_CPU_THREADS", "0") or 0)
        if threads <= 0:
            threads = max(1, min(4, (os.cpu_count() or 2) // 4))
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
