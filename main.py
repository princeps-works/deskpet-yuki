from __future__ import annotations

import os
import subprocess
import sys
import re
import random
import ctypes
import json
import time
import queue
import threading
import multiprocessing as mp
from ctypes import wintypes
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Optional


_SUBPROC_SCAN_CACHE: dict[str, object] = {
    "scan_target": None,
    "fingerprint": None,
    "brightness_fingerprint": None,
    "refreshed_at": 0.0,
    "ocr_hash": "",
    "ocr_text": "",
    "ocr_confidence": 0.0,
    "screen_context": "",
    "scene_summary": "",
    "scene_should_comment": False,
    "mode": "none",
    "mm_reason": "off",
    "mm_elapsed_ms": 0,
    "mm_summary_len": 0,
    "vision_route": "none",
    "capture_rect": None,
}

_SUBPROC_MM_RUNTIME: dict[str, object] = {
    "inited": False,
    "enabled": False,
    "client": None,
    "settings": None,
    "timeout_sec": 5.0,
    "max_edge": 1280,
    "mm_fail_streak": 0,
    "mm_cooldown_until": None,
    "mm_last_request_at": None,
}


def _init_subprocess_scan_runtime() -> None:
    """Load worker settings before the first timed scan; vision runs elsewhere."""
    if bool(_SUBPROC_MM_RUNTIME.get("inited", False)):
        return
    try:
        subproc_settings = load_settings(Path(__file__).resolve().parent)
        _SUBPROC_MM_RUNTIME["enabled"] = bool(
            subproc_settings.enable_multimodal_vision
            and subproc_settings.enable_mm_screen_comment
        )
        _SUBPROC_MM_RUNTIME["timeout_sec"] = float(subproc_settings.mm_timeout_sec)
        _SUBPROC_MM_RUNTIME["max_edge"] = int(subproc_settings.mm_image_max_edge)
        # Vision requests are handled asynchronously by the main process, so
        # constructing an API client here only delays the first OCR scan.
        _SUBPROC_MM_RUNTIME["client"] = None
        _SUBPROC_MM_RUNTIME["settings"] = subproc_settings
    except Exception:
        _SUBPROC_MM_RUNTIME["enabled"] = False
        _SUBPROC_MM_RUNTIME["client"] = None
        _SUBPROC_MM_RUNTIME["settings"] = None
    finally:
        _SUBPROC_MM_RUNTIME["inited"] = True

# 兼容两种启动方式：
# 1) python -m desktop_pet.main
# 2) python desktop_pet/main.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop_pet.config.settings import load_settings
from desktop_pet.vision.ocr import warmup_ocr_engine
from desktop_pet.config.prompts import SYSTEM_VOICEVOX_TRANSLATE_PROMPT


def _merge_chromium_flags(existing: str, extras: list[str]) -> str:
    tokens = [t.strip() for t in existing.split(" ") if t.strip()]
    present = set(tokens)
    for item in extras:
        if item not in present:
            tokens.append(item)
            present.add(item)
    return " ".join(tokens)


def configure_webengine_render_mode(mode: str) -> None:
    selected = (mode or "gpu").strip().lower()
    existing_flags = os.getenv("QTWEBENGINE_CHROMIUM_FLAGS", "")

    if selected == "software":
        flags = _merge_chromium_flags(
            existing_flags,
            ["--disable-gpu", "--disable-gpu-compositing", "--disable-software-rasterizer"],
        )
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
        os.environ["QT_OPENGL"] = "software"
        print("[STARTUP] WebEngine render mode: software")
        return

    if selected == "auto":
        print("[STARTUP] WebEngine render mode: auto")
        return

    flags = _merge_chromium_flags(
        existing_flags,
        [
            "--ignore-gpu-blocklist",
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--use-angle=d3d11",
        ],
    )
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
    os.environ.setdefault("QT_OPENGL", "desktop")
    print("[STARTUP] WebEngine render mode: gpu")


def _frame_fingerprint(image, *, text_sensitive: bool = False) -> bytes:
    # Downsample grayscale bytes as a lightweight scene fingerprint.
    size = (96, 36) if text_sensitive else (32, 18)
    gray = image.convert("L").resize(size)
    return gray.tobytes()


def _fingerprint_diff_ratio(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    changed = sum(1 for x, y in zip(a, b) if abs(x - y) > 10)
    return changed / len(a)


# Text-detection grid for a dialogue crop, and how far a cell must deviate from
# the crop's median brightness to count as containing a glyph. 192x48 over this
# game's 1979x369 box means each cell covers ~10x8 source pixels, small enough
# that a stroke dominates its cell instead of being averaged away.
_TEXT_INK_SIZE = (192, 48)
_TEXT_INK_DEVIATION = 0.38
_VN_OCR_CACHE_MAX_REUSE_SEC = 10.0


def _text_ink_fingerprint(image) -> bytes:
    """Binary ink map of a dialogue crop, for detecting text changes.

    ``_frame_fingerprint`` is tuned for whole-screen scene changes and averages
    ~21x10 source pixels per cell for a text-sensitive crop. That washes thin
    glyph strokes out: measured on this game's real 1979x369 box, two different
    short lines scored 0.0064 and a name-plate-only change 0.0032, both under
    the 0.008 threshold. The box therefore looked "unchanged", OCR was skipped,
    and the first line ever read was returned forever.

    Marking the cells whose brightness deviates from the box's own median keeps
    the signal. Measuring deviation rather than "brighter than a cut" means a
    box with dark text on a light plate is handled by the same code, and the
    threshold scales with the crop's contrast so a global brightness change
    cannot move it.
    """
    cols, rows = _TEXT_INK_SIZE
    data = image.convert("L").resize((cols, rows)).tobytes()
    ordered = sorted(data)
    low, high = ordered[0], ordered[-1]
    if high - low < 24:
        # Flat crop: no text-like contrast, so nothing can have changed.
        return bytes(len(data))
    median = ordered[len(ordered) // 2]
    cut = max(18.0, (high - low) * _TEXT_INK_DEVIATION)
    return bytes(abs(value - median) >= cut for value in data)


def _ink_diff_ratio(a: bytes, b: bytes) -> float:
    """Share of inked cells that differ between two ink maps.

    Dividing by the inked cells rather than by the crop area is what makes this
    sensitive: a dialogue box is mostly empty, so an area-normalised ratio
    shrinks every real change towards zero. Identical frames and pure background
    brightness shifts both score 0.0, while two different short lines score
    ~0.57 -- roughly 70x the 0.12 threshold this is used with.
    """
    if not a or not b or len(a) != len(b):
        return 1.0
    xor = 0
    union = 0
    for left, right in zip(a, b):
        if left or right:
            union += 1
            if left != right:
                xor += 1
    if union == 0:
        return 0.0
    return xor / union


def _vn_ocr_cache_can_reuse(
    ink_fingerprint: bytes,
    previous_ink_fingerprint: bytes,
    brightness_fingerprint: bytes,
    previous_brightness_fingerprint: bytes,
    *,
    ink_threshold: float,
    brightness_threshold: float,
    cache_age_sec: float,
) -> bool:
    """Reuse VN OCR only while both change signals agree and the cache is fresh."""
    return (
        0.0 <= float(cache_age_sec) < _VN_OCR_CACHE_MAX_REUSE_SEC
        and _ink_diff_ratio(ink_fingerprint, previous_ink_fingerprint)
        < float(ink_threshold)
        and _fingerprint_diff_ratio(
            brightness_fingerprint, previous_brightness_fingerprint
        )
        < float(brightness_threshold)
    )


def _vn_stable_cache_can_reuse(*args, ink_threshold: float, brightness_threshold: float, cache_age_sec: float, has_text: bool = True) -> bool:
    """A stricter comparison permits longer reuse of a genuinely static crop."""
    max_age = 60.0 if has_text else _VN_OCR_CACHE_MAX_REUSE_SEC
    return 0.0 <= cache_age_sec < max_age and _vn_ocr_cache_can_reuse(
        *args,
        ink_threshold=min(0.01, ink_threshold),
        brightness_threshold=min(0.002, brightness_threshold),
        cache_age_sec=min(cache_age_sec, _VN_OCR_CACHE_MAX_REUSE_SEC - 0.01),
    )


def _vn_text_is_settling(cache: dict, ink: bytes, brightness: bytes, now: float) -> bool:
    """Debounce growing glyphs before OCR, with a bounded wait for animation."""
    previous_ink = cache.get("settle_ink")
    previous_brightness = cache.get("settle_brightness")
    changed = (
        not isinstance(previous_ink, bytes)
        or not isinstance(previous_brightness, bytes)
        or _ink_diff_ratio(ink, previous_ink) >= 0.01
        or _fingerprint_diff_ratio(brightness, previous_brightness) >= 0.002
    )
    if not cache.get("settle_active", False):
        cache["settle_started_at"] = now
        cache["settle_quiet_at"] = now
        cache["settle_active"] = True
        cache["settle_growing"] = False
    elif changed:
        cache["settle_quiet_at"] = now
        if isinstance(previous_ink, bytes):
            added = sum(bool(left and not right) for left, right in zip(ink, previous_ink))
            removed = sum(bool(right and not left) for left, right in zip(ink, previous_ink))
            if added >= max(3, removed * 2):
                cache["settle_growing"] = True
    cache["settle_ink"] = ink
    cache["settle_brightness"] = brightness
    max_wait = 1.6 if cache["settle_growing"] else 0.8
    waiting = now - cache["settle_quiet_at"] < 0.25 and now - cache["settle_started_at"] < max_wait
    if not waiting:
        cache["settle_active"] = False
    return waiting


def _text_change_metrics(image, *, visual_novel: bool):
    """Pick the fingerprint and comparison appropriate to the captured crop.

    In visual-novel mode the captured image *is* the dialogue box, so the
    question is "did the text change" and the ink comparison answers it. Outside
    visual-novel mode the image is the whole screen, where the coarse brightness
    fingerprint is the right tool for spotting scene changes.
    """
    if visual_novel:
        return _text_ink_fingerprint(image), _ink_diff_ratio
    return _frame_fingerprint(image, text_sensitive=False), _fingerprint_diff_ratio


def _resize_for_ocr(image, max_edge: int):
    edge = int(max_edge)
    if edge <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= edge:
        return image
    ratio = float(edge) / float(longest)
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    return image.resize((new_w, new_h))


def _ocr_text_variants(image) -> list:
    """Pre-processing variants tried for visual-novel dialogue, best-first.

    Measured on a real 960x252 dialogue crop: grayscale + autocontrast takes
    1.3s while raw RGB takes 2.3s for the same text, because the recogniser has
    one channel instead of three. So the cheap grayscale form is tried first and
    the colour/polarity variants only serve as fallbacks when it reads too
    little. Padding matters too: text touching the crop edge is regularly
    dropped.
    """
    from PIL import ImageOps

    ordered: list = []
    try:
        gray = ImageOps.grayscale(image)
        ordered.append(ImageOps.autocontrast(gray).convert("RGB"))
    except Exception:
        pass
    ordered.append(image)
    try:
        ordered.append(ImageOps.expand(image, border=12, fill=(255, 255, 255)))
    except Exception:
        pass
    try:
        ordered.append(ImageOps.invert(ImageOps.grayscale(image)).convert("RGB"))
    except Exception:
        pass
    # De-duplicate while preserving order.
    unique: list = []
    for item in ordered:
        if not any(item is existing for existing in unique):
            unique.append(item)
    return unique


def _text_band_candidates(
    height: int,
    *,
    text_ratio: float,
    bottom_margin: float,
    mid_top_ratio: float,
) -> list[tuple[str, int, int]]:
    """Vertical crops to retry when the assumed dialogue box reads nothing.

    Two **disjoint** windows instead of a pile of overlapping bands. The previous
    version tried four bands anchored at 92/76/60/44% of the height; combined
    with the bottom margin, two of them re-read pixels the app's own band had
    already covered (at 800x500 the "70-92%" band sat entirely inside "58-92%"),
    so they paid a full OCR pass for identical pixels -- while the mid-screen
    coverage that justified the sweep in the first place went missing.

    Returns ``[(label, top, bottom), ...]`` in pixels for a frame ``height``
    tall: the lower band the app already assumes, then the strip directly above
    it. Because they are disjoint, every pass reads new pixels, and their union
    spans ``mid_top_ratio`` to the bottom margin.
    """
    if height <= 0:
        return []
    margin = max(0.0, min(0.40, float(bottom_margin)))
    bottom = min(height, int(round(height * (1.0 - margin))))
    lower_top = int(round(height * max(0.2, min(0.95, float(text_ratio)))))
    mid_top = int(round(height * max(0.0, min(0.90, float(mid_top_ratio)))))
    out: list[tuple[str, int, int]] = []
    if bottom - lower_top >= 32:
        out.append(("text_band", lower_top, bottom))
    if lower_top - mid_top >= 32:
        out.append(("mid_band", mid_top, lower_top))
    return out


def _crop_visual_novel_scan_image(image, *, enabled: bool, has_manual_region: bool):
    """Legacy fixed-ratio crop.

    Kept for callers that do not go through :func:`_resolve_vn_capture`; the
    live scan path now derives the text box from the resolved window rect.
    """
    if not enabled or has_manual_region:
        return image
    return _crop_lower_part(image, 0.58)


def _crop_lower_part(image, top_ratio: float):
    """Keep the lower part of an image; used for the visual-novel text box.

    ``top_ratio`` is where the kept region starts as a fraction of the height,
    so any value in (0, 1) must crop -- the previous version only cropped when
    the ratio happened to exceed roughly 0.6.
    """
    ratio = max(0.2, min(0.95, float(top_ratio)))
    width, height = image.size
    top = int(round(height * ratio))
    if top <= 0:
        return image
    if height - top < 24:
        return image
    return image.crop((0, top, width, height))


def _crop_text_band(image, top_ratio: float, *, bottom_margin: float = 0.0):
    """The dialogue band: lower ``1 - top_ratio`` of the frame, minus the UI bar.

    The automatic band used to run to the very bottom of the window, so it always
    contained the game's SAVE/LOAD/CONFIG bar. Measured on 青空下的加缪
    (2560x1600): dialogue text ends at ~91% of the height, the bar sits at
    93-95%, and below 95.5% is a black letterbox. OCR therefore returned the
    dialogue and the chrome glued together
    ("...没什么变化。 SAVELOADCONFIGSKIPAUTOLOGQ.SAVEQ.LOAD"), and the tracker
    stored that as dialogue. ``bottom_margin`` cuts the strip away.
    """
    ratio = max(0.2, min(0.95, float(top_ratio)))
    margin = max(0.0, min(0.40, float(bottom_margin)))
    width, height = image.size
    top = max(0, int(round(height * ratio)))
    bottom = min(height, int(round(height * (1.0 - margin))))
    if bottom - top < 24:
        # Degenerate request; better the whole frame than an empty crop.
        return image
    if top == 0 and bottom == height:
        return image
    return image.crop((0, top, width, bottom))


# The game's bottom bar is read as one glued run -- "SAVELOADCONFIGSKIPAUTOLOG
# Q.SAVEQ.LOAD" -- so the tokens are matched as a run first, which cannot damage
# a real word, and only then as standalone words.
_UI_CHROME_WORDS = (
    "Q.SAVE",
    "Q.LOAD",
    "QUICKSAVE",
    "QUICKLOAD",
    "BACKLOG",
    "CONFIG",
    "SKIP",
    "AUTO",
    "SAVE",
    "LOAD",
    "LOG",
    "MENU",
    "HISTORY",
    "TITLE",
    "HIDE",
)
_UI_CHROME_TOKEN = "(?:" + "|".join(re.escape(word) for word in _UI_CHROME_WORDS) + ")"
_UI_CHROME_RUN = re.compile(
    rf"(?i){_UI_CHROME_TOKEN}(?:[\s._\-|]*{_UI_CHROME_TOKEN})+"
)
_UI_CHROME_ALONE = re.compile(
    rf"(?i)(?<![A-Za-z0-9])(?:{_UI_CHROME_TOKEN})(?![A-Za-z0-9])"
)


def strip_ui_chrome(text: str) -> str:
    """Remove the game's menu-bar words from OCR text.

    The bottom bar sits inside the captured band, and OCR returns it glued to the
    dialogue. ``prepare_ocr_text`` accepts that mixture as ``quality_ok``, so
    without this the chrome is stored as if the player had read it out -- the
    user's own cache contains the fact "台词中混有SAVE/LOAD/Q.SAVE/Q.LOAD等界面
    文本". Stripping here keeps both the tracker and the panel on the real words.

    ASCII-only tokens, so Chinese dialogue is untouched, and a lone token needs a
    non-alphanumeric boundary (``DIALOGUE`` keeps its "LOG").
    """
    value = str(text or "")
    if not value:
        return ""
    value = _UI_CHROME_RUN.sub(" ", value)
    value = _UI_CHROME_ALONE.sub(" ", value)
    value = re.sub(r"[ \t\u3000]{2,}", " ", value)
    return value.strip(" \t\u3000")


def should_run_band_sweep(
    *,
    vn_mode: bool,
    has_vision_frame: bool,
    ocr_chars: int,
    quiet_sec: float,
    quiet_required_sec: float,
    frame_changed: bool,
    cooled_down: bool,
) -> tuple[bool, str]:
    """Decide whether the expensive dialogue-box sweep is worth running now.

    Pure so the pacing is testable without a live window. Ordered cheapest
    first: structural checks, then how long the box has been unreadable, then
    frame movement, then the between-sweeps cooldown.

    A single empty read is normal -- the box blinks out between lines, a scene
    transition has no text, a frame can be caught mid-fade -- so requiring
    ``quiet_required_sec`` of no text is what stops a brief blank from buying a
    ~10s recovery.
    """
    if not vn_mode or not has_vision_frame:
        return False, "no_vn_frame"
    if ocr_chars > 0:
        return False, "text_found"
    if quiet_sec < quiet_required_sec:
        return False, f"quiet({quiet_sec:.1f}s<{quiet_required_sec:.0f}s)"
    if not frame_changed:
        return False, "frame_unchanged"
    if not cooled_down:
        return False, "cooldown"
    return True, "sweep"


def _resolve_vn_capture(
    *,
    mode: str,
    resolver,
    ocr_region,
    vision_region,
    monitor_index: int,
    text_ratio: float,
    text_bottom_margin: float = 0.0,
) -> tuple[object, Optional[object], object, str]:
    """Capture the visual-novel target with separate OCR and vision framing.

    ``ocr_region`` is an optional *window-relative* rect and is authoritative
    when set; the fixed bottom-band heuristic is only a fallback, because
    guessing where the dialogue box lives is the main reason OCR reads nothing.

    ``vision_region`` is an optional rect used with ``mode="manual"``: it defines
    the game画面 area for cases where the window cannot be resolved or
    PrintWindow yields nothing (exclusive-fullscreen games). ``vision_region`` is
    given in *capture space* by the caller.

    Returns ``(image, window_rect, vision_image, source)`` where ``image`` is the
    region handed to OCR and ``vision_image`` is the frame handed to the vision
    model. ``_resolve_vn_capture.warning`` carries a user-facing message when a
    configured region could not be honoured.
    """
    from desktop_pet.vision.capture import (
        capture_absolute_rect,
        capture_primary_screen,
        capture_window_client,
        get_monitor_geometry,
    )

    _resolve_vn_capture.warning = ""

    target_mode = str(mode or "window").strip().lower()

    def _crop_region(image, region):
        _crop_region.warning = ""
        if region is None:
            return None
        try:
            left, top, width, height = (int(value) for value in region)
        except (TypeError, ValueError):
            return None
        img_w, img_h = image.size
        raw_right = left + max(1, width)
        raw_bottom = top + max(1, height)
        left = max(0, min(left, img_w - 1))
        top = max(0, min(top, img_h - 1))
        right = max(left + 1, min(img_w, raw_right))
        bottom = max(top + 1, min(img_h, raw_bottom))
        if right - left < 8 or bottom - top < 8:
            _crop_region.warning = "框选区域在画面之外，已退回自动"
            return None
        # A region that mostly falls outside the frame means the coordinates and
        # the frame disagree (DPI mismatch, or the window moved/shrunk).
        wanted = float(max(1, width) * max(1, height))
        got = float((right - left) * (bottom - top))
        if got / wanted < 0.6:
            _crop_region.warning = "框选区域大部分超出画面，请重新框选"
        return image.crop((left, top, right, bottom))

    def _finish(frame, window_rect, source, *, text_region, screen_relative_region):
        """Pick the vision frame and the OCR image for a captured frame."""
        vision_image = frame
        warnings: list[str] = []
        # 1. Optional manual vision box. It defines what counts as "the game
        #    画面" and everything downstream is relative to it.
        if vision_region is not None:
            boxed = _crop_region(frame, vision_region)
            box_warning = str(getattr(_crop_region, "warning", "") or "")
            if boxed is not None:
                vision_image = boxed
                if box_warning:
                    warnings.append(f"画面区域：{box_warning}")
            elif box_warning:
                warnings.append(f"画面区域：{box_warning}")
        # 2. OCR region is window-relative; translate it when the vision box
        #    shifted the origin so both stay aligned.
        interior_region = text_region
        if interior_region is not None and vision_image is not frame:
            try:
                offset_x = 0 if vision_region is None else -int(vision_region[0])
                offset_y = 0 if vision_region is None else -int(vision_region[1])
                interior_region = (
                    int(interior_region[0]) + offset_x,
                    int(interior_region[1]) + offset_y,
                    int(interior_region[2]),
                    int(interior_region[3]),
                )
            except Exception:
                interior_region = None
        from desktop_pet.vision.vn_ui_mask import mask_visual_novel_ui

        ocr_frame = vision_image
        if window_rect is not None and vision_image is frame:
            ocr_frame = mask_visual_novel_ui(
                vision_image, title=str(getattr(window_rect, "title", "") or ""),
                process_name=str(getattr(window_rect, "process_name", "") or ""),
            )
        cropped = _crop_region(ocr_frame, interior_region)
        ocr_warning = str(getattr(_crop_region, "warning", "") or "")
        if ocr_warning:
            warnings.append(f"文本区：{ocr_warning}")
        _finish.warning = "；".join(warnings)
        if cropped is not None:
            return cropped, window_rect, vision_image, source
        if window_rect is None and screen_relative_region is not None and vision_image is frame:
            # Screen grabs are monitor-relative, so an absolute rect needs the
            # monitor origin subtracted before cropping.
            try:
                monitor_left, monitor_top, _mw, _mh = get_monitor_geometry(int(monitor_index))
                shifted = (
                    int(screen_relative_region[0]) - int(monitor_left),
                    int(screen_relative_region[1]) - int(monitor_top),
                    int(screen_relative_region[2]),
                    int(screen_relative_region[3]),
                )
                cropped = _crop_region(ocr_frame, shifted)
                if cropped is not None:
                    return cropped, window_rect, vision_image, source
            except Exception:
                pass
        # Fall back to the lower band of whatever frame we ended up with, minus
        # the game's bottom UI bar (see _crop_text_band).
        return (
            _crop_text_band(ocr_frame, text_ratio, bottom_margin=text_bottom_margin),
            window_rect,
            vision_image,
            source,
        )

    if target_mode == "window" and resolver is not None:
        rect = resolver.resolve()
        if rect is not None:
            # PrintWindow first: screen capture would pick up whatever window is
            # in front of the game, silently feeding us the wrong content.
            frame = capture_window_client(int(getattr(rect, "hwnd", 0) or 0))
            if frame is not None:
                result = _finish(
                    frame, rect, "window_direct", text_region=ocr_region, screen_relative_region=None
                )
                _resolve_vn_capture.warning = str(getattr(_finish, "warning", "") or "")
                return result
            frame = capture_absolute_rect(rect.left, rect.top, rect.width, rect.height)
            if frame is not None:
                result = _finish(
                    frame, rect, "window", text_region=ocr_region, screen_relative_region=None
                )
                _resolve_vn_capture.warning = str(getattr(_finish, "warning", "") or "")
                return result
        if str(getattr(resolver, "target_title", "") or ""):
            _resolve_vn_capture.warning = "已锁定的游戏窗口不可用，视觉小说扫描已暂停"
            return None, None, None, "target_missing"

    # Manual mode, or window resolution failed: grab the monitor and let the
    # manual vision box define the game area. The OCR region is already in
    # capture space here, so it goes in as ``text_region`` -- passing it as the
    # monitor-relative fallback instead would skip cropping whenever a vision box
    # is set and silently fall back to the heuristic band.
    img = capture_primary_screen(monitor_index=int(monitor_index), region=None)
    source = "manual" if (target_mode == "manual" and vision_region is not None) else "screen_fallback"
    result = _finish(
        img,
        None,
        source,
        text_region=ocr_region,
        screen_relative_region=None,
    )
    notes = [str(getattr(_finish, "warning", "") or "")]
    if source == "screen_fallback" and resolver is not None:
        reason = str(getattr(resolver, "last_error", "") or "")
        if reason:
            notes.append(f"没找到游戏窗口（{reason}），当前抓的是整个屏幕")
    _resolve_vn_capture.warning = "；".join(item for item in notes if item)
    return result


def _enable_per_monitor_dpi_awareness() -> None:
    """Make Win32 window rects and mss captures share one physical coordinate space.

    Without this the process is DPI-virtualised: DwmGetWindowAttribute reports
    the game window as 1707x1067 (logical) while mss captures 2560x1600
    (physical). A text region selected on screen then lands outside the reported
    window rect, so the crop is rejected and OCR never sees the dialogue. Must
    run before QApplication is created.
    """
    if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows
        return
    try:
        user32 = ctypes.windll.user32
    except Exception:  # pragma: no cover
        return
    # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
    for name, arg in (
        ("SetProcessDpiAwarenessContext", ctypes.c_void_p(-4)),
        ("SetProcessDpiAwarenessContext", ctypes.c_void_p(-3)),
    ):
        func = getattr(user32, name, None)
        if not callable(func):
            continue
        try:
            if func(arg):
                print("[STARTUP] DPI awareness: per-monitor v2")
                return
        except Exception:
            continue
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            print("[STARTUP] DPI awareness: per-monitor (shcore)")
            return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        print("[STARTUP] DPI awareness: system")
    except Exception:
        print("[STARTUP] DPI awareness: unavailable (scaling may misalign regions)")


def vn_scan_transition(
    *,
    enabling: bool,
    scan_enabled: bool,
    autostarted: bool,
    prev_enabled: bool,
) -> tuple[bool | None, bool, bool, str]:
    """Decide how a visual-novel mode toggle should touch the auto-scan toggle.

    Entering VN mode is the user saying "watch the game window", but the vision
    sampler's activity gate and the whole OCR path key off auto-scan, which
    starts every launch switched off. Requiring a second, unrelated-looking
    toggle made VN mode appear completely inert, so entering it turns scanning
    on; leaving it hands the user's own preference back.

    Pure so the policy is testable. Returns
    ``(target, autostarted, prev_enabled, note)`` where ``target`` is the scan
    state to apply and ``None`` means "leave the toggle alone".
    """
    if enabling:
        if scan_enabled:
            # Already scanning: nothing to do, and nothing to undo later.
            return None, False, prev_enabled, ""
        return True, True, False, "并已开启自动扫描"
    if autostarted:
        # Only rewind a change this feature made, never a deliberate user press
        # (the scan toggle clears ``autostarted`` when pressed for real).
        restore = bool(prev_enabled)
        note = "" if restore else "；自动扫描已恢复关闭"
        return restore, False, prev_enabled, note
    return None, False, prev_enabled, ""


def normalize_game_title(title: str) -> str:
    """Reduce a game window title to just the game's name.

    Localisation groups and patch versions live in the window title --
    "青空下的加缪【天空鸽剧汉化组】v0.6-Beta" -- so comparing raw titles would
    report a mismatch whenever the user installed a different patch, which is
    noise rather than a wrong cache.
    """
    value = str(title or "")
    # Bracketed decorations: 【汉化组】, [patch], （体验版）.
    value = re.sub(r"[【\[（(][^】\]）)]*[】\]）)]", "", value)
    # Version-ish tails: v0.6-Beta, 1.2.3, v1.0_rc1. Anchored to the end rather
    # than using \b, because there is no word boundary between a CJK character
    # and the "v" ("青空下的加缪v0.6-Beta" must still lose its version).
    value = re.sub(
        r"\s*v?\d+(?:\.\d+)+(?:[-_ ]\S+)*\s*$", "", value, flags=re.IGNORECASE
    )
    return value.strip(" -_·|~—")


def game_title_changed(previous: str, current: str) -> bool:
    """Has the game window changed since it was last seen?

    A cache holds one game's facts, summary and character notes, so when the
    window title changes the loaded cache is probably from the previous game and
    continuing would blend two stories together.

    Titles are normalised first because the same game reports a slightly
    different string after a patch ("青空下的加缪【汉化组】v0.6-Beta" vs "v0.7");
    comparing raw would warn about a "new game" on every update. Pure, so the
    rule is testable without a live window. ``False`` when either side is
    unknown, since an empty title is not evidence of a change.
    """
    before = normalize_game_title(previous)
    after = normalize_game_title(current)
    if not before or not after:
        return False
    return before != after


def story_cache_game_status(bound_title: str, has_content: bool, captured_title: str) -> str:
    """Decide whether a captured game may write to the loaded story cache."""
    if not normalize_game_title(captured_title):
        return "unknown"
    if bound_title:
        return "mismatch" if game_title_changed(bound_title, captured_title) else "match"
    return "unbound" if has_content else "bind_empty"


def vision_should_capture(
    *,
    tokens: float,
    in_flight: bool,
    enabled: bool,
    active: bool,
    budget_ok: bool,
    threshold: float,
    shots_per_minute: float,
    delta: float | None = None,
    since_last_sec: float | None = None,
    min_gap_sec: float = 0.0,
) -> tuple[bool, str]:
    """Decide whether the vision sampler should look right now.

    Pure so the policy is testable: the sampler itself needs a live window, a
    thread and an API key. Order matters -- cheap structural checks first, then
    quota, then the content gate, so a skipped tick never costs a capture.

    Two independent pacing controls:

    * ``tokens``/``shots_per_minute`` bound the long-run total (3/min here).
    * ``since_last_sec``/``min_gap_sec`` bound how fast that total may be spent,
      so a burst saved up during a quiet spell cannot be dumped into the first
      seconds of an opening movie and leave the rest of it unsampled.

    ``delta`` is ``None`` before a frame exists; the caller captures first and
    re-checks the content gate with the measured value.
    """
    if not enabled:
        return False, "disabled"
    if in_flight:
        return False, "in_flight"
    if not active:
        return False, "scan_off"
    if tokens < 1.0:
        rate = max(0.1, float(shots_per_minute))
        wait = (1.0 - float(tokens)) * 60.0 / rate
        return False, f"quota(约{wait:.0f}秒后)"
    if not budget_ok:
        return False, "budget"
    if (
        min_gap_sec > 0
        and since_last_sec is not None
        and since_last_sec < min_gap_sec
    ):
        return False, f"gap(还需{min_gap_sec - since_last_sec:.0f}秒)"
    if delta is not None and delta < threshold:
        return False, f"unchanged(delta={delta:.2f})"
    return True, ""


def _compose_screen_context(vision_summary: str, ocr_text: str) -> tuple[str, str]:
    """Keep source-specific prefix while routing into one downstream processing path."""
    vision_clean = " ".join(str(vision_summary or "").split())
    ocr_clean = " ".join(str(ocr_text or "").split())

    if not vision_clean and not ocr_clean:
        return "", "none"

    parts: list[str] = []
    if ocr_clean and vision_clean:
        mode = "vision+ocr"
        parts.append(f"视觉摘要: {vision_clean}")
        parts.append(f"OCR文本: {ocr_clean}")
    elif ocr_clean:
        mode = "ocr"
        parts.append(f"OCR文本: {ocr_clean}")
    else:
        mode = "vision"
        parts.append(f"视觉摘要: {vision_clean}")
    return "\n".join(parts).strip(), mode


def _request_multimodal_summary(
    llm_client,
    image,
    settings_obj,
    runtime_state: dict,
    log_fn=None,
    *,
    ocr_context: str = "",
    route: str = "vision_only",
    respect_min_interval: bool = True,
    vision_enabled: bool | None = None,
    focus_query: str = "",
) -> tuple[str, str, int, int]:
    feature_enabled = (
        bool(getattr(settings_obj, "enable_mm_screen_comment", False))
        if vision_enabled is None
        else bool(vision_enabled)
    )
    mm_enabled = bool(getattr(settings_obj, "enable_multimodal_vision", False) and feature_enabled)
    if (not mm_enabled) or llm_client is None:
        return "", "off", 0, 0

    def _log(stage: str, detail: str) -> None:
        if callable(log_fn):
            try:
                log_fn(stage, detail)
            except Exception:
                pass

    def _mark_success() -> None:
        runtime_state["mm_fail_streak"] = 0
        runtime_state["mm_cooldown_until"] = None

    def _mark_failure(reason: str) -> None:
        streak = int(runtime_state.get("mm_fail_streak", 0)) + 1
        runtime_state["mm_fail_streak"] = streak
        threshold = max(1, int(getattr(settings_obj, "mm_failure_threshold", 1)))
        if streak >= threshold:
            cooldown_sec = max(10, int(getattr(settings_obj, "mm_cooldown_sec", 30)))
            runtime_state["mm_cooldown_until"] = datetime.now() + timedelta(seconds=cooldown_sec)
            _log("mm_cooldown", f"reason={reason}, streak={streak}, cooldown={cooldown_sec}s")
        else:
            _log("mm_fail", f"reason={reason}, streak={streak}/{threshold}")

    cooldown_until = runtime_state.get("mm_cooldown_until")
    now = datetime.now()
    if isinstance(cooldown_until, datetime) and now < cooldown_until:
        remain = int((cooldown_until - now).total_seconds())
        _log("mm_skip", f"cooldown_remain={max(1, remain)}s")
        return "", "cooldown", 0, 0

    if respect_min_interval:
        min_interval_sec = max(0, int(getattr(settings_obj, "mm_auto_min_interval_sec", 0)))
        last_request_at = runtime_state.get("mm_last_request_at")
        if min_interval_sec > 0 and isinstance(last_request_at, datetime):
            elapsed_sec = (now - last_request_at).total_seconds()
            if elapsed_sec < min_interval_sec:
                remain = int(round(min_interval_sec - elapsed_sec))
                _log("mm_skip", f"rate_limit_remain={max(1, remain)}s")
                return "", "rate_limited", 0, 0

    runtime_state["mm_last_request_at"] = now

    try:
        if bool(getattr(settings_obj, "enable_mm_compat_mode", True)):
            from desktop_pet.vision.multimodal import describe_screen_image_compat

            vision_result = describe_screen_image_compat(
                llm_client,
                image,
                timeout_sec=float(getattr(settings_obj, "mm_timeout_sec", 5.0)),
                max_edge=int(getattr(settings_obj, "mm_image_max_edge", 1280)),
                ocr_context=ocr_context,
                route=route,
                focus_query=focus_query,
            )
            summary = str(vision_result.summary or "").strip()
            reason = str(vision_result.reason)
            elapsed_ms = int(vision_result.elapsed_ms)
        else:
            from desktop_pet.vision.multimodal import describe_screen_image

            visual_summary = describe_screen_image(
                llm_client,
                image,
                ocr_context=ocr_context,
                route=route,
                focus_query=focus_query,
            )
            summary = str(visual_summary or "").strip()
            reason = "ok" if summary else "empty"
            elapsed_ms = 0
    except Exception:
        _mark_failure("error")
        _log("mm_result", "reason=error, elapsed_ms=0, len=0")
        return "", "error", 0, 0

    summary_len = len(summary)
    _log("mm_result", f"reason={reason}, elapsed_ms={elapsed_ms}, len={summary_len}")
    if summary:
        _mark_success()
        return summary, reason, elapsed_ms, summary_len

    _mark_failure(reason)
    return "", reason, elapsed_ms, 0


def _build_ocr_first_context(
    llm_client,
    image,
    ocr_result,
    settings_obj,
    runtime_state: dict,
    log_fn=None,
    *,
    respect_vision_interval: bool,
    vision_enabled: bool | None = None,
    focus_query: str = "",
) -> dict:
    from desktop_pet.vision.router import VisionRouteDecision, decide_vision_route

    ocr_text = str(getattr(ocr_result, "text", "") or "").strip()
    if bool(getattr(settings_obj, "enable_ocr_first_routing", True)):
        decision = decide_vision_route(
            ocr_result,
            ocr_only_min_chars=int(getattr(settings_obj, "ocr_only_min_chars", 120)),
            ocr_only_min_confidence=float(getattr(settings_obj, "ocr_only_min_confidence", 0.75)),
            hybrid_min_chars=int(getattr(settings_obj, "ocr_hybrid_min_chars", 30)),
        )
    else:
        decision = VisionRouteDecision("vision_only", "ocr_first_disabled")

    if callable(log_fn):
        try:
            log_fn(
                "vision_route",
                (
                    f"route={decision.route}, reason={decision.reason}, "
                    f"ocr_chars={int(getattr(ocr_result, 'char_count', 0))}, "
                    f"ocr_conf={float(getattr(ocr_result, 'average_confidence', 0.0)):.2f}"
                ),
            )
        except Exception:
            pass

    vision_summary = ""
    mm_reason = "ocr_only"
    mm_elapsed_ms = 0
    mm_summary_len = 0
    if decision.route != "ocr_only":
        ocr_context = ""
        if decision.route == "vision+ocr":
            ocr_context = ocr_result.to_prompt_text(
                max_chars=int(getattr(settings_obj, "ocr_context_max_chars", 1200))
            )
        vision_summary, mm_reason, mm_elapsed_ms, mm_summary_len = _request_multimodal_summary(
            llm_client,
            image,
            settings_obj,
            runtime_state,
            log_fn,
            ocr_context=ocr_context,
            route=decision.route,
            respect_min_interval=respect_vision_interval,
            vision_enabled=vision_enabled,
            focus_query=focus_query,
        )

    screen_context, mode = _compose_screen_context(vision_summary, ocr_text)
    return {
        "screen_context": screen_context,
        "mode": mode,
        "vision_route": decision.route,
        "vision_route_reason": decision.reason,
        "mm_reason": mm_reason,
        "mm_elapsed_ms": mm_elapsed_ms,
        "mm_summary_len": mm_summary_len,
        "ocr_hash": str(getattr(ocr_result, "text_hash", "") or ""),
        "ocr_text": ocr_text,
        "ocr_chars": int(getattr(ocr_result, "char_count", 0)),
        "ocr_confidence": float(getattr(ocr_result, "average_confidence", 0.0)),
    }


def _build_window_resolver(settings_obj, excluded_pids, *, monitor_index: int = 0):
    """Build a topmost-foreign-window resolver, or None when unsupported."""
    try:
        from desktop_pet.vision.game_window import GameWindowResolver
    except Exception:
        return None
    try:
        return GameWindowResolver(
            excluded_pids={int(pid) for pid in excluded_pids if int(pid) > 0},
            monitor_index=int(monitor_index),
            target_title=str(getattr(settings_obj, "vision_target_title", "") or ""),
        )
    except Exception:
        return None


def _subprocess_window_resolver(
    settings_obj,
    owner_pid: int = 0,
    extra_pids=None,
    target_title: str = "",
):
    """Lazily create the resolver used inside the scan worker process.

    Separate instance from the main process: the subprocess performs its own
    capture, so it needs its own resolver (and its own cache). Every pet-owned
    process must be excluded -- the pet UI lives in the owner process and the
    live2d renderer is yet another process. Miss one and the scan happily
    captures our own window, which surfaces as "OCR works but reads our UI".
    """
    global _SUBPROC_WINDOW_RESOLVER
    monitor_index = int(getattr(settings_obj, "vision_window_monitor_index", 0) or 0)
    excluded = {os.getpid()}
    if int(owner_pid) > 0:
        excluded.add(int(owner_pid))
    for pid in extra_pids or ():
        try:
            value = int(pid)
        except (TypeError, ValueError):
            continue
        if value > 0:
            excluded.add(value)
    if _SUBPROC_WINDOW_RESOLVER is not None:
        _SUBPROC_WINDOW_RESOLVER.set_excluded_pids(excluded)
        _SUBPROC_WINDOW_RESOLVER.set_monitor_index(monitor_index)
        _SUBPROC_WINDOW_RESOLVER.set_target_title(target_title)
        return _SUBPROC_WINDOW_RESOLVER
    resolver = _build_window_resolver(settings_obj, excluded, monitor_index=monitor_index)
    if resolver is not None:
        resolver.set_target_title(target_title)
    _SUBPROC_WINDOW_RESOLVER = resolver
    return resolver


_SUBPROC_WINDOW_RESOLVER = None


def run_scan_pipeline_subprocess(
    scan_monitor_index: int,
    scan_region: tuple[int, int, int, int] | None,
    ocr_cpu_threads: int = 0,
    ocr_cpu_affinity_count: int = 0,
    ocr_max_edge: int = 0,
    visual_novel_mode: bool = False,
    owner_pid: int = 0,
    owned_pids: tuple[int, ...] = (),
    ocr_region: tuple[int, int, int, int] | None = None,
    vision_region: tuple[int, int, int, int] | None = None,
    capture_mode: str = "",
    text_ratio: float | None = None,
    target_title: str = "",
) -> dict:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        proc.nice(
            psutil.NORMAL_PRIORITY_CLASS
            if bool(visual_novel_mode)
            else psutil.BELOW_NORMAL_PRIORITY_CLASS
        )
        if int(ocr_cpu_affinity_count) > 0 and hasattr(proc, "cpu_affinity"):
            current = list(proc.cpu_affinity())
            if current:
                proc.cpu_affinity(current[: int(ocr_cpu_affinity_count)])
    except Exception:
        pass

    if int(ocr_cpu_threads) > 0:
        thread_value = str(int(ocr_cpu_threads))
        os.environ["OMP_NUM_THREADS"] = thread_value
        os.environ["MKL_NUM_THREADS"] = thread_value
        os.environ["OPENBLAS_NUM_THREADS"] = thread_value
        os.environ["NUMEXPR_NUM_THREADS"] = thread_value
        os.environ["ORT_NUM_THREADS"] = thread_value

    from desktop_pet.vision.ocr import (
        extract_ocr_result,
        filter_visual_novel_ocr_result,
        get_ocr_runtime_status,
    )
    from desktop_pet.vision.scene_analyzer import analyze_scene
    _init_subprocess_scan_runtime()

    subproc_settings = _SUBPROC_MM_RUNTIME.get("settings")
    if text_ratio is None:
        text_ratio = float(getattr(subproc_settings, "visual_novel_text_ratio", 0.58))
    selected_capture_mode = str(capture_mode or "").strip().lower()
    if not selected_capture_mode:
        selected_capture_mode = "window" if bool(visual_novel_mode) else "screen"
        if bool(visual_novel_mode) and subproc_settings is not None:
            selected_capture_mode = str(
                getattr(subproc_settings, "vision_capture_mode", "window") or "window"
            )
    resolver = _subprocess_window_resolver(
        subproc_settings,
        owner_pid,
        owned_pids,
        target_title=target_title,
    )

    if bool(visual_novel_mode):
        image, window_rect, _vision_image, capture_source = _resolve_vn_capture(
            mode=selected_capture_mode,
            resolver=resolver,
            ocr_region=ocr_region,
            vision_region=vision_region,
            monitor_index=int(scan_monitor_index),
            text_ratio=float(text_ratio),
        text_bottom_margin=float(
            getattr(subproc_settings, "vn_text_band_bottom_margin", 0.0) or 0.0
        ),
        )
        if capture_source == "target_missing":
            return {
                "ocr_ok": True,
                "ocr_info": "target_missing",
                "capture_source": capture_source,
                "capture_title": "",
                "capture_process_name": "",
                "capture_hwnd": 0,
                "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
                "ocr_chars": 0,
                "ocr_note": "target_missing",
            }
    else:
        from desktop_pet.vision.capture import capture_primary_screen

        window_rect = None
        capture_source = "screen"
        image = capture_primary_screen(monitor_index=scan_monitor_index, region=scan_region)

    scan_target = (
        int(scan_monitor_index),
        tuple(scan_region) if scan_region is not None else None,
        bool(visual_novel_mode),
        capture_source,
        None if window_rect is None else window_rect.as_region(),
        str(getattr(window_rect, "title", "") or ""),
    )
    if _SUBPROC_SCAN_CACHE.get("scan_target") != scan_target:
        _SUBPROC_SCAN_CACHE["scan_target"] = scan_target
        _SUBPROC_SCAN_CACHE["fingerprint"] = None
        _SUBPROC_SCAN_CACHE["brightness_fingerprint"] = None
        _SUBPROC_SCAN_CACHE["refreshed_at"] = 0.0
        _SUBPROC_SCAN_CACHE["ocr_hash"] = ""
        _SUBPROC_SCAN_CACHE["ocr_text"] = ""
        _SUBPROC_SCAN_CACHE["ocr_confidence"] = 0.0
        _SUBPROC_SCAN_CACHE["screen_context"] = ""
        _SUBPROC_SCAN_CACHE["scene_summary"] = ""
        _SUBPROC_SCAN_CACHE["vision_route"] = "none"
        _SUBPROC_SCAN_CACHE["capture_rect"] = None
        _SUBPROC_SCAN_CACHE["capture_source"] = ""
        _SUBPROC_SCAN_CACHE["capture_title"] = ""
        _SUBPROC_SCAN_CACHE["capture_process_name"] = ""
        _SUBPROC_SCAN_CACHE["capture_hwnd"] = 0
        _SUBPROC_SCAN_CACHE["capture_warning"] = ""
        _SUBPROC_SCAN_CACHE["vn_ocr_filter"] = {}
        _SUBPROC_SCAN_CACHE["settle_active"] = False
    sampled_at = time.monotonic()
    fingerprint, text_diff = _text_change_metrics(image, visual_novel=bool(visual_novel_mode))
    brightness_fingerprint = (
        _frame_fingerprint(image, text_sensitive=True) if visual_novel_mode else None
    )
    prev_fingerprint = _SUBPROC_SCAN_CACHE.get("fingerprint")
    if isinstance(prev_fingerprint, bytes):
        diff_ratio = text_diff(fingerprint, prev_fingerprint)
        if visual_novel_mode:
            diff_threshold = float(
                getattr(subproc_settings, "ocr_text_ink_change_threshold", 0.12)
            )
            previous_brightness = _SUBPROC_SCAN_CACHE.get("brightness_fingerprint")
            refreshed_at = float(_SUBPROC_SCAN_CACHE.get("refreshed_at", 0.0) or 0.0)
            reuse_frame = bool(
                isinstance(brightness_fingerprint, bytes)
                and isinstance(previous_brightness, bytes)
                and _vn_stable_cache_can_reuse(
                    fingerprint,
                    prev_fingerprint,
                    brightness_fingerprint,
                    previous_brightness,
                    ink_threshold=diff_threshold,
                    brightness_threshold=float(
                        getattr(subproc_settings, "ocr_text_change_threshold", 0.008)
                    ),
                    cache_age_sec=time.monotonic() - refreshed_at,
                    has_text=bool(_SUBPROC_SCAN_CACHE.get("ocr_text")),
                )
            )
        else:
            diff_threshold = 0.04
            reuse_frame = diff_ratio < diff_threshold
        if reuse_frame and (visual_novel_mode or str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")).strip()):
            return {
                "ocr_ok": True,
                "ocr_info": "cache_reuse",
                "mode": str(_SUBPROC_SCAN_CACHE.get("mode", "none")),
                "screen_context": str(_SUBPROC_SCAN_CACHE.get("screen_context", "")),
                "ocr_text": str(_SUBPROC_SCAN_CACHE.get("ocr_text", "")),
                "ocr_confidence": float(_SUBPROC_SCAN_CACHE.get("ocr_confidence", 0.0) or 0.0),
                "scene_summary": str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")),
                "scene_should_comment": bool(_SUBPROC_SCAN_CACHE.get("scene_should_comment", False)),
                "mm_reason": str(_SUBPROC_SCAN_CACHE.get("mm_reason", "off")),
                "mm_elapsed_ms": int(_SUBPROC_SCAN_CACHE.get("mm_elapsed_ms", 0) or 0),
                "mm_summary_len": int(_SUBPROC_SCAN_CACHE.get("mm_summary_len", 0) or 0),
                "vision_route": str(_SUBPROC_SCAN_CACHE.get("vision_route", "cache")),
                "capture_rect": _SUBPROC_SCAN_CACHE.get("capture_rect"),
                "capture_source": str(_SUBPROC_SCAN_CACHE.get("capture_source", "")),
                "capture_title": str(_SUBPROC_SCAN_CACHE.get("capture_title", "")),
                "capture_process_name": str(_SUBPROC_SCAN_CACHE.get("capture_process_name", "")),
                "capture_hwnd": int(_SUBPROC_SCAN_CACHE.get("capture_hwnd", 0) or 0),
                "capture_warning": str(_SUBPROC_SCAN_CACHE.get("capture_warning", "")),
                "ocr_chars": len(str(_SUBPROC_SCAN_CACHE.get("ocr_text", "") or "")),
                "ocr_note": "cache_reuse",
                "sampled_at": sampled_at,
                "frame_stable": True,
                "ocr_elapsed_ms": 0,
                "reused_cache": True,
            }

    if visual_novel_mode and _vn_text_is_settling(_SUBPROC_SCAN_CACHE, fingerprint, brightness_fingerprint, sampled_at):
        return {
            "ocr_ok": True, "ocr_info": "text_settling", "mode": "none",
            "screen_context": "", "ocr_text": "", "scene_summary": "",
            "scene_should_comment": False, "vision_route": "none",
            "capture_source": capture_source,
            "capture_title": str(getattr(window_rect, "title", "") or ""),
            "sampled_at": sampled_at, "scan_deferred": True,
            "capture_process_name": str(getattr(window_rect, "process_name", "") or ""),
            "capture_hwnd": int(getattr(window_rect, "hwnd", 0) or 0),
            "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
            "ocr_chars": 0, "ocr_note": "text_settling",
            "frame_stable": False, "ocr_elapsed_ms": 0, "reused_cache": False,
        }

    ocr_ok, ocr_info = get_ocr_runtime_status()
    ocr_started_at = time.perf_counter()
    if visual_novel_mode and image.info.get("vn_ui_mask"):
        ocr_info += f"+ui_mask={image.info['vn_ui_mask']}:{image.info['vn_ui_mask_ms']:g}ms"
    if visual_novel_mode:
        text_edge = int(getattr(subproc_settings, "ocr_text_max_edge", 800) or 800)
        text_image = _resize_for_ocr(image, text_edge)
        ocr_result = extract_ocr_result(text_image)
    else:
        ocr_image = _resize_for_ocr(image, int(ocr_max_edge))
        ocr_result = extract_ocr_result(ocr_image)
    if visual_novel_mode:
        ocr_result, filter_note = filter_visual_novel_ocr_result(
            ocr_result,
            _SUBPROC_SCAN_CACHE.setdefault("vn_ocr_filter", {}),
        )
        if filter_note:
            ocr_info = f"{ocr_info}+vn_filter({filter_note})"
    ocr_elapsed_ms = round((time.perf_counter() - ocr_started_at) * 1000)
    previous_ocr_hash = str(_SUBPROC_SCAN_CACHE.get("ocr_hash", "") or "")
    if (
        ocr_result.text_hash
        and ocr_result.text_hash == previous_ocr_hash
        and str(_SUBPROC_SCAN_CACHE.get("vision_route", "")) == "ocr_only"
        and str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")).strip()
    ):
        _SUBPROC_SCAN_CACHE["fingerprint"] = fingerprint
        _SUBPROC_SCAN_CACHE["brightness_fingerprint"] = brightness_fingerprint
        _SUBPROC_SCAN_CACHE["refreshed_at"] = time.monotonic()
        return {
            "ocr_ok": ocr_ok,
            "ocr_info": "ocr_hash_reuse",
            "mode": str(_SUBPROC_SCAN_CACHE.get("mode", "none")),
            "screen_context": str(_SUBPROC_SCAN_CACHE.get("screen_context", "")),
            "ocr_text": str(_SUBPROC_SCAN_CACHE.get("ocr_text", "")),
            "ocr_confidence": float(_SUBPROC_SCAN_CACHE.get("ocr_confidence", 0.0) or 0.0),
            "scene_summary": str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")),
            "scene_should_comment": bool(_SUBPROC_SCAN_CACHE.get("scene_should_comment", False)),
            "mm_reason": "ocr_hash_cache",
            "mm_elapsed_ms": 0,
            "mm_summary_len": int(_SUBPROC_SCAN_CACHE.get("mm_summary_len", 0) or 0),
            "vision_route": "cache",
            "capture_rect": _SUBPROC_SCAN_CACHE.get("capture_rect"),
            "capture_source": str(_SUBPROC_SCAN_CACHE.get("capture_source", "")),
            "capture_title": str(_SUBPROC_SCAN_CACHE.get("capture_title", "")),
            "capture_process_name": str(_SUBPROC_SCAN_CACHE.get("capture_process_name", "")),
            "capture_hwnd": int(_SUBPROC_SCAN_CACHE.get("capture_hwnd", 0) or 0),
            "capture_warning": str(_SUBPROC_SCAN_CACHE.get("capture_warning", "")),
            "ocr_chars": len(str(_SUBPROC_SCAN_CACHE.get("ocr_text", "") or "")),
            "ocr_note": "ocr_hash_reuse",
            "sampled_at": sampled_at,
            "frame_stable": False,
            "ocr_elapsed_ms": ocr_elapsed_ms,
            "reused_cache": True,
        }

    mm_client = _SUBPROC_MM_RUNTIME.get("client")
    subproc_settings = _SUBPROC_MM_RUNTIME.get("settings")
    if subproc_settings is not None:
        pipeline = _build_ocr_first_context(
            mm_client,
            image,
            ocr_result,
            subproc_settings,
            _SUBPROC_MM_RUNTIME,
            None,
            respect_vision_interval=True,
            # Visual requests are issued asynchronously from the main process so
            # a slow vision API can never stall the scan worker.
            vision_enabled=False,
        )
    else:
        screen_context, mode = _compose_screen_context("", ocr_result.text)
        pipeline = {
            "screen_context": screen_context,
            "mode": mode,
            "vision_route": "ocr_only",
            "mm_reason": "off",
            "mm_elapsed_ms": 0,
            "mm_summary_len": 0,
            "ocr_hash": ocr_result.text_hash,
            "ocr_text": ocr_result.text,
            "ocr_confidence": ocr_result.average_confidence,
        }

    screen_context = str(pipeline["screen_context"])
    mode = str(pipeline["mode"])
    scene = analyze_scene(screen_context)

    _SUBPROC_SCAN_CACHE["fingerprint"] = fingerprint
    _SUBPROC_SCAN_CACHE["brightness_fingerprint"] = brightness_fingerprint
    _SUBPROC_SCAN_CACHE["refreshed_at"] = time.monotonic()
    _SUBPROC_SCAN_CACHE["ocr_hash"] = str(pipeline.get("ocr_hash", ""))
    _SUBPROC_SCAN_CACHE["ocr_text"] = str(pipeline.get("ocr_text", ""))
    _SUBPROC_SCAN_CACHE["ocr_confidence"] = float(pipeline.get("ocr_confidence", 0.0) or 0.0)
    _SUBPROC_SCAN_CACHE["screen_context"] = screen_context
    _SUBPROC_SCAN_CACHE["scene_summary"] = scene.summary
    _SUBPROC_SCAN_CACHE["scene_should_comment"] = scene.should_comment
    _SUBPROC_SCAN_CACHE["mode"] = mode
    _SUBPROC_SCAN_CACHE["mm_reason"] = str(pipeline.get("mm_reason", "off"))
    _SUBPROC_SCAN_CACHE["mm_elapsed_ms"] = int(pipeline.get("mm_elapsed_ms", 0) or 0)
    _SUBPROC_SCAN_CACHE["mm_summary_len"] = int(pipeline.get("mm_summary_len", 0) or 0)
    _SUBPROC_SCAN_CACHE["vision_route"] = str(pipeline.get("vision_route", "none"))
    _SUBPROC_SCAN_CACHE["capture_rect"] = None if window_rect is None else window_rect.as_region()
    _SUBPROC_SCAN_CACHE["capture_source"] = capture_source
    _SUBPROC_SCAN_CACHE["capture_title"] = str(getattr(window_rect, "title", "") or "")
    _SUBPROC_SCAN_CACHE["capture_process_name"] = str(
        getattr(window_rect, "process_name", "") or ""
    )
    _SUBPROC_SCAN_CACHE["capture_hwnd"] = int(getattr(window_rect, "hwnd", 0) or 0)
    _SUBPROC_SCAN_CACHE["capture_warning"] = str(
        getattr(_resolve_vn_capture, "warning", "") or ""
    )

    return {
        "ocr_ok": ocr_ok,
        "ocr_info": ocr_info,
        "mode": mode,
        "screen_context": screen_context,
        "ocr_text": str(pipeline.get("ocr_text", "")),
        "ocr_confidence": float(pipeline.get("ocr_confidence", 0.0) or 0.0),
        "scene_summary": scene.summary,
        "scene_should_comment": scene.should_comment,
        "mm_reason": str(pipeline.get("mm_reason", "off")),
        "mm_elapsed_ms": int(pipeline.get("mm_elapsed_ms", 0) or 0),
        "mm_summary_len": int(pipeline.get("mm_summary_len", 0) or 0),
        "vision_route": str(pipeline.get("vision_route", "none")),
        "capture_rect": None if window_rect is None else window_rect.as_region(),
        "capture_source": capture_source,
        "capture_title": str(getattr(window_rect, "title", "") or ""),
        "capture_process_name": str(getattr(window_rect, "process_name", "") or ""),
        "capture_hwnd": int(getattr(window_rect, "hwnd", 0) or 0),
        "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
        "ocr_chars": int(getattr(ocr_result, "char_count", 0)),
        "ocr_note": ocr_info,
        "sampled_at": sampled_at,
        "frame_stable": False,
        "ocr_elapsed_ms": ocr_elapsed_ms,
        "reused_cache": False,
    }


def _warm_scan_worker_ocr() -> None:
    """Initialize ONNX sessions and run one throwaway inference before scanning."""
    started = time.perf_counter()
    try:
        from PIL import Image, ImageDraw, ImageFont

        from desktop_pet.vision.ocr import extract_ocr_result, warmup_ocr_engine

        _init_subprocess_scan_runtime()
        ok, info = warmup_ocr_engine()
        if ok:
            sample = Image.new("RGB", (800, 160), (255, 255, 255))
            draw = ImageDraw.Draw(sample)
            try:
                font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 36)
            except Exception:
                font = ImageFont.load_default()
            draw.text((24, 48), "OCR 预热文本 Test 123", fill=(0, 0, 0), font=font)
            extract_ocr_result(sample)
        print(
            "[SCAN WORKER] OCR warmup:",
            f"{'ok' if ok else 'fail'} | {info} | {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    except Exception as exc:
        print(f"[SCAN WORKER] OCR warmup failed: {exc}", flush=True)


def run_scan_pipeline_worker_loop(task_queue, result_queue, ready_event=None) -> None:
    if ready_event is not None:
        try:
            _warm_scan_worker_ocr()
        finally:
            ready_event.set()
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id = int(task.get("task_id", 0))
        monitor_index = int(task.get("scan_monitor_index", 0))
        scan_region = task.get("scan_region")
        ocr_cpu_threads = int(task.get("ocr_cpu_threads", 0))
        ocr_cpu_affinity_count = int(task.get("ocr_cpu_affinity_count", 0))
        ocr_max_edge = int(task.get("ocr_max_edge", 0))
        visual_novel_mode = bool(task.get("visual_novel_mode", False))
        capture_mode = str(task.get("capture_mode", "") or "")
        text_ratio = task.get("text_ratio")
        target_title = str(task.get("target_title", "") or "")
        owner_pid = int(task.get("owner_pid", 0))
        owned_pids = tuple(task.get("owned_pids", ()) or ())
        raw_region = task.get("ocr_region")
        ocr_region = None
        if isinstance(raw_region, (list, tuple)) and len(raw_region) == 4:
            try:
                ocr_region = tuple(int(value) for value in raw_region)
            except (TypeError, ValueError):
                ocr_region = None
        raw_vision = task.get("vision_region")
        vision_region = None
        if isinstance(raw_vision, (list, tuple)) and len(raw_vision) == 4:
            try:
                vision_region = tuple(int(value) for value in raw_vision)
            except (TypeError, ValueError):
                vision_region = None
        started_ts = time.time()
        try:
            result = run_scan_pipeline_subprocess(
                monitor_index,
                scan_region,
                ocr_cpu_threads=ocr_cpu_threads,
                ocr_cpu_affinity_count=ocr_cpu_affinity_count,
                ocr_max_edge=ocr_max_edge,
                visual_novel_mode=visual_novel_mode,
                owner_pid=owner_pid,
                owned_pids=owned_pids,
                ocr_region=ocr_region,
                vision_region=vision_region,
                capture_mode=capture_mode,
                text_ratio=None if text_ratio is None else float(text_ratio),
                target_title=target_title,
            )
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": True,
                    "result": result,
                    "duration_sec": max(0.0, time.time() - started_ts),
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": False,
                    "error": str(exc),
                    "duration_sec": max(0.0, time.time() - started_ts),
                }
            )


def _enable_utf8_streams() -> None:
    """Make stdout/stderr UTF-8 so logs survive the Windows console code page.

    The live2d-py child prints emoji in its banner. Under a GBK code page the
    parent raised "'gbk' codec can't encode character '\U0001f43e'" while piping
    it, which both lost the log and mangled every Chinese line in the console.
    ``errors="replace"`` keeps a stray glyph from ever killing a log write.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _warn_if_qt_loaded_too_early() -> None:
    """Detect the unrecoverable PyQt6-before-onnxruntime ordering mistake.

    Qt's bundled DLLs break initialisation of onnxruntime and torch in the same
    process (WinError 1114), and the failure cannot be undone by retrying or
    reloading. Anything that imports PyQt6 before ``main()`` -- an extra launch
    wrapper, an IDE runner, a helper module -- will therefore disable OCR,
    semantic memory and VOICEVOX translation with a cryptic message. Report it
    in terms of the actual cause.
    """
    premature = sorted(
        name
        for name in sys.modules
        if name == "PyQt6" or name.startswith("PyQt6.")
    )
    if not premature:
        return
    if all(getattr(sys.modules.get(name), "__version__", None) for name in ("onnxruntime", "torch")):
        return
    print(
        "[STARTUP] WARNING: PyQt6 was already imported before native ML runtimes "
        f"({', '.join(premature[:3])}{'...' if len(premature) > 3 else ''})."
    )
    print(
        "[STARTUP] WARNING: native ML runtime initialisation may fail in this "
        "process, affecting OCR, semantic memory and VOICEVOX translation. "
        "Start the app with `python main.py` (or pet.bat) instead "
        "of importing Qt before main()."
    )


def _warmup_native_dependencies() -> None:
    """Load native ML runtimes before any PyQt6 import.

    PyQt6 ships its own Qt DLL set. Once it has been imported, initialising
    onnxruntime or torch in the same process fails with
    ``DLL load failed ... 动态链接库(DLL)初始化例程失败`` (WinError 1114).
    Loading them first is the only reliable ordering, so this must run before
    ``PyQt6`` is touched -- which also means Qt imports have to stay inside
    ``main()`` rather than at module scope.
    """
    try:
        import onnxruntime

        print(f"[STARTUP] onnxruntime preloaded: {onnxruntime.__version__}")
    except Exception as exc:
        print(f"[STARTUP] onnxruntime preload failed: {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"[STARTUP] torch preloaded: {torch.__version__}")
    except Exception as exc:
        print(f"[STARTUP] torch preload failed (VOICEVOX translation disabled): {type(exc).__name__}")


def main() -> int:
    # 先预热 OCR，避免 Qt 相关原生库先加载导致 onnxruntime DLL 冲突。
    _enable_utf8_streams()
    _warn_if_qt_loaded_too_early()
    _enable_per_monitor_dpi_awareness()
    ocr_ok, ocr_info = warmup_ocr_engine()
    print(f"[STARTUP] OCR status: {'ok' if ocr_ok else 'fail'} | {ocr_info}")
    _warmup_native_dependencies()

    base_dir = Path(__file__).resolve().parent
    settings = load_settings(base_dir)
    from desktop_pet.audio.speech import SpeechService

    def translate_tts_with_api(text: str) -> str:
        try:
            translated = llm_client.chat(
                user_text=text,
                system_prompt=SYSTEM_VOICEVOX_TRANSLATE_PROMPT,
            ).strip()
        except Exception:
            return ""
        if not translated or translated.startswith("[离线回声]"):
            return ""
        return translated[: settings.tts_translation_max_chars]

    speech = SpeechService(
        settings,
        translation_api_fallback=translate_tts_with_api,
    )

    tts_translation_warmed = threading.Event()

    def warmup_tts_translation() -> None:
        try:
            status = speech.warmup_translation()
            print(f"[STARTUP] TTS translation: {'ok' if status.ready else 'fail'} | {status.detail}")
        finally:
            tts_translation_warmed.set()

    threading.Thread(target=warmup_tts_translation, name="tts-warmup", daemon=True).start()
    configure_webengine_render_mode(settings.webengine_gpu_mode)

    live2d_py_process = None
    live2d_py_retry_used = False
    live2d_py_retry_count = [0]
    live2d_py_max_restarts = 3
    live2d_shutdown_requested = {"flag": 0, "ts": 0.0}
    # Clear any shutdown marker left behind by a previous run; otherwise the
    # renderer would exit the moment it starts.
    for _stale in (
        base_dir / "data" / "live2d_py_shutdown_ack.json",
        base_dir / "data" / "live2d_py_target_rect.json",
    ):
        try:
            if _stale.exists():
                _stale.unlink()
        except Exception:
            pass
    filter_motion_noise_log = os.getenv("LIVE2D_FILTER_MOTION_NOISE_LOG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    filter_live2d_startup_verbose = os.getenv("LIVE2D_FILTER_STARTUP_VERBOSE_LOG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _pipe_live2d_stream(stream, *, name: str) -> None:
        if stream is None:
            return
        try:
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                lowered = line.lower()
                noisy_motion = ("start motion" in lowered) and (("can't" in lowered) or ("cant" in lowered))
                if filter_motion_noise_log and noisy_motion:
                    continue
                if filter_live2d_startup_verbose and (
                    "create buffer:" in lowered
                    or "delete buffer:" in lowered
                    or "load motion:" in lowered
                ):
                    continue
                print(f"[LIVE2D-PY/{name}] {line}")
        except Exception as exc:
            print(f"[STARTUP] Live2D-py log pipe {name} failed: {exc}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def start_live2d_py_process(*, force_gl_init: bool) -> subprocess.Popen | None:
        child_env = os.environ.copy()
        # live2d-py's Cubism binding requires an explicit EGL/GL init in its
        # in-process context. Launching without it makes the native SDK fault
        # (0xC0000005) a moment after "normalized empty motion group", which is
        # what produced the "0x00000000 ... 内存不能为 written" dialog. Always
        # request it; the flag keeps the parameter meaningful for callers.
        child_env["LIVE2D_PY_FORCE_GL_INIT"] = "true"
        # Force UTF-8 on the child's streams. Its banner contains emoji; under the
        # console's GBK code page the child died with "'gbk' codec can't encode
        # character '\U0001f43e'", which made the renderer crash and the watchdog
        # restart it on every launch.
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "desktop_pet.live2d.live2d_py_runner",
            "--model",
            settings.live2d_model_json,
            "--width",
            str(settings.live2d_py_window_width),
            "--height",
            str(settings.live2d_py_window_height),
            "--title",
            "Live2D-py",
            "--borderless",
            "1",
            "--self-topmost",
            "0",
            "--window-drag",
            "0",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(base_dir.parent),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if proc.stdout is not None:
            threading.Thread(
                target=_pipe_live2d_stream,
                args=(proc.stdout,),
                kwargs={"name": "STDOUT"},
                daemon=True,
            ).start()
        if proc.stderr is not None:
            threading.Thread(
                target=_pipe_live2d_stream,
                args=(proc.stderr,),
                kwargs={"name": "STDERR"},
                daemon=True,
            ).start()
        return proc

    if settings.enable_live2d_py:
        try:
            live2d_py_process = start_live2d_py_process(force_gl_init=False)
            # live2d-py mode keeps WebEngine path disabled to avoid dual-render contention.
            settings.enable_live2d = False
            print(f"[STARTUP] Live2D-py process started (pid={live2d_py_process.pid})")
        except Exception as exc:
            print(f"[STARTUP] Live2D-py start failed: {exc}")

    from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QCursor, QGuiApplication
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

    from desktop_pet.core.scheduler import ScanScheduler
    from desktop_pet.llm.comment_engine import CommentEngine
    from desktop_pet.llm.client import LLMClient
    from desktop_pet.llm.dialog_manager import DialogManager
    from desktop_pet.llm.semantic_attention import SemanticAttentionRouter
    from desktop_pet.llm.visual_novel import DedupCalibrator, VisualNovelStoryLibrary, VisualNovelTracker
    from desktop_pet.policy.comment_policy import can_emit_comment
    from desktop_pet.ui.chat_panel import ChatPanel
    from desktop_pet.ui.pet_window import DesktopPet
    from desktop_pet.ui.region_selector import RegionSelectOverlay
    from desktop_pet.vision.capture import capture_primary_screen, find_monitor_index, get_monitor_geometry
    from desktop_pet.vision.multimodal import describe_screen_image, describe_screen_image_compat
    from desktop_pet.vision.ocr import (
        extract_ocr_result,
        filter_visual_novel_ocr_result,
        get_ocr_runtime_status,
    )
    from desktop_pet.vision.scene_analyzer import analyze_scene

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    def maybe_retry_live2d_py() -> None:
        """Restart the renderer if it faults. Native GL faults are survivable
        only by restarting the process, so the watchdog retries a few times
        instead of giving up after the first failure."""
        nonlocal live2d_py_process, live2d_py_retry_used
        if live2d_py_process is None:
            return
        if live2d_py_process.poll() is None:
            return

        exit_code = live2d_py_process.poll()
        if int(live2d_shutdown_requested["flag"]):
            return
        if live2d_py_retry_count[0] >= live2d_py_max_restarts:
            if not live2d_py_retry_used:
                live2d_py_retry_used = True
                print(
                    f"[STARTUP] Live2D-py crashed {live2d_py_max_restarts} times "
                    f"(last code={exit_code}); giving up and continuing without the model"
                )
            return

        live2d_py_retry_count[0] += 1
        live2d_py_retry_used = True
        print(
            "[STARTUP] Live2D-py exited with code="
            f"{exit_code} (hex={exit_code & 0xFFFFFFFF:08X}), restart "
            f"{live2d_py_retry_count[0]}/{live2d_py_max_restarts}"
        )
        try:
            live2d_py_process = start_live2d_py_process(force_gl_init=True)
            print(f"[STARTUP] Live2D-py restarted (pid={live2d_py_process.pid})")
        except Exception as exc:
            print(f"[STARTUP] Live2D-py restart failed: {exc}")

    if settings.enable_live2d_py:
        QTimer.singleShot(2000, maybe_retry_live2d_py)

    print(
        "[STARTUP] Scan target:",
        f"monitor={settings.scan_monitor_index if settings.scan_monitor_index else 'virtual-desktop'}",
        f"region={settings.scan_region if settings.scan_region else 'full-target'}",
    )
    if settings.enable_visual_novel_mode:
        roi_mode = "manual-region" if settings.scan_region else "bottom-42-percent"
        print(f"[STARTUP] Visual novel mode: enabled | roi={roi_mode}")

    llm_client = LLMClient(settings)
    dialog_memory_path = base_dir / "data" / "chat_long_memory.json"
    semantic_attention = SemanticAttentionRouter(
        enabled=settings.enable_semantic_attention,
        model_path=Path(settings.semantic_attention_model_path),
        top_k=settings.semantic_attention_top_k,
        min_score=settings.semantic_attention_min_score,
        max_length=settings.semantic_attention_max_length,
        cache_size=settings.semantic_attention_cache_size,
        cpu_threads=settings.semantic_attention_cpu_threads,
    )
    visual_novel_runtime = {"enabled": bool(settings.enable_visual_novel_mode)}
    visual_novel_library = VisualNovelStoryLibrary(
        base_dir / "data" / "visual_novel_stories",
        legacy_path=base_dir / "data" / "visual_novel_story.json",
    )
    dedup_calibrator = DedupCalibrator(
        Path(settings.dedup_calibration_path),
        enabled=bool(settings.dedup_calibration_enabled),
    )
    visual_novel_tracker = VisualNovelTracker(
        visual_novel_library.active_path,
        similarity_fn=semantic_attention.similarity,
        similarities_fn=semantic_attention.similarities,
        min_context_similarity=settings.visual_novel_min_context_similarity,
        max_facts=int(settings.visual_novel_max_facts),
        dedup_ngram_merge=float(settings.dedup_ngram_merge),
        dedup_ngram_guard=float(settings.dedup_ngram_guard),
        dedup_semantic_merge=float(settings.dedup_semantic_merge),
        calibrator=dedup_calibrator,
    )
    print(f"[STARTUP] Visual novel story cache: {visual_novel_library.active_name}")
    print(
        "[STARTUP] Fact dedup:",
        f"ngram>={settings.dedup_ngram_merge:.2f}",
        f"guard>={settings.dedup_ngram_guard:.2f}",
        f"cosine>={settings.dedup_semantic_merge:.2f}",
        f"calibration={dedup_calibrator.status}",
    )

    def provide_visual_novel_planner_context() -> str:
        if not visual_novel_runtime["enabled"]:
            return ""
        return visual_novel_tracker.build_planner_hint(max_chars=600)

    dialog = DialogManager(
        llm_client,
        memory_path=dialog_memory_path,
        tutor_enabled=settings.enable_tutor_persona,
        semantic_attention=semantic_attention,
        web_soft_deadline_sec=settings.web_search_soft_deadline_sec,
        web_hard_deadline_sec=settings.web_search_hard_deadline_sec,
        web_circuit_failure_threshold=settings.web_search_circuit_failure_threshold,
        web_circuit_cooldown_sec=settings.web_search_circuit_cooldown_sec,
        web_max_results=settings.web_search_max_results,
        web_context_max_chars=settings.web_search_context_max_chars,
        baidu_ai_search_api_key=settings.baidu_ai_search_api_key,
        long_memory_limit=settings.long_memory_limit,
        long_memory_context_window=settings.long_memory_context_window,
        visual_novel_context_provider=visual_novel_tracker.retrieve_for_query,
        visual_novel_planner_context_provider=provide_visual_novel_planner_context,
    )
    if settings.enable_semantic_attention:
        def _warmup_semantic_attention() -> None:
            # Both backends first import transformers' lazy AutoTokenizer.
            tts_translation_warmed.wait()
            ok = semantic_attention.warmup()
            print(
                "[STARTUP] Semantic attention:",
                "ready" if ok else "fallback",
                f"| {semantic_attention.status}",
            )

        threading.Thread(
            target=_warmup_semantic_attention,
            name="semantic-attention-warmup",
            daemon=True,
        ).start()
    comment_engine = CommentEngine(
        llm_client,
        tutor_enabled=settings.enable_tutor_persona,
        style_weights_text=settings.auto_comment_style_weights,
        stressed_keywords_text=settings.emotion_keywords_stressed,
        positive_keywords_text=settings.emotion_keywords_positive,
        focused_keywords_text=settings.emotion_keywords_focused,
        enable_api_understanding=settings.enable_comment_api_understanding,
        embeddings_fn=semantic_attention.embeddings if settings.enable_semantic_comment_candidates else None,
    )

    def estimate_bubble_duration_ms(display_text: str) -> int:
        visual_chars = len(re.sub(r"\s+", "", display_text))

        if settings.tts_provider.lower() == "voicevox":
            chars_per_sec = 4.8
        elif settings.tts_provider.lower() == "azure":
            chars_per_sec = 5.8
        else:
            chars_per_sec = 5.5

        speech_ms = int((max(1, visual_chars) / chars_per_sec) * 1000) + 1200
        return max(2600, min(22000, speech_ms + 800))

    def _speak_async(display_text: str) -> None:
        if speech.is_muted():
            return
        speech.speak(display_text)

    def _speak_async_with_callback(display_text: str, on_start=None, on_error=None, is_valid=None) -> None:
        if is_valid is not None and not is_valid():
            return
        if speech.is_muted():
            if callable(on_error):
                on_error()
            return
        if not display_text.strip():
            if callable(on_error):
                on_error()
            return
        if is_valid is None:
            queued = speech.speak(display_text, on_start=on_start, on_error=on_error)
        else:
            queued = speech.speak(display_text, on_start=on_start, on_error=on_error, is_valid=is_valid)
        if (not queued) and callable(on_error):
            on_error()

    class _SpeechUiBridge(QObject):
        ready = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.ready.connect(self.deliver, Qt.ConnectionType.QueuedConnection)

        @pyqtSlot(object)
        def deliver(self, callback):
            callback()

    speech_ui = _SpeechUiBridge()

    def show_and_speak(display_text: str, *, role: str | None = None, duration_ms: int | None = None, is_valid=None, on_display=None) -> None:
        bubble_ms = duration_ms if duration_ms is not None else estimate_bubble_duration_ms(display_text)

        shown_lock = threading.Lock()
        shown_state = {"done": False, "started": False}
        queued_at = time.perf_counter()

        def _emit_ui_once() -> None:
            if is_valid is not None and not shown_state["started"] and not is_valid():
                return
            with shown_lock:
                if shown_state["done"]:
                    return
                shown_state["done"] = True
            if role is not None:
                chat.append_message(role, display_text)
            pet.show_comment_bubble(display_text, duration_ms=bubble_ms)
            if on_display is not None:
                on_display()
            print(f"[SPEECH-SYNC] displayed wait_ms={(time.perf_counter() - queued_at) * 1000:.1f}")

        def _ready(reason):
            if reason == "playback_started":
                shown_state["started"] = True
            print(f"[SPEECH-SYNC] {reason} wait_ms={(time.perf_counter() - queued_at) * 1000:.1f}")
            speech_ui.ready.emit(_emit_ui_once)

        if speech.is_muted() or not speech.get_status()[0]:
            _emit_ui_once()
            return
        args = (_speak_async_with_callback, display_text,
                lambda: _ready("playback_started"), lambda: _ready("audio_unavailable"))
        if is_valid is None:
            tts_executor.submit(*args)
        else:
            tts_executor.submit(*args, is_valid)

    def speak_text(text: str) -> None:
        tts_executor.submit(_speak_async, text)

    def on_web_search_toggled(enabled: bool) -> None:
        dialog.set_web_search_enabled(bool(enabled))
        if enabled:
            pet.show_comment_bubble("联网检索已开启", duration_ms=1400)
        else:
            pet.show_comment_bubble("联网检索已关闭", duration_ms=1400)

    def on_multimodal_chat_toggled(enabled: bool) -> None:
        if enabled:
            pet.show_comment_bubble("手动聊天多模态已开启", duration_ms=1600)
        else:
            pet.show_comment_bubble("手动聊天多模态已关闭", duration_ms=1600)

    chat_screen_lock = threading.Lock()

    def _capture_for_context():
        """Frame used by manual chat context, matching the scan's capture mode.

        A plain screen grab includes the always-on-top pet window, so the model
        would describe the pet instead of the game. When visual-novel mode is on
        with window capture, use the same resolved window frame as the scanner.
        """
        mode = str(state.get("vision_capture_mode", settings.vision_capture_mode))
        if bool(state.get("visual_novel_mode_enabled")) and mode == "window" and window_resolver is not None:
            _refresh_vision_exclusions()
            rect = window_resolver.resolve()
            if rect is not None:
                from desktop_pet.vision.capture import capture_window_client

                frame = capture_window_client(int(getattr(rect, "hwnd", 0) or 0))
                if frame is None:
                    from desktop_pet.vision.capture import capture_absolute_rect

                    frame = capture_absolute_rect(rect.left, rect.top, rect.width, rect.height)
                if frame is not None:
                    return frame
        return capture_primary_screen(
            monitor_index=int(state["scan_monitor_index"]),
            region=state["scan_region"],
        )

    def provide_chat_screen_context(user_text: str) -> str:
        # This callback runs after startup in ChatPanel's worker thread. It deliberately
        # captures a fresh frame instead of reusing the automatic scan cache.
        with chat_screen_lock:
            image = _capture_for_context()
            pipeline = build_screen_context(
                image,
                respect_vision_interval=False,
                vision_enabled=True,
                focus_query=user_text,
            )
        screen_context = str(pipeline.get("screen_context", "") or "").strip()
        if not screen_context:
            return ""
        mode = str(pipeline.get("mode", "none") or "none")
        return f"屏幕识别模式: {mode}\n{screen_context}"

    def provide_uploaded_image_context(image, user_text: str, ocr_text: str) -> str:
        compact_ocr = str(ocr_text or "").strip()
        vision_summary = ""
        vision_reason = "off"
        if bool(settings.enable_multimodal_vision):
            route = "vision+ocr" if compact_ocr else "vision_only"
            vision_result = describe_screen_image_compat(
                llm_client,
                image,
                timeout_sec=float(settings.mm_timeout_sec),
                max_edge=int(settings.mm_image_max_edge),
                ocr_context=compact_ocr[: int(settings.ocr_context_max_chars)],
                route=route,
                focus_query=user_text,
                content_kind="uploaded_image",
            )
            vision_summary = str(vision_result.summary or "").strip()
            vision_reason = str(vision_result.reason or "empty")

        parts = [f"上传图片识别模式: {'vision+ocr' if compact_ocr else 'vision'}"]
        if compact_ocr:
            parts.append(f"本地OCR文字:\n{compact_ocr[: int(settings.ocr_context_max_chars)]}")
        if vision_summary:
            parts.append(f"Vision视觉摘要:\n{vision_summary}")
        else:
            parts.append(f"Vision视觉摘要不可用（{vision_reason}）")
        return "\n".join(parts)

    def on_archive_state_change(active: bool, message: str) -> None:
        if active:
            hint = message.strip() or "妹妹写日记中，请不要关闭"
            pet.show_comment_bubble(hint, duration_ms=120000)
        else:
            pet.hide_comment_bubble()
            done_hint = message.strip()
            if done_hint:
                pet.show_comment_bubble(done_hint, duration_ms=1800)

    pet = DesktopPet(settings=settings)
    chat = ChatPanel(
        dialog_manager=dialog,
        on_pet_reply=speak_text,
        on_archive_state_change=on_archive_state_change,
        on_web_search_toggled=on_web_search_toggled,
        screen_context_provider=provide_chat_screen_context,
        uploaded_image_context_provider=provide_uploaded_image_context,
        on_multimodal_toggled=on_multimodal_chat_toggled,
        show_system_messages=settings.chat_show_system_messages,
        web_search_enabled=False,
        multimodal_chat_enabled=settings.enable_chat_multimodal,
        screen_context_max_chars=settings.chat_screen_context_max_chars,
    )
    dialog.set_web_search_enabled(False)

    if settings.enable_live2d_py:
        chat.enable_live2d_overlay_mode()

    tts_ok, tts_info = speech.get_status()
    print(f"[STARTUP] TTS status: {'ok' if tts_ok else 'fail'} | {tts_info}")

    base_comment_interval_sec = max(10, settings.screen_scan_interval_sec)
    if settings.scan_tick_interval_sec > 0:
        normal_scan_tick_interval_sec = settings.scan_tick_interval_sec
    else:
        normal_scan_tick_interval_sec = max(5, min(15, max(1, base_comment_interval_sec // 6)))

    if settings.scan_submit_min_interval_sec > 0:
        normal_scan_submit_min_interval_sec = settings.scan_submit_min_interval_sec
    else:
        if settings.enable_scan_subprocess:
            # Keep sample cadence close to tick interval; subprocess cache reuse avoids heavy OCR each submit.
            normal_scan_submit_min_interval_sec = max(5, normal_scan_tick_interval_sec)
        else:
            normal_scan_submit_min_interval_sec = max(
                normal_scan_tick_interval_sec,
                min(15, max(1, base_comment_interval_sec // 3)),
            )
    scan_tick_interval_sec = 1 if settings.enable_visual_novel_mode else normal_scan_tick_interval_sec
    scan_submit_min_interval_sec = (
        1 if settings.enable_visual_novel_mode else normal_scan_submit_min_interval_sec
    )

    def schedule_next_comment(now: datetime) -> datetime:
        jitter = random.randint(-30, 30)
        next_sec = max(10, base_comment_interval_sec + jitter)
        return now + timedelta(seconds=next_sec)

    next_comment_at = schedule_next_comment(datetime.now())
    scan_busy_timeout_sec = max(5, int(settings.scan_busy_timeout_sec))
    print(
        "[STARTUP] Auto comment schedule:",
        f"base={base_comment_interval_sec}s",
        f"scan_tick={scan_tick_interval_sec}s",
        f"scan_submit_min={scan_submit_min_interval_sec}s",
        f"first_due={next_comment_at.strftime('%H:%M:%S')}",
    )

    # Hand-picked OCR boxes survive restarts: re-selecting the dialogue box on
    # every launch made OCR failures hard to diagnose and annoying to fix.
    _region_state_path = base_dir / "data" / "vn_regions.json"

    def _load_persisted_regions() -> dict:
        try:
            raw = json.loads(_region_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, object] = {}
        for key in ("ocr_region", "ocr_screen_region", "vision_region"):
            value = raw.get(key)
            if isinstance(value, list) and len(value) == 4:
                try:
                    out[key] = tuple(int(item) for item in value)
                except (TypeError, ValueError):
                    continue
        title = raw.get("target_title")
        if isinstance(title, str) and title.strip():
            out["target_title"] = title.strip()
        # The game window seen last time, so a cache carried over from another
        # game can be flagged on the next launch.
        last_game = raw.get("last_game_title")
        if isinstance(last_game, str) and last_game.strip():
            out["last_game_title"] = last_game.strip()
        return out

    def _save_persisted_regions() -> None:
        try:
            _region_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {}
            for key, value in _persisted_regions.items():
                if isinstance(value, tuple) and len(value) == 4:
                    payload[key] = list(value)
                elif isinstance(value, str) and value.strip():
                    payload[key] = value.strip()
            temp = _region_state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(_region_state_path)
        except Exception:
            pass

    _persisted_regions: dict[str, object] = _load_persisted_regions()

    state = {
        "last_comment_at": None,
        "last_summary": "",
        "last_comment_text": "",
        "scan_enabled": False,
        "visual_novel_mode_enabled": visual_novel_runtime["enabled"],
        "last_pipeline_mode": "",
        "scan_monitor_index": settings.scan_monitor_index,
        "scan_region": settings.scan_region,
        "region_overlay": None,
        "next_comment_at": next_comment_at,
        "cycle_memories": [],
        "scan_future": None,
        "scan_started_at": None,
        "last_scan_submit_at": None,
        "vn_scan_due_at": 0.0,
        "scan_timeout_streak": 0,
        "scan_backoff_until": None,
        "adaptive_submit_interval_sec": float(scan_submit_min_interval_sec),
        "adaptive_busy_timeout_sec": float(scan_busy_timeout_sec),
        "last_scan_duration_sec": 0.0,
        "scan_cache_fingerprint": None,
        "scan_cache_brightness_fingerprint": None,
        "scan_cache_refreshed_at": 0.0,
        "scan_cache_text_fp": None,
        "scan_cache_ocr_hash": "",
        "scan_cache_ocr_text": "",
        "scan_cache_ocr_confidence": 0.0,
        "scan_cache_screen_context": "",
        "scan_cache_scene_summary": "",
        "scan_cache_scene_should_comment": False,
        "scan_cache_mode": "none",
        "scan_cache_vision_route": "none",
        "vn_ocr_filter": {},
        "comment_future": None,
        "pending_comment_meta": None,
        "vn_last_evaluation_at": None,
        "vn_eval_failure_streak": 0,
        "vn_eval_retry_after": None,
        "mm_fail_streak": 0,
        "mm_cooldown_until": None,
        "mm_last_request_at": None,
        # --- asynchronous vision (visual-novel mode) ---
        "vision_future": None,
        "vision_started_at": None,
        "vision_last_done_at": None,
        "vision_last_request_at": None,
        "vision_context": "",
        "vision_context_at": None,
        "vision_budget_window_started": None,
        "vision_budget_used": 0,
        "vision_skip_reason": "",
        # Visual-novel mode is the master switch for "watch the game window", so
        # enabling it turns auto-scan on. Remember what the user had chosen so
        # leaving VN mode restores it instead of leaving the toggle flipped.
        "vn_scan_autostarted": False,
        "vn_scan_prev_enabled": False,
        # GUI-thread snapshot of the pet's own HWNDs; worker threads read this
        # instead of calling QWidget.winId() off the GUI thread (see
        # _collect_pet_owned_hwnds).
        "pet_owned_hwnds": set(),
        "plot_memory_synced_line": None,
        # Runtime overrides driven by the visual-novel side panel.
        "vision_capture_mode": str(settings.vision_capture_mode),
        "vision_change_threshold": float(settings.vision_change_threshold),
        "vision_shots_per_minute": float(settings.vision_shots_per_minute),
        "vision_burst_cap": float(settings.vision_burst_cap),
        "vision_min_gap_sec": float(settings.vision_min_gap_sec),
        "visual_novel_text_ratio": float(settings.visual_novel_text_ratio),
        "vision_window_margin_px": int(settings.vision_window_margin_px),
        # Window-relative OCR box; None means "use the bottom-band heuristic".
        "ocr_region": _persisted_regions.get("ocr_region"),
        # Capture-space box defining "the game画面" for manual capture mode.
        "vision_region": _persisted_regions.get("vision_region"),
        "last_capture_source": "",
        "last_capture_title": "",
        # The game window title seen last time (persisted in vn_regions.json), so
        # a cache left over from another game is flagged on the next launch.
        "last_game_title": str(_persisted_regions.get("last_game_title", "") or ""),
        # Cache/game-change warning currently in force, so the side panel can
        # keep showing it after the bubble has gone.
        "story_game_mismatch": "",
        "story_game_pending": None,
        # When the dialogue box last became unreadable; the band sweep waits
        # ocr_fallback_min_quiet_sec before running.
        "no_text_since": None,
        "last_capture_warning": "",
        "last_nonempty_ocr_text": "",
        "last_nonempty_ocr_at": None,
        # Independent vision sampler state.
        "vision_sampler_fingerprint": None,
        "vision_last_delta": 0.0,
    }

    if settings.enable_scan_subprocess:
        print("[STARTUP] Auto scan execution: subprocess-persistent")
    else:
        print("[STARTUP] Auto scan execution: thread")
    scan_executor = None
    scan_worker_ctx = None
    scan_task_queue = None
    scan_result_queue = None
    scan_worker_ready = None
    scan_worker_process = None
    scan_task_seq = 0
    comment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="comment-worker")
    tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-worker")
    vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-worker")
    window_resolver = _build_window_resolver(
        settings,
        {os.getpid()},
        monitor_index=int(settings.vision_window_monitor_index),
    )
    # A pin saved from the side panel wins over the env default, so the user
    # does not have to re-pin on every launch.
    _pinned_title = str(_persisted_regions.get("target_title") or settings.vision_target_title or "")
    if window_resolver is not None and _pinned_title:
        window_resolver.set_target_title(_pinned_title)
    print(
        "[STARTUP] Vision capture:",
        f"mode={settings.vision_capture_mode}",
        f"target={_pinned_title or 'auto (topmost)'}",
        f"ocr_text_edge={settings.ocr_text_max_edge}",
        f"quota={settings.vision_shots_per_minute:g}/min, burst={settings.vision_burst_cap:g}",
        f"change_threshold={settings.vision_change_threshold:.2f}",
        f"budget/h={settings.vision_budget_per_hour}",
        f"resolver={'ok' if window_resolver is not None else 'unavailable'}",
    )

    def _pet_owned_pids() -> tuple[int, ...]:
        """Every process whose windows must never be captured."""
        pids = [int(os.getpid())]
        try:
            if live2d_py_process is not None and live2d_py_process.poll() is None:
                pids.append(int(live2d_py_process.pid))
        except Exception:
            pass
        # De-duplicate while keeping the order stable.
        seen: set[int] = set()
        ordered: list[int] = []
        for pid in pids:
            if pid > 0 and pid not in seen:
                seen.add(pid)
                ordered.append(pid)
        return tuple(ordered)

    def _collect_pet_owned_hwnds() -> None:
        """Snapshot the pet's own window handles. GUI THREAD ONLY.

        ``QWidget.winId()`` is not thread-safe: calling it from a worker forces
        native window creation there, and when that happens for a widget the GUI
        thread has not realised yet (the side panel, before its first hover) Qt's
        event loop wedges outright -- no timers, no paint, no output, while the
        main thread still looks idle inside ``app.exec``. Both the vision sampler
        and the scan executor used to reach this through
        ``_refresh_vision_exclusions``. The handles are therefore cached here, on
        the GUI thread, and worker threads only ever read the snapshot.
        """
        handles: set[int] = set()
        for widget in (pet, chat, getattr(pet, "vn_sidebar", None)):
            try:
                if widget is None:
                    continue
                handle = int(widget.winId())
                if handle:
                    handles.add(handle)
            except Exception:
                continue
        state["pet_owned_hwnds"] = handles

    def _refresh_vision_exclusions() -> None:
        """Keep pet-owned windows out of the capture target.

        Safe from any thread: it only combines the GUI-thread HWND snapshot with
        the process list, and never touches a Qt object.
        """
        if window_resolver is None:
            return
        pids = set(_pet_owned_pids())
        cached = state.get("pet_owned_hwnds")
        hwnds = set(cached) if isinstance(cached, (set, frozenset, list, tuple)) else set()
        window_resolver.set_excluded_pids(pids)
        window_resolver.set_excluded_hwnds(hwnds)

    resource_policy_reapply_min_sec = float(settings.resource_policy_reapply_min_sec)
    follow_activate_distance_px = int(settings.live2d_follow_activate_distance_px)
    follow_enter_distance_px = max(140, follow_activate_distance_px)
    follow_exit_distance_px = max(follow_enter_distance_px + 80, int(follow_enter_distance_px * 1.8))
    follow_hold_sec = 0.9
    print(f"[STARTUP] Scan busy timeout: {scan_busy_timeout_sec}s")
    logical_cpu_count = int(os.cpu_count() or 0)
    all_cpu_ids = list(range(logical_cpu_count)) if logical_cpu_count > 0 else []
    if logical_cpu_count >= 4:
        split = max(1, logical_cpu_count // 2)
        app_cpu_ids = list(range(0, split))
        scan_cpu_ids = list(range(split, logical_cpu_count))
    else:
        app_cpu_ids = []
        scan_cpu_ids = []
    if app_cpu_ids and scan_cpu_ids:
        print(
            "[STARTUP] CPU affinity partition:",
            f"app={app_cpu_ids}",
            f"scan={scan_cpu_ids}",
        )
    resource_policy_last_apply_ts = 0.0
    resource_policy_last_mode = ""
    resource_policy_last_switch_ts = 0.0
    resource_policy_follow_grace_until_ts = 0.0
    resource_policy_drag_grace_until_ts = 0.0
    runtime_follow_near = False
    runtime_follow_effective = False
    runtime_follow_hold_until_ts = 0.0
    runtime_follow_near_on_streak = 0
    runtime_follow_near_off_streak = 0
    follow_near_enter_frames = 2
    follow_near_exit_frames = 4
    runtime_drag_active = False
    drag_lock_prev_active = False
    drag_lock_last_rect = None
    drag_lock_last_write_ts = 0.0
    last_zorder_sync_ts = 0.0
    resource_policy_heavy_scan_interval_sec = 20.0
    voicevox_pid_cache = 0
    voicevox_policy_last_scan_ts = 0.0
    proc_policy_cache: dict[int, tuple[object, tuple[int, ...]]] = {}

    def _apply_runtime_resource_policy(
        scan_active: bool,
        *,
        follow_active: bool = False,
        drag_active: bool = False,
        force: bool = False,
    ) -> None:
        nonlocal resource_policy_last_apply_ts, resource_policy_last_mode
        nonlocal resource_policy_last_switch_ts
        nonlocal resource_policy_follow_grace_until_ts, resource_policy_drag_grace_until_ts
        nonlocal voicevox_pid_cache, voicevox_policy_last_scan_ts
        if os.name != "nt":
            return

        now_ts = time.time()
        if drag_active:
            resource_policy_drag_grace_until_ts = now_ts + 0.28
        if follow_active:
            resource_policy_follow_grace_until_ts = now_ts + 0.42

        if drag_active or now_ts < resource_policy_drag_grace_until_ts:
            candidate_mode = "drag_priority"
        elif follow_active or now_ts < resource_policy_follow_grace_until_ts:
            candidate_mode = "follow_priority"
        elif scan_active:
            candidate_mode = "ocr_priority"
        else:
            candidate_mode = "idle"

        mode = candidate_mode
        if (
            (not force)
            and resource_policy_last_mode
            and candidate_mode != resource_policy_last_mode
            and (now_ts - resource_policy_last_switch_ts < 0.30)
        ):
            # Avoid rapid mode flapping around follow/ocr boundaries.
            mode = resource_policy_last_mode

        if (not force) and (resource_policy_last_mode == mode) and (now_ts - resource_policy_last_apply_ts < resource_policy_reapply_min_sec):
            return

        try:
            import psutil
        except Exception:
            return

        idle_cls = getattr(psutil, "IDLE_PRIORITY_CLASS", None)
        below_cls = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None)
        normal_cls = getattr(psutil, "NORMAL_PRIORITY_CLASS", None)
        above_cls = getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", None)
        if not any([idle_cls, below_cls, normal_cls, above_cls]):
            return

        def _set_proc_priority(pid: int, cls) -> None:
            if not pid or cls is None:
                return
            try:
                p = psutil.Process(int(pid))
                if not p.is_running():
                    return
                p.nice(cls)
            except Exception:
                pass

        def _set_proc_affinity(pid: int, cpu_ids: list[int]) -> None:
            if not pid or not cpu_ids:
                return
            try:
                p = psutil.Process(int(pid))
                if not p.is_running() or not hasattr(p, "cpu_affinity"):
                    return
                p.cpu_affinity(cpu_ids)
            except Exception:
                pass

        def _apply_proc_policy_cached(pid: int, cls, cpu_ids: list[int]) -> None:
            if not pid:
                return
            target_affinity = tuple(sorted(int(x) for x in cpu_ids)) if cpu_ids else tuple()
            target = (cls, target_affinity)
            prev = proc_policy_cache.get(int(pid))
            if (not force) and prev == target:
                return
            _set_proc_priority(pid, cls)
            if cpu_ids:
                _set_proc_affinity(pid, cpu_ids)
            proc_policy_cache[int(pid)] = target

        # Keep main UI responsive while scan worker does heavy OCR.
        if mode == "drag_priority":
            main_cls = above_cls
            scan_cls = below_cls
            live2d_cls = normal_cls
            scan_affinity = all_cpu_ids
            live2d_affinity = all_cpu_ids
        elif mode == "follow_priority":
            main_cls = normal_cls
            # Hovering the pet used to drop the scan worker to below-normal
            # priority, which visibly slows OCR exactly when the user reaches for
            # the control strip. In visual-novel mode the scan is the priority,
            # so it keeps its normal/above-normal class; other modes still yield
            # to the cursor follow.
            vn_scanning = bool(state.get("visual_novel_mode_enabled")) and bool(state.get("scan_enabled"))
            scan_cls = normal_cls if vn_scanning else below_cls
            live2d_cls = above_cls
            scan_affinity = all_cpu_ids
            live2d_affinity = all_cpu_ids
        elif mode == "ocr_priority":
            main_cls = normal_cls
            scan_cls = above_cls
            # OCR priority mode explicitly pushes gaze follow process to the lowest level.
            live2d_cls = idle_cls
            # Only pin cores when enough logical CPUs are available.
            if scan_cpu_ids and app_cpu_ids:
                scan_affinity = scan_cpu_ids
                live2d_affinity = app_cpu_ids
            else:
                scan_affinity = all_cpu_ids
                live2d_affinity = all_cpu_ids
        else:
            main_cls = normal_cls
            scan_cls = below_cls
            live2d_cls = below_cls
            scan_affinity = all_cpu_ids
            live2d_affinity = all_cpu_ids

        _apply_proc_policy_cached(os.getpid(), main_cls, all_cpu_ids)

        # scan worker is a persistent standalone process when subprocess mode is enabled.
        try:
            if settings.enable_scan_subprocess:
                if scan_worker_process is not None and scan_worker_process.is_alive():
                    _apply_proc_policy_cached(int(scan_worker_process.pid), scan_cls, scan_affinity)
        except Exception:
            pass

        # Follow process priority is controlled by mode.
        if live2d_py_process is not None and live2d_py_process.poll() is None:
            _apply_proc_policy_cached(int(live2d_py_process.pid), live2d_cls, live2d_affinity)

        # Keep VOICEVOX at normal as requested, even in OCR-priority mode.
        #
        # Finding the engine used to mean enumerating every process on the machine
        # -- measured ~1.9s here for 386 processes (exe + cmdline each). That ran
        # on the GUI thread every 20s *whether or not* the cached pid was still
        # valid, which is what produced a ~2s freeze roughly once every 20s
        # (measured 13 stalls in 255 ticks; 0-1 with the Live2D loop disabled).
        #
        # The scan is now only a fallback for an engine we did not launch: the
        # cached pid is validated on every call, so a restarted engine is still
        # noticed (its pid dies, the cache clears, and the scan runs at once).
        try:
            if voicevox_pid_cache <= 0:
                launched_pid = int(getattr(speech, "voicevox_engine_pid", 0) or 0)
                if launched_pid > 0:
                    voicevox_pid_cache = launched_pid
            if voicevox_pid_cache > 0:
                try:
                    proc = psutil.Process(int(voicevox_pid_cache))
                    if proc.is_running() and str(proc.name()).lower() == "run.exe":
                        _apply_proc_policy_cached(int(voicevox_pid_cache), normal_cls, [])
                    else:
                        voicevox_pid_cache = 0
                        # Rescan on this call rather than waiting out the interval.
                        voicevox_policy_last_scan_ts = 0.0
                except Exception:
                    voicevox_pid_cache = 0
                    voicevox_policy_last_scan_ts = 0.0

            need_full_scan = voicevox_pid_cache <= 0 and (
                voicevox_policy_last_scan_ts <= 0.0
                or (now_ts - voicevox_policy_last_scan_ts >= resource_policy_heavy_scan_interval_sec)
            )
            if need_full_scan:
                voicevox_policy_last_scan_ts = now_ts
                for proc in psutil.process_iter(attrs=["pid", "name", "exe", "cmdline"]):
                    name = str((proc.info.get("name") or "")).lower()
                    exe = str((proc.info.get("exe") or "")).lower()
                    cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                    if name != "run.exe":
                        continue
                    if "voicevox" not in exe and "voicevox" not in cmdline and "vv-engine" not in exe and "vv-engine" not in cmdline:
                        continue
                    voicevox_pid_cache = int(proc.info.get("pid") or 0)
                    if voicevox_pid_cache > 0:
                        _apply_proc_policy_cached(voicevox_pid_cache, normal_cls, [])
                        break
        except Exception:
            pass

        if resource_policy_last_mode != mode:
            log_heartbeat(
                "resource_policy",
                f"mode={mode}, scan={int(bool(scan_active))}, follow={int(bool(follow_active))}, drag={int(bool(drag_active))}",
            )
            resource_policy_last_switch_ts = now_ts
        resource_policy_last_mode = mode
        resource_policy_last_apply_ts = now_ts

    def _start_scan_worker() -> None:
        nonlocal scan_worker_ctx, scan_task_queue, scan_result_queue, scan_worker_process, scan_worker_ready
        if not settings.enable_scan_subprocess:
            return
        if scan_worker_process is not None and scan_worker_process.is_alive():
            return
        scan_worker_ctx = mp.get_context("spawn")
        scan_task_queue = scan_worker_ctx.Queue(maxsize=2)
        scan_result_queue = scan_worker_ctx.Queue(maxsize=4)
        scan_worker_ready = scan_worker_ctx.Event()
        scan_worker_process = scan_worker_ctx.Process(
            target=run_scan_pipeline_worker_loop,
            args=(scan_task_queue, scan_result_queue, scan_worker_ready),
            daemon=True,
        )
        scan_worker_process.start()
        print(f"[STARTUP] Scan worker process started (pid={scan_worker_process.pid})")

    def _stop_scan_worker() -> None:
        nonlocal scan_task_queue, scan_result_queue, scan_worker_process, scan_worker_ctx, scan_worker_ready
        if not settings.enable_scan_subprocess:
            return
        try:
            if scan_task_queue is not None:
                try:
                    scan_task_queue.put_nowait(None)
                except Exception:
                    pass
            if scan_worker_process is not None and scan_worker_process.is_alive():
                scan_worker_process.terminate()
                scan_worker_process.join(timeout=1.0)
        except Exception:
            pass
        scan_task_queue = None
        scan_result_queue = None
        scan_worker_process = None
        scan_worker_ctx = None
        scan_worker_ready = None

    def _drain_scan_results() -> None:
        if scan_result_queue is None:
            return
        while True:
            try:
                scan_result_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

    def _submit_scan_worker_task(scan_monitor_index: int, scan_region: tuple[int, int, int, int] | None) -> int:
        nonlocal scan_task_seq
        if scan_task_queue is None:
            return 0
        if scan_worker_ready is not None and not scan_worker_ready.is_set():
            return 0
        scan_task_seq += 1
        payload = {
            "task_id": int(scan_task_seq),
            "scan_monitor_index": int(scan_monitor_index),
            "scan_region": scan_region,
            "ocr_cpu_threads": int(settings.ocr_cpu_threads),
            "ocr_cpu_affinity_count": int(settings.ocr_cpu_affinity_count),
            "ocr_max_edge": int(settings.ocr_max_edge),
            "visual_novel_mode": bool(state["visual_novel_mode_enabled"]),
            "owner_pid": int(os.getpid()),
            "owned_pids": _pet_owned_pids(),
            "ocr_region": list(state.get("ocr_region") or ()),
            "vision_region": list(state.get("vision_region") or ()),
            "capture_mode": str(state.get("vision_capture_mode", settings.vision_capture_mode)),
            "text_ratio": float(
                state.get("visual_novel_text_ratio", settings.visual_novel_text_ratio)
            ),
            "target_title": str(_persisted_regions.get("target_title") or state.get("last_capture_title") or ""),
        }
        try:
            scan_task_queue.put_nowait(payload)
        except queue.Full:
            return 0
        except Exception:
            return 0
        return int(scan_task_seq)

    def _poll_scan_worker_result(expected_task_id: int):
        if scan_result_queue is None:
            return None
        while True:
            try:
                msg = scan_result_queue.get_nowait()
            except queue.Empty:
                return None
            except Exception:
                return None
            if int(msg.get("task_id", 0)) == int(expected_task_id):
                return msg

    def _reset_scan_worker(reason: str) -> None:
        nonlocal scan_executor
        pet.set_scan_busy(False)
        streak = int(state.get("scan_timeout_streak", 0)) + 1
        state["scan_timeout_streak"] = streak
        cooldown_sec = min(90, 6 * (2 ** (streak - 1)))
        state["scan_backoff_until"] = datetime.now() + timedelta(seconds=cooldown_sec)
        state["adaptive_submit_interval_sec"] = max(
            float(state.get("adaptive_submit_interval_sec", scan_submit_min_interval_sec)),
            float(cooldown_sec),
        )
        state["adaptive_busy_timeout_sec"] = min(
            90.0,
            max(
                float(scan_busy_timeout_sec),
                float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec)) * 1.4,
            ),
        )
        future = state.get("scan_future")
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass
        state["scan_future"] = None
        state["scan_started_at"] = None
        state["last_scan_submit_at"] = None
        if settings.enable_scan_subprocess:
            _stop_scan_worker()
            _start_scan_worker()
            _drain_scan_results()
        else:
            try:
                if scan_executor is not None:
                    scan_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")
        log_heartbeat(
            "scan_reset",
            (
                f"{reason}, streak={streak}, cooldown={cooldown_sec}s, "
                f"next_busy_timeout={state['adaptive_busy_timeout_sec']:.1f}s"
            ),
        )
        chat.append_message("系统", f"扫描任务卡住，已自动重置扫描器（{reason}）")

    def log_heartbeat(stage: str, detail: str = "") -> None:
        if not settings.enable_auto_comment_heartbeat:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"[HEARTBEAT {ts}] {stage}"
        if detail:
            msg += f" | {detail}"
        print(msg)

    def build_screen_context(
        image,
        *,
        respect_vision_interval: bool,
        ocr_result=None,
        vision_enabled: bool | None = None,
        focus_query: str = "",
    ) -> dict:
        if ocr_result is None:
            ocr_image = _resize_for_ocr(image, int(settings.ocr_max_edge))
            ocr_result = extract_ocr_result(ocr_image)
        return _build_ocr_first_context(
            llm_client,
            image,
            ocr_result,
            settings,
            state,
            log_heartbeat,
            respect_vision_interval=respect_vision_interval,
            vision_enabled=vision_enabled,
            focus_query=focus_query,
        )

    def maybe_emit_pipeline_status(mode: str):
        if mode == state["last_pipeline_mode"]:
            return
        state["last_pipeline_mode"] = mode

        if mode == "vision+ocr":
            chat.append_message("系统", "识别状态: 多模态视觉+OCR")
        elif mode == "vision":
            chat.append_message("系统", "识别状态: 仅多模态视觉")
        elif mode == "ocr":
            chat.append_message("系统", "识别状态: 仅本地OCR（省Token）")
        else:
            chat.append_message("系统", "识别状态: 未识别到有效内容")

    def _emit_comment(comment: str, role: str = "桌宠(自动)", *, is_valid=None, on_display=None) -> None:
        display_role = role
        if settings.chat_show_session_debug_marker:
            seg_id = max(1, int(dialog.get_current_session_segment_id()))
            display_role = f"{role}[S{seg_id}]"
        if on_display is None:
            show_and_speak(comment, role=display_role, is_valid=is_valid)
            dialog.record_session_message(role, comment)
        else:
            def record():
                dialog.record_session_message(role, comment)
                on_display()
            show_and_speak(comment, role=display_role, is_valid=is_valid, on_display=record)

    def _normalize_memory_summary(summary: str) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        text = text.replace("屏幕内容摘要:", "")
        text = " ".join(text.split())
        return text

    def _append_cycle_memory(summary: str) -> None:
        normalized = _normalize_memory_summary(summary)
        if not normalized:
            return
        captured_at = datetime.now()
        memories = state["cycle_memories"]

        if memories and memories[-1]["summary"] == normalized:
            memories[-1]["captured_at"] = captured_at
            return

        # Fuzzy dedup across nearby screenshots: keep one representative for highly similar text.
        for item in reversed(memories[-12:]):
            ratio = text_similarity(normalized, str(item.get("summary", "")))
            if ratio >= 0.90:
                item["captured_at"] = captured_at
                if len(normalized) > len(str(item.get("summary", ""))):
                    item["summary"] = normalized
                return
        memories.append({"summary": normalized, "captured_at": captured_at})
        if len(memories) > 24:
            del memories[:-24]

    def _compose_cycle_summary(now: datetime) -> tuple[str, str]:
        memories = state["cycle_memories"]
        if not memories:
            return "", ""

        recency_window_sec = settings.memory_recency_window_sec
        if recency_window_sec <= 0:
            recency_window_sec = max(45, base_comment_interval_sec + 30)
        min_weight = settings.memory_min_weight
        merged: dict[str, dict] = {}

        for item in memories:
            summary = _normalize_memory_summary(item["summary"])
            if not summary:
                continue
            captured_at = item["captured_at"]
            age_sec = max(0.0, (now - captured_at).total_seconds())
            weight = max(min_weight, 1.0 - (age_sec / recency_window_sec))

            canonical_key = summary
            for key in merged.keys():
                if text_similarity(summary, key) >= 0.90:
                    canonical_key = key
                    break

            prev = merged.get(canonical_key)
            if prev is None:
                merged[canonical_key] = {
                    "summary": summary,
                    "weight": weight,
                    "captured_at": captured_at,
                }
                continue
            if weight > prev["weight"]:
                prev["weight"] = weight
            if captured_at > prev["captured_at"]:
                prev["captured_at"] = captured_at
            if len(summary) > len(str(prev.get("summary", ""))):
                prev["summary"] = summary

        ranked = sorted(
            merged.values(),
            key=lambda x: (x["weight"], x["captured_at"]),
            reverse=True,
        )[:12]

        signature = "；".join(item["summary"] for item in ranked[:8])
        # Weight tags used to be written into the prompt and then stripped again
        # by CommentEngine._normalize_screen_summary, so only the ranking (which
        # already happened above) is meaningful.
        lines: list[str] = []
        for idx, item in enumerate(ranked, start=1):
            tag = "重点" if idx <= 3 else "参考"
            lines.append(f"{tag} {item['summary']}")

        composed = "近期扫描记忆（越靠近当前时刻权重越高）:\n" + "\n".join(lines)
        return composed[:6000], signature[:2400]

    def run_scan_pipeline() -> dict:
        vn_mode = bool(state["visual_novel_mode_enabled"])
        window_rect = None
        capture_source = "screen"
        _vision_image = None
        # Fetched before any early return: the cache-reuse paths below report it
        # as ``ocr_note``, and referencing it first raised UnboundLocalError and
        # aborted every single scan cycle.
        ocr_ok, ocr_info = get_ocr_runtime_status()
        if vn_mode:
            _refresh_vision_exclusions()
            image, window_rect, _vision_image, capture_source = _resolve_vn_capture(
                mode=str(state.get("vision_capture_mode", settings.vision_capture_mode)),
                resolver=window_resolver,
                ocr_region=state.get("ocr_region"),
                vision_region=state.get("vision_region"),
                monitor_index=int(state["scan_monitor_index"]),
                text_ratio=float(state.get("visual_novel_text_ratio", settings.visual_novel_text_ratio)),
                text_bottom_margin=float(settings.vn_text_band_bottom_margin),
            )
            if capture_source == "target_missing":
                return {"ocr_ok": True, "ocr_info": "target_missing", "capture_source": capture_source}
        else:
            image = capture_primary_screen(
                monitor_index=int(state["scan_monitor_index"]),
                region=state["scan_region"],
            )
        capture_rect = None if window_rect is None else window_rect.as_region()
        if vn_mode and image.info.get("vn_ui_mask"):
            ocr_info += f"+ui_mask={image.info['vn_ui_mask']}:{image.info['vn_ui_mask_ms']:g}ms"
        sampled_at = time.monotonic()
        fingerprint, text_diff = _text_change_metrics(image, visual_novel=vn_mode)
        brightness_fingerprint = (
            _frame_fingerprint(image, text_sensitive=True) if vn_mode else None
        )
        text_change_threshold = (
            float(settings.ocr_text_ink_change_threshold) if vn_mode else 0.04
        )
        prev_fingerprint = state.get("scan_cache_fingerprint")
        vn_reuse_frame = False
        if isinstance(prev_fingerprint, bytes):
            diff_ratio = text_diff(fingerprint, prev_fingerprint)
            diff_threshold = text_change_threshold
            if vn_mode:
                previous_brightness = state.get("scan_cache_brightness_fingerprint")
                refreshed_at = float(state.get("scan_cache_refreshed_at", 0.0) or 0.0)
                vn_reuse_frame = bool(
                    isinstance(brightness_fingerprint, bytes)
                    and isinstance(previous_brightness, bytes)
                    and _vn_stable_cache_can_reuse(
                        fingerprint,
                        prev_fingerprint,
                        brightness_fingerprint,
                        previous_brightness,
                        ink_threshold=diff_threshold,
                        brightness_threshold=float(settings.ocr_text_change_threshold),
                        cache_age_sec=time.monotonic() - refreshed_at,
                        has_text=bool(state.get("scan_cache_ocr_text")),
                    )
                )
                reuse_frame = vn_reuse_frame
            else:
                reuse_frame = diff_ratio < diff_threshold
            if reuse_frame and (vn_mode or str(state.get("scan_cache_scene_summary", "")).strip()):
                cached_context = str(state.get("scan_cache_screen_context", "")).strip()
                scene = analyze_scene(cached_context)
                return {
                    "ocr_ok": True,
                    "ocr_info": "cache_reuse",
                    "mode": str(state.get("scan_cache_mode", "none")),
                    "screen_context": cached_context,
                    "ocr_text": str(state.get("scan_cache_ocr_text", "")),
                    "ocr_confidence": float(state.get("scan_cache_ocr_confidence", 0.0) or 0.0),
                    "scene": scene,
                    "vision_route": "cache",
                    "capture_rect": capture_rect,
                    "capture_source": capture_source,
                    "capture_title": str(getattr(window_rect, "title", "") or ""),
                    "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
                    "ocr_fingerprint": fingerprint,
                    "ocr_note": ocr_info,
                    "sampled_at": sampled_at,
                    "frame_stable": True,
                    "ocr_elapsed_ms": 0,
                    "reused_cache": True,
                }

        if vn_mode and _vn_text_is_settling(state, fingerprint, brightness_fingerprint, sampled_at):
            return {
                "ocr_ok": True, "ocr_info": "text_settling", "mode": "none",
                "screen_context": "", "ocr_text": "", "scene": analyze_scene(""),
                "vision_route": "none", "capture_source": capture_source,
                "capture_title": str(getattr(window_rect, "title", "") or ""),
                "sampled_at": sampled_at, "scan_deferred": True,
                "frame_stable": False, "ocr_elapsed_ms": 0, "reused_cache": False,
            }

        ocr_started_at = time.perf_counter()
        if vn_mode:
            # Visual-novel mode is a latency-sensitive loop, so the cheap
            # grayscale form runs first (measured ~1.3s vs ~2.3s for RGB) and is
            # accepted as soon as it reads anything usable. Raw colour and the
            # polarity/padding variants only run as fallbacks.
            text_image = _resize_for_ocr(image, int(settings.ocr_text_max_edge))
            ocr_result = None
            ocr_chars = 0
            variant_used = ""
            for index, variant in enumerate(_ocr_text_variants(text_image)):
                candidate = extract_ocr_result(variant)
                if ocr_result is None or int(candidate.char_count) > ocr_chars:
                    ocr_result = candidate
                    ocr_chars = int(candidate.char_count)
                    variant_used = "raw" if variant is text_image else f"variant{index}"
                if ocr_chars >= 4:
                    break
            if ocr_result is None:
                ocr_result = extract_ocr_result(text_image)
                ocr_chars = int(ocr_result.char_count)
                variant_used = "raw"
            if variant_used != "raw":
                ocr_info = f"{ocr_info}+{variant_used}"
        else:
            ocr_image = _resize_for_ocr(image, int(settings.ocr_max_edge))
            ocr_result = extract_ocr_result(ocr_image)
            ocr_chars = int(ocr_result.char_count)

        # Safety net: a hand-picked box that misses the dialogue, or a window
        # whose text is not in the lower band, would silently produce zero text.
        # When the focused read is *completely* empty, sweep a couple of
        # candidate bands and keep the richest reading.
        #
        # The gate used to be "fewer than 12 characters", which meant a read that
        # picked up a little junk paid for the whole sweep: measured 5 candidates
        # x 4 variants = up to 20 OCR passes, ~12s when the machine was quiet and
        # 78s while the pet itself was busy. Requiring an empty read keeps the
        # recovery while making the common noisy case cheap.
        #
        # A single empty read is still normal -- the box blinks out between lines,
        # a scene transition has no text, one frame can be caught mid-fade -- and
        # sweeping on it meant a ~10s recovery every time the box was briefly
        # blank. The sweep now waits until no text has arrived for
        # ``ocr_fallback_min_quiet_sec`` (5s by default) before it is worth paying
        # for. The clock comes from ``last_nonempty_ocr_at``, which the tick
        # handler maintains for every source and either scan path.
        now_scan = datetime.now()
        quiet_required = float(settings.ocr_fallback_min_quiet_sec)
        last_text_at = state.get("last_nonempty_ocr_at")
        if isinstance(last_text_at, datetime):
            quiet_sec = max(0.0, (now_scan - last_text_at).total_seconds())
        else:
            # Nothing read yet this session: start the clock on the first empty
            # read, so one blank frame at startup does not buy a sweep either.
            if not isinstance(state.get("no_text_since"), datetime):
                state["no_text_since"] = now_scan
            quiet_sec = max(0.0, (now_scan - state["no_text_since"]).total_seconds())

        # Each OCR pass costs ~4-5s measured, so the sweep is expensive. It only
        # runs on a changed frame and at a bounded rate; otherwise one unreadable
        # frame would stretch the scan interval to tens of seconds.
        # The whole decision below is should_run_band_sweep(); this block only
        # gathers its inputs.
        fallback_interval = float(settings.ocr_fallback_min_interval_sec)
        last_fallback = state.get("last_ocr_fallback_at")
        cooled_down = not (
            isinstance(last_fallback, datetime)
            and (now_scan - last_fallback).total_seconds() < fallback_interval
        )
        # The cached fingerprint belongs to the previous frame, so comparing
        # against it also tells us whether the picture moved at all.
        cached_frame_fp = state.get("scan_cache_fingerprint")
        if isinstance(cached_frame_fp, (bytes, bytearray)) and isinstance(
            fingerprint, (bytes, bytearray)
        ):
            # Same comparison the cache gate uses, so "did the picture move"
            # means the same thing here as everywhere else in this pipeline.
            frame_changed = (
                text_diff(bytes(fingerprint), bytes(cached_frame_fp))
                >= text_change_threshold
            )
        else:
            frame_changed = True
        run_sweep, sweep_reason = should_run_band_sweep(
            vn_mode=bool(vn_mode),
            has_vision_frame=_vision_image is not None,
            ocr_chars=int(ocr_chars),
            quiet_sec=quiet_sec,
            quiet_required_sec=quiet_required,
            frame_changed=frame_changed,
            cooled_down=cooled_down,
        )
        state["band_sweep_skip"] = "" if run_sweep else sweep_reason
        best_band = ""
        if run_sweep:
            state["last_ocr_fallback_at"] = datetime.now()
            from desktop_pet.vision.vn_ui_mask import mask_visual_novel_ui

            fallback_frame = _vision_image
            if window_rect is not None and not state.get("vision_region"):
                fallback_frame = mask_visual_novel_ui(
                    _vision_image, title=str(getattr(window_rect, "title", "") or ""),
                    process_name=str(getattr(window_rect, "process_name", "") or ""),
                )
            fallback_image = _resize_for_ocr(
                fallback_frame, int(settings.ocr_text_max_edge)
            )
            fb_w, _fb_h = fallback_image.size
            # Two disjoint windows: the lower band the app assumes, then the
            # strip above it (centred text boxes). Deliberately no full-frame
            # candidate -- its SAVE/LOAD/CONFIG chrome used to win the
            # "richest reading" comparison over the clean bands.
            candidates: list[tuple[str, object]] = [
                (label, fallback_image.crop((0, top, fb_w, bottom)))
                for label, top, bottom in _text_band_candidates(
                    _fb_h,
                    text_ratio=float(settings.visual_novel_text_ratio),
                    bottom_margin=float(settings.vn_text_band_bottom_margin),
                    mid_top_ratio=float(settings.vn_text_band_mid_top_ratio),
                )
            ]
            for label, candidate_image in candidates:
                # Two variants per candidate, not all four: the cheap grayscale
                # pass and the raw colour one. The polarity and padding variants
                # cost the same as a full pass each and rarely change the
                # outcome, and this loop is the most expensive thing in the
                # pipeline.
                best_variant = None
                for variant in _ocr_text_variants(candidate_image)[:2]:
                    probed = extract_ocr_result(variant)
                    if best_variant is None or int(probed.char_count) > int(
                        best_variant.char_count
                    ):
                        best_variant = probed
                if best_variant is None:
                    continue
                # Compare on the text with the menu bar removed, so chrome
                # cannot out-rank real dialogue.
                if len(strip_ui_chrome(best_variant.text)) > len(
                    strip_ui_chrome(ocr_result.text)
                ):
                    ocr_result = best_variant
                    ocr_chars = int(best_variant.char_count)
                    best_band = label
                # OCR is the expensive part here; stop at the first usable hit.
                if ocr_chars >= 12:
                    break
        elif sweep_reason == "cooldown":
            ocr_info = f"{ocr_info}+fallback_throttled"
        if ocr_chars >= 12 and best_band:
            print(
                "[PROBE-OCR]",
                f"text region read too few chars; used {best_band} instead ({ocr_chars} chars)",
            )
            ocr_info = f"{ocr_info}+{best_band}"
        elif ocr_chars <= 0 and vn_mode:
            # Record the exact crop so an empty reading can be inspected.
            try:
                debug_dir = base_dir / "data" / "vn_debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                image.save(debug_dir / "last_ocr_input.png")
                _vision_image.save(debug_dir / "last_frame.png")
            except Exception:
                pass

        if vn_mode:
            ocr_result, filter_note = filter_visual_novel_ocr_result(
                ocr_result,
                state["vn_ocr_filter"],
            )
            ocr_chars = int(ocr_result.char_count)
            if filter_note:
                ocr_info = f"{ocr_info}+vn_filter({filter_note})"

        previous_ocr_hash = str(state.get("scan_cache_ocr_hash", "") or "")

        # Skip the OCR pass entirely while the text area looks unchanged.
        # OCR on a real dialogue frame costs ~4.5s (the engine is ~1.2s on clean
        # text; the surrounding art makes the detector work much harder), so
        # re-reading a static box every tick is what makes the scan feel slow.
        #
        # "Unchanged" must mean the same thing the cache gate above means, or a
        # change one of them can see and the other cannot freezes the pipeline.
        # The shared decision requires both the ink map and coarse brightness
        # map to stay quiet, and expires after ten seconds.
        cached_text = str(state.get("scan_cache_ocr_text", "") or "")
        reuse_ok = False
        if vn_mode and cached_text:
            gallery_fp = fingerprint
            prev_gallery_fp = state.get("scan_cache_text_fp")
            if isinstance(prev_gallery_fp, (bytes, bytearray)) and isinstance(
                gallery_fp, (bytes, bytearray)
            ):
                if vn_reuse_frame:
                    reuse_ok = True
            state["scan_cache_text_fp"] = gallery_fp
        if reuse_ok:
            cached_context = str(state.get("scan_cache_screen_context", "")).strip()
            if cached_context:
                scene = analyze_scene(cached_context)
                return {
                    "ocr_ok": ocr_ok,
                    "ocr_info": "text_region_unchanged",
                    "mode": str(state.get("scan_cache_mode", "none")),
                    "screen_context": cached_context,
                    "ocr_text": cached_text,
                    "ocr_confidence": float(state.get("scan_cache_ocr_confidence", 0.0) or 0.0),
                    "scene": scene,
                    "vision_route": "cache",
                    "capture_rect": capture_rect,
                    "capture_source": capture_source,
                    "capture_title": str(getattr(window_rect, "title", "") or ""),
                    "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
                    "ocr_fingerprint": fingerprint,
                    "ocr_note": "reused_unchanged_text_area",
                    "sampled_at": sampled_at,
                "frame_stable": False,
                "ocr_elapsed_ms": round((time.perf_counter() - ocr_started_at) * 1000),
                "reused_cache": True,
                }

        if (
        ocr_result.text_hash
        and ocr_result.text_hash == previous_ocr_hash
        and str(state.get("scan_cache_vision_route", "")) == "ocr_only"
        and str(state.get("scan_cache_scene_summary", "")).strip()
        ):
            state["scan_cache_fingerprint"] = fingerprint
            state["scan_cache_brightness_fingerprint"] = brightness_fingerprint
            state["scan_cache_refreshed_at"] = time.monotonic()
            cached_context = str(state.get("scan_cache_screen_context", "")).strip()
            scene = analyze_scene(cached_context)
            return {
                "ocr_ok": ocr_ok,
                "ocr_info": "ocr_hash_reuse",
                "mode": str(state.get("scan_cache_mode", "none")),
                "screen_context": cached_context,
                "ocr_text": str(state.get("scan_cache_ocr_text", "")),
                "ocr_confidence": float(state.get("scan_cache_ocr_confidence", 0.0) or 0.0),
                "scene": scene,
                "vision_route": "cache",
                "capture_rect": capture_rect,
                "capture_source": capture_source,
                "capture_title": str(getattr(window_rect, "title", "") or ""),
                "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
                "ocr_fingerprint": fingerprint,
                "ocr_note": ocr_info,
                "sampled_at": sampled_at,
                    "frame_stable": False,
                    "ocr_elapsed_ms": round((time.perf_counter() - ocr_started_at) * 1000),
                    "reused_cache": True,
            }

        ocr_elapsed_ms = round((time.perf_counter() - ocr_started_at) * 1000)
        pipeline = build_screen_context(
            image,
            respect_vision_interval=True,
            ocr_result=ocr_result,
            # Vision runs on its own worker thread so it cannot stall scanning.
            vision_enabled=False,
        )
        screen_context = str(pipeline["screen_context"])
        mode = str(pipeline["mode"])
        scene = analyze_scene(screen_context)
        state["scan_cache_fingerprint"] = fingerprint
        state["scan_cache_brightness_fingerprint"] = brightness_fingerprint
        state["scan_cache_refreshed_at"] = time.monotonic()
        state["scan_cache_ocr_hash"] = str(pipeline.get("ocr_hash", ""))
        state["scan_cache_ocr_text"] = str(pipeline.get("ocr_text", ""))
        state["scan_cache_ocr_confidence"] = float(pipeline.get("ocr_confidence", 0.0) or 0.0)
        state["scan_cache_screen_context"] = screen_context
        state["scan_cache_scene_summary"] = scene.summary
        state["scan_cache_scene_should_comment"] = scene.should_comment
        state["scan_cache_mode"] = mode
        state["scan_cache_vision_route"] = str(pipeline.get("vision_route", "none"))
        return {
            "ocr_ok": ocr_ok,
            "ocr_info": ocr_info,
            "mode": mode,
            "screen_context": screen_context,
            "ocr_text": str(pipeline.get("ocr_text", "")),
            "ocr_confidence": float(pipeline.get("ocr_confidence", 0.0) or 0.0),
            "scene": scene,
            "vision_route": str(pipeline.get("vision_route", "none")),
            "capture_rect": capture_rect,
            "capture_source": capture_source,
            "capture_title": str(getattr(window_rect, "title", "") or ""),
            "capture_warning": str(getattr(_resolve_vn_capture, "warning", "") or ""),
            "ocr_fingerprint": fingerprint,
            "ocr_note": ocr_info,
            "sampled_at": sampled_at,
            "frame_stable": False,
            "ocr_elapsed_ms": ocr_elapsed_ms,
            "reused_cache": False,
        }

    def text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    # -- asynchronous vision -------------------------------------------------

    def _vision_budget_available(now: datetime) -> bool:
        """Optional hard ceiling on top of the token bucket (0 = unlimited)."""
        if settings.vision_budget_per_hour <= 0:
            return True
        window_start = state.get("vision_budget_window_started")
        if not isinstance(window_start, datetime) or (now - window_start).total_seconds() >= 3600:
            state["vision_budget_window_started"] = now
            state["vision_budget_used"] = 0
            return True
        return int(state.get("vision_budget_used", 0)) < int(settings.vision_budget_per_hour)

    def _vision_worker(frame, source: str) -> str:
        """Describe the frame the sampler captured.

        The frame is passed in rather than re-captured: a fresh screen grab reads
        the composited desktop, which includes the pet window and whatever else
        is on top, so the model would describe the wrong content.
        """
        from desktop_pet.vision.multimodal import describe_screen_image_compat

        if frame is None:
            return ""
        ocr_context = str(state.get("scan_cache_ocr_text", "") or "")
        result = describe_screen_image_compat(
            llm_client,
            frame,
            timeout_sec=float(settings.mm_timeout_sec),
            max_edge=int(settings.mm_image_max_edge),
            # Hybrid keeps the model from re-transcribing dialogue we already
            # have, so the budget is spent on art/expressions/UI instead.
            ocr_context=ocr_context[: int(settings.ocr_context_max_chars)],
            route="vision+ocr" if ocr_context else "vision_only",
            focus_query=str(settings.vision_focus_query),
            content_kind="image",
        )
        return str(result.summary or "").strip()

    def poll_vision_future() -> None:
        future = state.get("vision_future")
        if future is None:
            return
        if not future.done():
            started = state.get("vision_started_at")
            if isinstance(started, datetime):
                elapsed = int((datetime.now() - started).total_seconds())
                # Long calls are fine: they no longer block scanning. Only give
                # up on genuinely stuck requests so the slot frees up.
                if elapsed > max(30.0, float(settings.mm_timeout_sec) * 3.0):
                    log_heartbeat("vision_timeout", f"elapsed={elapsed}s")
                    state["vision_future"] = None
                    state["vision_started_at"] = None
            return
        state["vision_future"] = None
        state["vision_started_at"] = None
        try:
            summary = str(future.result() or "").strip()
        except Exception as exc:
            log_heartbeat("vision_error", str(exc))
            return
        if not summary:
            log_heartbeat("vision_empty")
            return
        state["vision_context"] = summary[:600]
        state["vision_context_at"] = datetime.now()
        log_heartbeat("vision_result", f"len={len(summary)}")

    def _grab_vision_frame():
        """Capture the frame the vision model should see, or None.

        PrintWindow first so an occluding window (a browser, the pet itself)
        cannot become the subject; screen grab only as a fallback.
        """
        from desktop_pet.vision.capture import (
            capture_absolute_rect,
            capture_window_client,
        )

        mode = str(state.get("vision_capture_mode", settings.vision_capture_mode))
        if mode == "manual":
            manual_box = state.get("vision_region")
            if manual_box:
                try:
                    frame = capture_primary_screen(
                        monitor_index=int(state["scan_monitor_index"]), region=None
                    )
                    left, top, width, height = (int(value) for value in manual_box)
                    img_w, img_h = frame.size
                    left = max(0, min(left, img_w - 1))
                    top = max(0, min(top, img_h - 1))
                    right = max(left + 8, min(img_w, left + max(8, width)))
                    bottom = max(top + 8, min(img_h, top + max(8, height)))
                    return frame.crop((left, top, right, bottom))
                except Exception:
                    return None
            mode = "window"

        if window_resolver is None:
            return None
        rect = window_resolver.resolve()
        if rect is None:
            return None
        frame = capture_window_client(int(getattr(rect, "hwnd", 0) or 0))
        if frame is None:
            frame = capture_absolute_rect(rect.left, rect.top, rect.width, rect.height)
        if frame is None:
            return None
        margin = int(state.get("vision_window_margin_px", settings.vision_window_margin_px))
        if margin:
            try:
                left = max(0, -margin)
                top = max(0, -margin)
                right = min(frame.size[0], frame.size[0] + margin)
                bottom = min(frame.size[1], frame.size[1] + margin)
                frame = frame.crop((left, top, right, bottom))
            except Exception:
                pass
        return frame

    def _vision_rate_per_minute() -> float:
        """Sampling quota, preferring the overrides the side panel sets."""
        return float(
            state.get("vision_shots_per_minute", settings.vision_shots_per_minute)
        )

    def _vision_quota_state(now: datetime) -> tuple[float, datetime]:
        """Token bucket: returns (tokens, last_refill_at)."""
        rate_per_sec = _vision_rate_per_minute() / 60.0
        cap = float(state.get("vision_burst_cap", settings.vision_burst_cap))
        tokens = state.get("vision_tokens")
        last = state.get("vision_tokens_at")
        if not isinstance(tokens, (int, float)) or not isinstance(last, datetime):
            # Start with a full burst so the first notable scene is not delayed.
            return cap, now
        tokens = float(tokens) + max(0.0, (now - last).total_seconds()) * rate_per_sec
        return min(cap, tokens), now

    def _vision_since_last_sec(now: datetime) -> float | None:
        """Seconds since the last submitted vision request, or None."""
        last = state.get("vision_last_request_at")
        if not isinstance(last, datetime):
            return None
        return max(0.0, (now - last).total_seconds())

    def vision_sampling_step() -> None:
        """One tick of the independent vision sampler.

        Deliberately separate from the OCR scan loop: vision wants "does the
        picture look interesting right now", OCR wants "read the text". Sharing
        one pipeline made vision depend on text arriving, which meant a CG cut
        or an opening movie could never be looked at.

        All gating lives in the pure :func:`vision_should_capture`; this wrapper
        only performs the actions.
        """
        try:
            if visual_novel_tracker is None:
                return

            now = datetime.now()
            tokens, refill_at = _vision_quota_state(now)
            state["vision_tokens"] = tokens
            state["vision_tokens_at"] = refill_at
            threshold = float(
                state.get("vision_change_threshold", settings.vision_change_threshold)
            )
            min_gap = float(state.get("vision_min_gap_sec", settings.vision_min_gap_sec))
            since_last = _vision_since_last_sec(now)

            should_look, reason = vision_should_capture(
                tokens=tokens,
                delta=None,
                in_flight=state.get("vision_future") is not None,
                enabled=bool(
                    settings.enable_multimodal_vision and settings.enable_mm_screen_comment
                ),
                active=bool(state.get("scan_enabled")),
                budget_ok=_vision_budget_available(now),
                threshold=threshold,
                shots_per_minute=_vision_rate_per_minute(),
                since_last_sec=since_last,
                min_gap_sec=min_gap,
            )
            if not should_look:
                state["vision_skip_reason"] = reason
                return

            _refresh_vision_exclusions()
            frame = _grab_vision_frame()
            if frame is None:
                state["vision_skip_reason"] = "no_frame"
                return

            fingerprint = _frame_fingerprint(frame)
            cached = state.get("vision_sampler_fingerprint")
            delta = 1.0
            if isinstance(cached, (bytes, bytearray)):
                delta = _fingerprint_diff_ratio(bytes(fingerprint), bytes(cached))
            state["vision_last_delta"] = float(delta)

            should_look, reason = vision_should_capture(
                tokens=tokens,
                delta=delta,
                in_flight=False,
                enabled=True,
                active=True,
                budget_ok=True,
                threshold=threshold,
                shots_per_minute=_vision_rate_per_minute(),
            )
            if not should_look:
                # Do NOT move the sampler fingerprint on a skip: keeping the last
                # *submitted* frame as the baseline is what lets a later big
                # change still register as a change.
                state["vision_skip_reason"] = f"画面没怎么变(delta={delta:.2f})"
                return
            state["vision_sampler_fingerprint"] = fingerprint

            # Consume quota only on a real submission.
            state["vision_tokens"] = max(0.0, tokens - 1.0)
            state["vision_future"] = vision_executor.submit(
                _vision_worker, frame, str(state.get("last_capture_source", "") or "")
            )
            state["vision_started_at"] = now
            state["vision_last_request_at"] = now
            state["vision_budget_used"] = int(state.get("vision_budget_used", 0)) + 1
            if state.get("vision_budget_window_started") is None or (
                (now - state["vision_budget_window_started"]).total_seconds() >= 3600
                if isinstance(state.get("vision_budget_window_started"), datetime)
                else True
            ):
                state["vision_budget_window_started"] = now
                state["vision_budget_used"] = 1
            state["vision_skip_reason"] = ""
            log_heartbeat(
                "vision_submit",
                f"delta={delta:.2f}, frame={frame.size[0]}x{frame.size[1]}, "
                f"tokens={state['vision_tokens']:.2f}/{float(state.get('vision_burst_cap', settings.vision_burst_cap)):.0f}",
            )
        except Exception as exc:
            log_heartbeat("vision_sampler_error", str(exc))

    def _current_visual_context() -> str:
        summary = str(state.get("vision_context", "") or "").strip()
        if not summary:
            return ""
        captured_at = state.get("vision_context_at")
        if isinstance(captured_at, datetime):
            age = (datetime.now() - captured_at).total_seconds()
            # Stale observations would describe a scene the player has left, and
            # a cutscene changes faster than a dialogue box does.
            if age > float(settings.vision_context_ttl_sec):
                return ""
        return summary

    def _check_game_window_change() -> bool:
        """Check the loaded cache against the captured game before writing OCR."""
        if not state.get("visual_novel_mode_enabled"):
            return True
        # Only a resolved game window counts as evidence; a full-screen grab is
        # not, and would compare against whatever happened to be on top.
        if str(state.get("last_capture_source", "")) != "window_direct":
            return True
        title = str(state.get("last_capture_title", "") or "").strip()
        if not title:
            return True

        cache_name = Path(visual_novel_tracker.path).name
        bound_title = visual_novel_tracker.game_title
        status = story_cache_game_status(
            bound_title, visual_novel_tracker.has_story_content, title
        )
        if status == "bind_empty":
            visual_novel_tracker.bind_game_title(title)
            log_heartbeat("story_game", f"bound_empty_cache={cache_name!r}, game={title!r}")
        elif status in {"unbound", "mismatch"}:
            state["story_game_pending"] = {"cache": cache_name, "title": title}
            state["story_game_mismatch"] = (
                f"缓存「{Path(cache_name).stem}」"
                + (f"绑定「{bound_title}」" if bound_title else "尚未确认所属游戏")
                + f"；当前窗口「{title}」"
            )
            on_scan_toggle(False, announce=False)
            log_heartbeat("story_game", f"scan_paused: {status}, cache={cache_name!r}, game={title!r}")
            chat.append_message(
                "系统",
                f"⚠ {state['story_game_mismatch']}。为避免串台，视觉小说扫描已暂停。"
                "请载入或新建一个剧情缓存；若确认当前缓存属于这款游戏，"
                "再次开启自动扫描并确认绑定。",
            )
            pet.show_comment_bubble("剧情缓存尚未匹配，扫描暂停", duration_ms=5000)
            refresh_vn_sidebar_state()
            return False

        state["story_game_pending"] = None
        state["story_game_mismatch"] = ""
        previous = str(state.get("last_game_title", "") or "").strip()
        # Keep the old window baseline for diagnostics and region reset, but
        # the per-cache binding above is the authority for memory isolation.
        state["last_game_title"] = title
        if previous and game_title_changed(previous, title):
            state["vn_ocr_filter"] = {}
            # The vision resolver pins to the game it last resolved. Left alone
            # it keeps aiming at the previous game's window -- the startup line
            # read "target=星空列车与白的旅行" while 青空下的加缪 was on screen --
            # so drop the stale pin and let it re-resolve.
            if window_resolver is not None:
                try:
                    if str(getattr(window_resolver, "sticky_title", "") or "") == previous:
                        window_resolver.forget_target()
                except Exception:
                    pass
            if str(_persisted_regions.get("target_title", "") or "") == previous:
                _persisted_regions.pop("target_title", None)
        if _persisted_regions.get("last_game_title") != title:
            _persisted_regions["last_game_title"] = title
            _save_persisted_regions()
        return True

    def _process_visual_novel_text(result: dict, now: datetime) -> None:
        if visual_novel_tracker is None:
            return
        poll_vision_future()
        state["last_capture_source"] = str(result.get("capture_source", "") or "")
        state["last_capture_title"] = str(result.get("capture_title", "") or "")
        if window_resolver is not None and state["last_capture_title"]:
            window_resolver.set_target_title(state["last_capture_title"])
        if not _check_game_window_change():
            return
        state["last_capture_warning"] = str(result.get("capture_warning", "") or "")

        # Vision is sampled by its own loop (vision_sampling_step), driven by
        # scene change and a token bucket -- not by dialogue text. Nothing to do
        # here for vision.

        ocr_text = str(result.get("ocr_text", "") or "").strip()
        if not ocr_text:
            # Surfacing the empty case is essential: without it an OCR failure
            # looks identical to "the story is quiet".
            log_heartbeat(
                "vn_ocr",
                f"empty, source={state['last_capture_source']}, "
                f"window={state['last_capture_title']!r}, "
                f"conf={float(result.get('ocr_confidence', 0.0) or 0.0):.2f}, "
                f"route={result.get('vision_route', '?')}",
            )
            return
        log_heartbeat(
            "vn_ocr",
            f"chars={len(ocr_text)}, source={state['last_capture_source']}, "
            f"window={state['last_capture_title']!r}, "
            f"conf={float(result.get('ocr_confidence', 0.0) or 0.0):.2f}",
        )
        state["last_nonempty_ocr_text"] = ocr_text
        state["last_nonempty_ocr_at"] = now
        # Text arrived, so the "how long has the box been unreadable" clock that
        # gates the band sweep starts over.
        state["no_text_since"] = None
        ocr_text, quality_reason = visual_novel_tracker.prepare_ocr_text(
            ocr_text,
            confidence=float(result.get("ocr_confidence", 0.0) or 0.0),
        )
        if not ocr_text:
            log_heartbeat("vn_skip", quality_reason)
            return
        if quality_reason != "quality_ok":
            log_heartbeat("vn_filter", quality_reason)
        observation = visual_novel_tracker.observe_ocr(
            ocr_text,
            confidence=float(result.get("ocr_confidence", 0.0) or 0.0),
            now=float(result.get("sampled_at", time.monotonic())),
            frame_stable=bool(result.get("frame_stable", False)),
        )
        if not observation.accepted:
            log_heartbeat("vn_stabilize" if observation.reason == "stabilizing" else "vn_skip", observation.reason)
        else:
            log_heartbeat(
                "vn_buffer",
                f"reason={observation.reason}, semantic_score={observation.semantic_score:.2f}, "
                f"frame_age_ms={round((time.monotonic() - float(result.get('sampled_at', time.monotonic()))) * 1000)}",
            )

        if not visual_novel_tracker.has_pending_evaluation or state["comment_future"] is not None:
            return

        retry_after = state.get("vn_eval_retry_after")
        if isinstance(retry_after, datetime) and now < retry_after:
            log_heartbeat("vn_wait", "evaluation_backoff")
            return

        last_evaluation_at = state.get("vn_last_evaluation_at")
        if isinstance(last_evaluation_at, datetime) and (now - last_evaluation_at).total_seconds() < 12.0:
            log_heartbeat("vn_wait", "evaluation_rate_limit")
            return

        # The cooldown is a backstop, not the pacing mechanism: the semantic peak
        # detector and the moment judgement decide *whether* to speak. Waiting
        # out a fixed clock here used to delay comments past the scene that
        # triggered them, so the reply described content the player had left.
        cooldown_sec = max(8.0, float(settings.auto_comment_cooldown_sec) * 0.33)
        allow_comment = can_emit_comment(state["last_comment_at"], int(cooldown_sec))
        payload = visual_novel_tracker.build_evaluation_payload()
        if payload is None:
            return
        payload["personalization_hint"] = dialog.build_personalization_hint()
        visual_context = _current_visual_context()
        if visual_context:
            payload["visual_context"] = visual_context
        state["vn_last_evaluation_at"] = now
        meta = {
            "kind": "visual_novel", "now": now, "allow_comment": allow_comment,
            "reason": str(payload.get("reason", "")),
            "summary_len": len(str(payload.get("scene_summary", ""))),
            "visual_used": bool(visual_context),
            "story_path": str(visual_novel_tracker.path),
            "submitted_at": time.monotonic(),
        }
        state["pending_comment_meta"] = meta
        def on_reaction(evaluation):
            def deliver():
                if state.get("pending_comment_meta") is not meta:
                    return
                log_heartbeat("vn_reaction_ready", f"elapsed_ms={round((time.monotonic() - meta['submitted_at']) * 1000)}")
                _publish_comment(evaluation.get("comment", ""), meta)
            speech_ui.ready.emit(deliver)
        state["comment_future"] = comment_executor.submit(
            comment_engine.evaluate_visual_novel, payload, on_reaction,
        )
        log_heartbeat(
            "vn_evaluate",
            f"reason={payload.get('reason', 'unknown')}, visual={int(bool(visual_context))}, "
            f"cooldown_ok={int(allow_comment)}",
        )

    def _sync_plot_memory(evaluation: dict) -> None:
        """Mirror newly learned plot facts into the long-term memory store.

        The story cache remains the fast working set; this makes plot knowledge
        reachable from the ordinary chat path, which only ranks SQLite rows.
        Only newly merged facts are forwarded so the store does not fill up with
        the paraphrased duplicates the tracker already collapsed.
        """
        if visual_novel_tracker is None:
            return
        added = evaluation.get("added_facts")
        new_facts = [str(item) for item in added if str(item).strip()] if isinstance(added, list) else []
        last_sync = state.get("plot_memory_synced_line")
        total_lines = visual_novel_tracker.total_lines
        due = (
            not isinstance(last_sync, int)
            or (total_lines - last_sync) >= 24
            or len(new_facts) >= 3
        )
        if not due and str(evaluation.get("moment_type", "")) not in {"twist", "choice", "tender"}:
            return
        digest = visual_novel_tracker.scene_session_digest(max_chars=420)
        if not digest and not new_facts:
            return
        try:
            dialog.append_plot_memory(
                story_title=visual_novel_tracker.story_title,
                digest=digest,
                facts=new_facts[:4],
                confidence=0.68,
            )
            state["plot_memory_synced_line"] = total_lines
            log_heartbeat("vn_memory_sync", f"facts={len(new_facts[:4])}, line={total_lines}")
        except Exception as exc:
            log_heartbeat("vn_memory_sync_error", str(exc))

    def _retry_visual_novel_evaluation(reason: str) -> None:
        if visual_novel_tracker is not None:
            visual_novel_tracker.restore_evaluation_cursor()
        streak = int(state["vn_eval_failure_streak"]) + 1
        state["vn_eval_failure_streak"] = streak
        if streak >= 3:
            state["vn_eval_retry_after"] = datetime.now() + timedelta(seconds=90)
        log_heartbeat("vn_retry", f"reason={reason}, streak={streak}, backoff={int(streak >= 3)}")

    def _publish_comment(comment, meta) -> bool:
        is_valid = None
        if meta.get("kind") == "visual_novel":
            if meta.get("reaction_emitted"):
                return False
            is_valid = lambda: (state.get("visual_novel_mode_enabled", True)
                                and str(visual_novel_tracker.path) == meta.get("story_path")
                                and time.monotonic() - meta["submitted_at"] <= 20.0)
            if not is_valid():
                log_heartbeat("vn_skip", "comment_expired_memory_preserved")
                return False
            cooldown_sec = max(8.0, float(settings.auto_comment_cooldown_sec) * 0.33)
            if not can_emit_comment(state["last_comment_at"], int(cooldown_sec)):
                log_heartbeat("vn_skip", "cooldown")
                return False
        comment = str(comment or "").strip()
        if not comment:
            log_heartbeat("skip", "comment_empty_or_fallback_suppressed")
            return

        similarity = text_similarity(comment, state["last_comment_text"])
        if similarity >= settings.comment_similarity_skip_threshold:
            log_heartbeat(
                "skip",
                f"duplicate_comment similarity={similarity:.2f} threshold={settings.comment_similarity_skip_threshold:.2f}",
            )
            return

        state["last_comment_at"] = datetime.now() if meta.get("kind") == "visual_novel" else meta.get("now", datetime.now())
        state["last_summary"] = meta.get("cycle_signature", "")
        state["last_comment_text"] = comment
        on_display = None
        if meta.get("kind") == "visual_novel" and visual_novel_tracker is not None:
            def on_display():
                visual_novel_tracker.remember_comment(comment)
                log_heartbeat("vn_comment_displayed", f"request_to_display_ms={round((time.monotonic() - meta['submitted_at']) * 1000)}")
        role = "桌宠(视觉小说)" if meta.get("kind") == "visual_novel" else "桌宠(自动)"
        _emit_comment(comment, role=role, is_valid=is_valid, on_display=on_display)
        log_heartbeat(
            "emit",
            f"cycle_items={meta.get('cycle_items', 0)}, summary_len={meta.get('summary_len', 0)}",
        )
        meta["reaction_emitted"] = True
        return True

    def poll_comment_future() -> None:
        future = state["comment_future"]
        if future is None:
            return
        if not future.done():
            log_heartbeat("comment_busy")
            return

        state["comment_future"] = None
        meta = state["pending_comment_meta"] or {}
        state["pending_comment_meta"] = None

        try:
            generated = future.result()
        except Exception as exc:
            if meta.get("kind") == "visual_novel":
                _retry_visual_novel_evaluation(f"future_error:{type(exc).__name__}")
                return
            log_heartbeat("error", f"comment_generate_failed: {exc}")
            return

        if meta.get("kind") == "visual_novel":
            evaluation = generated if isinstance(generated, dict) else {}
            if visual_novel_tracker is not None:
                if not evaluation:
                    _retry_visual_novel_evaluation(comment_engine.last_visual_novel_error or "empty_evaluation")
                    return
                state["vn_eval_failure_streak"] = 0
                state["vn_eval_retry_after"] = None
                visual_novel_tracker.apply_evaluation(evaluation)
                _sync_plot_memory(evaluation)
            if not bool(evaluation.get("should_comment", False)):
                log_heartbeat(
                    "vn_skip",
                    f"moment={evaluation.get('moment_type', 'ordinary')}, "
                    f"via={evaluation.get('reaction_source', 'none')}, "
                    f"reaction_repeats={int(bool(evaluation.get('reaction_repeats', False)))}",
                )
                return
            comment = str(evaluation.get("comment", "")).strip()
        else:
            comment = generated

        _publish_comment(comment, meta)

    def do_auto_comment() -> None:
        try:
            log_heartbeat("tick")
            poll_comment_future()
            if not state["scan_enabled"]:
                log_heartbeat("skip", "scan_disabled")
                return

            future = state["scan_future"]
            if future is None:
                now = datetime.now()
                if (
                    settings.enable_live2d_py
                    and not state["visual_novel_mode_enabled"]
                    and (runtime_follow_effective or runtime_drag_active)
                ):
                    # Keep interaction smooth by postponing OCR submission while
                    # follow/drag is active. Visual-novel mode is deliberately
                    # exempt: the panel is opened by hovering the pet, so this
                    # gate would stall the scan every time the user reaches for
                    # it -- and there the steady cadence matters more than
                    # shaving latency off the cursor follow.
                    wait_reason = "drag" if runtime_drag_active else "follow"
                    log_heartbeat("scan_wait", f"interaction_priority={wait_reason}")
                    return

                backoff_until = state.get("scan_backoff_until")
                if isinstance(backoff_until, datetime) and now < backoff_until:
                    remain = int((backoff_until - now).total_seconds())
                    log_heartbeat("scan_cooldown", f"remain={max(1, remain)}s")
                    return

                adaptive_submit_interval_sec = max(
                    float(scan_submit_min_interval_sec),
                    float(state.get("adaptive_submit_interval_sec", scan_submit_min_interval_sec)),
                )
                if state["visual_novel_mode_enabled"]:
                    if time.monotonic() < float(state["vn_scan_due_at"]):
                        return
                    adaptive_submit_interval_sec = 0.0
                last_submit_at = state.get("last_scan_submit_at")
                if last_submit_at is not None:
                    elapsed_sec = (now - last_submit_at).total_seconds()
                    if elapsed_sec < adaptive_submit_interval_sec:
                        remaining = int(round(adaptive_submit_interval_sec - elapsed_sec))
                        log_heartbeat("scan_wait", f"submit_in={max(1, remaining)}s")
                        return

                if settings.enable_scan_subprocess:
                    task_id = _submit_scan_worker_task(
                        int(state["scan_monitor_index"]),
                        state["scan_region"],
                    )
                    if task_id <= 0:
                        warming = scan_worker_ready is not None and not scan_worker_ready.is_set()
                        log_heartbeat(
                            "scan_wait",
                            "worker_warming" if warming else "worker_queue_full",
                        )
                        return
                    state["scan_future"] = int(task_id)
                else:
                    state["scan_future"] = scan_executor.submit(run_scan_pipeline)
                state["scan_started_at"] = now
                state["last_scan_submit_at"] = now
                pet.set_scan_busy(True)
                log_heartbeat("scan_submit")
                return

            if settings.enable_scan_subprocess:
                result_msg = _poll_scan_worker_result(int(future))
                if result_msg is None:
                    started_at = state["scan_started_at"]
                    elapsed = int((datetime.now() - started_at).total_seconds()) if started_at else 0
                    active_busy_timeout_sec = max(
                        float(scan_busy_timeout_sec),
                        float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec)),
                    )
                    if elapsed >= active_busy_timeout_sec:
                        _reset_scan_worker(f"timeout={elapsed}s")
                        return
                    log_heartbeat("scan_busy", f"elapsed={elapsed}s/{int(active_busy_timeout_sec)}s")
                    return
            elif not future.done():
                started_at = state["scan_started_at"]
                elapsed = int((datetime.now() - started_at).total_seconds()) if started_at else 0
                active_busy_timeout_sec = max(
                    float(scan_busy_timeout_sec),
                    float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec)),
                )
                if elapsed >= active_busy_timeout_sec:
                    _reset_scan_worker(f"timeout={elapsed}s")
                    return
                log_heartbeat("scan_busy", f"elapsed={elapsed}s/{int(active_busy_timeout_sec)}s")
                return

            scan_started_at = state.get("scan_started_at")
            state["scan_future"] = None
            state["scan_started_at"] = None
            pet.set_scan_busy(False)
            if isinstance(scan_started_at, datetime):
                duration_sec = max(0.0, (datetime.now() - scan_started_at).total_seconds())
            else:
                duration_sec = 0.0
            state["last_scan_duration_sec"] = duration_sec
            state["scan_timeout_streak"] = 0
            state["scan_backoff_until"] = None
            measured_busy_timeout = max(float(scan_busy_timeout_sec), duration_sec * 2.2)
            prev_busy_timeout = float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec))
            state["adaptive_busy_timeout_sec"] = max(
                float(scan_busy_timeout_sec),
                min(90.0, (prev_busy_timeout * 0.7) + (measured_busy_timeout * 0.3)),
            )

            target_interval = max(float(scan_submit_min_interval_sec), duration_sec * 1.35)
            prev_interval = float(state.get("adaptive_submit_interval_sec", scan_submit_min_interval_sec))
            # In visual-novel mode the user wants a steady fast cadence. The
            # adaptive gate would otherwise stretch the interval to ~1.35x the
            # OCR duration (6-9s), which reads as "the scan is far too slow".
            max_interval = max(
                float(scan_submit_min_interval_sec),
                float(settings.scan_adaptive_max_interval_sec)
                if bool(state.get("visual_novel_mode_enabled"))
                else 90.0,
            )
            if target_interval > prev_interval:
                new_interval = min(max_interval, target_interval)
            else:
                # Recover gradually when scan load becomes lighter.
                new_interval = max(float(scan_submit_min_interval_sec), prev_interval * 0.82)
            state["adaptive_submit_interval_sec"] = new_interval

            if settings.enable_scan_subprocess:
                if not result_msg.get("ok", False):
                    _reset_scan_worker(f"worker_error={result_msg.get('error', 'unknown')}")
                    return
                result = result_msg.get("result", {})
                duration_sec = max(float(duration_sec), float(result_msg.get("duration_sec", 0.0)))
            else:
                result = future.result()
            if state["visual_novel_mode_enabled"] and result.get("capture_source") == "target_missing":
                on_scan_toggle(False, announce=False)
                log_heartbeat("vn_target_missing", "scan_paused")
                chat.append_message("系统", "已锁定的游戏窗口不可用，视觉小说扫描已暂停。重新打开游戏后可再开启扫描。")
                pet.show_comment_bubble("游戏窗口已关闭，扫描暂停", duration_ms=2400)
                return
            if not result["ocr_ok"]:
                log_heartbeat("ocr_status", result["ocr_info"])

            # Strip the game's menu-bar words before anything else sees the text.
            # OCR returns them glued to the dialogue and the quality filter
            # accepts the mixture, so without this the chrome becomes "dialogue"
            # in the story cache.
            raw_ocr_text = str(result.get("ocr_text", "") or "")
            cleaned_ocr_text = strip_ui_chrome(raw_ocr_text)
            if cleaned_ocr_text != raw_ocr_text:
                log_heartbeat(
                    "ocr_chrome",
                    f"removed={len(raw_ocr_text) - len(cleaned_ocr_text)}chars",
                )
                result["ocr_text"] = cleaned_ocr_text

            # Publish this frame's reading for the side panel. Only the
            # in-process scan path used to write these, so with
            # ENABLE_SCAN_SUBPROCESS (the default) the panel reported
            # "当前帧没有文字" forever while still showing the last recognised
            # line from last_nonempty_ocr_text -- the text and the "no text"
            # warning appeared together. Writing it here covers both paths, and
            # is a no-op repeat for the in-process one, which already stored it.
            state["scan_cache_ocr_text"] = str(result.get("ocr_text", "") or "")
            state["scan_cache_ocr_confidence"] = float(
                result.get("ocr_confidence", 0.0) or 0.0
            )

            mode = result["mode"]
            screen_context = result["screen_context"]
            vision_route = str(result.get("vision_route", "none"))
            reused_cache = bool(result.get("reused_cache", False))
            if settings.enable_scan_subprocess:
                scene_summary = str(result.get("scene_summary", "")).strip()
                scene_should_comment = bool(result.get("scene_should_comment", False))
                mm_reason = str(result.get("mm_reason", "off"))
                mm_elapsed_ms = int(result.get("mm_elapsed_ms", 0) or 0)
                mm_summary_len = int(result.get("mm_summary_len", 0) or 0)
                log_heartbeat(
                    "mm_result",
                    f"route={vision_route}, reason={mm_reason}, elapsed_ms={mm_elapsed_ms}, len={mm_summary_len}",
                )
            else:
                scene_obj = result["scene"]
                scene_summary = scene_obj.summary
                scene_should_comment = scene_obj.should_comment
            cache_tag = "cache" if reused_cache else "fresh"
            ocr_note = str(result.get("ocr_note", "") or "")
            log_heartbeat(
                "context_built",
                (
                    f"mode={mode}, route={vision_route}, src={cache_tag}, len={len(screen_context)}, "
                    f"ocr={result.get('ocr_chars', 0)}chars{('/' + ocr_note) if ocr_note else ''}, "
                    f"scan_dur={duration_sec:.2f}s, ocr_ms={result.get('ocr_elapsed_ms', 0)}, "
                    f"worker_ms={round(float(result_msg.get('duration_sec', 0)) * 1000) if settings.enable_scan_subprocess else 0}, "
                    f"stable={int(bool(result.get('frame_stable', False)))}, next_submit_min={state['adaptive_submit_interval_sec']:.1f}s, "
                    f"busy_timeout={state['adaptive_busy_timeout_sec']:.1f}s"
                ),
            )
            maybe_emit_pipeline_status(mode)

            if state["visual_novel_mode_enabled"]:
                if result.get("scan_deferred", False):
                    log_heartbeat("vn_settling", "waiting_for_glyphs")
                else:
                    _process_visual_novel_text(result, datetime.now())
                pending_ocr = visual_novel_tracker is not None and visual_novel_tracker.has_pending_ocr
                needs_recheck = (
                    result.get("scan_deferred", False)
                    or (pending_ocr and result.get("ocr_text"))
                    or not result.get("frame_stable", False)
                )
                delay_sec = 0.2 if needs_recheck else 1.0
                state["vn_scan_due_at"] = time.monotonic() + delay_sec
                return

            if scene_should_comment:
                _append_cycle_memory(scene_summary)

            now = datetime.now()
            if now < state["next_comment_at"]:
                due = state["next_comment_at"].strftime("%H:%M:%S")
                log_heartbeat("wait", f"next_due={due}, memory_count={len(state['cycle_memories'])}")
                return

            cycle_summary, cycle_signature = _compose_cycle_summary(now)
            if not cycle_summary:
                log_heartbeat("skip", "cycle_summary_empty")
                state["next_comment_at"] = schedule_next_comment(now)
                return

            if cycle_signature == state["last_summary"]:
                log_heartbeat("skip", "duplicate_cycle_summary")
                state["cycle_memories"] = []
                state["next_comment_at"] = schedule_next_comment(now)
                return

            if not can_emit_comment(state["last_comment_at"], settings.auto_comment_cooldown_sec):
                log_heartbeat("skip", "cooldown")
                return

            if state["comment_future"] is not None:
                log_heartbeat("skip", "comment_worker_busy")
                return

            long_memory_hint = dialog.build_light_long_memory_hint(
                limit=settings.screen_comment_memory_limit,
                query=cycle_summary,
            )
            recent_dialog_hint = dialog.build_recent_session_hint(limit=8)
            state["comment_future"] = comment_executor.submit(
                comment_engine.comment_on_summary,
                cycle_summary,
                long_memory_hint,
                settings.screen_comment_memory_weight,
                recent_dialog_hint,
                True,
                dialog.build_personalization_hint(),
            )
            state["pending_comment_meta"] = {
                "now": now,
                "cycle_signature": cycle_signature,
                "cycle_items": len(state["cycle_memories"]),
                "summary_len": len(cycle_summary),
            }
            state["cycle_memories"] = []
            state["next_comment_at"] = schedule_next_comment(now)
            log_heartbeat("comment_submit", f"next_due={state['next_comment_at'].strftime('%H:%M:%S')}")
        except Exception as exc:
            pet.set_scan_busy(False)
            log_heartbeat("error", str(exc))
            chat.append_message("系统", f"自动评论失败: {exc}")

    def do_manual_comment() -> None:
        try:
            image = capture_primary_screen(
                monitor_index=int(state["scan_monitor_index"]),
                region=state["scan_region"],
            )
            pipeline = build_screen_context(image, respect_vision_interval=False)
            screen_context = str(pipeline["screen_context"])
            mode = str(pipeline["mode"])
            maybe_emit_pipeline_status(mode)
            scene = analyze_scene(screen_context)
            if not scene.summary:
                chat.append_message("系统", "手动评论失败: 未识别到有效内容")
                return

            long_memory_hint = dialog.build_light_long_memory_hint(
                limit=settings.screen_comment_memory_limit,
                query=scene.summary,
            )
            recent_dialog_hint = dialog.build_recent_session_hint(limit=8)
            comment = comment_engine.comment_on_summary(
                scene.summary,
                long_memory_hint,
                settings.screen_comment_memory_weight,
                recent_dialog_hint,
                personalization_hint=dialog.build_personalization_hint(),
            )
            _emit_comment(comment, role="桌宠(手动)")
        except Exception as exc:
            chat.append_message("系统", f"手动评论失败: {exc}")

    def on_scan_toggle(enabled: bool, announce: bool = True) -> None:
        # ``announce`` is False when visual-novel mode drives this toggle on the
        # user's behalf: the VN handler folds the scan state into its own single
        # message instead of stacking two bubbles on top of each other.
        nonlocal scan_executor
        enabled = bool(enabled)
        pending = state.get("story_game_pending")
        if enabled and announce and state["visual_novel_mode_enabled"] and isinstance(pending, dict):
            cache_name = Path(visual_novel_tracker.path).name
            if pending.get("cache") == cache_name:
                title = str(pending.get("title", "") or "")
                answer = QMessageBox.question(
                    pet,
                    "确认剧情缓存所属游戏",
                    f"将缓存「{Path(cache_name).stem}」绑定到当前游戏「{title}」并继续扫描？\n"
                    "确认后，此缓存不会自动用于其他游戏。",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    pet.set_auto_scan_enabled(False)
                    return
                try:
                    visual_novel_tracker.bind_game_title(title)
                except Exception as exc:
                    QMessageBox.warning(pet, "绑定剧情缓存失败", str(exc))
                    pet.set_auto_scan_enabled(False)
                    return
                state["story_game_pending"] = None
                state["story_game_mismatch"] = ""
                refresh_vn_sidebar_state()
            else:
                state["story_game_pending"] = None
        state["scan_enabled"] = enabled
        if announce:
            # A deliberate user press outranks visual-novel mode's auto-start, so
            # leaving VN mode must not undo what they just chose.
            state["vn_scan_autostarted"] = False
        pet.set_auto_scan_enabled(enabled)
        if enabled:
            if settings.enable_scan_subprocess:
                _start_scan_worker()
            elif scan_executor is None:
                scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")
            state["next_comment_at"] = schedule_next_comment(datetime.now())
            state["cycle_memories"] = []
            state["scan_timeout_streak"] = 0
            state["scan_backoff_until"] = None
            state["adaptive_submit_interval_sec"] = float(scan_submit_min_interval_sec)
            state["adaptive_busy_timeout_sec"] = float(scan_busy_timeout_sec)
            pet.set_scan_busy(False)
            _apply_runtime_resource_policy(scan_active=False, force=True)
            scheduler.start()
            if announce:
                chat.append_message("系统", "已开启自动扫描")
                pet.show_comment_bubble("自动扫描已开启", duration_ms=1800)
        else:
            scheduler.stop()
            scan_future = state.get("scan_future")
            if scan_future is not None and hasattr(scan_future, "cancel"):
                try:
                    scan_future.cancel()
                except Exception:
                    pass
            state["scan_future"] = None
            state["scan_started_at"] = None
            state["last_scan_submit_at"] = None
            comment_future = state.get("comment_future")
            if comment_future is not None and hasattr(comment_future, "cancel"):
                try:
                    comment_future.cancel()
                except Exception:
                    pass
            state["comment_future"] = None
            state["pending_comment_meta"] = None
            state["vision_future"] = None
            state["vision_started_at"] = None
            state["cycle_memories"] = []
            if settings.enable_scan_subprocess:
                _stop_scan_worker()
            elif scan_executor is not None:
                scan_executor.shutdown(wait=False, cancel_futures=True)
                scan_executor = None
            pet.set_scan_busy(False)
            _apply_runtime_resource_policy(scan_active=False, force=True)
            if announce:
                chat.append_message("系统", "已关闭自动扫描；手动评论和聊天多模态仍可使用")
                pet.show_comment_bubble("自动扫描已关闭", duration_ms=1800)

    def _loaded_story_name() -> str:
        """Name of the cache actually loaded in memory.

        The panel used to print ``visual_novel_library.active_name``, which is
        read from the marker file. The tracker is what actually loads and owns a
        story, so whenever those two disagreed -- a failed or partial switch --
        the label kept naming the old cache while the tracker was on the new one.
        The loaded file is the truth; the marker is only the fallback.
        """
        if visual_novel_tracker is not None:
            try:
                name = Path(visual_novel_tracker.path).name
                if name:
                    return name
            except Exception:
                pass
        try:
            return str(visual_novel_library.active_name or "")
        except Exception:
            return ""

    def refresh_vn_sidebar_state() -> None:
        """Push live state into the visual-novel side panel."""
        sidebar = getattr(pet, "vn_sidebar", None)
        if sidebar is None:
            return
        try:
            sidebar.set_story_name(_loaded_story_name())
        except Exception:
            pass
        _refresh_ocr_status(sidebar)
        if visual_novel_tracker is None:
            return
        sidebar.set_memory_status(
            f"事实 {len(visual_novel_tracker.facts)} 条 · 场景 {len(visual_novel_tracker.scene_log)} 段\n"
            f"已扫描 {visual_novel_tracker.total_lines} 行 · 已归档 {visual_novel_tracker.summarized_line_count} 行"
        )
        vision_ctx = str(state.get("vision_context", "") or "").strip()
        if not settings.enable_multimodal_vision or not settings.enable_mm_screen_comment:
            status = "未启用（需 ENABLE_MULTIMODAL_VISION 与 ENABLE_MM_SCREEN_COMMENT）"
        elif vision_ctx:
            captured_at = state.get("vision_context_at")
            age = int((datetime.now() - captured_at).total_seconds()) if isinstance(captured_at, datetime) else -1
            status = f"最近一次：{age}s 前（{len(vision_ctx)}字）\n{vision_ctx[:80]}"
        else:
            reason = str(state.get("vision_skip_reason", "") or "等待触发")
            mapping = {
                "in_flight": "正在请求中",
                "disabled": "未启用",
                "scan_off": "未开启自动扫描",
                "no_window_rect": "未找到可截取的游戏窗口",
                "story_idle": "剧情没有推进",
                "min_interval": "未到最小间隔",
                "unchanged": "画面没有明显变化",
                "budget": "本小时额度已用完",
            }
            status = mapping.get(reason, reason)
            if not state.get("scan_enabled"):
                status += (
                    "\n提示：画面理解依赖自动扫描。"
                    "开启视觉小说模式会自动打开它，也可在桌宠面板手动开启。"
                )
        sidebar.set_vision_status(status)

    def _refresh_ocr_status(sidebar) -> None:
        """Report what OCR last produced, so an empty result is visible."""
        region = state.get("ocr_region")
        if isinstance(region, tuple) and len(region) == 4:
            area = f"已框选 {int(region[2])}×{int(region[3])}"
        else:
            ratio = float(state.get("visual_novel_text_ratio", settings.visual_novel_text_ratio))
            area = f"自动（窗口下方 {max(0.0, 1.0 - ratio) * 100:.0f}%）"
        pinned = ""
        if window_resolver is not None:
            pinned = str(getattr(window_resolver, "sticky_title", "") or "")
        target_title = str(state.get("last_capture_title", "") or "") or pinned
        mismatch = str(state.get("story_game_mismatch", "") or "")
        if mismatch:
            # Keep it visible after the bubble has gone: this is the state that
            # silently corrupts memory if it is forgotten.
            sidebar.set_target_status(f"⚠ {mismatch[:80]}")
        elif pinned:
            sidebar.set_target_status(f"已锁定：{pinned[:34]}")
        elif target_title:
            sidebar.set_target_status(f"自动选择：{target_title[:34]}")
        else:
            sidebar.set_target_status("自动选择（尚未抓到窗口）")
        text = str(state.get("scan_cache_ocr_text", "") or "").strip()
        confidence = float(state.get("scan_cache_ocr_confidence", 0.0) or 0.0)
        source = str(state.get("last_capture_source", "") or "-")
        title = str(state.get("last_capture_title", "") or "")
        target = f"{source}" + (f"「{title}」" if title else "")
        previous = str(state.get("last_nonempty_ocr_text", "") or "").strip()
        previous_at = state.get("last_nonempty_ocr_at")
        # A scan in flight is why the panel can sit unchanged for many seconds:
        # one OCR pass is ~3s, and the fallback sweep is several of them. Saying
        # so beats looking stuck -- and an empty-looking panel during startup was
        # reported as "OCR is not working" when it was simply still running.
        scanning_for = -1
        if state.get("scan_future") is not None:
            started_at = state.get("scan_started_at")
            if isinstance(started_at, datetime):
                scanning_for = int((datetime.now() - started_at).total_seconds())
        if scanning_for >= 0:
            head = f"扫描中…（已 {scanning_for}s）"
            if text:
                preview = re.sub(r"\s+", " ", text)[:60]
                body = f"{head}\n上一轮：{len(text)} 字\n{preview}"
            elif previous:
                previous_preview = re.sub(r"\s+", " ", previous)[:60]
                body = f"{head}\n上一轮成功：{len(previous)} 字\n{previous_preview}"
            else:
                body = f"{head}\n首次识别需要几秒，请稍候"
        elif text:
            preview = re.sub(r"\s+", " ", text)[:60]
            body = f"最近：{len(text)} 字 / 置信度 {confidence:.2f}\n{preview}"
        else:
            # Keep the last successful reading visible: dialogue boxes come and
            # go (scene transitions, menus), and showing "no text" then hides the
            # fact that OCR is working.
            if previous:
                age = (
                    int((datetime.now() - previous_at).total_seconds())
                    if isinstance(previous_at, datetime)
                    else -1
                )
                previous_preview = re.sub(r"\s+", " ", previous)[:60]
                body = (
                    f"当前帧没有文字（{age}s 前读到过 {len(previous)} 字）\n"
                    f"{previous_preview}"
                )
            else:
                body = "最近：还没有读到文字"
            # Say why the retry sweep is not running: an unreadable box that does
            # nothing looks like a bug, when it is usually the quiet window or an
            # unchanged frame.
            skip = str(state.get("band_sweep_skip", "") or "")
            if skip.startswith("quiet"):
                body += f"\n扫带等待中：连续无文字满 {settings.ocr_fallback_min_quiet_sec:g}s 才重扫"
            elif skip == "frame_unchanged":
                body += "\n画面未变化，不重扫"
        warning = str(state.get("last_capture_warning", "") or "")
        if warning:
            body += f"\n⚠ {warning}"
        sidebar.set_ocr_status(f"{area} · {target}\n{body}")

    def on_vn_capture_mode_changed(mode: str) -> None:
        selected = str(mode or "window").strip().lower()
        if selected not in {"window", "manual"}:
            selected = "window"
        state["vision_capture_mode"] = selected
        if window_resolver is not None:
            window_resolver.invalidate()
        label = {"window": "跟随游戏窗口", "manual": "手动框选画面"}.get(selected, selected)
        chat.append_message("系统", f"截图来源：{label}")
        if selected == "manual" and not state.get("vision_region"):
            chat.append_message("系统", "提示：还没有框选画面区域，请点「框选画面区域」")
        _invalidate_scan_cache()
        refresh_vn_sidebar_state()

    def on_vn_vision_quota_changed(value: int) -> None:
        """Vision shots per minute; the sampler's token refill rate."""
        state["vision_shots_per_minute"] = max(1, min(60, int(value)))

    def on_vn_change_threshold_changed(value: float) -> None:
        state["vision_change_threshold"] = max(0.0, min(1.0, float(value)))

    def on_vn_text_ratio_changed(value: float) -> None:
        state["visual_novel_text_ratio"] = max(0.2, min(0.95, float(value)))
        state["scan_cache_fingerprint"] = None
        refresh_vn_sidebar_state()

    def on_vn_window_margin_changed(value: int) -> None:
        state["vision_window_margin_px"] = int(value)
        if window_resolver is not None:
            window_resolver.invalidate()

    def on_vn_pin_target() -> None:
        """Pin capture to the topmost window right now (i.e. the game)."""
        if window_resolver is None:
            chat.append_message("系统", "窗口解析不可用")
            return
        window_resolver.forget_target()
        rect = window_resolver.resolve()
        if rect is None:
            chat.append_message("系统", f"没有可锁定的窗口（{window_resolver.last_error}）")
            return
        title = str(rect.title or "")
        if not title:
            chat.append_message("系统", "该窗口没有标题，无法锁定")
            return
        window_resolver.set_target_title(title)
        _persisted_regions["target_title"] = title
        state["story_game_pending"] = None
        state["story_game_mismatch"] = ""
        _save_persisted_regions()
        chat.append_message("系统", f"已锁定截图窗口：{title}")
        pet.show_comment_bubble(f"已锁定：{title[:16]}", duration_ms=2200)
        _invalidate_scan_cache()
        refresh_vn_sidebar_state()

    def on_vn_unpin_target() -> None:
        if window_resolver is not None:
            window_resolver.set_target_title("")
            window_resolver.forget_target()
        _persisted_regions.pop("target_title", None)
        state["last_capture_title"] = ""
        state["story_game_pending"] = None
        state["story_game_mismatch"] = ""
        _save_persisted_regions()
        chat.append_message("系统", "已取消锁定，改回自动挑最上层窗口")
        pet.show_comment_bubble("截图窗口已改为自动", duration_ms=1800)
        _invalidate_scan_cache()
        refresh_vn_sidebar_state()

    def on_vn_add_memory(text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        if not dialog.append_plot_memory(
            story_title=visual_novel_library.active_path.stem,
            digest=value,
            facts=[],
            confidence=0.85,
        ):
            chat.append_message("系统", "这条剧情记忆没有写入")
            return
        visual_novel_tracker.apply_evaluation({"facts": [value]})
        visual_novel_tracker.flush()
        pet.show_comment_bubble("记下了", duration_ms=1400)
        chat.append_message("系统", f"已新增剧情记忆：{value}")
        refresh_vn_sidebar_state()

    def _discard_pending_vn_evaluation() -> None:
        pending_meta = state.get("pending_comment_meta")
        if not isinstance(pending_meta, dict) or pending_meta.get("kind") != "visual_novel":
            return
        comment_future = state.get("comment_future")
        if comment_future is not None and hasattr(comment_future, "cancel"):
            try:
                comment_future.cancel()
            except Exception:
                pass
        state["comment_future"] = None
        state["pending_comment_meta"] = None

    def on_vn_reload_memory() -> None:
        _discard_pending_vn_evaluation()
        visual_novel_tracker.flush()
        visual_novel_tracker.switch_story(visual_novel_library.active_path)
        comment_engine.reset_scene_state()
        state["vn_eval_failure_streak"] = 0
        state["vn_eval_retry_after"] = None
        refresh_vn_sidebar_state()
        chat.append_message("系统", "已重新载入剧情缓存")
        pet.show_comment_bubble("剧情缓存已重新载入", duration_ms=1600)

    def on_vn_view_memory() -> None:
        from PyQt6.QtWidgets import QDialog, QTextEdit, QVBoxLayout
        visual_novel_tracker.flush()
        path = visual_novel_tracker.path
        viewer = QDialog(pet)
        viewer.setWindowTitle(f"剧情记忆：{path.name}")
        viewer.resize(700, 600)
        text = QTextEdit(viewer)
        text.setReadOnly(True)
        text.setPlainText(path.read_text(encoding="utf-8"))
        layout = QVBoxLayout(viewer)
        layout.addWidget(text)
        viewer.exec()

    def on_vn_reset_story() -> None:
        try:
            path = visual_novel_library.reset_story()
        except Exception as exc:
            chat.append_message("系统", f"清空剧情缓存失败：{exc}")
            return
        _discard_pending_vn_evaluation()
        visual_novel_tracker.switch_story(path)
        comment_engine.reset_scene_state()
        state["story_game_pending"] = None
        state["story_game_mismatch"] = ""
        state["vn_eval_failure_streak"] = 0
        state["vn_eval_retry_after"] = None
        state["scan_cache_fingerprint"] = None
        state["vision_sampler_fingerprint"] = None
        state["vision_context"] = ""
        refresh_vn_sidebar_state()
        pet.show_comment_bubble("当前剧情缓存已清空", duration_ms=2000)
        chat.append_message("系统", f"已清空剧情缓存：{path.stem}")

    def on_visual_novel_mode_toggled(enabled: bool) -> None:
        nonlocal scan_executor, scan_tick_interval_sec, scan_submit_min_interval_sec
        enabled = bool(enabled)
        if enabled == bool(state["visual_novel_mode_enabled"]):
            pet.set_visual_novel_mode_enabled(enabled)
            return

        state["visual_novel_mode_enabled"] = enabled
        visual_novel_runtime["enabled"] = enabled
        scan_tick_interval_sec = 1 if enabled else normal_scan_tick_interval_sec
        scan_submit_min_interval_sec = 1 if enabled else normal_scan_submit_min_interval_sec
        scheduler.set_interval(scan_tick_interval_sec)
        state["adaptive_submit_interval_sec"] = float(scan_submit_min_interval_sec)
        state["scan_timeout_streak"] = 0
        state["scan_backoff_until"] = None
        state["last_scan_submit_at"] = None
        state["cycle_memories"] = []
        state["scan_cache_fingerprint"] = None
        state["scan_cache_ocr_hash"] = ""
        state["scan_cache_ocr_text"] = ""
        state["scan_cache_ocr_confidence"] = 0.0
        state["scan_cache_screen_context"] = ""
        state["scan_cache_scene_summary"] = ""
        state["scan_cache_scene_should_comment"] = False
        state["scan_cache_mode"] = "none"
        state["scan_cache_vision_route"] = "none"
        state["vn_ocr_filter"] = {}

        scan_future = state.get("scan_future")
        if scan_future is not None and hasattr(scan_future, "cancel"):
            try:
                scan_future.cancel()
            except Exception:
                pass
        state["scan_future"] = None
        state["scan_started_at"] = None
        pet.set_scan_busy(False)

        comment_future = state.get("comment_future")
        if comment_future is not None and hasattr(comment_future, "cancel"):
            try:
                comment_future.cancel()
            except Exception:
                pass
        state["comment_future"] = None
        state["pending_comment_meta"] = None
        state["vision_future"] = None
        state["vision_started_at"] = None
        if enabled:
            state["vision_last_request_at"] = None
            state["vision_sampler_fingerprint"] = None

        if state["scan_enabled"] and settings.enable_scan_subprocess:
            _stop_scan_worker()
            _start_scan_worker()
            _drain_scan_results()
        elif state["scan_enabled"] and scan_executor is not None:
            scan_executor.shutdown(wait=False, cancel_futures=True)
            scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")

        if not enabled:
            visual_novel_tracker.flush()

        # Auto-scan is what makes OCR read the dialogue band and what the vision
        # sampler's activity gate keys off, so without it VN mode looks inert --
        # no text, no picture understanding. Turning VN mode on therefore turns
        # scanning on too, and turning it off hands the user's own preference
        # back. This happens only at the toggle, never as a re-assertion, so a
        # manual "off" pressed while VN mode is active still sticks.
        scan_target, autostarted, prev_enabled, scan_autostart_note = vn_scan_transition(
            enabling=enabled,
            scan_enabled=bool(state["scan_enabled"]),
            autostarted=bool(state.get("vn_scan_autostarted")),
            prev_enabled=bool(state.get("vn_scan_prev_enabled")),
        )
        state["vn_scan_autostarted"] = autostarted
        state["vn_scan_prev_enabled"] = prev_enabled
        if scan_target is not None:
            on_scan_toggle(scan_target, announce=False)

        pet.set_visual_novel_mode_enabled(enabled)
        mode_text = "已开启" if enabled else "已关闭"
        chat.append_message("系统", f"视觉小说模式{mode_text}{scan_autostart_note}")
        pet.show_comment_bubble(f"视觉小说模式{mode_text}", duration_ms=1800)
        refresh_vn_sidebar_state()
        log_heartbeat(
            "vn_mode",
            f"enabled={enabled}, scan_tick={scan_tick_interval_sec}s, "
            f"submit_min={scan_submit_min_interval_sec}s, "
            f"scan_enabled={bool(state['scan_enabled'])}",
        )

    def _switch_visual_novel_story(filename: str) -> None:
        nonlocal scan_executor
        target_path = visual_novel_library.directory / filename
        if target_path == visual_novel_tracker.path:
            # Already the live cache, but the panel reads the *marker*, so a
            # stale marker here would keep showing the old name indefinitely
            # while the tracker was happily on the new file. Make them agree.
            try:
                visual_novel_library.set_active(filename)
            except Exception:
                pass
            pet.set_visual_novel_story_name(filename)
            refresh_vn_sidebar_state()
            return

        _discard_pending_vn_evaluation()

        scan_future = state.get("scan_future")
        if scan_future is not None and hasattr(scan_future, "cancel"):
            try:
                scan_future.cancel()
            except Exception:
                pass
        state["scan_future"] = None
        state["scan_started_at"] = None
        state["last_scan_submit_at"] = None
        pet.set_scan_busy(False)
        if state["scan_enabled"] and settings.enable_scan_subprocess:
            _stop_scan_worker()
            _start_scan_worker()
            _drain_scan_results()
        elif state["scan_enabled"] and scan_executor is not None:
            scan_executor.shutdown(wait=False, cancel_futures=True)
            scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")

        visual_novel_tracker.switch_story(target_path)
        comment_engine.reset_scene_state()
        visual_novel_library.set_active(filename)
        state["story_game_pending"] = None
        state["story_game_mismatch"] = ""
        state["vn_last_evaluation_at"] = None
        state["vn_eval_failure_streak"] = 0
        state["vn_eval_retry_after"] = None
        state["scan_cache_fingerprint"] = None
        state["scan_cache_ocr_hash"] = ""
        state["scan_cache_ocr_text"] = ""
        state["scan_cache_ocr_confidence"] = 0.0
        state["scan_cache_screen_context"] = ""
        state["scan_cache_scene_summary"] = ""
        state["scan_cache_scene_should_comment"] = False
        state["scan_cache_mode"] = "none"
        state["scan_cache_vision_route"] = "none"
        state["vn_ocr_filter"] = {}
        pet.set_visual_novel_story_name(filename)
        # The panel label is pushed by refresh_vn_sidebar_state(); without this
        # the switch only became visible on the next 2.5s status tick, and not at
        # all if that timer was never started (pet.vn_sidebar is created lazily,
        # so it can be None at the point the timer start is evaluated).
        refresh_vn_sidebar_state()

    def on_visual_novel_story_manager_requested() -> None:
        action, ok = QInputDialog.getItem(
            pet,
            "剧情缓存",
            f"当前缓存：{visual_novel_library.active_name}\n请选择操作：",
            ["新建缓存", "载入缓存", "删除缓存"],
            0,
            False,
        )
        if not ok:
            return
        try:
            if action == "新建缓存":
                name, accepted = QInputDialog.getText(
                    pet,
                    "新建剧情缓存",
                    "输入游戏名或缓存名：",
                )
                if not accepted:
                    return
                path = visual_novel_library.create(name)
                _switch_visual_novel_story(path.name)
                chat.append_message("系统", f"已新建并载入剧情缓存：{path.name}")
                pet.show_comment_bubble(f"已切换缓存：{path.stem}", duration_ms=2200)
                return

            stories = visual_novel_library.list_stories()
            current_index = stories.index(visual_novel_library.active_name)
            filename, accepted = QInputDialog.getItem(
                pet,
                "载入剧情缓存" if action == "载入缓存" else "删除剧情缓存",
                "选择缓存文件：",
                stories,
                current_index,
                False,
            )
            if not accepted:
                return
            if action == "载入缓存":
                _switch_visual_novel_story(filename)
                chat.append_message("系统", f"已载入剧情缓存：{filename}")
                pet.show_comment_bubble(f"已切换缓存：{Path(filename).stem}", duration_ms=2200)
                return

            if len(stories) <= 1:
                raise ValueError("至少需要保留一个剧情缓存")
            answer = QMessageBox.question(
                pet,
                "删除剧情缓存",
                f"确定永久删除“{filename}”吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            if filename == visual_novel_library.active_name:
                replacement = next(item for item in stories if item != filename)
                _switch_visual_novel_story(replacement)
            visual_novel_library.delete(filename)
            chat.append_message("系统", f"已删除剧情缓存：{filename}")
            pet.show_comment_bubble(f"已删除缓存：{Path(filename).stem}", duration_ms=2200)
        except Exception as exc:
            QMessageBox.warning(pet, "剧情缓存操作失败", str(exc))

    def on_tts_mute_toggled(muted: bool) -> None:
        speech.set_muted(muted)
        pet.set_tts_muted(muted)
        if muted:
            chat.append_message("系统", "语音已静音")
            pet.show_comment_bubble("语音已静音", duration_ms=1800)
        else:
            chat.append_message("系统", "语音已开启")
            pet.show_comment_bubble("语音已开启", duration_ms=1800)

    def on_gaze_follow_toggled(enabled: bool) -> None:
        pet.set_gaze_follow_enabled(enabled)
        if enabled:
            chat.append_message("系统", "视线跟随已开启")
            pet.show_comment_bubble("视线跟随已开启", duration_ms=1600)
        else:
            chat.append_message("系统", "视线跟随已关闭")
            pet.show_comment_bubble("视线跟随已关闭", duration_ms=1600)

    def on_tutor_mode_toggled(enabled: bool) -> None:
        dialog.set_tutor_enabled(enabled)
        comment_engine.set_tutor_enabled(enabled)
        pet.set_tutor_mode_enabled(enabled)
        if enabled:
            chat.append_message("系统", "教学模式已开启")
            pet.show_comment_bubble("教学模式已开启", duration_ms=1600)
        else:
            chat.append_message("系统", "教学模式已关闭")
            pet.show_comment_bubble("教学模式已关闭", duration_ms=1600)

    def _to_window_relative(region, monitor_index: int):
        """Convert an overlay selection into coordinates inside the game window."""
        if window_resolver is None:
            return None
        try:
            rect = window_resolver.resolve()
        except Exception:
            return None
        if rect is None:
            return None
        try:
            monitor_left, monitor_top, _mw, _mh = get_monitor_geometry(int(monitor_index))
        except Exception:
            monitor_left, monitor_top = 0, 0
        absolute_left = int(monitor_left) + int(region[0])
        absolute_top = int(monitor_top) + int(region[1])
        rel_left = absolute_left - int(rect.left)
        rel_top = absolute_top - int(rect.top)
        # Allow a little slack for window borders and DPI rounding, but reject a
        # selection that clearly was not made on top of the game window.
        slack = 48
        if rel_left + int(region[2]) < -slack or rel_top + int(region[3]) < -slack:
            return None
        if rel_left > int(rect.width) + slack or rel_top > int(rect.height) + slack:
            return None
        return (rel_left, rel_top, int(region[2]), int(region[3]))

    def _prompt_region_selection(*, hint: str, on_selected) -> None:
        """Shared region picker used by both the scan region and the OCR box."""
        try:
            qt_screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            if qt_screen is None:
                raise RuntimeError("未找到可用显示器")
            qt_geometry = qt_screen.geometry()
            logical_geometry = (
                int(qt_geometry.x()),
                int(qt_geometry.y()),
                int(qt_geometry.width()),
                int(qt_geometry.height()),
            )
            monitor_index = find_monitor_index(qt_geometry.x(), qt_geometry.y())
            _left, _top, capture_width, capture_height = get_monitor_geometry(monitor_index)
            overlay = RegionSelectOverlay(
                logical_geometry,
                capture_size=(int(capture_width), int(capture_height)),
            )
            state["region_overlay"] = overlay

            def _on_selected(region: tuple[int, int, int, int]) -> None:
                state["region_overlay"] = None
                try:
                    on_selected(region, monitor_index)
                except Exception as exc:
                    chat.append_message("系统", f"应用区域失败: {exc}")

            def _on_cancelled() -> None:
                chat.append_message("系统", "已取消区域选择")
                state["region_overlay"] = None

            overlay.region_selected.connect(_on_selected)
            overlay.cancelled.connect(_on_cancelled)
            overlay.show_for_selection()
            pet.show_comment_bubble(hint, duration_ms=3500)
        except Exception as exc:
            chat.append_message("系统", f"区域选择失败: {exc}")

    def _invalidate_scan_cache() -> None:
        state["scan_cache_fingerprint"] = None
        state["scan_cache_text_fp"] = None
        state["scan_cache_ocr_hash"] = ""
        state["scan_cache_ocr_text"] = ""
        state["scan_cache_ocr_confidence"] = 0.0
        state["scan_cache_screen_context"] = ""
        state["scan_cache_scene_summary"] = ""
        state["scan_cache_vision_route"] = "none"
        state["vn_ocr_filter"] = {}

    def on_vn_select_vision_region() -> None:
        """Pick the game 画面 area used by manual capture mode."""

        def _apply(region: tuple[int, int, int, int], monitor_index: int) -> None:
            # Stored in capture space so it lines up directly with the frame we
            # crop, independent of where the window happens to be.
            capture_region = (int(region[0]), int(region[1]), int(region[2]), int(region[3]))
            _persisted_regions["vision_region"] = capture_region
            _save_persisted_regions()
            state["vision_region"] = capture_region
            if str(state.get("vision_capture_mode", "")) != "manual":
                state["vision_capture_mode"] = "manual"
            _invalidate_scan_cache()
            chat.append_message(
                "系统",
                f"已设置画面区域 {capture_region[2]}×{capture_region[3]}，截图来源已切到「手动框选画面」",
            )
            pet.show_comment_bubble(
                f"画面区域：{capture_region[2]}×{capture_region[3]}", duration_ms=2200
            )
            refresh_vn_sidebar_state()

        _prompt_region_selection(hint="拖拽框出游戏画面范围，右键或Esc取消", on_selected=_apply)

    def on_vn_reset_vision_region() -> None:
        _persisted_regions.pop("vision_region", None)
        _save_persisted_regions()
        state["vision_region"] = None
        _invalidate_scan_cache()
        chat.append_message("系统", "已清除画面区域")
        pet.show_comment_bubble("画面区域已清除", duration_ms=1600)
        refresh_vn_sidebar_state()

    def on_vn_select_ocr_region() -> None:
        """Pick the dialogue box; this is what makes OCR actually work."""

        def _apply(region: tuple[int, int, int, int], monitor_index: int) -> None:
            window_region = _to_window_relative(region, monitor_index)
            if window_region is not None:
                _persisted_regions["ocr_region"] = window_region
            _persisted_regions["ocr_screen_region"] = region
            _save_persisted_regions()
            state["ocr_region"] = _persisted_regions.get("ocr_region")
            _invalidate_scan_cache()
            target = window_region or region
            chat.append_message(
                "系统",
                f"已设置文本区 {target[2]}×{target[3]}（窗口内坐标）",
            )
            pet.show_comment_bubble(f"文本区已更新：{target[2]}×{target[3]}", duration_ms=2400)
            refresh_vn_sidebar_state()

        _prompt_region_selection(hint="拖拽框住游戏对话框，右键或Esc取消", on_selected=_apply)

    def on_vn_reset_ocr_region() -> None:
        _persisted_regions.pop("ocr_region", None)
        _persisted_regions.pop("ocr_screen_region", None)
        _save_persisted_regions()
        state["ocr_region"] = None
        _invalidate_scan_cache()
        chat.append_message("系统", "已清除文本区，改回自动猜测（窗口下方一定比例）")
        pet.show_comment_bubble("文本区已改回自动", duration_ms=1800)
        refresh_vn_sidebar_state()

    def on_vn_probe_ocr() -> None:
        """Read the text region once and report what OCR actually sees."""
        try:
            # GUI thread, so it may refresh the handle snapshot directly.
            _collect_pet_owned_hwnds()
            _refresh_vision_exclusions()
            image, _rect, _vision, source = _resolve_vn_capture(
                mode=str(state.get("vision_capture_mode", settings.vision_capture_mode)),
                resolver=window_resolver,
                ocr_region=state.get("ocr_region"),
                vision_region=state.get("vision_region"),
                monitor_index=int(state["scan_monitor_index"]),
                text_ratio=float(state.get("visual_novel_text_ratio", settings.visual_novel_text_ratio)),
                text_bottom_margin=float(settings.vn_text_band_bottom_margin),
            )
            if source == "target_missing":
                chat.append_message("系统", "试读失败：已锁定的游戏窗口不可用")
                return
            ocr_image = _resize_for_ocr(image, int(settings.ocr_max_edge))
            result = extract_ocr_result(ocr_image)
            text = str(result.text or "").strip()
            line_count = len(result.lines)
            confidence = float(result.average_confidence or 0.0)
            preview = re.sub(r"\s+", " ", text)[:120] or "(没有识别到文字)"
            print(
                "[PROBE-OCR]",
                f"source={source}",
                f"window={str(getattr(_rect, 'title', '') or '(unknown)')!r}",
                f"crop={image.size[0]}x{image.size[1]}",
                f"lines={line_count}",
                f"chars={result.char_count}",
                f"conf={confidence:.2f}",
                f"text={preview}",
            )
            chat.append_message(
                "系统",
                f"试读结果：来源 {source}｜窗口「{str(getattr(_rect, 'title', '') or '未知')}」\n"
                f"裁剪 {image.size[0]}×{image.size[1]}，"
                f"{line_count} 行 / {result.char_count} 字 / 置信度 {confidence:.2f}\n{preview}",
            )
            pet.show_comment_bubble(
                f"识别到 {result.char_count} 字" if text else "没有识别到文字",
                duration_ms=2600,
            )
        except Exception as exc:
            chat.append_message("系统", f"试读失败: {exc}")
            print(f"[PROBE-OCR] failed: {exc}")

    def on_select_scan_region() -> None:
        def _apply(region: tuple[int, int, int, int], monitor_index: int) -> None:
            state["scan_monitor_index"] = monitor_index
            state["scan_region"] = region
            _invalidate_scan_cache()
            chat.append_message("系统", f"已设置显示器{monitor_index}扫描区域: {region}")
            pet.show_comment_bubble(
                f"扫描区域已更新：显示器{monitor_index}，{region[2]}×{region[3]}",
                duration_ms=2400,
            )

        _prompt_region_selection(hint="拖拽框选扫描区域，右键或Esc取消", on_selected=_apply)

    def on_clear_scan_region() -> None:
        state["scan_monitor_index"] = 0
        state["scan_region"] = None
        _invalidate_scan_cache()
        chat.append_message("系统", "已清除扫描区域，恢复整个虚拟桌面扫描")
        pet.show_comment_bubble("已恢复所有屏幕扫描")

    def on_quit_requested() -> None:
        scheduler.stop()
        if visual_novel_tracker is not None:
            visual_novel_tracker.flush()
        pet.set_scan_busy(False)
        _apply_runtime_resource_policy(scan_active=False, force=True)
        future = state.get("scan_future")
        if future is not None and hasattr(future, "cancel"):
            future.cancel()
        comment_future = state.get("comment_future")
        if comment_future is not None:
            comment_future.cancel()
        if settings.enable_scan_subprocess:
            _stop_scan_worker()
        elif scan_executor is not None:
            scan_executor.shutdown(wait=False, cancel_futures=True)
        comment_executor.shutdown(wait=False, cancel_futures=True)
        tts_executor.shutdown(wait=False, cancel_futures=True)
        speech.shutdown()
        chat.set_disable_auto_archive_on_close(True)
        chat.close()
        pet.close()
        _stop_live2d_py()
        app.quit()

    HWND_TOPMOST = -1
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOZORDER = 0x0004
    SWP_ASYNCWINDOWPOS = 0x4000
    SW_RESTORE = 9
    live2d_hwnd_cache = 0
    last_input_diag_ts = 0.0
    input_diag_interval_sec = 20.0
    last_input_write_ts = 0.0
    last_input_signature = None
    last_live2d_relaunch_ts = 0.0
    last_live2d_applied_rect = None

    class _WinRect(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def _set_live2d_window_pos(
        user32,
        hwnd: int,
        x: int,
        y: int,
        w: int,
        h: int,
        insert_after: int = HWND_TOPMOST,
    ) -> None:
        user32.SetWindowPos(
            hwnd,
            insert_after,
            int(x),
            int(y),
            int(w),
            int(h),
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def _write_json_state_file(state_path: Path, payload: dict) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, ensure_ascii=True)

        # On Windows, os.replace may fail with WinError 5 when destination is briefly locked.
        # Use unique temp names + short retry; finally fall back to direct overwrite.
        last_exc: Exception | None = None
        for attempt in range(4):
            tmp_path = state_path.with_suffix(f".json.tmp.{os.getpid()}.{int(time.time() * 1000)}.{attempt}")
            try:
                tmp_path.write_text(body, encoding="utf-8")
                os.replace(tmp_path, state_path)
                return
            except Exception as exc:
                last_exc = exc
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                time.sleep(0.006 * (attempt + 1))

        try:
            state_path.write_text(body, encoding="utf-8")
            return
        except Exception as exc:
            last_exc = exc

        if last_exc is not None:
            raise last_exc

    def _write_live2d_target_rect(x: int, y: int, w: int, h: int) -> None:
        try:
            state_path = base_dir / "data" / "live2d_py_target_rect.json"
            payload = {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "shutdown": int(live2d_shutdown_requested["flag"]),
                "shutdown_ts": float(live2d_shutdown_requested["ts"]),
            }
            _write_json_state_file(state_path, payload)
        except Exception:
            pass

    def _write_live2d_shutdown_marker() -> None:
        """Ask the live2d-py renderer to exit instead of killing it.

        Force-terminating that process tears down native GL/Cubism state
        mid-frame, which raises an access-violation dialog on exit. The renderer
        polls the state files it already reads every frame, so a flag plus a
        short grace period is enough.
        """
        live2d_shutdown_requested["flag"] = 1
        live2d_shutdown_requested["ts"] = time.time()
        try:
            ack_path = base_dir / "data" / "live2d_py_shutdown_ack.json"
            if ack_path.exists():
                ack_path.unlink()
        except Exception:
            pass
        # The renderer picks the flag up from whichever state file it polls.
        try:
            _write_json_state_file(
                base_dir / "data" / "live2d_py_input_state.json",
                {
                    "ts": time.time(),
                    "x": 0,
                    "y": 0,
                    "w": 1,
                    "h": 1,
                    "inside": False,
                    "left_down": 0,
                    "right_down": 0,
                    "scan_busy": 0,
                    "follow_enabled": 0,
                    "drag_active": 0,
                    "shutdown": 1,
                    "shutdown_ts": float(live2d_shutdown_requested["ts"]),
                },
            )
        except Exception:
            pass
        try:
            geometry = _get_live2d_target_geometry_native(ctypes.windll.user32, int(pet.winId()))
            _write_live2d_target_rect(*geometry)
        except Exception:
            pass

    def _stop_live2d_py(grace_sec: float = 2.5) -> None:
        """Graceful shutdown with a hard-kill fallback."""
        nonlocal live2d_py_process
        process = live2d_py_process
        if process is None:
            return
        if process.poll() is not None:
            live2d_py_process = None
            return
        _write_live2d_shutdown_marker()
        deadline = time.time() + max(0.2, float(grace_sec))
        while time.time() < deadline:
            if process.poll() is not None:
                print("[STARTUP] Live2D-py exited cleanly")
                live2d_py_process = None
                return
            time.sleep(0.05)
        ack_path = base_dir / "data" / "live2d_py_shutdown_ack.json"
        if ack_path.exists():
            # The renderer acknowledged and is unwinding; give it a moment more.
            try:
                ack = json.loads(ack_path.read_text(encoding="utf-8"))
                if int(ack.get("pid", 0)) == int(process.pid):
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        if process.poll() is not None:
                            print("[STARTUP] Live2D-py exited cleanly")
                            live2d_py_process = None
                            return
                        time.sleep(0.05)
            except Exception:
                pass
        print("[STARTUP] Live2D-py did not exit in time, terminating")
        try:
            process.terminate()
        except Exception:
            pass
        live2d_py_process = None

    def _get_live2d_target_geometry_native(user32, host_hwnd: int) -> tuple[int, int, int, int]:
        # Use Win32 host rect as ground truth to avoid cross-monitor DPI drift.
        _x, _y, model_w_logical, model_h_logical = pet.get_live2d_py_target_geometry()
        logical_host_w = max(1, int(pet.width()))
        logical_host_h = max(1, int(pet.height()))

        rect = _WinRect()
        if host_hwnd and user32.GetWindowRect(host_hwnd, ctypes.byref(rect)):
            native_x = int(rect.left)
            native_y = int(rect.top)
            native_host_w = max(1, int(rect.right - rect.left))
            native_host_h = max(1, int(rect.bottom - rect.top))
            scale_x = float(native_host_w) / float(logical_host_w)
            scale_y = float(native_host_h) / float(logical_host_h)
            native_w = max(1, int(round(float(model_w_logical) * scale_x)))
            native_h = max(1, int(round(float(model_h_logical) * scale_y)))
            return native_x, native_y, native_w, native_h

        # Fallback if host rect temporarily unavailable.
        return int(_x), int(_y), max(1, int(model_w_logical)), max(1, int(model_h_logical))

    def _write_live2d_input_state(
        user32,
        x: int,
        y: int,
        w: int,
        h: int,
        follow_enabled: bool,
        drag_active: bool = False,
    ) -> None:
        nonlocal last_input_diag_ts, last_input_write_ts, last_input_signature
        try:
            scan_busy = bool(pet.scan_busy)
            now_ts = time.time()

            # When follow is disabled, stop polling mouse position/buttons to reduce OCR contention.
            # Keep a low-frequency heartbeat so runner can switch to non-follow mode immediately.
            if not follow_enabled:
                follow_off_keepalive = 3.0 if scan_busy else 1.8
                signature = (int(scan_busy), int(follow_enabled), int(drag_active), int(max(1, w)), int(max(1, h)))
                if signature == last_input_signature and (now_ts - last_input_write_ts) < follow_off_keepalive:
                    return

                payload = {
                    "ts": now_ts,
                    "x": int(max(0, min(max(1, w) - 1, w // 2))),
                    "y": int(max(0, min(max(1, h) - 1, h // 2))),
                    "w": int(max(1, w)),
                    "h": int(max(1, h)),
                    "inside": False,
                    "left_down": 0,
                    "right_down": 0,
                    "scan_busy": int(scan_busy),
                    "follow_enabled": 0,
                    "drag_active": int(bool(drag_active)),
                    "shutdown": int(live2d_shutdown_requested["flag"]),
                    "shutdown_ts": float(live2d_shutdown_requested["ts"]),
                }
                state_path = base_dir / "data" / "live2d_py_input_state.json"
                _write_json_state_file(state_path, payload)
                last_input_write_ts = now_ts
                last_input_signature = signature
                return

            # Keep follow responsive even when OCR is busy; avoid near-stop behavior.
            if scan_busy and (now_ts - last_input_write_ts) < 0.12:
                return

            pt = wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(pt)):
                return
            local_x = int(pt.x - x)
            local_y = int(pt.y - y)
            inside = 0 <= local_x < max(1, w) and 0 <= local_y < max(1, h)
            local_x = max(0, min(max(1, w) - 1, local_x))
            local_y = max(0, min(max(1, h) - 1, local_y))

            left_down = 1 if (user32.GetAsyncKeyState(0x01) & 0x8000) else 0
            right_down = 1 if (user32.GetAsyncKeyState(0x02) & 0x8000) else 0
            min_interval = 0.12 if scan_busy else 0.06
            keepalive_interval = 0.7 if scan_busy else 0.6
            signature = (
                local_x,
                local_y,
                left_down,
                right_down,
                inside,
                int(scan_busy),
                int(follow_enabled),
                int(drag_active),
            )
            changed = signature != last_input_signature
            if (not changed and (now_ts - last_input_write_ts) < keepalive_interval) or (
                changed and (now_ts - last_input_write_ts) < min_interval
            ):
                return

            payload = {
                "ts": now_ts,
                "x": int(local_x),
                "y": int(local_y),
                "w": int(max(1, w)),
                "h": int(max(1, h)),
                "inside": bool(inside),
                "left_down": int(left_down),
                "right_down": int(right_down),
                "scan_busy": int(scan_busy),
                "follow_enabled": int(follow_enabled),
                "drag_active": int(bool(drag_active)),
                "shutdown": int(live2d_shutdown_requested["flag"]),
                "shutdown_ts": float(live2d_shutdown_requested["ts"]),
            }
            state_path = base_dir / "data" / "live2d_py_input_state.json"
            _write_json_state_file(state_path, payload)
            last_input_write_ts = now_ts
            last_input_signature = signature

            if now_ts - last_input_diag_ts >= input_diag_interval_sec:
                print(
                    "[DIAG][MAIN][INPUT_WRITE]",
                    f"inside={inside}",
                    f"local=({local_x},{local_y})",
                    f"wh=({w},{h})",
                    f"left={left_down}",
                    f"right={right_down}",
                    f"scan_busy={scan_busy}",
                    f"follow={follow_enabled}",
                    f"drag={int(bool(drag_active))}",
                )
                last_input_diag_ts = now_ts
        except Exception as exc:
            now_ts = time.time()
            if now_ts - last_input_diag_ts >= input_diag_interval_sec:
                print(f"[DIAG][MAIN][INPUT_WRITE] failed: {exc}")
                last_input_diag_ts = now_ts

    def _is_cursor_near_rect(user32, x: int, y: int, w: int, h: int, threshold_px: int) -> bool:
        try:
            pt = wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(pt)):
                return False
            left = int(x)
            top = int(y)
            right = int(x + max(1, w) - 1)
            bottom = int(y + max(1, h) - 1)
            dx = max(left - int(pt.x), 0, int(pt.x) - right)
            dy = max(top - int(pt.y), 0, int(pt.y) - bottom)
            return (dx * dx + dy * dy) <= int(threshold_px * threshold_px)
        except Exception:
            return False

    def _raise_topmost(user32, hwnd: int, above_hwnd: int = 0) -> None:
        if not hwnd:
            return
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_ASYNCWINDOWPOS,
        )
        if above_hwnd:
            user32.SetWindowPos(
                hwnd,
                above_hwnd,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_ASYNCWINDOWPOS,
            )

    def _resolve_live2d_hwnd(user32) -> int:
        nonlocal live2d_hwnd_cache
        if live2d_hwnd_cache and user32.IsWindow(live2d_hwnd_cache):
            return int(live2d_hwnd_cache)

        pid = live2d_py_process.pid if live2d_py_process is not None else 0
        if pid <= 0:
            return 0

        # Preferred path: exact hwnd reported by renderer process.
        try:
            state_path = base_dir / "data" / "live2d_py_window_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state_pid = int(state.get("pid", 0))
                state_hwnd = int(state.get("hwnd", 0))
                if state_pid == int(pid) and state_hwnd and user32.IsWindow(state_hwnd):
                    live2d_hwnd_cache = state_hwnd
                    return state_hwnd
        except Exception:
            pass

        hwnd_result = {"value": 0, "area": 0}
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        rect_type = _WinRect

        def _enum_cb(hwnd, _lparam):
            proc_id = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
            if int(proc_id.value) != int(pid):
                return True
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.GetParent(hwnd):
                return True
            rect = rect_type()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            width = max(0, int(rect.right - rect.left))
            height = max(0, int(rect.bottom - rect.top))
            area = width * height
            if area <= hwnd_result["area"]:
                return True
            hwnd_result["value"] = int(hwnd)
            hwnd_result["area"] = area
            return True

        user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
        resolved = int(hwnd_result["value"])
        if resolved:
            live2d_hwnd_cache = resolved
        return resolved

    def _try_sync_live2d_py_windows() -> None:
        nonlocal live2d_py_process, live2d_py_retry_used, live2d_hwnd_cache, last_live2d_relaunch_ts
        nonlocal last_live2d_applied_rect
        nonlocal runtime_follow_near, runtime_follow_effective, runtime_follow_hold_until_ts, runtime_drag_active
        nonlocal runtime_follow_near_on_streak, runtime_follow_near_off_streak
        nonlocal drag_lock_prev_active, drag_lock_last_rect, drag_lock_last_write_ts, last_zorder_sync_ts
        if not settings.enable_live2d_py:
            return
        if not hasattr(ctypes, "windll"):
            return
        user32 = ctypes.windll.user32
        host_hwnd = int(pet.winId())
        chat_hwnd = int(chat.winId()) if chat.isVisible() else 0
        exp_x, exp_y, exp_w, exp_h = _get_live2d_target_geometry_native(user32, host_hwnd)
        now_ts = time.time()
        runtime_drag_active = bool(pet.is_live2d_py_interacting())
        near_enter_raw = _is_cursor_near_rect(user32, exp_x, exp_y, exp_w, exp_h, follow_enter_distance_px)
        near_exit_raw = _is_cursor_near_rect(user32, exp_x, exp_y, exp_w, exp_h, follow_exit_distance_px)
        if near_enter_raw:
            runtime_follow_hold_until_ts = now_ts + follow_hold_sec
        raw_near = bool(near_enter_raw or near_exit_raw)
        if raw_near:
            runtime_follow_near_on_streak += 1
            runtime_follow_near_off_streak = 0
            if (not runtime_follow_near) and runtime_follow_near_on_streak >= follow_near_enter_frames:
                runtime_follow_near = True
        else:
            runtime_follow_near_off_streak += 1
            runtime_follow_near_on_streak = 0
            if runtime_follow_near and runtime_follow_near_off_streak >= follow_near_exit_frames:
                runtime_follow_near = False

        follow_gate_ok = bool(runtime_follow_near or (now_ts < runtime_follow_hold_until_ts))
        runtime_follow_effective = bool(pet.gaze_follow_enabled and follow_gate_ok and (not runtime_drag_active))
        target_rect = (int(exp_x), int(exp_y), int(exp_w), int(exp_h))

        # Drag lock: avoid cross-process z-order churn while dragging to prevent flicker/disappear.
        if runtime_drag_active:
            if (target_rect != drag_lock_last_rect) and (now_ts - drag_lock_last_write_ts >= 0.05):
                _write_live2d_target_rect(exp_x, exp_y, exp_w, exp_h)
                drag_lock_last_rect = target_rect
                drag_lock_last_write_ts = now_ts
            _write_live2d_input_state(user32, exp_x, exp_y, exp_w, exp_h, False, True)

            # Main process becomes the single geometry writer during drag to avoid cross-process jitter.
            hwnd_drag = _resolve_live2d_hwnd(user32)
            if hwnd_drag:
                try:
                    user32.SetWindowPos(
                        hwnd_drag,
                        0,
                        int(exp_x),
                        int(exp_y),
                        int(exp_w),
                        int(exp_h),
                        SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER | SWP_ASYNCWINDOWPOS,
                    )
                    last_live2d_applied_rect = target_rect
                except Exception:
                    pass
            _apply_runtime_resource_policy(
                scan_active=bool(pet.scan_busy),
                follow_active=False,
                drag_active=True,
                force=False,
            )
            drag_lock_prev_active = True
            return

        force_post_drag_resync = False
        if drag_lock_prev_active:
            force_post_drag_resync = True
            drag_lock_prev_active = False

        _write_live2d_target_rect(exp_x, exp_y, exp_w, exp_h)
        drag_lock_last_rect = target_rect
        drag_lock_last_write_ts = now_ts
        _write_live2d_input_state(user32, exp_x, exp_y, exp_w, exp_h, runtime_follow_effective, False)
        # Cursor-distance driven policy:
        # - near model => follow priority
        # - far from model => idle
        # This avoids tick-driven policy updates that can cause periodic UI stutter.
        _apply_runtime_resource_policy(
            scan_active=False,
            follow_active=bool(runtime_follow_near),
            drag_active=runtime_drag_active,
            force=False,
        )
        hwnd = _resolve_live2d_hwnd(user32)
        if not hwnd:
            process_dead = (live2d_py_process is None) or (live2d_py_process.poll() is not None)
            if (
                process_dead
                and not int(live2d_shutdown_requested["flag"])
                and live2d_py_retry_count[0] < live2d_py_max_restarts
                and (now_ts - last_live2d_relaunch_ts >= 3.0)
            ):
                try:
                    live2d_py_retry_count[0] += 1
                    live2d_py_process = start_live2d_py_process(force_gl_init=True)
                    live2d_py_retry_used = True
                    live2d_hwnd_cache = 0
                    last_live2d_relaunch_ts = now_ts
                    print(
                        "[STARTUP] Live2D-py auto relaunch started "
                        f"(pid={live2d_py_process.pid}, "
                        f"attempt {live2d_py_retry_count[0]}/{live2d_py_max_restarts})"
                    )
                except Exception as exc:
                    print(f"[STARTUP] Live2D-py auto relaunch failed: {exc}")
            if host_hwnd:
                _raise_topmost(user32, host_hwnd)
            return

        try:
            if not user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetWindowPos(
                    hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_ASYNCWINDOWPOS,
                )
        except Exception:
            pass

        # Single geometry writer: always drive model rect from host side.
        if target_rect != last_live2d_applied_rect:
            try:
                user32.SetWindowPos(
                    hwnd,
                    0,
                    int(exp_x),
                    int(exp_y),
                    int(exp_w),
                    int(exp_h),
                    SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER | SWP_ASYNCWINDOWPOS,
                )
                last_live2d_applied_rect = target_rect
            except Exception:
                pass

        # Throttle z-order relock to avoid unnecessary flashing; force once after drag end.
        if force_post_drag_resync or (now_ts - last_zorder_sync_ts >= 0.45):
            _raise_topmost(user32, host_hwnd)
            user32.SetWindowPos(
                hwnd,
                host_hwnd or HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_ASYNCWINDOWPOS,
            )
            last_zorder_sync_ts = now_ts

    def _sync_live2d_drag_geometry() -> None:
        nonlocal last_live2d_applied_rect
        if not settings.enable_live2d_py or not hasattr(ctypes, "windll"):
            return
        user32 = ctypes.windll.user32
        host_hwnd = int(pet.winId())
        target_rect = _get_live2d_target_geometry_native(user32, host_hwnd)
        _write_live2d_input_state(user32, *target_rect, False, True)
        hwnd = _resolve_live2d_hwnd(user32)
        if hwnd and user32.SetWindowPos(
            hwnd, 0, *target_rect, SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER | SWP_ASYNCWINDOWPOS
        ):
            last_live2d_applied_rect = target_rect

    if state["scan_enabled"] and settings.enable_scan_subprocess:
        _start_scan_worker()

    scheduler = ScanScheduler(scan_tick_interval_sec)
    scheduler.tick.connect(do_auto_comment)
    if state["scan_enabled"]:
        scheduler.start()

    def poll_vn_scan() -> None:
        if not state["scan_enabled"] or not state["visual_novel_mode_enabled"]:
            return
        future = state["scan_future"]
        if future is not None:
            if settings.enable_scan_subprocess:
                if scan_result_queue is None or scan_result_queue.empty():
                    return
            elif not future.done():
                return
        if state["scan_future"] is not None or time.monotonic() >= float(state["vn_scan_due_at"]):
            do_auto_comment()

    vn_scan_result_timer = QTimer()
    vn_scan_result_timer.setInterval(100)
    vn_scan_result_timer.timeout.connect(poll_vn_scan)
    vn_scan_result_timer.start()

    chat.comment_btn.clicked.disconnect()
    chat.comment_btn.clicked.connect(do_manual_comment)

    def on_new_chat_requested() -> None:
        if settings.enable_live2d_py:
            mx, my, mw, _mh = pet.get_live2d_py_target_geometry()
            target_x = max(20, int(mx - chat.width() - 14))
            target_y = max(20, int(my + 24))
            chat.move(target_x, target_y)
        chat.show_and_focus()

    pet.open_chat_requested.connect(on_new_chat_requested)
    pet.auto_comment_requested.connect(do_manual_comment)
    pet.auto_scan_toggled.connect(on_scan_toggle)
    pet.visual_novel_mode_toggled.connect(on_visual_novel_mode_toggled)
    pet.visual_novel_story_manager_requested.connect(on_visual_novel_story_manager_requested)
    pet.tts_mute_toggled.connect(on_tts_mute_toggled)
    pet.gaze_follow_toggled.connect(on_gaze_follow_toggled)
    pet.tutor_mode_toggled.connect(on_tutor_mode_toggled)
    pet.select_scan_region_requested.connect(on_select_scan_region)
    pet.clear_scan_region_requested.connect(on_clear_scan_region)
    pet.quit_requested.connect(on_quit_requested)

    pet.set_auto_scan_enabled(state["scan_enabled"])
    pet.set_visual_novel_mode_enabled(state["visual_novel_mode_enabled"])

    # Auto-scan stays OFF at startup unless the user explicitly opts in with
    # AUTO_START_SCAN. That includes the case where visual-novel mode is enabled
    # from configuration: turning the scan on by itself is surprising, and it
    # also hides the manual off->on toggle that the user wants to be able to
    # exercise deliberately.
    #
    # Deferred into the event loop rather than run here. Starting the scheduler
    # and the scan executor during startup setup (before ``app.exec``) wedged the
    # process; the toggle drives the identical call from inside the loop, so
    # deferring keeps one proven code path instead of a second startup-only one.
    def _apply_startup_scan_policy() -> None:
        if settings.auto_start_scan and not state["scan_enabled"]:
            on_scan_toggle(True, announce=False)
            print("[STARTUP] AUTO_START_SCAN: auto-scan enabled")
        elif state["visual_novel_mode_enabled"]:
            # Everything is still off, and VN mode looks inert without the scan.
            # Say so once so the reason is discoverable from the log as well as
            # from the side panel hint.
            print(
                "[STARTUP] VN mode on, auto-scan left OFF "
                "(click 自动扫描 in the pet panel, or set AUTO_START_SCAN=true)"
            )

    QTimer.singleShot(0, _apply_startup_scan_policy)

    pet.set_visual_novel_story_name(visual_novel_library.active_name)
    pet.set_tts_muted(False)
    pet.set_gaze_follow_enabled(settings.live2d_follow_cursor)
    pet.set_tutor_mode_enabled(settings.enable_tutor_persona)

    vn_sidebar = getattr(pet, "vn_sidebar", None)
    if vn_sidebar is not None:
        vn_sidebar.capture_mode_changed.connect(on_vn_capture_mode_changed)
        vn_sidebar.vision_quota_changed.connect(on_vn_vision_quota_changed)
        vn_sidebar.vision_change_threshold_changed.connect(on_vn_change_threshold_changed)
        vn_sidebar.text_ratio_changed.connect(on_vn_text_ratio_changed)
        vn_sidebar.window_margin_changed.connect(on_vn_window_margin_changed)
        vn_sidebar.pin_target_requested.connect(on_vn_pin_target)
        vn_sidebar.unpin_target_requested.connect(on_vn_unpin_target)
        vn_sidebar.select_vision_region_requested.connect(on_vn_select_vision_region)
        vn_sidebar.reset_vision_region_requested.connect(on_vn_reset_vision_region)
        vn_sidebar.select_ocr_region_requested.connect(on_vn_select_ocr_region)
        vn_sidebar.reset_ocr_region_requested.connect(on_vn_reset_ocr_region)
        vn_sidebar.probe_ocr_requested.connect(on_vn_probe_ocr)
        vn_sidebar.select_story_requested.connect(on_visual_novel_story_manager_requested)
        vn_sidebar.add_memory_requested.connect(on_vn_add_memory)
        vn_sidebar.reload_memory_requested.connect(on_vn_reload_memory)
        vn_sidebar.reset_story_requested.connect(on_vn_reset_story)
        vn_sidebar.view_memory_requested.connect(on_vn_view_memory)
        vn_sidebar.select_scan_region_requested.connect(on_select_scan_region)
        vn_sidebar.clear_scan_region_requested.connect(on_clear_scan_region)
        refresh_vn_sidebar_state()

    pet.show()
    pet.raise_()
    pet.activateWindow()

    # Snapshot the pet's HWNDs from the GUI thread, then keep the snapshot fresh
    # so lazily-created windows (the side panel appears on first hover) get
    # excluded. Worker threads must never call winId() themselves.
    _collect_pet_owned_hwnds()
    _refresh_vision_exclusions()
    pet_hwnd_refresh_timer = QTimer()
    pet_hwnd_refresh_timer.setInterval(1500)
    pet_hwnd_refresh_timer.timeout.connect(_collect_pet_owned_hwnds)
    pet_hwnd_refresh_timer.start()

    def emit_opening_greeting() -> None:
        greeting = dialog.build_opening_greeting().strip()
        if not greeting:
            return
        show_and_speak(greeting, role="桌宠")
        # Start a fresh auto-comment cycle after greeting so timing feels natural.
        state["cycle_memories"] = []
        state["next_comment_at"] = schedule_next_comment(datetime.now())
        log_heartbeat("schedule_reset", f"next_due={state['next_comment_at'].strftime('%H:%M:%S')}")

    QTimer.singleShot(1200, emit_opening_greeting)

    if settings.enable_live2d_py:
        pet.live2d_drag_moved.connect(_sync_live2d_drag_geometry)
        live2d_ui_sync_timer = QTimer()
        live2d_ui_sync_timer.setInterval(90)
        live2d_ui_sync_timer.timeout.connect(_try_sync_live2d_py_windows)
        live2d_ui_sync_timer.start()
        QTimer.singleShot(500, _try_sync_live2d_py_windows)

    vn_sidebar_status_timer = QTimer()
    vn_sidebar_status_timer.setInterval(2500)
    vn_sidebar_status_timer.timeout.connect(refresh_vn_sidebar_state)
    if vn_sidebar is not None:
        vn_sidebar_status_timer.start()

    # Independent 1s vision sampler. Kept off the Qt thread because it captures a
    # window and does a little image work; the model call itself is handed to
    # vision_executor. Running it on its own loop means OCR load no longer
    # perturbs how often the picture is looked at.
    vision_sampler_stop = threading.Event()

    def _vision_sampler_loop() -> None:
        while not vision_sampler_stop.is_set():
            poll_vision_future()
            vision_sampling_step()
            # While the minimum-gap floor is the binding constraint there is
            # nothing to do for that whole window, so sleep it out instead of
            # ticking every second.
            wait = 1.0
            now = datetime.now()
            since_last = _vision_since_last_sec(now)
            min_gap = float(state.get("vision_min_gap_sec", settings.vision_min_gap_sec))
            if since_last is not None and min_gap > 0 and since_last < min_gap:
                wait = max(0.2, (min_gap - since_last) + 0.05)
            vision_sampler_stop.wait(min(wait, 30.0))

    vision_sampler_thread = threading.Thread(
        target=_vision_sampler_loop,
        name="vision-sampler",
        daemon=True,
    )
    vision_sampler_thread.start()
    print(
        "[STARTUP] Vision sampler:",
        f"1s tick, quota={settings.vision_shots_per_minute:g}/min, "
        f"burst={settings.vision_burst_cap:g}, "
        f"gap={settings.vision_min_gap_sec:g}s, "
        f"change_threshold={settings.vision_change_threshold:.2f}, "
        f"context_ttl={settings.vision_context_ttl_sec:g}s",
    )

    app.aboutToQuit.connect(lambda: vision_sampler_stop.set())
    app.aboutToQuit.connect(speech.shutdown)
    if visual_novel_tracker is not None:
        app.aboutToQuit.connect(visual_novel_tracker.flush)
    app.aboutToQuit.connect(lambda: _stop_live2d_py())
    if settings.enable_scan_subprocess:
        app.aboutToQuit.connect(_stop_scan_worker)
    else:
        app.aboutToQuit.connect(lambda: scan_executor and scan_executor.shutdown(wait=False, cancel_futures=True))
    app.aboutToQuit.connect(lambda: comment_executor.shutdown(wait=False, cancel_futures=True))
    app.aboutToQuit.connect(lambda: tts_executor.shutdown(wait=False, cancel_futures=True))
    app.aboutToQuit.connect(lambda: vision_executor.shutdown(wait=False, cancel_futures=True))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
