"""Regression tests for the dialogue-change detector.

The scan skips OCR while the dialogue box "looks unchanged", to avoid paying
~1.3s per tick on a static box. That gate used a 96x36 grayscale average of the
crop (1979x369 for this game) with a 0.008 threshold. Averaging ~21x10 source
pixels per cell washed thin glyph strokes away, so two different short lines
scored 0.0064 and a name-plate-only change 0.0032 -- both under the threshold.
The box therefore looked unchanged, OCR never re-ran, and the first line ever
read was returned forever: the pet appeared to read one sentence and stop.

The same coarse fingerprint gated the frame-level cache in *both* the
in-process and the subprocess scan paths, so whichever path ran, the pipeline
froze the same way.

The detector uses an ink map compared by Jaccard overlap over the inked cells,
with the old coarse brightness fingerprint retained as a second signal. Cache
reuse is allowed only while both signals agree that the frame is unchanged.

Glyphs are drawn as thin bars instead of real text so the tests do not depend on
a system CJK font, and the two "lines" differ only in stroke layout *within the
same footprint*, which is what a real dialogue change looks like (same number of
characters in the same place, different glyph shapes). A horizontal shift would
be an unrealistically easy case: the legacy gate happens to notice those.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from desktop_pet.config.settings import load_settings  # noqa: E402
from desktop_pet.main import (  # noqa: E402
    _VN_OCR_CACHE_MAX_REUSE_SEC,
    _frame_fingerprint,
    _fingerprint_diff_ratio,
    _ink_diff_ratio,
    _text_change_metrics,
    _text_ink_fingerprint,
    _vn_ocr_cache_can_reuse,
    _vn_stable_cache_can_reuse,
    _vn_text_is_settling,
)

_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS = load_settings(_ROOT)
INK_THRESHOLD = float(_SETTINGS.ocr_text_ink_change_threshold)
LEGACY_THRESHOLD = float(_SETTINGS.ocr_text_change_threshold)

# The real dialogue box the user framed on 星空列车与白的旅行.
WIDTH, HEIGHT = 1979, 369
# Glyph stroke thickness. At the 192x48 ink grid each cell covers ~10x8 source
# pixels, so a 4px stroke is thin relative to a cell -- the shape that the old
# 96x36 average erased.
STROKE = 4


def _box(
    glyphs: int = 3,
    *,
    variant: int = 0,
    light_text: bool = True,
    brightness: int = 0,
) -> Image.Image:
    """A dialogue box holding one line of ``glyphs`` thin-stroke characters.

    ``variant`` selects between two glyph shapes that occupy an identical
    bounding box, so only the ink distribution inside changes.
    """
    bg = (18, 18, 26) if light_text else (232, 232, 236)
    fg = (238, 238, 238) if light_text else (24, 24, 30)
    img = Image.new("RGB", (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(img)
    x = 60
    for _ in range(glyphs):
        if variant == 0:  # "T"-ish: top bar plus centre stem
            draw.rectangle([x, 60, x + 30, 60 + STROKE], fill=fg)
            draw.rectangle([x + 13, 60, x + 13 + STROKE, 100], fill=fg)
        else:  # "L"-ish: left stem plus bottom bar
            draw.rectangle([x, 60, x + STROKE, 100], fill=fg)
            draw.rectangle([x, 100 - STROKE, x + 30, 100], fill=fg)
        x += 46
    if brightness:
        img = img.point(lambda value: min(255, max(0, value + brightness)))
    return img


def _ink(image: Image.Image) -> bytes:
    return _text_ink_fingerprint(image)


# --------------------------------------------------------------------------
# Sensitivity: the changes that used to be invisible
# --------------------------------------------------------------------------


def test_short_line_change_is_detected():
    """The exact regression: two different short lines."""
    ratio = _ink_diff_ratio(_ink(_box(glyphs=2, variant=0)), _ink(_box(glyphs=2, variant=1)))
    assert ratio >= INK_THRESHOLD, (
        f"a short-line change scored {ratio:.5f}, below {INK_THRESHOLD}; "
        "this is the freeze regression"
    )


def test_legacy_fingerprint_was_blind_to_that_same_change():
    """Documents the premise for replacing it, on identical geometry."""
    left = _frame_fingerprint(_box(glyphs=2, variant=0), text_sensitive=True)
    right = _frame_fingerprint(_box(glyphs=2, variant=1), text_sensitive=True)
    ratio = _fingerprint_diff_ratio(left, right)
    assert ratio < LEGACY_THRESHOLD, (
        f"expected the legacy gate to miss this (scored {ratio:.5f} vs "
        f"{LEGACY_THRESHOLD}); if it now sees it, the premise changed"
    )


def test_new_detector_sees_what_the_legacy_one_missed():
    """Same frame pair, both metrics: the new one must win decisively."""
    pair = (_box(glyphs=2, variant=0), _box(glyphs=2, variant=1))
    legacy = _fingerprint_diff_ratio(
        _frame_fingerprint(pair[0], text_sensitive=True),
        _frame_fingerprint(pair[1], text_sensitive=True),
    )
    new = _ink_diff_ratio(_ink(pair[0]), _ink(pair[1]))
    assert legacy < LEGACY_THRESHOLD < INK_THRESHOLD <= new


def test_name_plate_change_is_detected():
    """A different speaker with the same ellipsis must count as a change."""
    ratio = _ink_diff_ratio(_ink(_box(glyphs=2, variant=0)), _ink(_box(glyphs=2, variant=1)))
    assert ratio >= INK_THRESHOLD, f"name-plate change scored {ratio:.5f}"


def test_longer_line_change_is_detected():
    ratio = _ink_diff_ratio(_ink(_box(glyphs=6, variant=0)), _ink(_box(glyphs=6, variant=1)))
    assert ratio >= INK_THRESHOLD, f"longer-line change scored {ratio:.5f}"


def test_dark_text_on_light_plate_works_too():
    """Deviation from the median is polarity-agnostic."""
    ratio = _ink_diff_ratio(
        _ink(_box(glyphs=3, variant=0, light_text=False)),
        _ink(_box(glyphs=3, variant=1, light_text=False)),
    )
    assert ratio >= INK_THRESHOLD, f"dark-on-light change scored {ratio:.5f}"


# --------------------------------------------------------------------------
# Specificity: the optimisation must survive
# --------------------------------------------------------------------------


def test_static_box_stays_quiet():
    """An unchanged box must not re-run OCR -- that is the whole point."""
    frame = _box(glyphs=3, variant=0)
    assert _ink_diff_ratio(_ink(frame), _ink(_box(glyphs=3, variant=0))) == 0.0


def test_vn_cache_requires_both_change_signals_to_be_quiet():
    ink = bytes([0, 1, 0, 1])
    stable_brightness = bytes([20, 20, 20, 20])
    changed_brightness = bytes([40, 40, 20, 20])
    assert _vn_ocr_cache_can_reuse(
        ink,
        ink,
        stable_brightness,
        stable_brightness,
        ink_threshold=INK_THRESHOLD,
        brightness_threshold=LEGACY_THRESHOLD,
        cache_age_sec=1.0,
    )
    assert not _vn_ocr_cache_can_reuse(
        ink,
        ink,
        changed_brightness,
        stable_brightness,
        ink_threshold=INK_THRESHOLD,
        brightness_threshold=LEGACY_THRESHOLD,
        cache_age_sec=1.0,
    )


def test_vn_cache_is_forced_to_refresh_after_ten_seconds():
    ink = bytes([0, 1, 0, 1])
    brightness = bytes([20, 20, 20, 20])
    assert _vn_ocr_cache_can_reuse(
        ink,
        ink,
        brightness,
        brightness,
        ink_threshold=INK_THRESHOLD,
        brightness_threshold=LEGACY_THRESHOLD,
        cache_age_sec=_VN_OCR_CACHE_MAX_REUSE_SEC - 0.01,
    )
    assert not _vn_ocr_cache_can_reuse(
        ink,
        ink,
        brightness,
        brightness,
        ink_threshold=INK_THRESHOLD,
        brightness_threshold=LEGACY_THRESHOLD,
        cache_age_sec=_VN_OCR_CACHE_MAX_REUSE_SEC,
    )


def test_background_brightness_change_does_not_look_like_text():
    """Gentle background animation must not defeat the skip."""
    base = _ink(_box(glyphs=3, variant=0))
    for brightness in (-4, -2, 2, 3, 5):
        ratio = _ink_diff_ratio(base, _ink(_box(glyphs=3, variant=0, brightness=brightness)))
        assert ratio < INK_THRESHOLD, (
            f"brightness {brightness:+d} scored {ratio:.5f}; the cut must stay "
            "relative to the crop's own contrast"
        )


def test_flat_crop_is_stable_not_explosive():
    """A crop with no contrast has no ink on either side, so nothing changed."""
    flat_a = _ink(Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20)))
    flat_b = _ink(Image.new("RGB", (WIDTH, HEIGHT), (24, 24, 24)))
    assert sum(flat_a) == 0
    assert _ink_diff_ratio(flat_a, flat_b) == 0.0


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_metrics_helper_picks_the_right_tool_per_mode():
    frame = _box(glyphs=3)
    vn_fp, vn_diff = _text_change_metrics(frame, visual_novel=True)
    screen_fp, screen_diff = _text_change_metrics(frame, visual_novel=False)
    assert vn_diff is _ink_diff_ratio, "VN crops must use the ink comparison"
    assert screen_diff is _fingerprint_diff_ratio, "full screens keep the scene comparison"
    assert len(vn_fp) == 192 * 48, "VN crops use the 192x48 ink map"
    assert len(screen_fp) == 32 * 18, "full screens keep the coarse 32x18 map"


def test_both_scan_paths_use_the_ink_metrics():
    """Both scan paths must use the shared dual-signal cache decision."""
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert source.count("_text_change_metrics(") >= 3, (
        "expected the definition plus both scan paths (subprocess and in-process)"
    )
    assert source.count("_vn_stable_cache_can_reuse(") >= 3, (
        "expected the helper definition plus both scan paths"
    )


def test_static_crop_reuses_ocr_beyond_ten_seconds_but_still_expires():
    ink = bytes([0, 1, 0, 1])
    brightness = bytes([20, 20, 20, 20])
    kwargs = dict(ink_threshold=INK_THRESHOLD, brightness_threshold=LEGACY_THRESHOLD)
    assert _vn_stable_cache_can_reuse(ink, ink, brightness, brightness, cache_age_sec=30.0, **kwargs)
    assert not _vn_stable_cache_can_reuse(ink, ink, brightness, brightness, cache_age_sec=60.0, **kwargs)
    assert not _vn_stable_cache_can_reuse(ink, ink, brightness, brightness, cache_age_sec=10.0, has_text=False, **kwargs)


def test_small_typewriter_extension_cannot_reuse_partial_ocr():
    # One additional glyph among many can pass the old 12% gate.
    before = bytes([1] * 100 + [0] * 100)
    after = bytes([1] * 103 + [0] * 97)
    brightness = bytes([20] * 200)
    kwargs = dict(ink_threshold=INK_THRESHOLD, brightness_threshold=LEGACY_THRESHOLD, cache_age_sec=1.0)
    assert _vn_ocr_cache_can_reuse(after, before, brightness, brightness, **kwargs)
    assert not _vn_stable_cache_can_reuse(after, before, brightness, brightness, **kwargs)


def test_growing_text_waits_until_a_quiet_capture_before_ocr():
    cache = {}
    brightness = bytes([20] * 200)
    first = bytes([1] * 100 + [0] * 100)
    full = bytes([1] * 130 + [0] * 70)
    assert _vn_text_is_settling(cache, first, brightness, 0.0)
    assert _vn_text_is_settling(cache, full, brightness, 0.2)
    assert _vn_text_is_settling(cache, full, brightness, 0.4)
    assert not _vn_text_is_settling(cache, full, brightness, 0.6)


def test_animated_crop_cannot_defer_ocr_indefinitely():
    cache = {}
    ink = bytes([1] * 100 + [0] * 100)
    for index in range(4):
        brightness = bytes([20 + index * 20] * 100 + [20] * 100)
        assert _vn_text_is_settling(cache, ink, brightness, index * 0.2)
    assert not _vn_text_is_settling(cache, ink, bytes([120] * 100 + [20] * 100), 0.81)


def test_long_typewriter_growth_gets_a_bounded_extension():
    cache = {}
    brightness = bytes([20] * 200)
    for index in range(7):
        ink = bytes([1] * (40 + index * 10) + [0] * (160 - index * 10))
        assert _vn_text_is_settling(cache, ink, brightness, index * 0.2)
    assert not _vn_text_is_settling(cache, bytes([1] * 150 + [0] * 50), brightness, 1.61)


def test_threshold_is_configured_and_sane():
    assert 0.0 < INK_THRESHOLD <= 0.5, (
        f"ink threshold {INK_THRESHOLD} should sit far below real changes "
        "(~0.45-0.87) and far above noise (0.00)"
    )
