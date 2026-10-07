"""Tests for the automatic text band and the menu-bar chrome it used to include.

Reported from the running app: "without doing a manual box selection first, OCR
has a hard time reading text".

Measured on the live frame (青空下的加缪, 2560x1600):

  * the automatic band is the lower 42% of the window, so it always contains the
    game's bottom UI bar;
  * OCR returns the narration and the bar glued together --
    "燠微微地睁开眼睛。…把燠弄醒了。 SAVELOADCONFIGSKIPAUTOLOGQ.SAVEQ.LOAD";
  * ``prepare_ocr_text`` accepts that mixture as ``quality_ok``, so the chrome is
    stored as dialogue. The user's own cache contains the resulting fact
    "台词中混有SAVE/LOAD/Q.SAVE/Q.LOAD等界面文本";
  * a manual box that stops above the bar returns 45 characters at confidence
    0.99 with zero chrome tokens.

Pixel geometry of that frame: dialogue ends at ~91% of the height, the bar text
sits at 93-95%, and below 95.5% is a black letterbox -- hence the 8% margin.
"""

from __future__ import annotations

import ast
from pathlib import Path

from PIL import Image

import desktop_pet.main as main_module
from desktop_pet.config.settings import load_settings
from desktop_pet.main import (
    _crop_text_band,
    _text_band_candidates,
    should_run_band_sweep,
    strip_ui_chrome,
)

_MAIN = Path(main_module.__file__)
_SOURCE = _MAIN.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_SETTINGS = load_settings(Path(__file__).resolve().parent.parent)

# What OCR actually returned on the live frame.
REAL_MIXED = (
    "燠微微地睁开眼睛。 萤在旁边睡得正香，把头靠在了燠的肩膀上，把燠弄醒了。 "
    "SAVELOADCONFIGSKIPAUTOLOGQ.SAVEQ.LOAD"
)
REAL_CHROME_ONLY = "SAVELOADCONFIGSKIPAUTOLOGQ.SAVEQ.LOAD"


def _segment(name: str) -> str:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SOURCE, node) or ""
    raise AssertionError(f"function {name} not found")


# --------------------------------------------------------------------------
# Chrome stripping
# --------------------------------------------------------------------------


def test_real_mixed_reading_keeps_only_the_dialogue():
    cleaned = strip_ui_chrome(REAL_MIXED)
    assert "SAVE" not in cleaned and "LOAD" not in cleaned
    assert "CONFIG" not in cleaned and "SKIP" not in cleaned
    assert "燠微微地睁开眼睛" in cleaned
    assert "把燠弄醒了" in cleaned
    assert len(cleaned) < len(REAL_MIXED)


def test_chrome_only_reading_becomes_empty():
    assert strip_ui_chrome(REAL_CHROME_ONLY) == ""
    assert strip_ui_chrome("SAVELOADCONFIGSKIPAUTOLOGQ.SAVE Q.LOAD") == ""


def test_plain_dialogue_is_untouched():
    for line in (
        "我从口袋里掏出了被汗液略微沾湿的手机。",
        "窗外的夜色很深，星星一闪一闪。",
    ):
        assert strip_ui_chrome(line) == line


def test_ellipsis_dialogue_survives():
    """A silent line is legitimate dialogue and must not be eaten."""
    assert strip_ui_chrome("......") == "......"
    assert strip_ui_chrome("晓 ......") == "晓 ......"


def test_a_word_containing_a_token_is_not_mangled():
    """'DIALOGUE' contains 'LOG' but is not a menu-bar word."""
    assert "DIALOGUE" in strip_ui_chrome("他走进了 DIALOGUE 场景。")


# --------------------------------------------------------------------------
# Band geometry
# --------------------------------------------------------------------------


def _frame(height: int = 1600, width: int = 2560) -> Image.Image:
    return Image.new("RGB", (width, height), (10, 10, 10))


def test_band_excludes_the_bottom_ui_strip():
    band = _crop_text_band(_frame(), 0.58, bottom_margin=0.08)
    assert band.size == (2560, int(round(1600 * (0.92 - 0.58))))


def test_band_keeps_the_dialogue_rows():
    """Dialogue measured at 80-91% of the height must stay inside the band."""
    band = _crop_text_band(_frame(), 0.58, bottom_margin=0.08)
    top = int(round(1600 * 0.58))
    assert top <= int(1600 * 0.80)
    assert top + band.size[1] >= int(1600 * 0.91)
    # ...and the bar (93-95%) must be outside it.
    assert top + band.size[1] <= int(1600 * 0.93)


def test_zero_margin_reproduces_the_old_behaviour():
    band = _crop_text_band(_frame(), 0.58, bottom_margin=0.0)
    assert band.size[1] == 1600 - int(round(1600 * 0.58))


def test_degenerate_margin_returns_the_frame_rather_than_nothing():
    band = _crop_text_band(_frame(), 0.58, bottom_margin=0.40)
    assert band.size[0] == 2560 and band.size[1] > 0


def test_configured_margin_matches_the_measurement():
    assert abs(_SETTINGS.vn_text_band_bottom_margin - 0.08) < 1e-9, (
        "the default must clear the bar measured at 93-95% of the height"
    )


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_resolve_vn_capture_accepts_and_uses_the_margin():
    source = _segment("_resolve_vn_capture")
    assert "text_bottom_margin" in source
    assert "_crop_text_band" in source, "the automatic band must use the margin"
    assert "settings.vn_text_band_bottom_margin" in _SOURCE, (
        "the live call sites must pass the configured margin"
    )


def test_both_scan_paths_pass_the_margin():
    assert _SOURCE.count("text_bottom_margin=") >= 3, (
        "in-process pipeline, subprocess worker and the probe button"
    )


def test_chrome_is_stripped_before_the_text_is_used():
    handler = _segment("do_auto_comment")
    assert "strip_ui_chrome" in handler
    # ...and before the panel field is published, so both agree.
    assert handler.index("strip_ui_chrome") < handler.index("scan_cache_ocr_text")


def test_sweep_only_runs_when_the_read_is_completely_empty():
    """It used to run below 12 chars, paying ~20 OCR passes for a little junk."""
    source = _segment("run_scan_pipeline")
    assert "ocr_chars <= 0" in source
    assert "ocr_chars < 12" not in source


def test_sweep_no_longer_prefers_the_chrome_heavy_full_frame():
    source = _segment("run_scan_pipeline")
    assert '"full_window"' not in source, (
        "the full frame won the 'richest reading' comparison because of the bar"
    )
    assert "_text_band_candidates" in source


def test_sweep_is_bounded_to_two_variants_per_candidate():
    source = _segment("run_scan_pipeline")
    assert "_ocr_text_variants(candidate_image)[:2]" in source, (
        "each extra variant is a full OCR pass; the sweep was up to 20 of them"
    )


def test_sweep_candidates_are_disjoint_and_cover_the_text_zone():
    """The retry bands must read new pixels, not repeat the primary band.

    The old sweep used four overlapping bands; with the bottom margin applied,
    its "70-92%" band sat entirely inside the app's own "58-92%" band, so it paid
    a full OCR pass to re-read identical pixels.
    """
    rows = _text_band_candidates(
        500, text_ratio=0.58, bottom_margin=0.08, mid_top_ratio=0.30
    )
    assert [label for label, _t, _b in rows] == ["text_band", "mid_band"]
    (_, lower_top, lower_bottom), (_, mid_top, mid_bottom) = rows
    assert lower_top == mid_bottom, "must abut exactly, never overlap"
    assert mid_top < lower_top < lower_bottom
    # The union spans the mid band down to the bottom margin.
    assert mid_top == int(round(500 * 0.30))
    assert lower_bottom == int(round(500 * 0.92))


def test_sweep_covers_where_dialogue_actually_sits():
    """Dialogue measured at 80-91% must be inside the first candidate."""
    rows = _text_band_candidates(
        1600, text_ratio=0.58, bottom_margin=0.08, mid_top_ratio=0.30
    )
    _label, top, bottom = rows[0]
    assert top <= int(1600 * 0.80)
    assert bottom >= int(1600 * 0.91)


def test_mid_band_is_dropped_when_the_frame_is_too_short():
    """A degenerate 30-pixel strip is not worth an OCR pass."""
    rows = _text_band_candidates(
        100, text_ratio=0.58, bottom_margin=0.08, mid_top_ratio=0.30
    )
    assert [label for label, _t, _b in rows] == ["text_band"]


def test_no_candidates_for_a_degenerate_height():
    assert _text_band_candidates(0, text_ratio=0.58, bottom_margin=0.08, mid_top_ratio=0.30) == []


def test_sweep_no_longer_uses_overlapping_anchored_bands():
    """The anchored helper is gone; keeping it would be dead code."""
    assert "_ocr_band_regions" not in _SOURCE
    assert "_text_band_candidates" in _SOURCE


def test_sweep_waits_for_a_sustained_absence_of_text():
    """One blank read is normal; the sweep must not fire on it.

    A dialogue box blinks out between lines, scene transitions have no text, and
    a frame can be caught mid-fade. Sweeping on the first empty read cost a ~10s
    recovery every time the box was briefly blank.
    """
    base = dict(
        vn_mode=True,
        has_vision_frame=True,
        ocr_chars=0,
        quiet_sec=5.0,
        quiet_required_sec=5.0,
        frame_changed=True,
        cooled_down=True,
    )
    should, reason = should_run_band_sweep(**base)
    assert should, reason

    # 4.9s of blank is not yet worth a sweep...
    should, reason = should_run_band_sweep(**{**base, "quiet_sec": 4.9})
    assert not should and reason.startswith("quiet"), reason

    # ...and a single blank read right after text arrived must not trigger it.
    should, reason = should_run_band_sweep(**{**base, "quiet_sec": 0.0})
    assert not should and reason.startswith("quiet"), reason


def test_sweep_policy_declines_for_each_other_reason():
    base = dict(
        vn_mode=True,
        has_vision_frame=True,
        ocr_chars=0,
        quiet_sec=30.0,
        quiet_required_sec=5.0,
        frame_changed=True,
        cooled_down=True,
    )
    should, _ = should_run_band_sweep(**base)
    assert should

    for override, expected in (
        ({"vn_mode": False}, "no_vn_frame"),
        ({"has_vision_frame": False}, "no_vn_frame"),
        ({"ocr_chars": 3}, "text_found"),
        ({"frame_changed": False}, "frame_unchanged"),
        ({"cooled_down": False}, "cooldown"),
    ):
        should, reason = should_run_band_sweep(**{**base, **override})
        assert not should and reason == expected, (override, reason)


def test_the_quiet_clock_restarts_when_text_arrives():
    handler = _segment("_process_visual_novel_text")
    assert 'state["no_text_since"] = None' in handler


def test_quiet_window_defaults_to_five_seconds():
    assert abs(_SETTINGS.ocr_fallback_min_quiet_sec - 5.0) < 1e-9


def test_sweep_is_still_bounded_by_its_own_cooldown():
    """The quiet window and the between-sweeps cooldown are separate knobs."""
    assert _SETTINGS.ocr_fallback_min_interval_sec >= _SETTINGS.ocr_fallback_min_quiet_sec
    source = _segment("run_scan_pipeline")
    assert "should_run_band_sweep(" in source


def test_panel_reports_that_a_scan_is_in_flight():
    source = _segment("_refresh_ocr_status")
    assert "扫描中" in source
    assert 'state.get("scan_future")' in source


def test_panel_only_reports_first_scan_without_any_successful_history():
    source = _segment("_refresh_ocr_status")
    previous_lookup = 'state.get("last_nonempty_ocr_text"'
    assert source.index(previous_lookup) < source.index("if scanning_for >= 0:")
    assert "elif previous:" in source
    assert "上一轮成功" in source
    assert "首次识别需要几秒" in source
