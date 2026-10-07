"""Tests for the independent vision sampler's trigger policy.

Vision used to be triggered from the OCR scan callback, which meant it required
dialogue text to arrive -- so a CG cut or a text-free opening movie could never
be looked at. It is now its own 1s loop whose policy is the pure function
``vision_should_capture``, exercised here.

Also guards the structural properties that regressed before:
  * the trigger must be reachable without OCR text,
  * the sampler must key off the visual frame, not the OCR text,
  * the loop must actually be started on a thread.
"""

from __future__ import annotations

import ast
from pathlib import Path

import desktop_pet.main as main_module
from desktop_pet.main import vision_should_capture

_MAIN = Path(main_module.__file__)
_BASE = {
    "tokens": 3.0,
    "delta": 0.60,
    "in_flight": False,
    "enabled": True,
    "active": True,
    "budget_ok": True,
    "threshold": 0.10,
    "shots_per_minute": 3.0,
}


def test_big_scene_change_fires_immediately():
    should, reason = vision_should_capture(**_BASE)
    assert should, reason


def test_quiet_scene_is_skipped():
    should, reason = vision_should_capture(**{**_BASE, "delta": 0.02})
    assert not should
    assert reason.startswith("unchanged")


def test_cg_cut_fires_even_with_no_dialogue():
    """The whole point: content comes from the picture, not from text."""
    # No parameter expresses "no new text" -- text is not part of the decision.
    should, _ = vision_should_capture(**{**_BASE, "delta": 0.92})
    assert should


def test_quota_shortfall_is_reported_with_a_wait():
    should, reason = vision_should_capture(**{**_BASE, "tokens": 0.25})
    assert not should
    assert "quota" in reason
    # 0.75 tokens short at 3/min is 15s.
    assert "15" in reason


def test_in_flight_blocks_a_second_request():
    should, reason = vision_should_capture(**{**_BASE, "in_flight": True})
    assert not should and reason == "in_flight"


def test_disabled_and_inactive_short_circuit():
    assert vision_should_capture(**{**_BASE, "enabled": False})[1] == "disabled"
    assert vision_should_capture(**{**_BASE, "active": False})[1] == "scan_off"


def test_budget_ceiling_still_applies():
    should, reason = vision_should_capture(**{**_BASE, "budget_ok": False})
    assert not should and reason == "budget"


def test_delta_none_means_capture_before_measuring():
    """The first call has no frame yet, so it must not be rejected by the gate."""
    should, reason = vision_should_capture(**{**_BASE, "delta": None})
    assert should, reason


def test_sampler_is_started_on_its_own_thread():
    source = _MAIN.read_text(encoding="utf-8")
    assert "_vision_sampler_loop" in source
    assert 'name="vision-sampler"' in source
    tree = ast.parse(source)
    starts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
    ]
    assert starts, "no thread start found"


def test_sampler_loop_ticks_at_one_second_by_default():
    source = _MAIN.read_text(encoding="utf-8")
    start = source.find("def _vision_sampler_loop")
    assert start != -1
    body = source[start : source.find("vision_sampler_thread", start)]
    assert "wait = 1.0" in body, "the default sampler tick should be one second"
    assert "vision_sampler_stop.wait(" in body


def test_sampler_sleeps_out_the_gap_instead_of_polling():
    """While the floor is binding there is nothing to do, so it may sleep."""
    source = _MAIN.read_text(encoding="utf-8")
    start = source.find("def _vision_sampler_loop")
    body = source[start : source.find("vision_sampler_thread", start)]
    assert "min_gap" in body, "the loop should account for the minimum gap"
    assert "wait = max(" in body, "the loop should extend its sleep to clear the gap"


# --- minimum gap floor -------------------------------------------------------


def test_gap_floor_blocks_a_request_that_just_fired():
    should, reason = vision_should_capture(
        **{**_BASE, "since_last_sec": 2.0, "min_gap_sec": 6.0}
    )
    assert not should
    assert reason.startswith("gap")


def test_gap_floor_allows_once_elapsed():
    should, reason = vision_should_capture(
        **{**_BASE, "since_last_sec": 6.5, "min_gap_sec": 6.0}
    )
    assert should, reason


def test_gap_floor_spreads_the_saved_burst():
    """The point of the floor: a full bucket must not be dumped in 3 seconds."""
    fired_at: list[int] = []
    tokens = 3.0
    last = None
    rate = 3.0 / 60.0  # tokens per second
    for second in range(90):
        tokens = min(3.0, tokens + rate)
        since = None if last is None else second - last
        should, _ = vision_should_capture(
            tokens=tokens,
            delta=0.9,
            in_flight=False,
            enabled=True,
            active=True,
            budget_ok=True,
            threshold=0.10,
            shots_per_minute=3.0,
            since_last_sec=since,
            min_gap_sec=6.0,
        )
        if should:
            tokens -= 1.0
            last = second
            fired_at.append(second)

    assert len(fired_at) >= 2, fired_at
    # Nothing may be closer together than the floor.
    gaps = [b - a for a, b in zip(fired_at, fired_at[1:])]
    assert all(gap >= 6 for gap in gaps), f"gap violated: {fired_at}"
    # And the opening burst is spread, not clustered into the first seconds.
    assert fired_at[1] - fired_at[0] >= 6, fired_at
    assert fired_at[2] - fired_at[1] >= 6, fired_at


def test_gap_zero_disables_the_floor():
    should, reason = vision_should_capture(
        **{**_BASE, "since_last_sec": 0.0, "min_gap_sec": 0.0}
    )
    assert should, reason


def test_first_ever_request_is_not_blocked_by_the_gap():
    should, reason = vision_should_capture(
        **{**_BASE, "since_last_sec": None, "min_gap_sec": 6.0}
    )
    assert should, reason


def test_trigger_no_longer_requires_new_story_text():
    source = _MAIN.read_text(encoding="utf-8")
    assert 'vision_skip_reason"] = "story_idle"' not in source, (
        "requiring accepted story text made text-free scenes impossible to see"
    )
    # And it must not live in the OCR callback any more.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_process_visual_novel_text":
            calls = {
                sub.func.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            assert "_maybe_start_vision" not in calls, (
                "vision must not be triggered from the OCR callback again"
            )
