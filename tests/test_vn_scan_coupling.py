"""Tests for the visual-novel mode / auto-scan coupling.

Enabling VN mode used to leave the auto-scan toggle exactly as it was. Because
that toggle starts every launch switched off, and because both the OCR path and
the vision sampler's activity gate key off it, a fresh launch with VN mode
turned on did nothing at all -- no dialogue text, no picture understanding --
while the side panel reported the raw reason token ``scan_off``.

The policy now lives in the pure function ``vn_scan_transition`` so it can be
checked here without a live window, and the structural properties that make it
work (the toggle can be driven without announcing, and the panel explains the
dependency in Chinese) are guarded too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import desktop_pet.main as main_module
from desktop_pet.main import vn_scan_transition

_MAIN = Path(main_module.__file__)
_SOURCE = _MAIN.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _find_function(name: str, tree: ast.AST = _TREE) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _calls_in(node: ast.AST, func_name: str) -> list[ast.Call]:
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            target = sub.func
            if isinstance(target, ast.Name) and target.id == func_name:
                found.append(sub)
            elif isinstance(target, ast.Attribute) and target.attr == func_name:
                found.append(sub)
    return found


# --------------------------------------------------------------------------
# Entering VN mode
# --------------------------------------------------------------------------


def test_enabling_vn_with_scan_off_turns_scan_on():
    """The regression: VN mode on a fresh launch must not stay inert."""
    target, autostarted, prev, note = vn_scan_transition(
        enabling=True, scan_enabled=False, autostarted=False, prev_enabled=False
    )
    assert target is True, "entering VN mode must switch auto-scan on"
    assert autostarted is True, "must remember that we made the change"
    assert prev is False, "the user's own preference was off"
    assert "自动扫描" in note, note


def test_enabling_vn_when_already_scanning_is_a_no_op():
    """If the user already had scanning on, VN mode must not claim credit."""
    target, autostarted, prev, note = vn_scan_transition(
        enabling=True, scan_enabled=True, autostarted=False, prev_enabled=True
    )
    assert target is None, "nothing to change"
    assert autostarted is False, "nothing to undo on exit"
    assert note == ""


def test_enabling_vn_twice_is_idempotent():
    """A redundant toggle must not overwrite the remembered preference."""
    first = vn_scan_transition(
        enabling=True, scan_enabled=False, autostarted=False, prev_enabled=False
    )
    assert first[0] is True
    # Second enable sees scan already on (set by the first) and does nothing.
    second = vn_scan_transition(
        enabling=True, scan_enabled=True, autostarted=first[1], prev_enabled=first[2]
    )
    assert second[0] is None
    assert second[2] is False, "the original preference survives"


# --------------------------------------------------------------------------
# Leaving VN mode
# --------------------------------------------------------------------------


def test_disabling_vn_restores_scan_off_when_it_autostarted():
    target, autostarted, prev, note = vn_scan_transition(
        enabling=False, scan_enabled=True, autostarted=True, prev_enabled=False
    )
    assert target is False, "must hand the user's original setting back"
    assert autostarted is False
    assert "恢复关闭" in note, note


def test_disabling_vn_restores_scan_on_when_that_was_the_preference():
    target, autostarted, prev, note = vn_scan_transition(
        enabling=False, scan_enabled=True, autostarted=True, prev_enabled=True
    )
    assert target is True
    assert autostarted is False
    assert note == "", "nothing surprising happened, so say nothing"


def test_disabling_vn_leaves_a_deliberate_user_press_alone():
    """A real press on the scan toggle clears the autostart flag, so exit is quiet."""
    target, autostarted, prev, note = vn_scan_transition(
        enabling=False, scan_enabled=True, autostarted=False, prev_enabled=False
    )
    assert target is None, "do not override an explicit user choice"
    assert autostarted is False
    assert note == ""


def test_disabling_vn_when_it_never_touched_scan_is_a_no_op():
    target, autostarted, prev, note = vn_scan_transition(
        enabling=False, scan_enabled=False, autostarted=False, prev_enabled=False
    )
    assert target is None
    assert note == ""


# --------------------------------------------------------------------------
# Structural guards
# --------------------------------------------------------------------------


def test_scan_toggle_can_be_driven_without_announcing():
    """VN mode drives the toggle, so it must not stack a second bubble."""
    func = _find_function("on_scan_toggle")
    args = func.args.args
    assert [a.arg for a in args][:2] == ["enabled", "announce"], [a.arg for a in args]

    defaults = func.args.defaults
    assert defaults, "announce needs a default so the Qt signal keeps working"
    assert isinstance(defaults[-1], ast.Constant) and defaults[-1].value is True

    # The Qt slot is still connected with the signal's single argument.
    assert "pet.auto_scan_toggled.connect(on_scan_toggle)" in _SOURCE


def test_vn_handler_uses_the_policy_and_announces_nothing():
    handler = _find_function("on_visual_novel_mode_toggled")
    assert _calls_in(handler, "vn_scan_transition"), "handler must use the pure policy"

    driven = _calls_in(handler, "on_scan_toggle")
    assert driven, "handler must be able to switch scanning on"
    for call in driven:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "announce" in keywords, "must suppress the duplicate bubble"
        assert isinstance(keywords["announce"], ast.Constant)
        assert keywords["announce"].value is False


def test_vn_mode_message_reports_the_scan_state():
    handler = _find_function("on_visual_novel_mode_toggled")
    source = ast.get_source_segment(_SOURCE, handler) or ""
    assert "scan_autostart_note" in source
    assert "scan_enabled={bool(state['scan_enabled'])}" in source, (
        "the heartbeat should record the resulting scan state"
    )


def test_scan_toggle_clears_autostart_on_a_real_press():
    """Guard the rule that a deliberate press outranks VN mode's bookkeeping."""
    func = _find_function("on_scan_toggle")
    source = ast.get_source_segment(_SOURCE, func) or ""
    assert 'state["vn_scan_autostarted"] = False' in source
    # ...and it must sit under the announce guard, not on the driven path.
    assert "if announce:" in source


def test_panel_explains_the_scan_dependency_in_chinese():
    assert '"scan_off": "未开启自动扫描"' in _SOURCE, (
        "the raw English reason token must not reach the panel"
    )
    assert "画面理解依赖自动扫描" in _SOURCE


def test_config_enabled_vn_mode_leaves_scan_off_at_startup():
    """Auto-scan must default OFF at startup, even with VN mode from config.

    Turning the scan on by itself was both surprising and it hid the manual
    off->on toggle, so startup only ever opts in through AUTO_START_SCAN.
    """
    startup = _find_function("_apply_startup_scan_policy")
    source = ast.get_source_segment(_SOURCE, startup) or ""
    assert "vn_scan_transition" not in source, (
        "startup must not drive the scan through the VN coupling"
    )
    assert "settings.auto_start_scan" in source, "AUTO_START_SCAN is the only opt-in"
    assert "left OFF" in source, "startup should explain why nothing is running"


def test_vn_toggle_still_couples_to_scan():
    """The interactive toggle keeps the coupling; only startup changed."""
    handler = _find_function("on_visual_novel_mode_toggled")
    assert _calls_in(handler, "vn_scan_transition"), "the toggle still uses the policy"


def test_vision_activity_gate_still_keys_off_scan():
    """Documents the dependency this coupling exists to satisfy."""
    source = ast.get_source_segment(_SOURCE, _find_function("vision_sampling_step")) or (
        ast.get_source_segment(_SOURCE, _find_function("_vision_sampler_loop")) or ""
    )
    assert source, "sampler step not found"
    assert "scan_enabled" in source


def test_auto_start_scan_setting_parses_both_ways():
    """AUTO_START_SCAN had no implementation behind it, only a stale comment."""
    import os
    import tempfile

    from desktop_pet.config.settings import load_settings

    saved = os.environ.get("AUTO_START_SCAN")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # No .env here, so nothing overrides the environment under test.
            os.environ["AUTO_START_SCAN"] = "1"
            assert load_settings(Path(tmp)).auto_start_scan is True
            os.environ["AUTO_START_SCAN"] = "false"
            assert load_settings(Path(tmp)).auto_start_scan is False
            os.environ.pop("AUTO_START_SCAN", None)
            assert load_settings(Path(tmp)).auto_start_scan is False, "default is off"
    finally:
        if saved is None:
            os.environ.pop("AUTO_START_SCAN", None)
        else:
            os.environ["AUTO_START_SCAN"] = saved


def test_startup_auto_start_scan_is_opt_in_only():
    """AUTO_START_SCAN defaults off, so a fresh launch never scans on its own."""
    assert "settings.auto_start_scan" in _SOURCE
    startup = _find_function("_apply_startup_scan_policy")
    source = ast.get_source_segment(_SOURCE, startup) or ""
    guard = "if settings.auto_start_scan and not state[\"scan_enabled\"]:"
    assert guard in source, "the enable must be gated on the explicit opt-in"
    # And the opt-in itself must default to false.
    settings_source = (Path(main_module.__file__).parent / "config" / "settings.py").read_text(
        encoding="utf-8"
    )
    assert 'os.getenv("AUTO_START_SCAN", "false")' in settings_source


def test_startup_scan_enable_is_deferred_into_the_event_loop():
    """Starting the scheduler during setup wedged the whole process.

    The identical call from the user's toggle runs inside the event loop and
    always worked, so startup must go through ``QTimer.singleShot`` rather than
    running the transition inline.
    """
    func = _find_function("_apply_startup_scan_policy")
    source = ast.get_source_segment(_SOURCE, func) or ""
    assert "on_scan_toggle" in source

    # ...and the only top-level call to it must be the deferred registration.
    registrations = [
        call
        for call in _calls_in(_TREE, "singleShot")
        if any(
            isinstance(a, ast.Name) and a.id == "_apply_startup_scan_policy"
            for a in call.args
        )
    ]
    assert len(registrations) == 1, "expected exactly one deferred registration"
    # It is only ever *referenced* (passed to singleShot), never called directly.
    assert not _calls_in(_TREE, "_apply_startup_scan_policy"), (
        "must not run inline during startup setup"
    )
