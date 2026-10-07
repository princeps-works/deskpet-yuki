"""Tests that worker threads never touch Qt widgets.

``QWidget.winId()`` is not thread-safe. The vision sampler and the scan executor
both used to reach it through ``_refresh_vision_exclusions``, which walked the
pet, chat and side-panel widgets. When the side panel had not been realised yet
(it only becomes native on first hover) the call forced native creation on a
worker thread and wedged Qt's event loop completely: the main thread still sat
idle inside ``app.exec``, but no timer, paint or log line ever ran again -- the
app looked hung while being, in Python terms, perfectly alive.

These tests keep the fix honest: the widget walk lives in a GUI-thread-only
collector, and the exclusion refresh that worker threads call must not mention
any Qt object.
"""

from __future__ import annotations

import ast
from pathlib import Path

import desktop_pet.main as main_module

_MAIN = Path(main_module.__file__)
_SOURCE = _MAIN.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _widget_names_in(node: ast.AST) -> set[str]:
    """Names of Qt widgets referenced anywhere inside ``node``."""
    watched = {"pet", "chat", "vn_sidebar"}
    used: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in watched:
            used.add(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in watched:
            used.add(sub.attr)
    return used


def test_collector_is_the_only_place_that_reads_win_id():
    collector = _find_function("_collect_pet_owned_hwnds")
    source = ast.get_source_segment(_SOURCE, collector) or ""
    assert "winId()" in source, "the collector must actually snapshot the handles"
    assert _widget_names_in(collector) >= {"pet", "chat"}, sorted(_widget_names_in(collector))


def test_exclusion_refresh_is_safe_for_worker_threads():
    """The refresh runs on the sampler and scan threads, so it must be Qt-free."""
    refresh = _find_function("_refresh_vision_exclusions")
    source = ast.get_source_segment(_SOURCE, refresh) or ""
    assert "winId" not in source, "must not call winId() off the GUI thread"
    assert not _widget_names_in(refresh), (
        f"must not touch widgets: {sorted(_widget_names_in(refresh))}"
    )
    # It should read the cached snapshot instead.
    assert "pet_owned_hwnds" in source


def test_worker_call_sites_go_through_the_safe_refresh():
    """Worker paths call the cached refresh and never winId() directly."""
    for name in ("_capture_for_context", "run_scan_pipeline", "vision_sampling_step"):
        func = _find_function(name)
        source = ast.get_source_segment(_SOURCE, func) or ""
        assert "_refresh_vision_exclusions" in source, f"{name} should refresh exclusions"
        assert "winId" not in source, f"{name} must not call winId()"


def test_only_gui_thread_functions_call_win_id():
    """An allowlist, because an off-thread winId() wedges the whole app.

    ``_collect_pet_owned_hwnds`` runs on the GUI thread by construction,
    ``_try_sync_live2d_py_windows`` is driven by a GUI-thread timer,
    ``_sync_live2d_drag_geometry`` by the pet's GUI-thread drag signal, and
    ``_write_live2d_shutdown_marker`` is reached from ``_stop_live2d_py`` during
    ``aboutToQuit``. Anything else appearing here is a new threading bug.
    ``main`` is exempt because its source segment nests every one of the
    functions above; the nested definitions are checked on their own.
    """
    allowed = {
        "main",
        "_collect_pet_owned_hwnds",
        "_try_sync_live2d_py_windows",
        "_sync_live2d_drag_geometry",
        "_write_live2d_shutdown_marker",
    }
    offenders = set()
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.FunctionDef):
            continue
        segment = ast.get_source_segment(_SOURCE, node) or ""
        if "winId" in segment and node.name not in allowed:
            offenders.add(node.name)
    assert not offenders, f"winId() found in non-GUI-thread functions: {sorted(offenders)}"


def test_sampler_never_touches_widgets():
    for name in ("vision_sampling_step", "_vision_sampler_loop", "_vision_worker"):
        func = _find_function(name)
        assert not _widget_names_in(func), f"{name} must stay Qt-free"


def test_snapshot_is_taken_on_the_gui_thread_at_startup_and_kept_fresh():
    assert 'state["pet_owned_hwnds"] = handles' in _SOURCE
    # Taken after the windows are shown, and refreshed by a GUI-thread timer.
    assert "_collect_pet_owned_hwnds()" in _SOURCE
    assert "pet_hwnd_refresh_timer.timeout.connect(_collect_pet_owned_hwnds)" in _SOURCE
    show_at = _SOURCE.find("    pet.show()")
    collect_at = _SOURCE.find("    _collect_pet_owned_hwnds()", show_at)
    assert show_at != -1 and collect_at > show_at, (
        "handles must be snapshotted only after the windows exist"
    )


def test_probe_refreshes_the_snapshot_from_the_gui_thread():
    probe = _find_function("on_vn_probe_ocr")
    source = ast.get_source_segment(_SOURCE, probe) or ""
    assert "_collect_pet_owned_hwnds" in source, (
        "the button handler runs on the GUI thread, so it may refresh directly"
    )
