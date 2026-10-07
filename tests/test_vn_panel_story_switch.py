"""Tests for three side-panel defects reported from the running app.

1. Switching the story cache left the panel naming the *old* cache. The panel
   printed ``visual_novel_library.active_name`` (read from the marker file) while
   the tracker is what actually loads and owns a story, so whenever the two
   disagreed the label named a file that was not the one in memory.

2. Loading a cache while a *different* game was on screen blended two stories'
   memory. The guard compares the game window seen last time with the one on
   screen now, and suggests a new cache.

3. The OCR panel reported "当前帧没有文字" while simultaneously displaying
   recognised text. ``scan_cache_ocr_text`` was only ever written by the
   in-process scan path; with ENABLE_SCAN_SUBPROCESS (the default) the panel's
   own field stayed empty while the fallback text came from
   ``last_nonempty_ocr_text``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import desktop_pet.main as main_module
from desktop_pet.main import game_title_changed, normalize_game_title

_MAIN = Path(main_module.__file__)
_SOURCE = _MAIN.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _segment(name: str) -> str:
    return ast.get_source_segment(_SOURCE, _find_function(name)) or ""


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
# Title normalisation (shared by the game-change guard)
# --------------------------------------------------------------------------


def test_normalisation_strips_localisation_group_and_version():
    assert normalize_game_title("青空下的加缪【天空鸽剧汉化组】v0.6-Beta") == "青空下的加缪"
    assert normalize_game_title("星空列车与白的旅行 v1.02") == "星空列车与白的旅行"
    assert normalize_game_title("Aokana [NekoNyan] v1.2.3") == "Aokana"


def test_normalisation_keeps_a_plain_name_intact():
    for name in ("星空列车与白的旅行", "青空下的加缪", "CLANNAD"):
        assert normalize_game_title(name) == name


def test_normalisation_handles_cjk_immediately_before_version():
    """There is no \\b between a CJK character and 'v', so it must still strip."""
    assert normalize_game_title("青空下的加缪v0.6") == "青空下的加缪"


# --------------------------------------------------------------------------
# Issue 2: previous window vs current window
# --------------------------------------------------------------------------


def test_different_games_are_a_change():
    assert game_title_changed("星空列车与白的旅行", "青空下的加缪【汉化组】v0.6-Beta")


def test_same_game_after_a_patch_is_not_a_change():
    """Otherwise every patch update would look like a new game."""
    assert not game_title_changed(
        "青空下的加缪【汉化组】v0.6-Beta", "青空下的加缪【汉化组】v0.7"
    )


def test_unknown_titles_are_not_a_change():
    """An empty title is not evidence; warning on it would be noise."""
    assert not game_title_changed("", "青空下的加缪")
    assert not game_title_changed("星空列车与白的旅行", "")
    assert not game_title_changed("", "")


def test_guard_is_called_when_the_window_title_becomes_known():
    handler = _find_function("_process_visual_novel_text")
    assert _calls_in(handler, "_check_game_window_change"), (
        "the guard must run when the capture title is learned"
    )


def test_guard_only_trusts_a_resolved_window():
    """A screen grab must not become the baseline for the next launch."""
    source = _segment("_check_game_window_change")
    assert '!= "window_direct"' in source
    assert 'state["last_game_title"] = title' in source, "the baseline must advance"


def test_last_game_title_is_persisted_across_launches():
    load = _segment("_load_persisted_regions")
    save = _segment("_save_persisted_regions")
    assert "last_game_title" in load, "must be read back at startup"
    assert 'last_game_title' in _SOURCE
    # The save helper writes every string entry, so the key round-trips.
    assert 'isinstance(value, str)' in save
    assert '"last_game_title"' in _SOURCE


def test_guard_suggests_creating_a_new_cache():
    source = _segment("_check_game_window_change")
    assert "新建一个剧情缓存" in source


# --------------------------------------------------------------------------
# Issue 1: the panel must name the cache that is actually loaded
# --------------------------------------------------------------------------


def test_panel_shows_the_loaded_cache_not_the_marker():
    source = _segment("_loaded_story_name")
    assert "visual_novel_tracker.path" in source, (
        "the loaded file is the source of truth"
    )
    assert "active_name" in source, "the marker is only the fallback"
    # And the panel uses that helper.
    assert "_loaded_story_name()" in _segment("refresh_vn_sidebar_state")


def test_switch_refreshes_the_panel_on_both_paths():
    """It used to rely on the 2.5s tick, and on a timer that may never start."""
    source = _segment("_switch_visual_novel_story")
    assert source.count("refresh_vn_sidebar_state()") >= 2, (
        "both the already-current path and the switching path must refresh"
    )


def test_already_current_path_still_writes_the_marker():
    """Otherwise the panel and the marker disagree forever after a partial switch."""
    source = _segment("_switch_visual_novel_story")
    assert "set_active" in source


# --------------------------------------------------------------------------
# Issue 3: publish the OCR reading from both scan paths
# --------------------------------------------------------------------------


def test_ocr_result_is_published_to_state_for_the_panel():
    handler = _find_function("do_auto_comment")
    source = ast.get_source_segment(_SOURCE, handler) or ""
    assert 'state["scan_cache_ocr_text"] = str(result.get("ocr_text"' in source, (
        "the subprocess path returns through a queue and never wrote this, so the "
        "panel said 'no text' while showing recognised text"
    )
    assert 'state["scan_cache_ocr_confidence"]' in source


def test_subprocess_default_is_the_reason_it_was_missed():
    """Documents why the in-thread-only write was not enough."""
    settings_source = (
        Path(main_module.__file__).parent / "config" / "settings.py"
    ).read_text(encoding="utf-8")
    assert 'os.getenv("ENABLE_SCAN_SUBPROCESS", "true")' in settings_source
