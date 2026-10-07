"""A game must be verified before its dialogue enters an existing story cache."""

from pathlib import Path
from tempfile import TemporaryDirectory

import desktop_pet.main as main_module
from desktop_pet.llm.visual_novel import VisualNovelStoryLibrary, VisualNovelTracker
from desktop_pet.vision.game_window import GameWindowResolver


def test_empty_cache_binds_to_first_game_and_survives_reload():
    with TemporaryDirectory() as temp_dir:
        library = VisualNovelStoryLibrary(Path(temp_dir))
        path = library.create("本次游戏")
        tracker = VisualNovelTracker(path)
        assert not tracker.has_story_content
        assert main_module.story_cache_game_status(tracker.game_title, tracker.has_story_content, "樱花、萌放") == "bind_empty"

        tracker.bind_game_title("樱花、萌放【汉化组】v1.0")
        restored = VisualNovelTracker(path)
        assert restored.game_title == "樱花、萌放【汉化组】v1.0"
        assert main_module.story_cache_game_status(restored.game_title, False, "樱花、萌放【汉化组】v1.1") == "match"
        assert main_module.story_cache_game_status(restored.game_title, False, "柠檬果酱乐队") == "mismatch"


def test_populated_legacy_cache_requires_confirmation_before_binding():
    with TemporaryDirectory() as temp_dir:
        library = VisualNovelStoryLibrary(Path(temp_dir))
        path = library.create("自定义旧缓存")
        tracker = VisualNovelTracker(path)
        tracker.apply_evaluation({"facts": ["角色准备出发。"]})
        tracker.flush()

        restored = VisualNovelTracker(path)
        assert restored.has_story_content
        assert restored.game_title == ""
        assert main_module.story_cache_game_status("", restored.has_story_content, "樱花、萌放") == "unbound"


def test_guard_runs_before_ocr_enters_story_memory():
    source = Path(main_module.__file__).read_text(encoding="utf-8")
    handler = source[source.index("    def _process_visual_novel_text("):source.index("    def _sync_plot_memory(")]
    assert handler.index("if not _check_game_window_change():") < handler.index("visual_novel_tracker.observe_ocr(")
    assert "return" in handler[handler.index("if not _check_game_window_change():"):handler.index("visual_novel_tracker.observe_ocr(")]
    assert GameWindowResolver._is_unlikely_target("ChatGPT")
