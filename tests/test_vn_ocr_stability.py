"""Typewriter OCR should enter story memory once per completed line."""

from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_pet.llm.visual_novel import VisualNovelTracker


def test_growing_dialogue_is_stored_once():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")
        assert not tracker.observe_ocr("姬织 我应该有非取回不可的", now=0.0).accepted
        assert not tracker.observe_ocr("姬织 我应该有非取回不可的东西", now=0.5).accepted
        assert tracker.observe_ocr("姬织 我应该有非取回不可的东西", now=1.4).accepted
        assert tracker.total_lines == 1
        assert not tracker.observe_ocr("姬织 我应该有非取回不可的东西", now=2.0).accepted
        assert not tracker.last_observation_accepted
        assert tracker.total_lines == 1


def test_later_extension_replaces_unsubmitted_line():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")
        tracker.observe_ocr("七七 那个时候我们要道别了", now=0.0)
        tracker.observe_ocr("七七 那个时候我们要道别了", now=0.9)
        tracker.observe_ocr("七七 那个时候我们要道别了，毕竟太阳与月亮无法相容", now=1.2)
        tracker.observe_ocr("七七 那个时候我们要道别了，毕竟太阳与月亮无法相容", now=2.1)
        assert tracker.total_lines == 1
        assert "太阳与月亮" in tracker._lines[-1]["text"]


def test_distinct_fast_lines_are_kept_in_order():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")
        tracker.observe_ocr("第一句话说明她找到了一张地图", now=0.0)
        tracker.observe_ocr("第二句话说明她想找回遗失的东西", now=0.5)
        tracker.observe_ocr("第二句话说明她想找回遗失的东西", now=1.4)
        assert [item["text"] for item in tracker._lines] == [
            "第一句话说明她找到了一张地图",
            "第二句话说明她想找回遗失的东西",
        ]


def test_old_ocr_without_unchanged_frame_cannot_confirm_stability():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")
        half = "七七 那个时候我们要道别了"
        full = half + "，毕竟太阳与月亮无法相容"
        tracker.observe_ocr(half, now=0.0, frame_stable=False)
        assert not tracker.observe_ocr(half, now=1.5, frame_stable=False).accepted
        assert tracker.total_lines == 0
        tracker.observe_ocr(full, now=1.6, frame_stable=False)
        assert tracker.observe_ocr(full, now=2.5, frame_stable=True).accepted
        assert [item["text"] for item in tracker._lines] == [full]


def test_empty_transition_does_not_confirm_partial_text():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")
        tracker.observe_ocr("她想找回那件遗失的东西", now=0.0, frame_stable=False)
        assert not tracker.observe_ocr("", now=2.0, frame_stable=True).accepted
        assert tracker.total_lines == 0
        assert tracker.has_pending_ocr
