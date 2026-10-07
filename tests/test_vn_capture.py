"""Tests for visual-novel capture resolution (window / manual modes)."""

from __future__ import annotations

from PIL import Image

from desktop_pet.vision import capture as capture_module
from desktop_pet.vision.game_window import CaptureRect
from desktop_pet.main import _resolve_vn_capture


class _StubResolver:
    def __init__(self, rect: CaptureRect | None, target_title: str = "") -> None:
        self.rect = rect
        self.target_title = target_title
        self.calls = 0

    def resolve(self):
        self.calls += 1
        return self.rect

    def invalidate(self) -> None:
        pass


def _patch_capture(calls: dict, *, direct_available: bool = False):
    def fake_capture_window_client(hwnd):
        calls["direct_hwnd"] = hwnd
        if not direct_available:
            return None
        return Image.new("RGB", (1600, 1000), (5, 6, 7))

    def fake_capture_absolute_rect(left, top, width, height):
        calls["absolute"] = (left, top, width, height)
        return Image.new("RGB", (max(1, width), max(1, height)), (10, 20, 30))

    def fake_capture_primary_screen(monitor_index=0, region=None):
        calls["primary"] = (monitor_index, region)
        if region is not None:
            _left, _top, width, height = region
            return Image.new("RGB", (max(1, width), max(1, height)), (40, 50, 60))
        return Image.new("RGB", (1920, 1080), (40, 50, 60))

    def fake_get_monitor_geometry(monitor_index=0):
        calls["monitor"] = monitor_index
        return (0, 0, 1920, 1080)

    capture_module.capture_window_client = fake_capture_window_client
    capture_module.capture_absolute_rect = fake_capture_absolute_rect
    capture_module.capture_primary_screen = fake_capture_primary_screen
    capture_module.get_monitor_geometry = fake_get_monitor_geometry


def test_window_mode_uses_bottom_band_when_no_ocr_region():
    calls: dict = {}
    _patch_capture(calls)
    resolver = _StubResolver(CaptureRect(100, 200, 1600, 1000, 1, "window"))

    image, rect, vision_image, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=None,
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "window"
    assert calls["absolute"] == (100, 200, 1600, 1000)
    # Vision gets the whole window; OCR gets the lower band only.
    assert vision_image.size == (1600, 1000)
    assert image.size[0] == 1600
    assert image.size[1] < 1000
    assert rect is not None and rect.as_region() == (100, 200, 1600, 1000)


def test_window_mode_prefers_direct_window_capture():
    calls: dict = {}
    _patch_capture(calls, direct_available=True)
    resolver = _StubResolver(
        CaptureRect(
            100,
            200,
            1600,
            1000,
            1,
            source="window",
            title="",
            uses_full_screen=False,
            process_name="game.exe",
            hwnd=777,
        )
    )

    _image, _rect, vision_image, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=None,
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    # PrintWindow avoids the composited desktop entirely.
    assert source == "window_direct"
    assert calls["direct_hwnd"] == 777
    assert "absolute" not in calls
    assert vision_image.size == (1600, 1000)


def test_window_mode_uses_hand_picked_ocr_region():
    calls: dict = {}
    _patch_capture(calls)
    resolver = _StubResolver(CaptureRect(100, 200, 1600, 1000, 1, "window"))

    image, _rect, vision_image, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=(80, 700, 1440, 260),
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "window"
    assert vision_image.size == (1600, 1000)
    assert image.size == (1440, 260)


def test_window_mode_clamps_out_of_range_ocr_region():
    calls: dict = {}
    _patch_capture(calls)
    resolver = _StubResolver(CaptureRect(0, 0, 800, 600, 1, "window"))

    image, _rect, _vision, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=(700, 500, 900, 900),
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "window"
    assert image.size == (100, 100)


def test_window_mode_falls_back_to_screen_without_window():
    calls: dict = {}
    _patch_capture(calls)
    resolver = _StubResolver(None)

    image, rect, _vision_image, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=None,
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert resolver.calls == 1
    assert source == "screen_fallback"
    assert rect is None
    assert calls["primary"] == (0, None)
    assert image.size[1] < 1080


def test_locked_window_missing_does_not_capture_other_apps():
    calls: dict = {}
    _patch_capture(calls)
    resolver = _StubResolver(None, target_title="Sakura.exe")

    image, rect, vision_image, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=None,
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert (image, rect, vision_image, source) == (None, None, None, "target_missing")
    assert "primary" not in calls


def test_manual_mode_crops_the_vision_box():
    """Manual mode must honour the hand-picked 画面 box for vision and OCR."""
    calls: dict = {}
    _patch_capture(calls)

    image, rect, vision_image, source = _resolve_vn_capture(
        mode="manual",
        resolver=_StubResolver(None),
        ocr_region=None,
        vision_region=(200, 300, 1000, 700),
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "manual"
    assert rect is None
    # Vision is the picked box, not the whole desktop.
    assert vision_image.size == (1000, 700)
    # OCR is a band *inside* that box.
    assert image.size[0] == 1000
    assert image.size[1] < 700


def test_manual_mode_translates_ocr_region_into_the_box():
    calls: dict = {}
    _patch_capture(calls)

    image, _rect, vision_image, source = _resolve_vn_capture(
        mode="manual",
        resolver=_StubResolver(None),
        # Window-relative OCR box, fully inside the vision box. The vision box
        # shifts the origin by (200,300) so the crop must be translated to
        # (100,100) to stay aligned with the dialogue.
        ocr_region=(300, 400, 500, 200),
        vision_region=(200, 300, 1000, 700),
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "manual"
    assert vision_image.size == (1000, 700), f"vision={vision_image.size}"
    assert image.size == (500, 200), f"ocr={image.size} source={source} calls={calls}"


def test_manual_mode_without_a_box_uses_screen_fallback():
    calls: dict = {}
    _patch_capture(calls)

    _image, rect, _vision, source = _resolve_vn_capture(
        mode="manual",
        resolver=_StubResolver(None),
        ocr_region=None,
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "screen_fallback"
    assert rect is None


def test_tiny_ocr_region_is_ignored():
    calls: dict = {}
    _patch_capture(calls)
    resolver = _StubResolver(CaptureRect(0, 0, 1600, 1000, 1, "window"))

    image, _rect, _vision, source = _resolve_vn_capture(
        mode="window",
        resolver=resolver,
        ocr_region=(10, 10, 4, 4),
        vision_region=None,
        monitor_index=0,
        text_ratio=0.58,
    )

    assert source == "window"
    # Falls back to the heuristic band rather than cropping a 4x4 box.
    assert image.size[1] > 4
