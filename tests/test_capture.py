from desktop_pet.ui.region_selector import scale_region_to_capture
from desktop_pet.vision.capture import _clamp_region, _resolve_monitor_index


def test_virtual_desktop_index_is_supported():
    assert _resolve_monitor_index(0, 3) == 0
    assert _resolve_monitor_index(2, 3) == 2
    assert _resolve_monitor_index(99, 3) == 0


def test_full_scaled_screen_maps_to_full_physical_capture():
    region = scale_region_to_capture(
        (0, 0, 1707, 1067),
        (1707, 1067),
        (2560, 1600),
    )
    assert region == (0, 0, 2560, 1600)


def test_scaled_region_uses_independent_dpi_axes():
    left, top, width, height = scale_region_to_capture(
        (100, 50, 400, 200),
        (1707, 1067),
        (2560, 1600),
    )
    assert (left, top) == (149, 74)
    assert (width, height) == (601, 301)


def test_capture_region_is_clamped_to_monitor_bounds():
    assert _clamp_region((2500, 1500, 500, 500), 2560, 1600) == (2500, 1500, 60, 100)
