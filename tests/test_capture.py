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


def test_printwindow_keeps_client_only_flag_on_primary_and_fallback():
    import ctypes
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import importlib.util
    from desktop_pet.vision import capture
    # Existing capture-fixture tests replace module globals without restoring
    # them; load this API test independently so it exercises the real function.
    spec = importlib.util.spec_from_file_location('capture_api_test', capture.__file__)
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    capture_window_client = isolated.capture_window_client

    for answers, expected in [([1], [3]), ([0, 1], [3, 1]), ([0, 0], [3, 1])]:
        user = MagicMock()
        gdi = MagicMock()
        user.IsWindow.return_value = 1
        user.GetDC.return_value = 1
        def rect(hwnd, pointer):
            pointer._obj.left = pointer._obj.top = 0
            pointer._obj.right, pointer._obj.bottom = 80, 60
            return 1
        user.GetClientRect.side_effect = rect
        user.PrintWindow.side_effect = answers
        gdi.CreateCompatibleDC.return_value = 1
        gdi.CreateCompatibleBitmap.return_value = 1
        gdi.GetDIBits.return_value = 60
        with patch.object(ctypes, 'windll', SimpleNamespace(user32=user, gdi32=gdi), create=True):
            image = capture_window_client(1)
        assert [call.args[2] for call in user.PrintWindow.call_args_list] == expected
        assert (image.size if image is not None else None) == ((80, 60) if any(answers) else None)
