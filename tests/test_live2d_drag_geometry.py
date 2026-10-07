"""Cross-screen drag uses one native pixel origin for the UI and model."""

from pathlib import Path
import ctypes
from ctypes import wintypes
from types import SimpleNamespace

from desktop_pet.ui.pet_window import native_drag_target
import desktop_pet.main as main_module
import desktop_pet.ui.pet_window as pet_window_module
from PyQt6.QtCore import QPoint


def test_native_drag_target_keeps_cursor_offset_across_monitors():
    anchor = (120, 340, 20, 200)
    assert native_drag_target(anchor, (120, 340)) == (20, 200)
    assert native_drag_target(anchor, (-300, 530)) == (-400, 390)


def test_model_sync_is_connected_to_drag_events():
    source = Path(main_module.__file__).read_text(encoding="utf-8")
    assert "pet.live2d_drag_moved.connect(_sync_live2d_drag_geometry)" in source
    assert "target_rect = _get_live2d_target_geometry_native(user32, host_hwnd)" in source


def test_drag_event_moves_native_host_and_requests_model_sync():
    calls = []

    class User32:
        def GetCursorPos(self, pointer):
            point = ctypes.cast(pointer, ctypes.POINTER(wintypes.POINT)).contents
            point.x, point.y = -300, 530
            return 1

        def SetWindowPos(self, *args):
            calls.append(args)
            return 1

    host = SimpleNamespace(
        _native_drag_anchor=(120, 340, 20, 200),
        live2d_drag_moved=SimpleNamespace(emit=lambda: calls.append("sync")),
        winId=lambda: 123,
    )
    original = pet_window_module.ctypes
    pet_window_module.ctypes = SimpleNamespace(
        windll=SimpleNamespace(user32=User32()), byref=ctypes.byref
    )
    try:
        pet_window_module.DesktopPet._move_by_drag_delta(host, QPoint(0, 0))
    finally:
        pet_window_module.ctypes = original
    assert calls == [(123, 0, -400, 390, 0, 0, 0x0015), "sync"]
