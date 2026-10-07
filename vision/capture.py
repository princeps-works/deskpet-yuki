from __future__ import annotations

from typing import Optional, Tuple

from mss import mss
from PIL import Image


def _resolve_monitor_index(monitor_index: int, monitor_count: int) -> int:
    """MSS index 0 is the complete virtual desktop; real displays start at 1."""
    if monitor_count <= 1:
        return 0
    index = int(monitor_index)
    if index == 0:
        return 0
    if 1 <= index < monitor_count:
        return index
    return 0


def _clamp_region(
    region: Tuple[int, int, int, int],
    monitor_width: int,
    monitor_height: int,
) -> Tuple[int, int, int, int]:
    rel_left, rel_top, width, height = [int(value) for value in region]
    if width <= 0 or height <= 0:
        raise ValueError("capture region width/height must be positive")

    left = max(0, min(rel_left, max(0, monitor_width - 1)))
    top = max(0, min(rel_top, max(0, monitor_height - 1)))
    right = max(left + 1, min(monitor_width, rel_left + width))
    bottom = max(top + 1, min(monitor_height, rel_top + height))
    return left, top, right - left, bottom - top


def get_monitor_geometries() -> tuple[Tuple[int, int, int, int], ...]:
    """Return MSS geometries, including virtual desktop at index 0."""
    with mss() as sct:
        return tuple(
            (int(item["left"]), int(item["top"]), int(item["width"]), int(item["height"]))
            for item in sct.monitors
        )


def find_monitor_index(left: int, top: int) -> int:
    """Match a Qt screen to a real MSS display by its virtual-desktop origin."""
    geometries = get_monitor_geometries()
    if len(geometries) <= 1:
        return 0
    target_left = int(left)
    target_top = int(top)
    exact = [
        index
        for index, geometry in enumerate(geometries[1:], start=1)
        if geometry[0] == target_left and geometry[1] == target_top
    ]
    if exact:
        return exact[0]
    return min(
        range(1, len(geometries)),
        key=lambda index: abs(geometries[index][0] - target_left) + abs(geometries[index][1] - target_top),
    )


def get_monitor_geometry(monitor_index: int = 0) -> Tuple[int, int, int, int]:
    with mss() as sct:
        monitors = sct.monitors
        idx = _resolve_monitor_index(monitor_index, len(monitors))
        monitor = monitors[idx]
        return monitor["left"], monitor["top"], monitor["width"], monitor["height"]


def capture_primary_screen(
    monitor_index: int = 0,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    with mss() as sct:
        monitors = sct.monitors
        # monitors[0] is the complete virtual desktop, real displays start from 1.
        idx = _resolve_monitor_index(monitor_index, len(monitors))
        monitor = monitors[idx]

        if region is None:
            grab_box = monitor
        else:
            rel_left, rel_top, width, height = _clamp_region(
                region,
                int(monitor["width"]),
                int(monitor["height"]),
            )
            left = monitor["left"] + rel_left
            top = monitor["top"] + rel_top
            grab_box = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }

        shot = sct.grab(grab_box)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def capture_absolute_rect(
    left: int,
    top: int,
    width: int,
    height: int,
) -> Optional[Image.Image]:
    """Grab an absolute virtual-desktop rectangle in physical pixels.

    Used by window-follow capture, where coordinates already come from the
    window manager instead of a monitor-relative region.
    """
    x = int(left)
    y = int(top)
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        return None
    try:
        with mss() as sct:
            shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
            return Image.frombytes("RGB", shot.size, shot.rgb)
    except Exception:
        return None


def capture_window_client(hwnd: int) -> Optional[Image.Image]:
    """Capture a window's own content via PrintWindow, ignoring occlusion.

    Screen capture returns whatever is composited on top of the game, so any
    window in front of it (a browser, an editor, the pet itself) silently
    becomes the OCR input -- which looks exactly like "OCR reads the wrong
    text". PrintWindow asks the target window to render itself instead.

    Returns None when the call is unsupported or produces an empty bitmap; the
    caller falls back to screen capture in that case.
    """
    if not hwnd:
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # pragma: no cover
        return None
    if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    PW_RENDERFULLCONTENT = 0x00000002

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

    hwnd = wintypes.HWND(int(hwnd))
    if not user32.IsWindow(hwnd):
        return None

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 16 or height < 16:
        return None

    window_dc = user32.GetDC(hwnd)
    if not window_dc:
        return None
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    if not memory_dc:
        user32.ReleaseDC(hwnd, window_dc)
        return None
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    if not bitmap:
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)
        return None

    try:
        gdi32.SelectObject(memory_dc, bitmap)
        ok = user32.PrintWindow(hwnd, memory_dc, PW_RENDERFULLCONTENT)
        if not ok:
            ok = user32.PrintWindow(hwnd, memory_dc, 0)
        if not ok:
            return None

        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = width
        header.biHeight = -height  # top-down
        header.biPlanes = 1
        header.biBitCount = 32
        header.biCompression = 0  # BI_RGB
        info = BITMAPINFO()
        info.bmiHeader = header

        buffer_size = width * height * 4
        buffer = ctypes.create_string_buffer(buffer_size)
        scanned = gdi32.GetDIBits(
            memory_dc, bitmap, 0, height, buffer, ctypes.byref(info), 0
        )
        if not scanned:
            return None
        image = Image.frombuffer(
            "RGBA", (width, height), buffer, "raw", "BGRA", 0, 1
        ).convert("RGB")
        return image
    except Exception:
        return None
    finally:
        try:
            gdi32.DeleteObject(bitmap)
        except Exception:
            pass
        try:
            gdi32.DeleteDC(memory_dc)
        except Exception:
            pass
        try:
            user32.ReleaseDC(hwnd, window_dc)
        except Exception:
            pass
