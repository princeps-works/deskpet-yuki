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
