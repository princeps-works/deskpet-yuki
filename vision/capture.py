from __future__ import annotations

from typing import Optional, Tuple

from mss import mss
from PIL import Image


def get_monitor_geometry(monitor_index: int = 1) -> Tuple[int, int, int, int]:
    with mss() as sct:
        monitors = sct.monitors
        idx = monitor_index if 1 <= monitor_index < len(monitors) else 1
        monitor = monitors[idx]
        return monitor["left"], monitor["top"], monitor["width"], monitor["height"]


def capture_primary_screen(
    monitor_index: int = 1,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    with mss() as sct:
        monitors = sct.monitors
        # monitors[0] is the virtual full area, real displays start from 1.
        idx = monitor_index if 1 <= monitor_index < len(monitors) else 1
        monitor = monitors[idx]

        if region is None:
            grab_box = monitor
        else:
            rel_left, rel_top, width, height = region
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
