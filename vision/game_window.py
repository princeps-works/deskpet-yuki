from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

from desktop_pet.vision.capture import find_monitor_index, get_monitor_geometries

# --- Win32 constants -------------------------------------------------------

_GWL_EXSTYLE = -20
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_DWMWA_EXTENDED_FRAME_BOUNDS = 9

_MIN_WINDOW_AREA_RATIO = 0.02
_MAX_WINDOW_AREA_RATIO = 1.5
_MIN_WINDOW_SIDE_PX = 160
_CACHE_TTL_SEC = 2.0
# A visual novel fills most of its screen; browser chrome and tool windows do not.
_FULLSCREEN_AREA_RATIO = 0.80

# Windows that belong to the pet's own tooling rather than to the game. Titles
# are matched case-insensitively as substrings.
_OWN_TITLES = (
    "live2d-py",
    "视觉小说侧栏",
    "桌宠聊天",
    "长期记忆日记",
    "区域选择",
)
# Titles that are very unlikely to be the game being read.
_UNLIKELY_TITLES = (
    "microsoft\u200b edge",
    "microsoft edge",
    "google chrome",
    "mozilla firefox",
    "windows powershell",
    "windows terminal",
    "命令提示符",
    "文件资源管理器",
    "file explorer",
    "program manager",
    "nvidia geforce overlay",
    "windows 输入体验",
    "settings",
    "设置",
    "任务管理器",
    "task manager",
    "visual studio code",
    "pycharm",
    "codex",
    "chatgpt",
)
# Process names that never host the game being read.
_UNLIKELY_PROCESSES = (
    "explorer.exe",
    "applicationframehost.exe",
    "textinputhost.exe",
    "systemsettings.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "codex.exe",
)

_process_name_cache: dict[int, str] = {}


def _process_name(pid: int) -> str:
    """Best-effort process name, cached because enumeration runs per resolve."""
    if pid in _process_name_cache:
        return _process_name_cache[pid]
    name = ""
    try:
        import psutil

        name = str(psutil.Process(pid).name() or "")
    except Exception:
        name = ""
    _process_name_cache[pid] = name
    if len(_process_name_cache) > 512:
        _process_name_cache.clear()
    return name


@dataclass(frozen=True)
class CaptureRect:
    """A physical-pixel rectangle plus the monitor it was clamped to."""

    left: int
    top: int
    width: int
    height: int
    monitor_index: int
    source: str = "window"
    title: str = ""
    uses_full_screen: bool = False
    process_name: str = ""
    hwnd: int = 0

    def as_region(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.width, self.height

    def clamp_to(self, monitor: tuple[int, int, int, int]) -> Optional["CaptureRect"]:
        mon_left, mon_top, mon_w, mon_h = monitor
        right = min(self.left + self.width, mon_left + mon_w)
        bottom = min(self.top + self.height, mon_top + mon_h)
        left = max(self.left, mon_left)
        top = max(self.top, mon_top)
        if right - left < _MIN_WINDOW_SIDE_PX or bottom - top < _MIN_WINDOW_SIDE_PX:
            return None
        return CaptureRect(
            left,
            top,
            right - left,
            bottom - top,
            self.monitor_index,
            source=self.source,
            title=self.title,
            uses_full_screen=self.uses_full_screen,
            process_name=self.process_name,
            hwnd=self.hwnd,
        )

    def inset(self, lr: float, tb: float) -> "CaptureRect":
        dx = int(round(self.width * lr))
        dy = int(round(self.height * tb))
        width = self.width - dx * 2
        height = self.height - dy * 2
        if width < _MIN_WINDOW_SIDE_PX or height < _MIN_WINDOW_SIDE_PX:
            return self
        return CaptureRect(
            self.left + dx,
            self.top + dy,
            width,
            height,
            self.monitor_index,
            source=self.source,
            title=self.title,
            uses_full_screen=self.uses_full_screen,
            process_name=self.process_name,
            hwnd=self.hwnd,
        )

    def expand(self, margin_px: int) -> "CaptureRect":
        if margin_px == 0:
            return self
        width = self.width + margin_px * 2
        height = self.height + margin_px * 2
        if width < _MIN_WINDOW_SIDE_PX or height < _MIN_WINDOW_SIDE_PX:
            return self
        return CaptureRect(
            self.left - margin_px,
            self.top - margin_px,
            width,
            height,
            self.monitor_index,
            source=self.source,
            title=self.title,
            uses_full_screen=self.uses_full_screen,
            process_name=self.process_name,
            hwnd=self.hwnd,
        )

    def text_box(self, top_ratio: float) -> "CaptureRect":
        """The lower part of the rect, used as the visual-novel text box."""
        ratio = max(0.2, min(0.95, float(top_ratio)))
        offset = int(round(self.height * ratio))
        height = self.height - offset
        if height < _MIN_WINDOW_SIDE_PX:
            return self
        return CaptureRect(
            self.left,
            self.top + offset,
            self.width,
            height,
            self.monitor_index,
            source=f"{self.source}:text",
            title=self.title,
            uses_full_screen=self.uses_full_screen,
            process_name=self.process_name,
            hwnd=self.hwnd,
        )


class _WinRect(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class GameWindowResolver:
    """Resolve the topmost *foreign* application window as a capture rectangle.

    The desktop pet owns several always-on-top windows (host overlay, live2d
    renderer, region selector, chat panel, VN settings panel). A plain
    GetForegroundWindow() call would happily return one of them, which would
    then be captured and fed to OCR/vision -- producing "OCR output" that is
    really our own UI text. Every foreign-window filter here therefore starts by
    excluding our own processes and window titles.
    """

    # Pet-owned window titles, used as a second line of defence in case a pid
    # (notably the live2d renderer's) was not registered in time.
    _OWN_TITLES = _OWN_TITLES

    def __init__(
        self,
        *,
        excluded_pids: set[int] | None = None,
        excluded_hwnds: set[int] | None = None,
        monitor_index: int = 0,
        min_area_ratio: float = _MIN_WINDOW_AREA_RATIO,
        cache_ttl_sec: float = _CACHE_TTL_SEC,
        target_title: str = "",
    ) -> None:
        self._excluded_pids = {int(pid) for pid in (excluded_pids or set()) if int(pid) > 0}
        self._excluded_hwnds = {int(h) for h in (excluded_hwnds or set()) if int(h) > 0}
        self._monitor_index = max(0, int(monitor_index))
        self._min_area_ratio = max(0.0, min(0.5, float(min_area_ratio)))
        self._cache_ttl_sec = max(0.0, float(cache_ttl_sec))
        self._target_title = str(target_title or "").strip().casefold()
        self._cache: Optional[tuple[float, Optional[CaptureRect], str]] = None
        self._last_error = ""
        # Window the resolver has settled on, keyed by process + title so it
        # survives the handle being recreated.
        self._sticky: Optional[tuple[int, str]] = None

    @classmethod
    def _is_own_ui_title(cls, title: str) -> bool:
        value = str(title or "").strip().casefold()
        if not value:
            return False
        return any(token.casefold() in value for token in cls._OWN_TITLES)

    @staticmethod
    def _is_unlikely_target(title: str) -> bool:
        value = str(title or "").strip().casefold()
        if not value:
            return False
        return any(token.casefold() in value for token in _UNLIKELY_TITLES)

    @staticmethod
    def _is_unlikely_process(process_name: str) -> bool:
        value = str(process_name or "").strip().casefold()
        if not value:
            return False
        return any(token.casefold() == value for token in _UNLIKELY_PROCESSES)

    def _score(self, item: CaptureRect) -> float:
        """Rank a candidate window by how likely it is to be the game.

        Pure z-order picks whatever happens to be on top -- typically a browser
        showing the documentation the user is reading, or the pet's own tool
        window. Games in visual-novel mode occupy (nearly) their whole screen,
        so area ratio dominates, with curated title/process penalties on top.
        """
        monitor = self._monitor_for(item.monitor_index)
        monitor_area = 1.0
        if monitor is not None:
            monitor_area = float(max(1, monitor[2] * monitor[3]))
        ratio = (item.width * item.height) / monitor_area
        score = min(1.0, ratio / _FULLSCREEN_AREA_RATIO)
        if self._matches_target_title(item.title):
            score += 10.0
        if self._is_unlikely_target(item.title):
            score -= 2.0
        if self._is_unlikely_process(item.process_name):
            score -= 2.0
        if item.uses_full_screen:
            score += 0.5
        return score

    def _matches_target_title(self, title: str) -> bool:
        if not self._target_title:
            return False
        return self._target_title in str(title or "").strip().casefold()

    def set_target_title(self, title: str) -> None:
        value = str(title or "").strip().casefold()
        if value != self._target_title:
            self._target_title = value
            self._sticky = None
            self.invalidate()

    @property
    def sticky_title(self) -> str:
        return str(self._sticky[1]) if self._sticky else ""

    @property
    def target_title(self) -> str:
        return self._target_title

    def _sticky_alive(self, wins: list[CaptureRect]) -> Optional[CaptureRect]:
        """Re-use the previously chosen window while it still exists.

        Z-order alone is the wrong signal: the desktop pet itself, a browser, or
        any tool window can sit above the game, and re-resolving every tick
        makes the capture target flap between windows. Once a window has been
        chosen it is kept until it disappears.
        """
        if self._sticky is None:
            return None
        pid, title = self._sticky
        for item in wins:
            if item.title == title:
                return item
        return None

    # -- configuration ------------------------------------------------------

    def set_excluded_pids(self, pids: set[int]) -> None:
        new = {int(pid) for pid in pids if int(pid) > 0}
        if new != self._excluded_pids:
            self._excluded_pids = new
            self.invalidate()

    def set_excluded_hwnds(self, hwnds: set[int]) -> None:
        new = {int(h) for h in hwnds if int(h) > 0}
        if new != self._excluded_hwnds:
            self._excluded_hwnds = new
            self.invalidate()

    def set_monitor_index(self, monitor_index: int) -> None:
        value = max(0, int(monitor_index))
        if value != self._monitor_index:
            self._monitor_index = value
            self.invalidate()

    def forget_target(self) -> None:
        """Drop the sticky choice so the next resolve re-evaluates z-order."""
        self._sticky = None
        self.invalidate()

    def invalidate(self) -> None:
        self._cache = None

    @property
    def last_error(self) -> str:
        return self._last_error

    # -- resolution ---------------------------------------------------------

    def resolve(self) -> Optional[CaptureRect]:
        """Return the topmost foreign window rect, or None when undeterminable."""
        if not hasattr(ctypes, "windll"):  # pragma: no cover - non-Windows fallback
            self._last_error = "not_windows"
            return None

        now = time.monotonic()
        cached = self._cache
        if cached is not None and (now - cached[0]) < self._cache_ttl_sec:
            self._last_error = cached[2]
            return cached[1]

        rect, error = self._resolve_uncached()
        self._cache = (now, rect, error)
        self._last_error = error
        return rect

    def _resolve_uncached(self) -> tuple[Optional[CaptureRect], str]:
        try:
            user32 = ctypes.windll.user32
        except Exception as exc:  # pragma: no cover
            return None, f"user32_unavailable:{type(exc).__name__}"

        candidates: list[CaptureRect] = []
        area_rejected = 0
        owned_rejected = 0

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        rect_type = _WinRect
        user32.GetWindowTextLengthW.restype = ctypes.c_int

        def _rect_of(hwnd: int) -> Optional[tuple[int, int, int, int]]:
            bounds = rect_type()
            try:
                ok = user32.DwmGetWindowAttribute(
                    wintypes.HWND(hwnd),
                    ctypes.c_uint(_DWMWA_EXTENDED_FRAME_BOUNDS),
                    ctypes.byref(bounds),
                    ctypes.sizeof(bounds),
                )
            except Exception:
                ok = 0
            if not ok:
                if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(bounds)):
                    return None
            width = int(bounds.right - bounds.left)
            height = int(bounds.bottom - bounds.top)
            if width <= 0 or height <= 0:
                return None
            return int(bounds.left), int(bounds.top), width, height

        def _enum_cb(hwnd, _lparam):
            nonlocal area_rejected, owned_rejected
            if not hwnd:
                return True
            hwnd_value = int(hwnd)
            if hwnd_value in self._excluded_hwnds:
                owned_rejected += 1
                return True
            if not user32.IsWindowVisible(wintypes.HWND(hwnd_value)):
                return True
            pid = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(wintypes.HWND(hwnd_value), ctypes.byref(pid))
            # Primary guard: our own UI must never become the capture target.
            # This process hosts the pet window and the VN panel; the live2d
            # renderer is a separate process whose pid must be passed in.
            if int(pid.value) in self._excluded_pids:
                owned_rejected += 1
                return True
            if user32.GetWindow(wintypes.HWND(hwnd_value), ctypes.c_uint(4)):  # GW_OWNER
                return True
            title_length = user32.GetWindowTextLengthW(wintypes.HWND(hwnd_value))
            if title_length <= 0:
                return True
            title = ""
            try:
                buffer = ctypes.create_unicode_buffer(int(title_length) + 2)
                user32.GetWindowTextW(wintypes.HWND(hwnd_value), buffer, len(buffer))
                title = str(buffer.value or "").strip()
            except Exception:
                title = ""
            if self._is_own_ui_title(title):
                owned_rejected += 1
                return True
            try:
                ex_style = int(user32.GetWindowLongW(wintypes.HWND(hwnd_value), _GWL_EXSTYLE))
            except Exception:
                ex_style = 0
            if ex_style & _WS_EX_TOOLWINDOW and not ex_style & _WS_EX_NOACTIVATE:
                # Plain tool windows are palettes/overlays; skip them. Fullscreen
                # exclusive games may also be tool windows, so only skip when the
                # window is also small enough to be a palette.
                geometry = _rect_of(hwnd_value)
                if geometry is None:
                    return True
                if geometry[2] * geometry[3] < 1280 * 720:
                    return True
            geometry = _rect_of(hwnd_value)
            if geometry is None:
                return True
            left, top, width, height = geometry
            monitor_index = 0
            try:
                monitor_index = find_monitor_index(left, top)
            except Exception:
                monitor_index = 0
            if self._monitor_index > 0 and monitor_index != self._monitor_index:
                return True
            monitor = self._monitor_for(monitor_index)
            if monitor is None:
                return True
            monitor_area = max(1, monitor[2] * monitor[3])
            ratio = (width * height) / float(monitor_area)
            if ratio < self._min_area_ratio or ratio > _MAX_WINDOW_AREA_RATIO:
                area_rejected += 1
                return True
            if width < _MIN_WINDOW_SIDE_PX or height < _MIN_WINDOW_SIDE_PX:
                area_rejected += 1
                return True
            candidates.append(
                CaptureRect(
                    left,
                    top,
                    width,
                    height,
                    monitor_index,
                    source="window",
                    title=title,
                    uses_full_screen=bool(ratio >= _FULLSCREEN_AREA_RATIO),
                    process_name=_process_name(int(pid.value)),
                    hwnd=hwnd_value,
                )
            )
            return True

        try:
            user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
        except Exception as exc:
            return None, f"enum_failed:{type(exc).__name__}"

        if os.getenv("VISION_DEBUG_RESOLVE", "").strip().lower() in {"1", "true", "yes", "on"}:
            for item in candidates:
                print(
                    "[RESOLVE]",
                    f"hwnd={item.hwnd} score={self._score(item):.2f}",
                    f"rect={item.width}x{item.height} title={item.title[:40]!r}",
                )

        if not candidates:
            if owned_rejected and not area_rejected:
                return None, "only_own_windows"
            if area_rejected:
                return None, "no_window_passed_size_filter"
            return None, "no_visible_window"

        # 1. An explicitly pinned title always wins.
        if self._target_title:
            pinned = [item for item in candidates if self._matches_target_title(item.title)]
            if pinned:
                pick = pinned[0]
                self._sticky = (0, pick.title)
                monitor = self._monitor_for(pick.monitor_index)
                if monitor is not None:
                    clamped = pick.clamp_to(monitor)
                    if clamped is not None:
                        return clamped, ""
                return pick, ""
            return None, f"target_missing:{self._target_title[:40]}"

        # 2. Keep the window chosen previously while it is still around.
        sticky = self._sticky_alive(candidates)
        if sticky is not None:
            monitor = self._monitor_for(sticky.monitor_index)
            if monitor is not None:
                clamped = sticky.clamp_to(monitor)
                if clamped is not None:
                    return clamped, ""
            return sticky, ""

        # 3. Fresh pick: prefer the window that looks like a game rather than
        #    whatever happens to be on top.
        pick = max(candidates, key=self._score)
        # Refuse a window we are confident is not the game (browser, explorer,
        # terminal...). Capturing it would feed the model an unrelated screen,
        # which is worse than reporting "no window" and letting the caller
        # decide. An explicit pin bypasses this via step 1.
        if self._is_unlikely_target(pick.title) or self._is_unlikely_process(pick.process_name):
            return None, f"only_unlikely_windows:{pick.title[:40]}"
        self._sticky = (0, pick.title)
        monitor = self._monitor_for(pick.monitor_index)
        if os.getenv("VISION_DEBUG_RESOLVE", "").strip().lower() in {"1", "true", "yes", "on"}:
            print(f"[RESOLVE] pick.hwnd={pick.hwnd} monitor={monitor}")
        if monitor is not None:
            clamped = pick.clamp_to(monitor)
            if os.getenv("VISION_DEBUG_RESOLVE", "").strip().lower() in {"1", "true", "yes", "on"}:
                print(
                    "[RESOLVE] clamped="
                    + ("None" if clamped is None else f"hwnd={clamped.hwnd} {clamped.as_region()}")
                )
            if clamped is not None:
                return clamped, ""
        return pick, ""

    @staticmethod
    def _monitor_for(monitor_index: int) -> Optional[tuple[int, int, int, int]]:
        try:
            geometries = get_monitor_geometries()
        except Exception:
            return None
        if not geometries:
            return None
        index = int(monitor_index)
        if 0 <= index < len(geometries):
            return geometries[index]
        return geometries[0]
