from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import tempfile
import traceback
from ctypes import wintypes
from pathlib import Path


class _LineFilterStream:
    def __init__(self, wrapped, blocked_tokens: list[str]):
        self._wrapped = wrapped
        self._blocked = [token.lower() for token in blocked_tokens if token]
        self._buf = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            lowered = line.lower()
            noisy_motion = ("start motion" in lowered) and (("can't" in lowered) or ("cant" in lowered))
            if noisy_motion or any(token in lowered for token in self._blocked):
                continue
            self._wrapped.write(line + "\n")
        return len(s)

    def flush(self):
        if self._buf:
            lowered = self._buf.lower()
            noisy_motion = ("start motion" in lowered) and (("can't" in lowered) or ("cant" in lowered))
            if (not noisy_motion) and (not any(token in lowered for token in self._blocked)):
                self._wrapped.write(self._buf)
            self._buf = ""
        if hasattr(self._wrapped, "flush"):
            self._wrapped.flush()

    def isatty(self):
        if hasattr(self._wrapped, "isatty"):
            return self._wrapped.isatty()
        return False

    @property
    def encoding(self):
        return getattr(self._wrapped, "encoding", "utf-8")


def _install_log_filters() -> None:
    filter_motion_noise = os.getenv("LIVE2D_FILTER_MOTION_NOISE_LOG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    blocked_tokens = ["can't start motion"] if filter_motion_noise else []
    try:
        sys.stdout = _LineFilterStream(sys.stdout, blocked_tokens)
    except Exception:
        pass
    try:
        sys.stderr = _LineFilterStream(sys.stderr, blocked_tokens)
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live2D-py standalone renderer process")
    parser.add_argument("--model", required=True, help="Path to .model3.json")
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument("--vsync", type=int, default=1)
    parser.add_argument("--title", default="Live2D-py")
    parser.add_argument("--borderless", type=int, default=1)
    parser.add_argument("--transparent-key", type=int, default=1)
    parser.add_argument("--self-topmost", type=int, default=0)
    parser.add_argument("--window-drag", type=int, default=0)
    return parser.parse_args()


def _log_line(message: str) -> None:
    print(message)
    try:
        log_path = Path(__file__).resolve().parent.parent / "data" / "live2d_py_poc.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(message + "\n")
    except Exception:
        pass


def _safe_call(obj, method_name: str, *args):
    func = getattr(obj, method_name, None)
    if callable(func):
        try:
            return func(*args)
        except Exception as exc:
            _log_line(f"[LIVE2D-PY] call failed: {method_name}{args} -> {exc}")
            return None
    return None


def _try_call_param_method(model, method_name: str, param_name: str, value: float) -> bool:
    func = getattr(model, method_name, None)
    if not callable(func):
        return False
    candidates = [
        (param_name, float(value)),
        (param_name, float(value), 1.0),
    ]
    for args in candidates:
        try:
            func(*args)
            return True
        except Exception:
            continue
    return False


def _set_model_param(model, param_name: str, value: float) -> bool:
    # Try common direct-set APIs first.
    for method_name in [
        "SetParameterValue",
        "SetParameterValueById",
        "SetParameterFloat",
        "SetParamFloat",
        "SetParamValue",
    ]:
        if _try_call_param_method(model, method_name, param_name, value):
            return True
    return False


def _diag_log(message: str) -> None:
    _log_line("[DIAG][RUNNER] " + message)


def _diag_enabled() -> bool:
    return os.getenv("LIVE2D_INPUT_DIAG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _write_window_state(pid: int, hwnd: int) -> None:
    try:
        state_path = Path(__file__).resolve().parent.parent / "data" / "live2d_py_window_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"pid": int(pid), "hwnd": int(hwnd)}
        state_path.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass


def _read_target_rect() -> tuple[int, int, int, int] | None:
    try:
        state_path = Path(__file__).resolve().parent.parent / "data" / "live2d_py_target_rect.json"
        if not state_path.exists():
            return None
        data = json.loads(state_path.read_text(encoding="utf-8"))
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        w = int(data.get("w", 0))
        h = int(data.get("h", 0))
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h
    except Exception:
        return None


def _read_input_state() -> dict | None:
    candidates = [
        Path(__file__).resolve().parent.parent / "data" / "live2d_py_input_state.json",
        Path.cwd() / "data" / "live2d_py_input_state.json",
        Path.cwd() / "desktop_pet" / "data" / "live2d_py_input_state.json",
    ]
    for state_path in candidates:
        try:
            if not state_path.exists():
                continue
            data = json.loads(state_path.read_text(encoding="utf-8"))
            ts = float(data.get("ts", 0.0))
            if time.time() - ts > 2.0:
                continue
            return {
                "x": int(data.get("x", 0)),
                "y": int(data.get("y", 0)),
                "w": max(1, int(data.get("w", 1))),
                "h": max(1, int(data.get("h", 1))),
                "inside": bool(data.get("inside", False)),
                "left_down": int(data.get("left_down", 0)),
                "right_down": int(data.get("right_down", 0)),
                "follow_enabled": int(data.get("follow_enabled", 1)),
                "scan_busy": int(data.get("scan_busy", 0)),
                "drag_active": int(data.get("drag_active", 0)),
            }
        except Exception:
            continue
    return None


def _prepare_model_json_for_runtime(model_path: Path) -> tuple[Path, dict[str, int], Path | None]:
    """Normalize motion groups and return a runtime-safe model json path.

    Returns (runtime_model_path, interactive_group_sizes, temp_file_path).
    """
    try:
        data = json.loads(model_path.read_text(encoding="utf-8"))
        refs = data.get("FileReferences") if isinstance(data, dict) else None
        motions = refs.get("Motions") if isinstance(refs, dict) else None
        if not isinstance(motions, dict):
            return model_path, {}, None

        changed = False
        empty_group_items = motions.get("")
        if isinstance(empty_group_items, list) and empty_group_items:
            if isinstance(motions.get("Special"), list):
                motions["Special"].extend(empty_group_items)
            else:
                motions["Special"] = list(empty_group_items)
            motions.pop("", None)
            changed = True
            _log_line("[LIVE2D-PY] normalized empty motion group -> Special")

        interactive_groups: dict[str, int] = {}
        for key, value in motions.items():
            if isinstance(value, list) and value and key and key.lower() != "idle":
                interactive_groups[str(key)] = int(len(value))

        if not changed:
            return model_path, interactive_groups, None

        fd, temp_name = tempfile.mkstemp(
            prefix="live2d_runtime_",
            suffix=".model3.json",
            dir=str(model_path.parent),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return temp_path, interactive_groups, temp_path
    except Exception as exc:
        _log_line(f"[LIVE2D-PY] model json normalize failed: {exc}")
        return model_path, {}, None


def run() -> int:
    try:
        _install_log_filters()
        _log_line("[LIVE2D-PY] runner start")
        args = _parse_args()
        model_path = Path(args.model).expanduser()
        if not model_path.exists():
            _log_line(f"[LIVE2D-PY] model file not found: {model_path}")
            return 2

        import pygame
        import live2d.v3 as live2d
        from pygame.locals import DOUBLEBUF, NOFRAME, OPENGL
        from OpenGL.GL import GL_TEXTURE_2D, glEnable

        if hasattr(live2d, "setLogEnable"):
            live2d.setLogEnable(False)
        elif hasattr(live2d, "setLogEnabled"):
            live2d.setLogEnabled(False)
        pygame.init()
        live2d.init()

        screen_size = (max(320, int(args.width)), max(320, int(args.height)))
        pygame.display.set_caption(str(args.title or "Live2D-py"))
        flags = DOUBLEBUF | OPENGL
        if int(args.borderless) != 0:
            flags |= NOFRAME
        pygame.display.set_mode(screen_size, flags, vsync=max(0, int(args.vsync)))

        win_hwnd = None
        win_user32 = None
        win_setpos_flags = 0
        win_nosize_flags = 0
        win_move_flags = 0
        point_type = None
        rect_type = None

        if os.name == "nt" and int(args.borderless) != 0:
            try:
                import ctypes

                GWL_STYLE = -16
                GWL_EXSTYLE = -20
                WS_CAPTION = 0x00C00000
                WS_THICKFRAME = 0x00040000
                WS_MINIMIZEBOX = 0x00020000
                WS_MAXIMIZEBOX = 0x00010000
                WS_SYSMENU = 0x00080000
                WS_EX_LAYERED = 0x00080000
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_NOZORDER = 0x0004
                SWP_FRAMECHANGED = 0x0020
                SWP_NOACTIVATE = 0x0010
                SWP_SHOWWINDOW = 0x0040
                LWA_COLORKEY = 0x00000001
                RGB_BLACK = 0x00000000
                HWND_TOPMOST = -1

                class _Point(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                class _Rect(ctypes.Structure):
                    _fields_ = [
                        ("left", ctypes.c_long),
                        ("top", ctypes.c_long),
                        ("right", ctypes.c_long),
                        ("bottom", ctypes.c_long),
                    ]

                point_type = _Point
                rect_type = _Rect

                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    # Keep hwnd/user32 available even if later style adjustments fail.
                    win_hwnd = int(hwnd)
                    win_user32 = ctypes.windll.user32
                    win_setpos_flags = SWP_NOACTIVATE | SWP_SHOWWINDOW
                    win_nosize_flags = SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW
                    win_move_flags = SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER

                    _write_window_state(os.getpid(), int(hwnd))
                    user32 = win_user32
                    style = user32.GetWindowLongW(hwnd, GWL_STYLE)
                    style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
                    user32.SetWindowLongW(hwnd, GWL_STYLE, style)

                    if int(args.transparent_key) != 0:
                        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                        ex_style |= WS_EX_LAYERED
                        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)
                        user32.SetLayeredWindowAttributes(hwnd, RGB_BLACK, 0, LWA_COLORKEY)

                    user32.SetWindowPos(
                        hwnd,
                        0,
                        0,
                        0,
                        0,
                        0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED,
                    )

                    if int(args.self_topmost) != 0:
                        win_user32.SetWindowPos(
                            win_hwnd,
                            HWND_TOPMOST,
                            0,
                            0,
                            0,
                            0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                        )
            except Exception as exc:
                _log_line(f"[LIVE2D-PY] borderless style adjust failed: {exc}")

        if getattr(live2d, "LIVE2D_VERSION", 3) == 3:
            force_gl_init = os.getenv("LIVE2D_PY_FORCE_GL_INIT", "false").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if force_gl_init:
                try:
                    if hasattr(live2d, "glInit"):
                        live2d.glInit()
                    elif hasattr(live2d, "glewInit"):
                        live2d.glewInit()
                except Exception as exc:
                    _log_line(f"[LIVE2D-PY] GL init failed: {exc}")

        runtime_model_path, interactive_motion_groups, temp_model_path = _prepare_model_json_for_runtime(model_path)
        model = live2d.LAppModel()
        try:
            model.LoadModelJson(str(runtime_model_path))
        except Exception as exc:
            # Fallback to original model json if normalized temp load fails.
            if runtime_model_path != model_path:
                _log_line(f"[LIVE2D-PY] runtime model load failed, fallback to original: {exc}")
                model.LoadModelJson(str(model_path))
                runtime_model_path = model_path
                interactive_motion_groups = {}
            else:
                raise
        _safe_call(model, "Resize", *screen_size)
        _safe_call(model, "SetAutoBlinkEnable", True)
        _safe_call(model, "SetAutoBreathEnable", True)
        if _safe_call(model, "StartRandomMotion") is None:
            _safe_call(model, "StartRandomMotion", "Idle", 1)

        # Prevent repeated motion requests from the same click path from colliding.
        motion_trigger_cooldown_sec = 0.5
        last_motion_trigger_ts = 0.0

        def _motion_started_ok(result) -> bool:
            # Typical failed markers in live2d wrappers are None/False/-1.
            return result not in (None, False, -1)

        def _trigger_interactive_motion(x: int, y: int) -> None:
            nonlocal last_motion_trigger_ts
            now = time.monotonic()
            if now - last_motion_trigger_ts < motion_trigger_cooldown_sec:
                return
            last_motion_trigger_ts = now

            started = False
            if interactive_motion_groups:
                group_names = list(interactive_motion_groups.keys())
                random.shuffle(group_names)
                for group in group_names:
                    count = max(1, int(interactive_motion_groups.get(group, 1)))
                    index = random.randrange(count)

                    # Try deterministic index first, with FORCE priority fallback to NORMAL.
                    for priority in (3, 2):
                        result = _safe_call(model, "StartMotion", group, int(index), int(priority))
                        if _motion_started_ok(result):
                            started = True
                            break
                        result = _safe_call(model, "StartRandomMotion", group, int(priority))
                        if _motion_started_ok(result):
                            started = True
                            break
                    if started:
                        break

            if not started:
                if _safe_call(model, "Touch", x, y) is None:
                    _safe_call(model, "Tap", x, y)

        clock = pygame.time.Clock()
        running = True
        frame_count = 0
        dragging_window = False
        drag_offset_x = 0
        drag_offset_y = 0
        last_target_rect = None
        angle_x = 0.0
        angle_y = 0.0
        body_x = 0.0
        prev_right_down = 0
        pending_angle_x = 0.0
        pending_angle_y = 0.0
        pending_body_x = 0.0
        pending_drag_x = 0
        pending_drag_y = 0
        has_pending_drag = False
        last_diag_ts = 0.0
        diag_interval_sec = 20.0
        diag_enabled = _diag_enabled()
        param_ok_x = False
        param_ok_y = False
        param_ok_body = False
        last_good_input_state = None
        cached_input_state = None
        next_input_read_ts = 0.0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if (
                    int(args.window_drag) != 0
                    and event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and win_hwnd
                    and win_user32
                ):
                    try:
                        pt = point_type()
                        rc = rect_type()
                        if win_user32.GetCursorPos(pt) and win_user32.GetWindowRect(win_hwnd, rc):
                            dragging_window = True
                            drag_offset_x = int(pt.x - rc.left)
                            drag_offset_y = int(pt.y - rc.top)
                    except Exception:
                        dragging_window = False

                if int(args.window_drag) != 0 and event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging_window = False

                if (
                    int(args.window_drag) != 0
                    and event.type == pygame.MOUSEMOTION
                    and dragging_window
                    and win_hwnd
                    and win_user32
                ):
                    try:
                        pt = point_type()
                        if win_user32.GetCursorPos(pt):
                            new_x = int(pt.x - drag_offset_x)
                            new_y = int(pt.y - drag_offset_y)
                            win_user32.SetWindowPos(
                                win_hwnd,
                                -1,
                                new_x,
                                new_y,
                                0,
                                0,
                                win_nosize_flags,
                            )
                    except Exception:
                        pass

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    x, y = pygame.mouse.get_pos()
                    _trigger_interactive_motion(int(x), int(y))

            # Drive gaze follow from global cursor position so it still works when a transparent host overlays the model.
            local_x = None
            local_y = None
            local_w = None
            local_h = None
            now_ts = time.time()
            if now_ts >= next_input_read_ts:
                cached_input_state = _read_input_state()
                scan_busy_flag = bool(cached_input_state and cached_input_state.get("scan_busy", 0))
                # Poll slower during scan_busy to reduce disk IO contention.
                next_input_read_ts = now_ts + (0.14 if scan_busy_flag else 0.08)
            input_state = cached_input_state
            if input_state is not None:
                last_good_input_state = input_state
            elif last_good_input_state is not None:
                input_state = last_good_input_state
            if input_state is not None:
                local_x = input_state["x"]
                local_y = input_state["y"]
                local_w = input_state["w"]
                local_h = input_state["h"]

                # Broadcast right-button click from host side to model tap.
                cur_right_down = input_state["right_down"]
                if cur_right_down == 1 and prev_right_down == 0 and input_state["inside"]:
                    _trigger_interactive_motion(int(input_state["x"]), int(input_state["y"]))
                prev_right_down = cur_right_down
            else:
                prev_right_down = 0

            if local_x is None and win_hwnd and win_user32 and point_type is not None and rect_type is not None:
                try:
                    pt = point_type()
                    rc = rect_type()
                    if win_user32.GetCursorPos(pt) and win_user32.GetWindowRect(win_hwnd, rc):
                        width = max(1, int(rc.right - rc.left))
                        height = max(1, int(rc.bottom - rc.top))
                        local_x = int(pt.x - rc.left)
                        local_y = int(pt.y - rc.top)
                        local_x = max(0, min(width - 1, local_x))
                        local_y = max(0, min(height - 1, local_y))
                        local_w = width
                        local_h = height
                except Exception:
                    pass
            elif local_x is None and os.name == "nt":
                # Fallback path: if hwnd mapping is unavailable, use target rect mapping.
                try:
                    import ctypes

                    target = _read_target_rect()
                    if target is not None:
                        tx, ty, tw, th = target
                        pt = wintypes.POINT()
                        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                            local_x = max(0, min(max(1, tw) - 1, int(pt.x - tx)))
                            local_y = max(0, min(max(1, th) - 1, int(pt.y - ty)))
                            local_w = max(1, int(tw))
                            local_h = max(1, int(th))
                except Exception:
                    pass

            scan_busy_mode = bool(input_state and input_state.get("scan_busy", 0))
            follow_enabled = bool(input_state is None or input_state.get("follow_enabled", 1))
            drag_active = bool(input_state and input_state.get("drag_active", 0))
            if follow_enabled and local_x is not None and local_y is not None and local_w and local_h:
                nx = (float(local_x) / float(local_w)) * 2.0 - 1.0
                ny = 1.0 - (float(local_y) / float(local_h)) * 2.0
                nx = max(-1.0, min(1.0, nx))
                ny = max(-1.0, min(1.0, ny))

                pending_drag_x = int(local_x)
                pending_drag_y = int(local_y)
                has_pending_drag = True

                target_angle_x = nx * 30.0
                target_angle_y = ny * 30.0
                target_body_x = nx * 10.0

                smooth = 0.08 if scan_busy_mode else 0.18
                body_smooth = 0.06 if scan_busy_mode else 0.12
                angle_x += (target_angle_x - angle_x) * smooth
                angle_y += (target_angle_y - angle_y) * smooth
                body_x += (target_body_x - body_x) * body_smooth
                pending_angle_x = angle_x
                pending_angle_y = angle_y
                pending_body_x = body_x

            live2d.clearBuffer()
            # Some v3 environments require explicit texture state enable before drawing.
            glEnable(GL_TEXTURE_2D)
            model.Update()
            if follow_enabled and has_pending_drag:
                _safe_call(model, "Drag", pending_drag_x, pending_drag_y)
            # Apply gaze parameters after Update so motions don't overwrite the values.
            if follow_enabled:
                param_ok_x = _set_model_param(model, "ParamAngleX", pending_angle_x)
                param_ok_y = _set_model_param(model, "ParamAngleY", pending_angle_y)
                param_ok_body = _set_model_param(model, "ParamBodyAngleX", pending_body_x)
            model.Draw()

            # Geometry synchronization is host-driven in main process to avoid cross-process position races.

            pygame.display.flip()
            clock.tick(40 if scan_busy_mode else 60)
            frame_count += 1
            if frame_count % 30 == 0 and int(args.self_topmost) != 0 and win_hwnd and win_user32:
                try:
                    win_user32.SetWindowPos(win_hwnd, -1, 0, 0, 0, 0, win_setpos_flags | 0x0002 | 0x0001)
                except Exception:
                    pass

            now_ts = time.time()
            if diag_enabled and (now_ts - last_diag_ts >= diag_interval_sec):
                if not follow_enabled:
                    last_diag_ts = now_ts
                    continue
                if input_state is None:
                    _diag_log("input_state=none")
                else:
                    _diag_log(
                        " ".join(
                            [
                                f"inside={input_state['inside']}",
                                f"xy=({input_state['x']},{input_state['y']})",
                                f"wh=({input_state['w']},{input_state['h']})",
                                f"drag={has_pending_drag}",
                                f"dragLock={int(drag_active)}",
                                f"follow={follow_enabled}",
                                f"paramX={param_ok_x}",
                                f"paramY={param_ok_y}",
                                f"paramBody={param_ok_body}",
                            ]
                        )
                    )
                last_diag_ts = now_ts

        try:
            live2d.dispose()
        except Exception:
            pass
        try:
            if "temp_model_path" in locals() and temp_model_path is not None:
                temp_model_path.unlink(missing_ok=True)
        except Exception:
            pass
        pygame.quit()
        return 0
    except Exception:
        _log_line("[LIVE2D-PY] fatal exception:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
