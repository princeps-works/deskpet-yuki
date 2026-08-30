from __future__ import annotations

import os
import subprocess
import sys
import re
import random
import ctypes
import json
import time
import queue
import threading
import multiprocessing as mp
from ctypes import wintypes
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from datetime import timedelta
from pathlib import Path


_SUBPROC_SCAN_CACHE: dict[str, object] = {
    "scan_target": None,
    "fingerprint": None,
    "ocr_hash": "",
    "ocr_text": "",
    "ocr_confidence": 0.0,
    "screen_context": "",
    "scene_summary": "",
    "scene_should_comment": False,
    "mode": "none",
    "mm_reason": "off",
    "mm_elapsed_ms": 0,
    "mm_summary_len": 0,
    "vision_route": "none",
}

_SUBPROC_MM_RUNTIME: dict[str, object] = {
    "inited": False,
    "enabled": False,
    "client": None,
    "settings": None,
    "timeout_sec": 5.0,
    "max_edge": 1280,
    "mm_fail_streak": 0,
    "mm_cooldown_until": None,
    "mm_last_request_at": None,
}

# 兼容两种启动方式：
# 1) python -m desktop_pet.main
# 2) python desktop_pet/main.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from desktop_pet.config.settings import load_settings
from desktop_pet.vision.ocr import warmup_ocr_engine
from desktop_pet.config.prompts import SYSTEM_VOICEVOX_TRANSLATE_PROMPT


def _merge_chromium_flags(existing: str, extras: list[str]) -> str:
    tokens = [t.strip() for t in existing.split(" ") if t.strip()]
    present = set(tokens)
    for item in extras:
        if item not in present:
            tokens.append(item)
            present.add(item)
    return " ".join(tokens)


def configure_webengine_render_mode(mode: str) -> None:
    selected = (mode or "gpu").strip().lower()
    existing_flags = os.getenv("QTWEBENGINE_CHROMIUM_FLAGS", "")

    if selected == "software":
        flags = _merge_chromium_flags(
            existing_flags,
            ["--disable-gpu", "--disable-gpu-compositing", "--disable-software-rasterizer"],
        )
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
        os.environ["QT_OPENGL"] = "software"
        print("[STARTUP] WebEngine render mode: software")
        return

    if selected == "auto":
        print("[STARTUP] WebEngine render mode: auto")
        return

    flags = _merge_chromium_flags(
        existing_flags,
        [
            "--ignore-gpu-blocklist",
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--use-angle=d3d11",
        ],
    )
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
    os.environ.setdefault("QT_OPENGL", "desktop")
    print("[STARTUP] WebEngine render mode: gpu")


def _frame_fingerprint(image, *, text_sensitive: bool = False) -> bytes:
    # Downsample grayscale bytes as a lightweight scene fingerprint.
    size = (96, 36) if text_sensitive else (32, 18)
    gray = image.convert("L").resize(size)
    return gray.tobytes()


def _fingerprint_diff_ratio(a: bytes, b: bytes) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    changed = sum(1 for x, y in zip(a, b) if abs(x - y) > 10)
    return changed / len(a)


def _resize_for_ocr(image, max_edge: int):
    edge = int(max_edge)
    if edge <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= edge:
        return image
    ratio = float(edge) / float(longest)
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    return image.resize((new_w, new_h))


def _crop_visual_novel_scan_image(image, *, enabled: bool, has_manual_region: bool):
    if not enabled or has_manual_region:
        return image
    width, height = image.size
    top = max(0, min(height - 1, int(round(height * 0.58))))
    return image.crop((0, top, width, height))


def _compose_screen_context(vision_summary: str, ocr_text: str) -> tuple[str, str]:
    """Keep source-specific prefix while routing into one downstream processing path."""
    vision_clean = " ".join(str(vision_summary or "").split())
    ocr_clean = " ".join(str(ocr_text or "").split())

    if not vision_clean and not ocr_clean:
        return "", "none"

    parts: list[str] = []
    if ocr_clean and vision_clean:
        mode = "vision+ocr"
        parts.append(f"视觉摘要: {vision_clean}")
        parts.append(f"OCR文本: {ocr_clean}")
    elif ocr_clean:
        mode = "ocr"
        parts.append(f"OCR文本: {ocr_clean}")
    else:
        mode = "vision"
        parts.append(f"视觉摘要: {vision_clean}")
    return "\n".join(parts).strip(), mode


def _request_multimodal_summary(
    llm_client,
    image,
    settings_obj,
    runtime_state: dict,
    log_fn=None,
    *,
    ocr_context: str = "",
    route: str = "vision_only",
    respect_min_interval: bool = True,
    vision_enabled: bool | None = None,
    focus_query: str = "",
) -> tuple[str, str, int, int]:
    feature_enabled = (
        bool(getattr(settings_obj, "enable_mm_screen_comment", False))
        if vision_enabled is None
        else bool(vision_enabled)
    )
    mm_enabled = bool(getattr(settings_obj, "enable_multimodal_vision", False) and feature_enabled)
    if (not mm_enabled) or llm_client is None:
        return "", "off", 0, 0

    def _log(stage: str, detail: str) -> None:
        if callable(log_fn):
            try:
                log_fn(stage, detail)
            except Exception:
                pass

    def _mark_success() -> None:
        runtime_state["mm_fail_streak"] = 0
        runtime_state["mm_cooldown_until"] = None

    def _mark_failure(reason: str) -> None:
        streak = int(runtime_state.get("mm_fail_streak", 0)) + 1
        runtime_state["mm_fail_streak"] = streak
        threshold = max(1, int(getattr(settings_obj, "mm_failure_threshold", 1)))
        if streak >= threshold:
            cooldown_sec = max(10, int(getattr(settings_obj, "mm_cooldown_sec", 30)))
            runtime_state["mm_cooldown_until"] = datetime.now() + timedelta(seconds=cooldown_sec)
            _log("mm_cooldown", f"reason={reason}, streak={streak}, cooldown={cooldown_sec}s")
        else:
            _log("mm_fail", f"reason={reason}, streak={streak}/{threshold}")

    cooldown_until = runtime_state.get("mm_cooldown_until")
    now = datetime.now()
    if isinstance(cooldown_until, datetime) and now < cooldown_until:
        remain = int((cooldown_until - now).total_seconds())
        _log("mm_skip", f"cooldown_remain={max(1, remain)}s")
        return "", "cooldown", 0, 0

    if respect_min_interval:
        min_interval_sec = max(0, int(getattr(settings_obj, "mm_auto_min_interval_sec", 0)))
        last_request_at = runtime_state.get("mm_last_request_at")
        if min_interval_sec > 0 and isinstance(last_request_at, datetime):
            elapsed_sec = (now - last_request_at).total_seconds()
            if elapsed_sec < min_interval_sec:
                remain = int(round(min_interval_sec - elapsed_sec))
                _log("mm_skip", f"rate_limit_remain={max(1, remain)}s")
                return "", "rate_limited", 0, 0

    runtime_state["mm_last_request_at"] = now

    try:
        if bool(getattr(settings_obj, "enable_mm_compat_mode", True)):
            from desktop_pet.vision.multimodal import describe_screen_image_compat

            vision_result = describe_screen_image_compat(
                llm_client,
                image,
                timeout_sec=float(getattr(settings_obj, "mm_timeout_sec", 5.0)),
                max_edge=int(getattr(settings_obj, "mm_image_max_edge", 1280)),
                ocr_context=ocr_context,
                route=route,
                focus_query=focus_query,
            )
            summary = str(vision_result.summary or "").strip()
            reason = str(vision_result.reason)
            elapsed_ms = int(vision_result.elapsed_ms)
        else:
            from desktop_pet.vision.multimodal import describe_screen_image

            visual_summary = describe_screen_image(
                llm_client,
                image,
                ocr_context=ocr_context,
                route=route,
                focus_query=focus_query,
            )
            summary = str(visual_summary or "").strip()
            reason = "ok" if summary else "empty"
            elapsed_ms = 0
    except Exception:
        _mark_failure("error")
        _log("mm_result", "reason=error, elapsed_ms=0, len=0")
        return "", "error", 0, 0

    summary_len = len(summary)
    _log("mm_result", f"reason={reason}, elapsed_ms={elapsed_ms}, len={summary_len}")
    if summary:
        _mark_success()
        return summary, reason, elapsed_ms, summary_len

    _mark_failure(reason)
    return "", reason, elapsed_ms, 0


def _build_ocr_first_context(
    llm_client,
    image,
    ocr_result,
    settings_obj,
    runtime_state: dict,
    log_fn=None,
    *,
    respect_vision_interval: bool,
    vision_enabled: bool | None = None,
    focus_query: str = "",
) -> dict:
    from desktop_pet.vision.router import VisionRouteDecision, decide_vision_route

    ocr_text = str(getattr(ocr_result, "text", "") or "").strip()
    if bool(getattr(settings_obj, "enable_ocr_first_routing", True)):
        decision = decide_vision_route(
            ocr_result,
            ocr_only_min_chars=int(getattr(settings_obj, "ocr_only_min_chars", 120)),
            ocr_only_min_confidence=float(getattr(settings_obj, "ocr_only_min_confidence", 0.75)),
            hybrid_min_chars=int(getattr(settings_obj, "ocr_hybrid_min_chars", 30)),
        )
    else:
        decision = VisionRouteDecision("vision_only", "ocr_first_disabled")

    if callable(log_fn):
        try:
            log_fn(
                "vision_route",
                (
                    f"route={decision.route}, reason={decision.reason}, "
                    f"ocr_chars={int(getattr(ocr_result, 'char_count', 0))}, "
                    f"ocr_conf={float(getattr(ocr_result, 'average_confidence', 0.0)):.2f}"
                ),
            )
        except Exception:
            pass

    vision_summary = ""
    mm_reason = "ocr_only"
    mm_elapsed_ms = 0
    mm_summary_len = 0
    if decision.route != "ocr_only":
        ocr_context = ""
        if decision.route == "vision+ocr":
            ocr_context = ocr_result.to_prompt_text(
                max_chars=int(getattr(settings_obj, "ocr_context_max_chars", 1200))
            )
        vision_summary, mm_reason, mm_elapsed_ms, mm_summary_len = _request_multimodal_summary(
            llm_client,
            image,
            settings_obj,
            runtime_state,
            log_fn,
            ocr_context=ocr_context,
            route=decision.route,
            respect_min_interval=respect_vision_interval,
            vision_enabled=vision_enabled,
            focus_query=focus_query,
        )

    screen_context, mode = _compose_screen_context(vision_summary, ocr_text)
    return {
        "screen_context": screen_context,
        "mode": mode,
        "vision_route": decision.route,
        "vision_route_reason": decision.reason,
        "mm_reason": mm_reason,
        "mm_elapsed_ms": mm_elapsed_ms,
        "mm_summary_len": mm_summary_len,
        "ocr_hash": str(getattr(ocr_result, "text_hash", "") or ""),
        "ocr_text": ocr_text,
        "ocr_chars": int(getattr(ocr_result, "char_count", 0)),
        "ocr_confidence": float(getattr(ocr_result, "average_confidence", 0.0)),
    }


def run_scan_pipeline_subprocess(
    scan_monitor_index: int,
    scan_region: tuple[int, int, int, int] | None,
    ocr_cpu_threads: int = 0,
    ocr_cpu_affinity_count: int = 0,
    ocr_max_edge: int = 0,
    visual_novel_mode: bool = False,
) -> dict:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        if int(ocr_cpu_affinity_count) > 0 and hasattr(proc, "cpu_affinity"):
            current = list(proc.cpu_affinity())
            if current:
                proc.cpu_affinity(current[: int(ocr_cpu_affinity_count)])
    except Exception:
        pass

    if int(ocr_cpu_threads) > 0:
        thread_value = str(int(ocr_cpu_threads))
        os.environ["OMP_NUM_THREADS"] = thread_value
        os.environ["MKL_NUM_THREADS"] = thread_value
        os.environ["OPENBLAS_NUM_THREADS"] = thread_value
        os.environ["NUMEXPR_NUM_THREADS"] = thread_value
        os.environ["ORT_NUM_THREADS"] = thread_value

    from desktop_pet.vision.capture import capture_primary_screen
    from desktop_pet.vision.ocr import extract_ocr_result, get_ocr_runtime_status
    from desktop_pet.vision.scene_analyzer import analyze_scene
    from desktop_pet.llm.client import LLMClient

    if not bool(_SUBPROC_MM_RUNTIME.get("inited", False)):
        try:
            subproc_settings = load_settings(Path(__file__).resolve().parent)
            mm_enabled = bool(subproc_settings.enable_multimodal_vision and subproc_settings.enable_mm_screen_comment)
            _SUBPROC_MM_RUNTIME["enabled"] = mm_enabled
            _SUBPROC_MM_RUNTIME["timeout_sec"] = float(subproc_settings.mm_timeout_sec)
            _SUBPROC_MM_RUNTIME["max_edge"] = int(subproc_settings.mm_image_max_edge)
            _SUBPROC_MM_RUNTIME["client"] = LLMClient(subproc_settings) if mm_enabled else None
            _SUBPROC_MM_RUNTIME["settings"] = subproc_settings
        except Exception:
            _SUBPROC_MM_RUNTIME["enabled"] = False
            _SUBPROC_MM_RUNTIME["client"] = None
            _SUBPROC_MM_RUNTIME["settings"] = None
        finally:
            _SUBPROC_MM_RUNTIME["inited"] = True

    image = capture_primary_screen(monitor_index=scan_monitor_index, region=scan_region)
    image = _crop_visual_novel_scan_image(
        image,
        enabled=bool(visual_novel_mode),
        has_manual_region=scan_region is not None,
    )
    scan_target = (
        int(scan_monitor_index),
        tuple(scan_region) if scan_region is not None else None,
        bool(visual_novel_mode),
    )
    if _SUBPROC_SCAN_CACHE.get("scan_target") != scan_target:
        _SUBPROC_SCAN_CACHE["scan_target"] = scan_target
        _SUBPROC_SCAN_CACHE["fingerprint"] = None
        _SUBPROC_SCAN_CACHE["ocr_hash"] = ""
        _SUBPROC_SCAN_CACHE["ocr_text"] = ""
        _SUBPROC_SCAN_CACHE["ocr_confidence"] = 0.0
        _SUBPROC_SCAN_CACHE["screen_context"] = ""
        _SUBPROC_SCAN_CACHE["scene_summary"] = ""
        _SUBPROC_SCAN_CACHE["vision_route"] = "none"
    fingerprint = _frame_fingerprint(image, text_sensitive=bool(visual_novel_mode))
    prev_fingerprint = _SUBPROC_SCAN_CACHE.get("fingerprint")
    if isinstance(prev_fingerprint, bytes):
        diff_ratio = _fingerprint_diff_ratio(fingerprint, prev_fingerprint)
        diff_threshold = 0.012 if visual_novel_mode else 0.04
        if diff_ratio < diff_threshold and str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")).strip():
            return {
                "ocr_ok": True,
                "ocr_info": "cache_reuse",
                "mode": str(_SUBPROC_SCAN_CACHE.get("mode", "none")),
                "screen_context": str(_SUBPROC_SCAN_CACHE.get("screen_context", "")),
                "ocr_text": str(_SUBPROC_SCAN_CACHE.get("ocr_text", "")),
                "ocr_confidence": float(_SUBPROC_SCAN_CACHE.get("ocr_confidence", 0.0) or 0.0),
                "scene_summary": str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")),
                "scene_should_comment": bool(_SUBPROC_SCAN_CACHE.get("scene_should_comment", False)),
                "mm_reason": str(_SUBPROC_SCAN_CACHE.get("mm_reason", "off")),
                "mm_elapsed_ms": int(_SUBPROC_SCAN_CACHE.get("mm_elapsed_ms", 0) or 0),
                "mm_summary_len": int(_SUBPROC_SCAN_CACHE.get("mm_summary_len", 0) or 0),
                "vision_route": str(_SUBPROC_SCAN_CACHE.get("vision_route", "cache")),
                "reused_cache": True,
            }

    ocr_ok, ocr_info = get_ocr_runtime_status()
    ocr_image = _resize_for_ocr(image, int(ocr_max_edge))
    ocr_result = extract_ocr_result(ocr_image)
    previous_ocr_hash = str(_SUBPROC_SCAN_CACHE.get("ocr_hash", "") or "")
    if (
        ocr_result.text_hash
        and ocr_result.text_hash == previous_ocr_hash
        and str(_SUBPROC_SCAN_CACHE.get("vision_route", "")) == "ocr_only"
        and str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")).strip()
    ):
        _SUBPROC_SCAN_CACHE["fingerprint"] = fingerprint
        return {
            "ocr_ok": ocr_ok,
            "ocr_info": "ocr_hash_reuse",
            "mode": str(_SUBPROC_SCAN_CACHE.get("mode", "none")),
            "screen_context": str(_SUBPROC_SCAN_CACHE.get("screen_context", "")),
            "ocr_text": str(_SUBPROC_SCAN_CACHE.get("ocr_text", "")),
            "ocr_confidence": float(_SUBPROC_SCAN_CACHE.get("ocr_confidence", 0.0) or 0.0),
            "scene_summary": str(_SUBPROC_SCAN_CACHE.get("scene_summary", "")),
            "scene_should_comment": bool(_SUBPROC_SCAN_CACHE.get("scene_should_comment", False)),
            "mm_reason": "ocr_hash_cache",
            "mm_elapsed_ms": 0,
            "mm_summary_len": int(_SUBPROC_SCAN_CACHE.get("mm_summary_len", 0) or 0),
            "vision_route": "cache",
            "reused_cache": True,
        }

    mm_client = _SUBPROC_MM_RUNTIME.get("client")
    subproc_settings = _SUBPROC_MM_RUNTIME.get("settings")
    if subproc_settings is not None:
        pipeline = _build_ocr_first_context(
            mm_client,
            image,
            ocr_result,
            subproc_settings,
            _SUBPROC_MM_RUNTIME,
            None,
            respect_vision_interval=True,
            vision_enabled=False if visual_novel_mode else None,
        )
    else:
        screen_context, mode = _compose_screen_context("", ocr_result.text)
        pipeline = {
            "screen_context": screen_context,
            "mode": mode,
            "vision_route": "ocr_only",
            "mm_reason": "off",
            "mm_elapsed_ms": 0,
            "mm_summary_len": 0,
            "ocr_hash": ocr_result.text_hash,
            "ocr_text": ocr_result.text,
            "ocr_confidence": ocr_result.average_confidence,
        }

    screen_context = str(pipeline["screen_context"])
    mode = str(pipeline["mode"])
    scene = analyze_scene(screen_context)

    _SUBPROC_SCAN_CACHE["fingerprint"] = fingerprint
    _SUBPROC_SCAN_CACHE["ocr_hash"] = str(pipeline.get("ocr_hash", ""))
    _SUBPROC_SCAN_CACHE["ocr_text"] = str(pipeline.get("ocr_text", ""))
    _SUBPROC_SCAN_CACHE["ocr_confidence"] = float(pipeline.get("ocr_confidence", 0.0) or 0.0)
    _SUBPROC_SCAN_CACHE["screen_context"] = screen_context
    _SUBPROC_SCAN_CACHE["scene_summary"] = scene.summary
    _SUBPROC_SCAN_CACHE["scene_should_comment"] = scene.should_comment
    _SUBPROC_SCAN_CACHE["mode"] = mode
    _SUBPROC_SCAN_CACHE["mm_reason"] = str(pipeline.get("mm_reason", "off"))
    _SUBPROC_SCAN_CACHE["mm_elapsed_ms"] = int(pipeline.get("mm_elapsed_ms", 0) or 0)
    _SUBPROC_SCAN_CACHE["mm_summary_len"] = int(pipeline.get("mm_summary_len", 0) or 0)
    _SUBPROC_SCAN_CACHE["vision_route"] = str(pipeline.get("vision_route", "none"))

    return {
        "ocr_ok": ocr_ok,
        "ocr_info": ocr_info,
        "mode": mode,
        "screen_context": screen_context,
        "ocr_text": str(pipeline.get("ocr_text", "")),
        "ocr_confidence": float(pipeline.get("ocr_confidence", 0.0) or 0.0),
        "scene_summary": scene.summary,
        "scene_should_comment": scene.should_comment,
        "mm_reason": str(pipeline.get("mm_reason", "off")),
        "mm_elapsed_ms": int(pipeline.get("mm_elapsed_ms", 0) or 0),
        "mm_summary_len": int(pipeline.get("mm_summary_len", 0) or 0),
        "vision_route": str(pipeline.get("vision_route", "none")),
        "reused_cache": False,
    }


def run_scan_pipeline_worker_loop(task_queue, result_queue) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id = int(task.get("task_id", 0))
        monitor_index = int(task.get("scan_monitor_index", 0))
        scan_region = task.get("scan_region")
        ocr_cpu_threads = int(task.get("ocr_cpu_threads", 0))
        ocr_cpu_affinity_count = int(task.get("ocr_cpu_affinity_count", 0))
        ocr_max_edge = int(task.get("ocr_max_edge", 0))
        visual_novel_mode = bool(task.get("visual_novel_mode", False))
        started_ts = time.time()
        try:
            result = run_scan_pipeline_subprocess(
                monitor_index,
                scan_region,
                ocr_cpu_threads=ocr_cpu_threads,
                ocr_cpu_affinity_count=ocr_cpu_affinity_count,
                ocr_max_edge=ocr_max_edge,
                visual_novel_mode=visual_novel_mode,
            )
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": True,
                    "result": result,
                    "duration_sec": max(0.0, time.time() - started_ts),
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "task_id": task_id,
                    "ok": False,
                    "error": str(exc),
                    "duration_sec": max(0.0, time.time() - started_ts),
                }
            )


def main() -> int:
    # 先预热 OCR，避免 Qt 相关原生库先加载导致 onnxruntime DLL 冲突。
    ocr_ok, ocr_info = warmup_ocr_engine()
    print(f"[STARTUP] OCR status: {'ok' if ocr_ok else 'fail'} | {ocr_info}")

    base_dir = Path(__file__).resolve().parent
    settings = load_settings(base_dir)
    configure_webengine_render_mode(settings.webengine_gpu_mode)

    live2d_py_process = None
    live2d_py_retry_used = False
    filter_motion_noise_log = os.getenv("LIVE2D_FILTER_MOTION_NOISE_LOG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    filter_live2d_startup_verbose = os.getenv("LIVE2D_FILTER_STARTUP_VERBOSE_LOG", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _pipe_live2d_stream(stream, *, name: str) -> None:
        if stream is None:
            return
        try:
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                lowered = line.lower()
                noisy_motion = ("start motion" in lowered) and (("can't" in lowered) or ("cant" in lowered))
                if filter_motion_noise_log and noisy_motion:
                    continue
                if filter_live2d_startup_verbose and (
                    "create buffer:" in lowered
                    or "delete buffer:" in lowered
                    or "load motion:" in lowered
                ):
                    continue
                print(f"[LIVE2D-PY/{name}] {line}")
        except Exception as exc:
            print(f"[STARTUP] Live2D-py log pipe {name} failed: {exc}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def start_live2d_py_process(*, force_gl_init: bool) -> subprocess.Popen | None:
        child_env = os.environ.copy()
        if force_gl_init:
            child_env["LIVE2D_PY_FORCE_GL_INIT"] = "true"
        else:
            child_env.pop("LIVE2D_PY_FORCE_GL_INIT", None)

        cmd = [
            sys.executable,
            "-m",
            "desktop_pet.live2d.live2d_py_runner",
            "--model",
            settings.live2d_model_json,
            "--width",
            str(settings.live2d_py_window_width),
            "--height",
            str(settings.live2d_py_window_height),
            "--title",
            "Live2D-py",
            "--borderless",
            "1",
            "--self-topmost",
            "0",
            "--window-drag",
            "0",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(base_dir.parent),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if proc.stdout is not None:
            threading.Thread(
                target=_pipe_live2d_stream,
                args=(proc.stdout,),
                kwargs={"name": "STDOUT"},
                daemon=True,
            ).start()
        if proc.stderr is not None:
            threading.Thread(
                target=_pipe_live2d_stream,
                args=(proc.stderr,),
                kwargs={"name": "STDERR"},
                daemon=True,
            ).start()
        return proc

    if settings.enable_live2d_py:
        try:
            live2d_py_process = start_live2d_py_process(force_gl_init=False)
            # live2d-py mode keeps WebEngine path disabled to avoid dual-render contention.
            settings.enable_live2d = False
            print(f"[STARTUP] Live2D-py process started (pid={live2d_py_process.pid})")
        except Exception as exc:
            print(f"[STARTUP] Live2D-py start failed: {exc}")

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QCursor, QGuiApplication
    from PyQt6.QtWidgets import QApplication, QInputDialog, QMessageBox

    from desktop_pet.core.scheduler import ScanScheduler
    from desktop_pet.audio.speech import SpeechService
    from desktop_pet.llm.comment_engine import CommentEngine
    from desktop_pet.llm.client import LLMClient
    from desktop_pet.llm.dialog_manager import DialogManager
    from desktop_pet.llm.semantic_attention import SemanticAttentionRouter
    from desktop_pet.llm.visual_novel import VisualNovelStoryLibrary, VisualNovelTracker
    from desktop_pet.policy.comment_policy import can_emit_comment
    from desktop_pet.ui.chat_panel import ChatPanel
    from desktop_pet.ui.pet_window import DesktopPet
    from desktop_pet.ui.region_selector import RegionSelectOverlay
    from desktop_pet.vision.capture import capture_primary_screen, find_monitor_index, get_monitor_geometry
    from desktop_pet.vision.multimodal import describe_screen_image, describe_screen_image_compat
    from desktop_pet.vision.ocr import extract_ocr_result, get_ocr_runtime_status
    from desktop_pet.vision.scene_analyzer import analyze_scene

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    def maybe_retry_live2d_py() -> None:
        nonlocal live2d_py_process, live2d_py_retry_used
        if live2d_py_process is None or live2d_py_retry_used:
            return
        if live2d_py_process.poll() is None:
            return

        exit_code = live2d_py_process.poll()
        print(f"[STARTUP] Live2D-py exited early with code={exit_code}, retry with forced gl init")
        try:
            live2d_py_process = start_live2d_py_process(force_gl_init=True)
            live2d_py_retry_used = True
            print(f"[STARTUP] Live2D-py retry started (pid={live2d_py_process.pid})")
        except Exception as exc:
            print(f"[STARTUP] Live2D-py retry failed: {exc}")

    if settings.enable_live2d_py:
        QTimer.singleShot(2500, maybe_retry_live2d_py)

    print(
        "[STARTUP] Scan target:",
        f"monitor={settings.scan_monitor_index if settings.scan_monitor_index else 'virtual-desktop'}",
        f"region={settings.scan_region if settings.scan_region else 'full-target'}",
    )
    if settings.enable_visual_novel_mode:
        roi_mode = "manual-region" if settings.scan_region else "bottom-42-percent"
        print(f"[STARTUP] Visual novel mode: enabled | roi={roi_mode}")

    llm_client = LLMClient(settings)
    dialog_memory_path = base_dir / "data" / "chat_long_memory.json"
    semantic_attention = SemanticAttentionRouter(
        enabled=settings.enable_semantic_attention,
        model_path=Path(settings.semantic_attention_model_path),
        top_k=settings.semantic_attention_top_k,
        min_score=settings.semantic_attention_min_score,
        max_length=settings.semantic_attention_max_length,
        cache_size=settings.semantic_attention_cache_size,
        cpu_threads=settings.semantic_attention_cpu_threads,
    )
    visual_novel_runtime = {"enabled": bool(settings.enable_visual_novel_mode)}
    visual_novel_library = VisualNovelStoryLibrary(
        base_dir / "data" / "visual_novel_stories",
        legacy_path=base_dir / "data" / "visual_novel_story.json",
    )
    visual_novel_tracker = VisualNovelTracker(
        visual_novel_library.active_path,
        similarity_fn=semantic_attention.similarity,
        similarities_fn=semantic_attention.similarities,
        min_context_similarity=settings.visual_novel_min_context_similarity,
    )
    print(f"[STARTUP] Visual novel story cache: {visual_novel_library.active_name}")

    def provide_visual_novel_planner_context() -> str:
        if not visual_novel_runtime["enabled"]:
            return ""
        return visual_novel_tracker.build_planner_hint(max_chars=600)

    dialog = DialogManager(
        llm_client,
        memory_path=dialog_memory_path,
        tutor_enabled=settings.enable_tutor_persona,
        semantic_attention=semantic_attention,
        web_soft_deadline_sec=settings.web_search_soft_deadline_sec,
        web_hard_deadline_sec=settings.web_search_hard_deadline_sec,
        web_circuit_failure_threshold=settings.web_search_circuit_failure_threshold,
        web_circuit_cooldown_sec=settings.web_search_circuit_cooldown_sec,
        web_max_results=settings.web_search_max_results,
        web_context_max_chars=settings.web_search_context_max_chars,
        long_memory_limit=settings.long_memory_limit,
        long_memory_context_window=settings.long_memory_context_window,
        visual_novel_context_provider=visual_novel_tracker.retrieve_for_query,
        visual_novel_planner_context_provider=provide_visual_novel_planner_context,
    )
    if settings.enable_semantic_attention:
        def _warmup_semantic_attention() -> None:
            ok = semantic_attention.warmup()
            print(
                "[STARTUP] Semantic attention:",
                "ready" if ok else "fallback",
                f"| {semantic_attention.status}",
            )

        threading.Thread(
            target=_warmup_semantic_attention,
            name="semantic-attention-warmup",
            daemon=True,
        ).start()
    comment_engine = CommentEngine(
        llm_client,
        tutor_enabled=settings.enable_tutor_persona,
        style_weights_text=settings.auto_comment_style_weights,
        stressed_keywords_text=settings.emotion_keywords_stressed,
        positive_keywords_text=settings.emotion_keywords_positive,
        focused_keywords_text=settings.emotion_keywords_focused,
        enable_api_understanding=settings.enable_comment_api_understanding,
    )

    def translate_tts_with_api(text: str) -> str:
        try:
            translated = llm_client.chat(
                user_text=text,
                system_prompt=SYSTEM_VOICEVOX_TRANSLATE_PROMPT,
            ).strip()
        except Exception:
            return ""
        if not translated or translated.startswith("[离线回声]"):
            return ""
        return translated[: settings.tts_translation_max_chars]

    speech = SpeechService(
        settings,
        translation_api_fallback=translate_tts_with_api,
    )

    def estimate_bubble_duration_ms(display_text: str) -> int:
        visual_chars = len(re.sub(r"\s+", "", display_text))

        if settings.tts_provider.lower() == "voicevox":
            chars_per_sec = 4.8
        elif settings.tts_provider.lower() == "azure":
            chars_per_sec = 5.8
        else:
            chars_per_sec = 5.5

        speech_ms = int((max(1, visual_chars) / chars_per_sec) * 1000) + 1200
        return max(2600, min(22000, speech_ms + 800))

    def _speak_async(display_text: str) -> None:
        if speech.is_muted():
            return
        speech.speak(display_text)

    def _speak_async_with_callback(display_text: str, on_start=None) -> None:
        if speech.is_muted():
            if callable(on_start):
                on_start()
            return
        if not display_text.strip():
            if callable(on_start):
                on_start()
            return
        queued = speech.speak(display_text, on_start=on_start)
        if (not queued) and callable(on_start):
            on_start()

    def show_and_speak(display_text: str, *, role: str | None = None, duration_ms: int | None = None) -> None:
        bubble_ms = duration_ms if duration_ms is not None else estimate_bubble_duration_ms(display_text)

        shown_lock = threading.Lock()
        shown_state = {"done": False}

        def _emit_ui_once() -> None:
            with shown_lock:
                if shown_state["done"]:
                    return
                shown_state["done"] = True
            if role is not None:
                chat.append_message(role, display_text)
            pet.show_comment_bubble(display_text, duration_ms=bubble_ms)

        # Fallback: if TTS generation/playback is blocked, still show message after a short wait.
        QTimer.singleShot(2200, _emit_ui_once)
        tts_executor.submit(_speak_async_with_callback, display_text, lambda: QTimer.singleShot(0, _emit_ui_once))

    def speak_text(text: str) -> None:
        tts_executor.submit(_speak_async, text)

    def on_web_search_toggled(enabled: bool) -> None:
        dialog.set_web_search_enabled(bool(enabled))
        if enabled:
            pet.show_comment_bubble("联网检索已开启", duration_ms=1400)
        else:
            pet.show_comment_bubble("联网检索已关闭", duration_ms=1400)

    def on_multimodal_chat_toggled(enabled: bool) -> None:
        if enabled:
            pet.show_comment_bubble("手动聊天多模态已开启", duration_ms=1600)
        else:
            pet.show_comment_bubble("手动聊天多模态已关闭", duration_ms=1600)

    chat_screen_lock = threading.Lock()

    def provide_chat_screen_context(user_text: str) -> str:
        # This callback runs after startup in ChatPanel's worker thread. It deliberately
        # captures a fresh frame instead of reusing the automatic scan cache.
        with chat_screen_lock:
            image = capture_primary_screen(
                monitor_index=int(state["scan_monitor_index"]),
                region=state["scan_region"],
            )
            pipeline = build_screen_context(
                image,
                respect_vision_interval=False,
                vision_enabled=True,
                focus_query=user_text,
            )
        screen_context = str(pipeline.get("screen_context", "") or "").strip()
        if not screen_context:
            return ""
        mode = str(pipeline.get("mode", "none") or "none")
        return f"屏幕识别模式: {mode}\n{screen_context}"

    def provide_uploaded_image_context(image, user_text: str, ocr_text: str) -> str:
        compact_ocr = str(ocr_text or "").strip()
        vision_summary = ""
        vision_reason = "off"
        if bool(settings.enable_multimodal_vision):
            route = "vision+ocr" if compact_ocr else "vision_only"
            vision_result = describe_screen_image_compat(
                llm_client,
                image,
                timeout_sec=float(settings.mm_timeout_sec),
                max_edge=int(settings.mm_image_max_edge),
                ocr_context=compact_ocr[: int(settings.ocr_context_max_chars)],
                route=route,
                focus_query=user_text,
                content_kind="uploaded_image",
            )
            vision_summary = str(vision_result.summary or "").strip()
            vision_reason = str(vision_result.reason or "empty")

        parts = [f"上传图片识别模式: {'vision+ocr' if compact_ocr else 'vision'}"]
        if compact_ocr:
            parts.append(f"本地OCR文字:\n{compact_ocr[: int(settings.ocr_context_max_chars)]}")
        if vision_summary:
            parts.append(f"Vision视觉摘要:\n{vision_summary}")
        else:
            parts.append(f"Vision视觉摘要不可用（{vision_reason}）")
        return "\n".join(parts)

    def on_archive_state_change(active: bool, message: str) -> None:
        if active:
            hint = message.strip() or "妹妹写日记中，请不要关闭"
            pet.show_comment_bubble(hint, duration_ms=120000)
        else:
            pet.hide_comment_bubble()
            done_hint = message.strip()
            if done_hint:
                pet.show_comment_bubble(done_hint, duration_ms=1800)

    pet = DesktopPet(settings=settings)
    chat = ChatPanel(
        dialog_manager=dialog,
        on_pet_reply=speak_text,
        on_archive_state_change=on_archive_state_change,
        on_web_search_toggled=on_web_search_toggled,
        screen_context_provider=provide_chat_screen_context,
        uploaded_image_context_provider=provide_uploaded_image_context,
        on_multimodal_toggled=on_multimodal_chat_toggled,
        show_system_messages=settings.chat_show_system_messages,
        web_search_enabled=False,
        multimodal_chat_enabled=settings.enable_chat_multimodal,
        screen_context_max_chars=settings.chat_screen_context_max_chars,
    )
    dialog.set_web_search_enabled(False)

    if settings.enable_live2d_py:
        chat.enable_live2d_overlay_mode()

    tts_ok, tts_info = speech.get_status()
    print(f"[STARTUP] TTS status: {'ok' if tts_ok else 'fail'} | {tts_info}")

    base_comment_interval_sec = max(10, settings.screen_scan_interval_sec)
    if settings.scan_tick_interval_sec > 0:
        normal_scan_tick_interval_sec = settings.scan_tick_interval_sec
    else:
        normal_scan_tick_interval_sec = max(5, min(15, max(1, base_comment_interval_sec // 6)))

    if settings.scan_submit_min_interval_sec > 0:
        normal_scan_submit_min_interval_sec = settings.scan_submit_min_interval_sec
    else:
        if settings.enable_scan_subprocess:
            # Keep sample cadence close to tick interval; subprocess cache reuse avoids heavy OCR each submit.
            normal_scan_submit_min_interval_sec = max(5, normal_scan_tick_interval_sec)
        else:
            normal_scan_submit_min_interval_sec = max(
                normal_scan_tick_interval_sec,
                min(15, max(1, base_comment_interval_sec // 3)),
            )
    scan_tick_interval_sec = 1 if settings.enable_visual_novel_mode else normal_scan_tick_interval_sec
    scan_submit_min_interval_sec = (
        1 if settings.enable_visual_novel_mode else normal_scan_submit_min_interval_sec
    )

    def schedule_next_comment(now: datetime) -> datetime:
        jitter = random.randint(-30, 30)
        next_sec = max(10, base_comment_interval_sec + jitter)
        return now + timedelta(seconds=next_sec)

    next_comment_at = schedule_next_comment(datetime.now())
    scan_busy_timeout_sec = max(5, int(settings.scan_busy_timeout_sec))
    print(
        "[STARTUP] Auto comment schedule:",
        f"base={base_comment_interval_sec}s",
        f"scan_tick={scan_tick_interval_sec}s",
        f"scan_submit_min={scan_submit_min_interval_sec}s",
        f"first_due={next_comment_at.strftime('%H:%M:%S')}",
    )

    state = {
        "last_comment_at": None,
        "last_summary": "",
        "last_comment_text": "",
        "scan_enabled": False,
        "visual_novel_mode_enabled": visual_novel_runtime["enabled"],
        "last_pipeline_mode": "",
        "scan_monitor_index": settings.scan_monitor_index,
        "scan_region": settings.scan_region,
        "region_overlay": None,
        "next_comment_at": next_comment_at,
        "cycle_memories": [],
        "scan_future": None,
        "scan_started_at": None,
        "last_scan_submit_at": None,
        "scan_timeout_streak": 0,
        "scan_backoff_until": None,
        "adaptive_submit_interval_sec": float(scan_submit_min_interval_sec),
        "adaptive_busy_timeout_sec": float(scan_busy_timeout_sec),
        "last_scan_duration_sec": 0.0,
        "scan_cache_fingerprint": None,
        "scan_cache_ocr_hash": "",
        "scan_cache_ocr_text": "",
        "scan_cache_ocr_confidence": 0.0,
        "scan_cache_screen_context": "",
        "scan_cache_scene_summary": "",
        "scan_cache_scene_should_comment": False,
        "scan_cache_mode": "none",
        "scan_cache_vision_route": "none",
        "comment_future": None,
        "pending_comment_meta": None,
        "vn_last_evaluation_at": None,
        "mm_fail_streak": 0,
        "mm_cooldown_until": None,
        "mm_last_request_at": None,
    }

    if settings.enable_scan_subprocess:
        print("[STARTUP] Auto scan execution: subprocess-persistent")
    else:
        print("[STARTUP] Auto scan execution: thread")
    scan_executor = None
    scan_worker_ctx = None
    scan_task_queue = None
    scan_result_queue = None
    scan_worker_process = None
    scan_task_seq = 0
    comment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="comment-worker")
    tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-worker")

    def warmup_tts_translation() -> None:
        status = speech.warmup_translation()
        print(f"[STARTUP] TTS translation: {'ok' if status.ready else 'fail'} | {status.detail}")

    tts_executor.submit(warmup_tts_translation)
    resource_policy_reapply_min_sec = float(settings.resource_policy_reapply_min_sec)
    follow_activate_distance_px = int(settings.live2d_follow_activate_distance_px)
    follow_enter_distance_px = max(140, follow_activate_distance_px)
    follow_exit_distance_px = max(follow_enter_distance_px + 80, int(follow_enter_distance_px * 1.8))
    follow_hold_sec = 0.9
    print(f"[STARTUP] Scan busy timeout: {scan_busy_timeout_sec}s")
    logical_cpu_count = int(os.cpu_count() or 0)
    all_cpu_ids = list(range(logical_cpu_count)) if logical_cpu_count > 0 else []
    if logical_cpu_count >= 4:
        split = max(1, logical_cpu_count // 2)
        app_cpu_ids = list(range(0, split))
        scan_cpu_ids = list(range(split, logical_cpu_count))
    else:
        app_cpu_ids = []
        scan_cpu_ids = []
    if app_cpu_ids and scan_cpu_ids:
        print(
            "[STARTUP] CPU affinity partition:",
            f"app={app_cpu_ids}",
            f"scan={scan_cpu_ids}",
        )
    resource_policy_last_apply_ts = 0.0
    resource_policy_last_mode = ""
    resource_policy_last_switch_ts = 0.0
    resource_policy_follow_grace_until_ts = 0.0
    resource_policy_drag_grace_until_ts = 0.0
    runtime_follow_near = False
    runtime_follow_effective = False
    runtime_follow_hold_until_ts = 0.0
    runtime_follow_near_on_streak = 0
    runtime_follow_near_off_streak = 0
    follow_near_enter_frames = 2
    follow_near_exit_frames = 4
    runtime_drag_active = False
    drag_lock_prev_active = False
    drag_lock_last_rect = None
    drag_lock_last_write_ts = 0.0
    last_zorder_sync_ts = 0.0
    resource_policy_heavy_scan_interval_sec = 20.0
    voicevox_pid_cache = 0
    voicevox_policy_last_scan_ts = 0.0
    proc_policy_cache: dict[int, tuple[object, tuple[int, ...]]] = {}

    def _apply_runtime_resource_policy(
        scan_active: bool,
        *,
        follow_active: bool = False,
        drag_active: bool = False,
        force: bool = False,
    ) -> None:
        nonlocal resource_policy_last_apply_ts, resource_policy_last_mode
        nonlocal resource_policy_last_switch_ts
        nonlocal resource_policy_follow_grace_until_ts, resource_policy_drag_grace_until_ts
        nonlocal voicevox_pid_cache, voicevox_policy_last_scan_ts
        if os.name != "nt":
            return

        now_ts = time.time()
        if drag_active:
            resource_policy_drag_grace_until_ts = now_ts + 0.28
        if follow_active:
            resource_policy_follow_grace_until_ts = now_ts + 0.42

        if drag_active or now_ts < resource_policy_drag_grace_until_ts:
            candidate_mode = "drag_priority"
        elif follow_active or now_ts < resource_policy_follow_grace_until_ts:
            candidate_mode = "follow_priority"
        elif scan_active:
            candidate_mode = "ocr_priority"
        else:
            candidate_mode = "idle"

        mode = candidate_mode
        if (
            (not force)
            and resource_policy_last_mode
            and candidate_mode != resource_policy_last_mode
            and (now_ts - resource_policy_last_switch_ts < 0.30)
        ):
            # Avoid rapid mode flapping around follow/ocr boundaries.
            mode = resource_policy_last_mode

        if (not force) and (resource_policy_last_mode == mode) and (now_ts - resource_policy_last_apply_ts < resource_policy_reapply_min_sec):
            return

        try:
            import psutil
        except Exception:
            return

        idle_cls = getattr(psutil, "IDLE_PRIORITY_CLASS", None)
        below_cls = getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", None)
        normal_cls = getattr(psutil, "NORMAL_PRIORITY_CLASS", None)
        above_cls = getattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS", None)
        if not any([idle_cls, below_cls, normal_cls, above_cls]):
            return

        def _set_proc_priority(pid: int, cls) -> None:
            if not pid or cls is None:
                return
            try:
                p = psutil.Process(int(pid))
                if not p.is_running():
                    return
                p.nice(cls)
            except Exception:
                pass

        def _set_proc_affinity(pid: int, cpu_ids: list[int]) -> None:
            if not pid or not cpu_ids:
                return
            try:
                p = psutil.Process(int(pid))
                if not p.is_running() or not hasattr(p, "cpu_affinity"):
                    return
                p.cpu_affinity(cpu_ids)
            except Exception:
                pass

        def _apply_proc_policy_cached(pid: int, cls, cpu_ids: list[int]) -> None:
            if not pid:
                return
            target_affinity = tuple(sorted(int(x) for x in cpu_ids)) if cpu_ids else tuple()
            target = (cls, target_affinity)
            prev = proc_policy_cache.get(int(pid))
            if (not force) and prev == target:
                return
            _set_proc_priority(pid, cls)
            if cpu_ids:
                _set_proc_affinity(pid, cpu_ids)
            proc_policy_cache[int(pid)] = target

        # Keep main UI responsive while scan worker does heavy OCR.
        if mode == "drag_priority":
            main_cls = above_cls
            scan_cls = below_cls
            live2d_cls = normal_cls
            scan_affinity = all_cpu_ids
            live2d_affinity = all_cpu_ids
        elif mode == "follow_priority":
            main_cls = normal_cls
            scan_cls = below_cls
            live2d_cls = above_cls
            scan_affinity = all_cpu_ids
            live2d_affinity = all_cpu_ids
        elif mode == "ocr_priority":
            main_cls = normal_cls
            scan_cls = above_cls
            # OCR priority mode explicitly pushes gaze follow process to the lowest level.
            live2d_cls = idle_cls
            # Only pin cores when enough logical CPUs are available.
            if scan_cpu_ids and app_cpu_ids:
                scan_affinity = scan_cpu_ids
                live2d_affinity = app_cpu_ids
            else:
                scan_affinity = all_cpu_ids
                live2d_affinity = all_cpu_ids
        else:
            main_cls = normal_cls
            scan_cls = below_cls
            live2d_cls = below_cls
            scan_affinity = all_cpu_ids
            live2d_affinity = all_cpu_ids

        _apply_proc_policy_cached(os.getpid(), main_cls, all_cpu_ids)

        # scan worker is a persistent standalone process when subprocess mode is enabled.
        try:
            if settings.enable_scan_subprocess:
                if scan_worker_process is not None and scan_worker_process.is_alive():
                    _apply_proc_policy_cached(int(scan_worker_process.pid), scan_cls, scan_affinity)
        except Exception:
            pass

        # Follow process priority is controlled by mode.
        if live2d_py_process is not None and live2d_py_process.poll() is None:
            _apply_proc_policy_cached(int(live2d_py_process.pid), live2d_cls, live2d_affinity)

        # Keep VOICEVOX at normal as requested, even in OCR-priority mode.
        # Use cached pid and low-frequency full scan to reduce periodic spikes.
        try:
            if voicevox_pid_cache > 0:
                try:
                    proc = psutil.Process(int(voicevox_pid_cache))
                    if proc.is_running() and str(proc.name()).lower() == "run.exe":
                        _apply_proc_policy_cached(int(voicevox_pid_cache), normal_cls, [])
                    else:
                        voicevox_pid_cache = 0
                except Exception:
                    voicevox_pid_cache = 0

            need_full_scan = force or voicevox_pid_cache <= 0 or (now_ts - voicevox_policy_last_scan_ts >= resource_policy_heavy_scan_interval_sec)
            if need_full_scan:
                voicevox_policy_last_scan_ts = now_ts
                for proc in psutil.process_iter(attrs=["pid", "name", "exe", "cmdline"]):
                    name = str((proc.info.get("name") or "")).lower()
                    exe = str((proc.info.get("exe") or "")).lower()
                    cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                    if name != "run.exe":
                        continue
                    if "voicevox" not in exe and "voicevox" not in cmdline and "vv-engine" not in exe and "vv-engine" not in cmdline:
                        continue
                    voicevox_pid_cache = int(proc.info.get("pid") or 0)
                    if voicevox_pid_cache > 0:
                        _apply_proc_policy_cached(voicevox_pid_cache, normal_cls, [])
                        break
        except Exception:
            pass

        if resource_policy_last_mode != mode:
            log_heartbeat(
                "resource_policy",
                f"mode={mode}, scan={int(bool(scan_active))}, follow={int(bool(follow_active))}, drag={int(bool(drag_active))}",
            )
            resource_policy_last_switch_ts = now_ts
        resource_policy_last_mode = mode
        resource_policy_last_apply_ts = now_ts

    def _start_scan_worker() -> None:
        nonlocal scan_worker_ctx, scan_task_queue, scan_result_queue, scan_worker_process
        if not settings.enable_scan_subprocess:
            return
        if scan_worker_process is not None and scan_worker_process.is_alive():
            return
        scan_worker_ctx = mp.get_context("spawn")
        scan_task_queue = scan_worker_ctx.Queue(maxsize=2)
        scan_result_queue = scan_worker_ctx.Queue(maxsize=4)
        scan_worker_process = scan_worker_ctx.Process(
            target=run_scan_pipeline_worker_loop,
            args=(scan_task_queue, scan_result_queue),
            daemon=True,
        )
        scan_worker_process.start()
        print(f"[STARTUP] Scan worker process started (pid={scan_worker_process.pid})")

    def _stop_scan_worker() -> None:
        nonlocal scan_task_queue, scan_result_queue, scan_worker_process, scan_worker_ctx
        if not settings.enable_scan_subprocess:
            return
        try:
            if scan_task_queue is not None:
                try:
                    scan_task_queue.put_nowait(None)
                except Exception:
                    pass
            if scan_worker_process is not None and scan_worker_process.is_alive():
                scan_worker_process.terminate()
                scan_worker_process.join(timeout=1.0)
        except Exception:
            pass
        scan_task_queue = None
        scan_result_queue = None
        scan_worker_process = None
        scan_worker_ctx = None

    def _drain_scan_results() -> None:
        if scan_result_queue is None:
            return
        while True:
            try:
                scan_result_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

    def _submit_scan_worker_task(scan_monitor_index: int, scan_region: tuple[int, int, int, int] | None) -> int:
        nonlocal scan_task_seq
        if scan_task_queue is None:
            return 0
        scan_task_seq += 1
        payload = {
            "task_id": int(scan_task_seq),
            "scan_monitor_index": int(scan_monitor_index),
            "scan_region": scan_region,
            "ocr_cpu_threads": int(settings.ocr_cpu_threads),
            "ocr_cpu_affinity_count": int(settings.ocr_cpu_affinity_count),
            "ocr_max_edge": int(settings.ocr_max_edge),
            "visual_novel_mode": bool(state["visual_novel_mode_enabled"]),
        }
        try:
            scan_task_queue.put_nowait(payload)
        except queue.Full:
            return 0
        except Exception:
            return 0
        return int(scan_task_seq)

    def _poll_scan_worker_result(expected_task_id: int):
        if scan_result_queue is None:
            return None
        while True:
            try:
                msg = scan_result_queue.get_nowait()
            except queue.Empty:
                return None
            except Exception:
                return None
            if int(msg.get("task_id", 0)) == int(expected_task_id):
                return msg

    def _reset_scan_worker(reason: str) -> None:
        nonlocal scan_executor
        pet.set_scan_busy(False)
        streak = int(state.get("scan_timeout_streak", 0)) + 1
        state["scan_timeout_streak"] = streak
        cooldown_sec = min(90, 6 * (2 ** (streak - 1)))
        state["scan_backoff_until"] = datetime.now() + timedelta(seconds=cooldown_sec)
        state["adaptive_submit_interval_sec"] = max(
            float(state.get("adaptive_submit_interval_sec", scan_submit_min_interval_sec)),
            float(cooldown_sec),
        )
        state["adaptive_busy_timeout_sec"] = min(
            90.0,
            max(
                float(scan_busy_timeout_sec),
                float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec)) * 1.4,
            ),
        )
        future = state.get("scan_future")
        if future is not None:
            try:
                future.cancel()
            except Exception:
                pass
        state["scan_future"] = None
        state["scan_started_at"] = None
        state["last_scan_submit_at"] = None
        if settings.enable_scan_subprocess:
            _stop_scan_worker()
            _start_scan_worker()
            _drain_scan_results()
        else:
            try:
                if scan_executor is not None:
                    scan_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")
        log_heartbeat(
            "scan_reset",
            (
                f"{reason}, streak={streak}, cooldown={cooldown_sec}s, "
                f"next_busy_timeout={state['adaptive_busy_timeout_sec']:.1f}s"
            ),
        )
        chat.append_message("系统", f"扫描任务卡住，已自动重置扫描器（{reason}）")

    def log_heartbeat(stage: str, detail: str = "") -> None:
        if not settings.enable_auto_comment_heartbeat:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        msg = f"[HEARTBEAT {ts}] {stage}"
        if detail:
            msg += f" | {detail}"
        print(msg)

    def build_screen_context(
        image,
        *,
        respect_vision_interval: bool,
        ocr_result=None,
        vision_enabled: bool | None = None,
        focus_query: str = "",
    ) -> dict:
        if ocr_result is None:
            ocr_image = _resize_for_ocr(image, int(settings.ocr_max_edge))
            ocr_result = extract_ocr_result(ocr_image)
        return _build_ocr_first_context(
            llm_client,
            image,
            ocr_result,
            settings,
            state,
            log_heartbeat,
            respect_vision_interval=respect_vision_interval,
            vision_enabled=vision_enabled,
            focus_query=focus_query,
        )

    def maybe_emit_pipeline_status(mode: str):
        if mode == state["last_pipeline_mode"]:
            return
        state["last_pipeline_mode"] = mode

        if mode == "vision+ocr":
            chat.append_message("系统", "识别状态: 多模态视觉+OCR")
        elif mode == "vision":
            chat.append_message("系统", "识别状态: 仅多模态视觉")
        elif mode == "ocr":
            chat.append_message("系统", "识别状态: 仅本地OCR（省Token）")
        else:
            chat.append_message("系统", "识别状态: 未识别到有效内容")

    def _emit_comment(comment: str, role: str = "桌宠(自动)") -> None:
        display_role = role
        if settings.chat_show_session_debug_marker:
            seg_id = max(1, int(dialog.get_current_session_segment_id()))
            display_role = f"{role}[S{seg_id}]"
        show_and_speak(comment, role=display_role)
        dialog.record_session_message(role, comment)

    def _normalize_memory_summary(summary: str) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        text = text.replace("屏幕内容摘要:", "")
        text = " ".join(text.split())
        return text

    def _append_cycle_memory(summary: str) -> None:
        normalized = _normalize_memory_summary(summary)
        if not normalized:
            return
        captured_at = datetime.now()
        memories = state["cycle_memories"]

        if memories and memories[-1]["summary"] == normalized:
            memories[-1]["captured_at"] = captured_at
            return

        # Fuzzy dedup across nearby screenshots: keep one representative for highly similar text.
        for item in reversed(memories[-12:]):
            ratio = text_similarity(normalized, str(item.get("summary", "")))
            if ratio >= 0.90:
                item["captured_at"] = captured_at
                if len(normalized) > len(str(item.get("summary", ""))):
                    item["summary"] = normalized
                return
        memories.append({"summary": normalized, "captured_at": captured_at})
        if len(memories) > 24:
            del memories[:-24]

    def _compose_cycle_summary(now: datetime) -> tuple[str, str]:
        memories = state["cycle_memories"]
        if not memories:
            return "", ""

        recency_window_sec = settings.memory_recency_window_sec
        if recency_window_sec <= 0:
            recency_window_sec = max(45, base_comment_interval_sec + 30)
        min_weight = settings.memory_min_weight
        merged: dict[str, dict] = {}

        for item in memories:
            summary = _normalize_memory_summary(item["summary"])
            if not summary:
                continue
            captured_at = item["captured_at"]
            age_sec = max(0.0, (now - captured_at).total_seconds())
            weight = max(min_weight, 1.0 - (age_sec / recency_window_sec))

            canonical_key = summary
            for key in merged.keys():
                if text_similarity(summary, key) >= 0.90:
                    canonical_key = key
                    break

            prev = merged.get(canonical_key)
            if prev is None:
                merged[canonical_key] = {
                    "summary": summary,
                    "weight": weight,
                    "captured_at": captured_at,
                }
                continue
            if weight > prev["weight"]:
                prev["weight"] = weight
            if captured_at > prev["captured_at"]:
                prev["captured_at"] = captured_at
            if len(summary) > len(str(prev.get("summary", ""))):
                prev["summary"] = summary

        ranked = sorted(
            merged.values(),
            key=lambda x: (x["weight"], x["captured_at"]),
            reverse=True,
        )[:12]

        signature = "；".join(item["summary"] for item in ranked[:8])
        lines: list[str] = []
        for idx, item in enumerate(ranked, start=1):
            tag = "重点" if idx <= 3 else "参考"
            lines.append(f"{tag}[权重{item['weight']:.2f}] {item['summary']}")

        composed = "近期扫描记忆（越靠近当前时刻权重越高）:\n" + "\n".join(lines)
        return composed[:6000], signature[:2400]

    def run_scan_pipeline() -> dict:
        image = capture_primary_screen(
            monitor_index=int(state["scan_monitor_index"]),
            region=state["scan_region"],
        )
        image = _crop_visual_novel_scan_image(
            image,
            enabled=bool(state["visual_novel_mode_enabled"]),
            has_manual_region=state["scan_region"] is not None,
        )
        fingerprint = _frame_fingerprint(
            image,
            text_sensitive=bool(state["visual_novel_mode_enabled"]),
        )
        prev_fingerprint = state.get("scan_cache_fingerprint")
        if isinstance(prev_fingerprint, bytes):
            diff_ratio = _fingerprint_diff_ratio(fingerprint, prev_fingerprint)
            diff_threshold = 0.012 if state["visual_novel_mode_enabled"] else 0.04
            if diff_ratio < diff_threshold and str(state.get("scan_cache_scene_summary", "")).strip():
                cached_context = str(state.get("scan_cache_screen_context", "")).strip()
                scene = analyze_scene(cached_context)
                return {
                    "ocr_ok": True,
                    "ocr_info": "cache_reuse",
                    "mode": str(state.get("scan_cache_mode", "none")),
                    "screen_context": cached_context,
                    "ocr_text": str(state.get("scan_cache_ocr_text", "")),
                    "ocr_confidence": float(state.get("scan_cache_ocr_confidence", 0.0) or 0.0),
                    "scene": scene,
                    "vision_route": "cache",
                    "reused_cache": True,
                }

        ocr_ok, ocr_info = get_ocr_runtime_status()
        ocr_image = _resize_for_ocr(image, int(settings.ocr_max_edge))
        ocr_result = extract_ocr_result(ocr_image)
        previous_ocr_hash = str(state.get("scan_cache_ocr_hash", "") or "")
        if (
        ocr_result.text_hash
        and ocr_result.text_hash == previous_ocr_hash
        and str(state.get("scan_cache_vision_route", "")) == "ocr_only"
        and str(state.get("scan_cache_scene_summary", "")).strip()
        ):
            state["scan_cache_fingerprint"] = fingerprint
            cached_context = str(state.get("scan_cache_screen_context", "")).strip()
            scene = analyze_scene(cached_context)
            return {
                "ocr_ok": ocr_ok,
                "ocr_info": "ocr_hash_reuse",
                "mode": str(state.get("scan_cache_mode", "none")),
                "screen_context": cached_context,
                "ocr_text": str(state.get("scan_cache_ocr_text", "")),
                "ocr_confidence": float(state.get("scan_cache_ocr_confidence", 0.0) or 0.0),
                "scene": scene,
                "vision_route": "cache",
                "reused_cache": True,
            }

        pipeline = build_screen_context(
            image,
            respect_vision_interval=True,
            ocr_result=ocr_result,
            vision_enabled=False if state["visual_novel_mode_enabled"] else None,
        )
        screen_context = str(pipeline["screen_context"])
        mode = str(pipeline["mode"])
        scene = analyze_scene(screen_context)
        state["scan_cache_fingerprint"] = fingerprint
        state["scan_cache_ocr_hash"] = str(pipeline.get("ocr_hash", ""))
        state["scan_cache_ocr_text"] = str(pipeline.get("ocr_text", ""))
        state["scan_cache_ocr_confidence"] = float(pipeline.get("ocr_confidence", 0.0) or 0.0)
        state["scan_cache_screen_context"] = screen_context
        state["scan_cache_scene_summary"] = scene.summary
        state["scan_cache_scene_should_comment"] = scene.should_comment
        state["scan_cache_mode"] = mode
        state["scan_cache_vision_route"] = str(pipeline.get("vision_route", "none"))
        return {
            "ocr_ok": ocr_ok,
            "ocr_info": ocr_info,
            "mode": mode,
            "screen_context": screen_context,
            "ocr_text": str(pipeline.get("ocr_text", "")),
            "ocr_confidence": float(pipeline.get("ocr_confidence", 0.0) or 0.0),
            "scene": scene,
            "vision_route": str(pipeline.get("vision_route", "none")),
            "reused_cache": False,
        }

    def text_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _process_visual_novel_text(result: dict, now: datetime) -> None:
        if visual_novel_tracker is None:
            return
        ocr_text = str(result.get("ocr_text", "") or "").strip()
        if not ocr_text:
            return
        ocr_text, quality_reason = visual_novel_tracker.prepare_ocr_text(
            ocr_text,
            confidence=float(result.get("ocr_confidence", 0.0) or 0.0),
        )
        if not ocr_text:
            log_heartbeat("vn_skip", quality_reason)
            return
        if quality_reason != "quality_ok":
            log_heartbeat("vn_filter", quality_reason)
        observation = visual_novel_tracker.observe(
            ocr_text,
            confidence=float(result.get("ocr_confidence", 0.0) or 0.0),
        )
        if not observation.accepted:
            log_heartbeat("vn_skip", observation.reason)
        else:
            log_heartbeat(
                "vn_buffer",
                f"reason={observation.reason}, semantic_score={observation.semantic_score:.2f}",
            )
        if not visual_novel_tracker.has_pending_evaluation or state["comment_future"] is not None:
            return

        last_evaluation_at = state.get("vn_last_evaluation_at")
        if isinstance(last_evaluation_at, datetime) and (now - last_evaluation_at).total_seconds() < 12.0:
            log_heartbeat("vn_wait", "evaluation_rate_limit")
            return

        allow_comment = can_emit_comment(state["last_comment_at"], settings.auto_comment_cooldown_sec)
        if not allow_comment:
            log_heartbeat("vn_wait", "comment_cooldown")
            return
        payload = visual_novel_tracker.build_evaluation_payload()
        if payload is None:
            return
        state["comment_future"] = comment_executor.submit(
            comment_engine.evaluate_visual_novel,
            payload,
        )
        state["vn_last_evaluation_at"] = now
        state["pending_comment_meta"] = {
            "kind": "visual_novel",
            "now": now,
            "allow_comment": allow_comment,
            "reason": str(payload.get("reason", "")),
            "summary_len": len(str(payload.get("scene_summary", ""))),
        }
        log_heartbeat("vn_evaluate", f"reason={payload.get('reason', 'unknown')}")

    def poll_comment_future() -> None:
        future = state["comment_future"]
        if future is None:
            return
        if not future.done():
            log_heartbeat("comment_busy")
            return

        state["comment_future"] = None
        meta = state["pending_comment_meta"] or {}
        state["pending_comment_meta"] = None

        try:
            generated = future.result()
        except Exception as exc:
            log_heartbeat("error", f"comment_generate_failed: {exc}")
            return

        if meta.get("kind") == "visual_novel":
            evaluation = generated if isinstance(generated, dict) else {}
            if visual_novel_tracker is not None:
                visual_novel_tracker.apply_evaluation(evaluation)
            if not bool(meta.get("allow_comment", False)):
                log_heartbeat("vn_skip", "summary_updated_during_cooldown")
                return
            if not bool(evaluation.get("should_comment", False)):
                log_heartbeat("vn_skip", f"moment={evaluation.get('moment_type', 'ordinary')}")
                return
            comment = str(evaluation.get("comment", "")).strip()
        else:
            comment = generated

        comment = str(comment or "").strip()
        if not comment:
            log_heartbeat("skip", "comment_empty_or_fallback_suppressed")
            return

        similarity = text_similarity(comment, state["last_comment_text"])
        if similarity >= settings.comment_similarity_skip_threshold:
            log_heartbeat(
                "skip",
                f"duplicate_comment similarity={similarity:.2f} threshold={settings.comment_similarity_skip_threshold:.2f}",
            )
            return

        state["last_comment_at"] = meta.get("now", datetime.now())
        state["last_summary"] = meta.get("cycle_signature", "")
        state["last_comment_text"] = comment
        role = "桌宠(视觉小说)" if meta.get("kind") == "visual_novel" else "桌宠(自动)"
        _emit_comment(comment, role=role)
        log_heartbeat(
            "emit",
            f"cycle_items={meta.get('cycle_items', 0)}, summary_len={meta.get('summary_len', 0)}",
        )

    def do_auto_comment() -> None:
        try:
            log_heartbeat("tick")
            poll_comment_future()
            if not state["scan_enabled"]:
                log_heartbeat("skip", "scan_disabled")
                return

            future = state["scan_future"]
            if future is None:
                now = datetime.now()
                if settings.enable_live2d_py and (runtime_follow_effective or runtime_drag_active):
                    # Keep interaction smooth: postpone OCR submission until follow/drag settles.
                    wait_reason = "drag" if runtime_drag_active else "follow"
                    log_heartbeat("scan_wait", f"interaction_priority={wait_reason}")
                    return

                backoff_until = state.get("scan_backoff_until")
                if isinstance(backoff_until, datetime) and now < backoff_until:
                    remain = int((backoff_until - now).total_seconds())
                    log_heartbeat("scan_cooldown", f"remain={max(1, remain)}s")
                    return

                adaptive_submit_interval_sec = max(
                    float(scan_submit_min_interval_sec),
                    float(state.get("adaptive_submit_interval_sec", scan_submit_min_interval_sec)),
                )
                last_submit_at = state.get("last_scan_submit_at")
                if last_submit_at is not None:
                    elapsed_sec = (now - last_submit_at).total_seconds()
                    if elapsed_sec < adaptive_submit_interval_sec:
                        remaining = int(round(adaptive_submit_interval_sec - elapsed_sec))
                        log_heartbeat("scan_wait", f"submit_in={max(1, remaining)}s")
                        return

                if settings.enable_scan_subprocess:
                    task_id = _submit_scan_worker_task(
                        int(state["scan_monitor_index"]),
                        state["scan_region"],
                    )
                    if task_id <= 0:
                        log_heartbeat("scan_wait", "worker_queue_full")
                        return
                    state["scan_future"] = int(task_id)
                else:
                    state["scan_future"] = scan_executor.submit(run_scan_pipeline)
                state["scan_started_at"] = now
                state["last_scan_submit_at"] = now
                pet.set_scan_busy(True)
                log_heartbeat("scan_submit")
                return

            if settings.enable_scan_subprocess:
                result_msg = _poll_scan_worker_result(int(future))
                if result_msg is None:
                    started_at = state["scan_started_at"]
                    elapsed = int((datetime.now() - started_at).total_seconds()) if started_at else 0
                    active_busy_timeout_sec = max(
                        float(scan_busy_timeout_sec),
                        float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec)),
                    )
                    if elapsed >= active_busy_timeout_sec:
                        _reset_scan_worker(f"timeout={elapsed}s")
                        return
                    log_heartbeat("scan_busy", f"elapsed={elapsed}s/{int(active_busy_timeout_sec)}s")
                    return
            elif not future.done():
                started_at = state["scan_started_at"]
                elapsed = int((datetime.now() - started_at).total_seconds()) if started_at else 0
                active_busy_timeout_sec = max(
                    float(scan_busy_timeout_sec),
                    float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec)),
                )
                if elapsed >= active_busy_timeout_sec:
                    _reset_scan_worker(f"timeout={elapsed}s")
                    return
                log_heartbeat("scan_busy", f"elapsed={elapsed}s/{int(active_busy_timeout_sec)}s")
                return

            scan_started_at = state.get("scan_started_at")
            state["scan_future"] = None
            state["scan_started_at"] = None
            pet.set_scan_busy(False)
            if isinstance(scan_started_at, datetime):
                duration_sec = max(0.0, (datetime.now() - scan_started_at).total_seconds())
            else:
                duration_sec = 0.0
            state["last_scan_duration_sec"] = duration_sec
            state["scan_timeout_streak"] = 0
            state["scan_backoff_until"] = None
            measured_busy_timeout = max(float(scan_busy_timeout_sec), duration_sec * 2.2)
            prev_busy_timeout = float(state.get("adaptive_busy_timeout_sec", scan_busy_timeout_sec))
            state["adaptive_busy_timeout_sec"] = max(
                float(scan_busy_timeout_sec),
                min(90.0, (prev_busy_timeout * 0.7) + (measured_busy_timeout * 0.3)),
            )

            target_interval = max(float(scan_submit_min_interval_sec), duration_sec * 1.35)
            prev_interval = float(state.get("adaptive_submit_interval_sec", scan_submit_min_interval_sec))
            if target_interval > prev_interval:
                new_interval = min(90.0, target_interval)
            else:
                # Recover gradually when scan load becomes lighter.
                new_interval = max(float(scan_submit_min_interval_sec), prev_interval * 0.82)
            state["adaptive_submit_interval_sec"] = new_interval

            if settings.enable_scan_subprocess:
                if not result_msg.get("ok", False):
                    _reset_scan_worker(f"worker_error={result_msg.get('error', 'unknown')}")
                    return
                result = result_msg.get("result", {})
                duration_sec = max(float(duration_sec), float(result_msg.get("duration_sec", 0.0)))
            else:
                result = future.result()
            if not result["ocr_ok"]:
                log_heartbeat("ocr_status", result["ocr_info"])

            mode = result["mode"]
            screen_context = result["screen_context"]
            vision_route = str(result.get("vision_route", "none"))
            reused_cache = bool(result.get("reused_cache", False))
            if settings.enable_scan_subprocess:
                scene_summary = str(result.get("scene_summary", "")).strip()
                scene_should_comment = bool(result.get("scene_should_comment", False))
                mm_reason = str(result.get("mm_reason", "off"))
                mm_elapsed_ms = int(result.get("mm_elapsed_ms", 0) or 0)
                mm_summary_len = int(result.get("mm_summary_len", 0) or 0)
                log_heartbeat(
                    "mm_result",
                    f"route={vision_route}, reason={mm_reason}, elapsed_ms={mm_elapsed_ms}, len={mm_summary_len}",
                )
            else:
                scene_obj = result["scene"]
                scene_summary = scene_obj.summary
                scene_should_comment = scene_obj.should_comment
            cache_tag = "cache" if reused_cache else "fresh"
            log_heartbeat(
                "context_built",
                (
                    f"mode={mode}, route={vision_route}, src={cache_tag}, len={len(screen_context)}, "
                    f"scan_dur={duration_sec:.1f}s, next_submit_min={state['adaptive_submit_interval_sec']:.1f}s, "
                    f"busy_timeout={state['adaptive_busy_timeout_sec']:.1f}s"
                ),
            )
            maybe_emit_pipeline_status(mode)

            if state["visual_novel_mode_enabled"]:
                _process_visual_novel_text(result, datetime.now())
                return

            if scene_should_comment:
                _append_cycle_memory(scene_summary)

            now = datetime.now()
            if now < state["next_comment_at"]:
                due = state["next_comment_at"].strftime("%H:%M:%S")
                log_heartbeat("wait", f"next_due={due}, memory_count={len(state['cycle_memories'])}")
                return

            cycle_summary, cycle_signature = _compose_cycle_summary(now)
            if not cycle_summary:
                log_heartbeat("skip", "cycle_summary_empty")
                state["next_comment_at"] = schedule_next_comment(now)
                return

            if cycle_signature == state["last_summary"]:
                log_heartbeat("skip", "duplicate_cycle_summary")
                state["cycle_memories"] = []
                state["next_comment_at"] = schedule_next_comment(now)
                return

            if not can_emit_comment(state["last_comment_at"], settings.auto_comment_cooldown_sec):
                log_heartbeat("skip", "cooldown")
                return

            if state["comment_future"] is not None:
                log_heartbeat("skip", "comment_worker_busy")
                return

            long_memory_hint = dialog.build_light_long_memory_hint(limit=settings.screen_comment_memory_limit)
            recent_dialog_hint = dialog.build_recent_session_hint(limit=8)
            state["comment_future"] = comment_executor.submit(
                comment_engine.comment_on_summary,
                cycle_summary,
                long_memory_hint,
                settings.screen_comment_memory_weight,
                recent_dialog_hint,
                True,
            )
            state["pending_comment_meta"] = {
                "now": now,
                "cycle_signature": cycle_signature,
                "cycle_items": len(state["cycle_memories"]),
                "summary_len": len(cycle_summary),
            }
            state["cycle_memories"] = []
            state["next_comment_at"] = schedule_next_comment(now)
            log_heartbeat("comment_submit", f"next_due={state['next_comment_at'].strftime('%H:%M:%S')}")
        except Exception as exc:
            pet.set_scan_busy(False)
            log_heartbeat("error", str(exc))
            chat.append_message("系统", f"自动评论失败: {exc}")

    def do_manual_comment() -> None:
        try:
            image = capture_primary_screen(
                monitor_index=int(state["scan_monitor_index"]),
                region=state["scan_region"],
            )
            pipeline = build_screen_context(image, respect_vision_interval=False)
            screen_context = str(pipeline["screen_context"])
            mode = str(pipeline["mode"])
            maybe_emit_pipeline_status(mode)
            scene = analyze_scene(screen_context)
            if not scene.summary:
                chat.append_message("系统", "手动评论失败: 未识别到有效内容")
                return

            long_memory_hint = dialog.build_light_long_memory_hint(limit=settings.screen_comment_memory_limit)
            recent_dialog_hint = dialog.build_recent_session_hint(limit=8)
            comment = comment_engine.comment_on_summary(
                scene.summary,
                long_memory_hint,
                settings.screen_comment_memory_weight,
                recent_dialog_hint,
            )
            _emit_comment(comment, role="桌宠(手动)")
        except Exception as exc:
            chat.append_message("系统", f"手动评论失败: {exc}")

    def on_scan_toggle(enabled: bool) -> None:
        nonlocal scan_executor
        enabled = bool(enabled)
        state["scan_enabled"] = enabled
        pet.set_auto_scan_enabled(enabled)
        if enabled:
            if settings.enable_scan_subprocess:
                _start_scan_worker()
            elif scan_executor is None:
                scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")
            state["next_comment_at"] = schedule_next_comment(datetime.now())
            state["cycle_memories"] = []
            state["scan_timeout_streak"] = 0
            state["scan_backoff_until"] = None
            state["adaptive_submit_interval_sec"] = float(scan_submit_min_interval_sec)
            state["adaptive_busy_timeout_sec"] = float(scan_busy_timeout_sec)
            pet.set_scan_busy(False)
            _apply_runtime_resource_policy(scan_active=False, force=True)
            scheduler.start()
            chat.append_message("系统", "已开启自动扫描")
            pet.show_comment_bubble("自动扫描已开启", duration_ms=1800)
        else:
            scheduler.stop()
            scan_future = state.get("scan_future")
            if scan_future is not None and hasattr(scan_future, "cancel"):
                try:
                    scan_future.cancel()
                except Exception:
                    pass
            state["scan_future"] = None
            state["scan_started_at"] = None
            state["last_scan_submit_at"] = None
            comment_future = state.get("comment_future")
            if comment_future is not None and hasattr(comment_future, "cancel"):
                try:
                    comment_future.cancel()
                except Exception:
                    pass
            state["comment_future"] = None
            state["pending_comment_meta"] = None
            state["cycle_memories"] = []
            if settings.enable_scan_subprocess:
                _stop_scan_worker()
            elif scan_executor is not None:
                scan_executor.shutdown(wait=False, cancel_futures=True)
                scan_executor = None
            pet.set_scan_busy(False)
            _apply_runtime_resource_policy(scan_active=False, force=True)
            chat.append_message("系统", "已关闭自动扫描；手动评论和聊天多模态仍可使用")
            pet.show_comment_bubble("自动扫描已关闭", duration_ms=1800)

    def on_visual_novel_mode_toggled(enabled: bool) -> None:
        nonlocal scan_executor, scan_tick_interval_sec, scan_submit_min_interval_sec
        enabled = bool(enabled)
        if enabled == bool(state["visual_novel_mode_enabled"]):
            pet.set_visual_novel_mode_enabled(enabled)
            return

        state["visual_novel_mode_enabled"] = enabled
        visual_novel_runtime["enabled"] = enabled
        scan_tick_interval_sec = 1 if enabled else normal_scan_tick_interval_sec
        scan_submit_min_interval_sec = 1 if enabled else normal_scan_submit_min_interval_sec
        scheduler.set_interval(scan_tick_interval_sec)
        state["adaptive_submit_interval_sec"] = float(scan_submit_min_interval_sec)
        state["scan_timeout_streak"] = 0
        state["scan_backoff_until"] = None
        state["last_scan_submit_at"] = None
        state["cycle_memories"] = []
        state["scan_cache_fingerprint"] = None
        state["scan_cache_ocr_hash"] = ""
        state["scan_cache_ocr_text"] = ""
        state["scan_cache_ocr_confidence"] = 0.0
        state["scan_cache_screen_context"] = ""
        state["scan_cache_scene_summary"] = ""
        state["scan_cache_scene_should_comment"] = False
        state["scan_cache_mode"] = "none"
        state["scan_cache_vision_route"] = "none"

        scan_future = state.get("scan_future")
        if scan_future is not None and hasattr(scan_future, "cancel"):
            try:
                scan_future.cancel()
            except Exception:
                pass
        state["scan_future"] = None
        state["scan_started_at"] = None
        pet.set_scan_busy(False)

        comment_future = state.get("comment_future")
        if comment_future is not None and hasattr(comment_future, "cancel"):
            try:
                comment_future.cancel()
            except Exception:
                pass
        state["comment_future"] = None
        state["pending_comment_meta"] = None

        if state["scan_enabled"] and settings.enable_scan_subprocess:
            _stop_scan_worker()
            _start_scan_worker()
            _drain_scan_results()
        elif state["scan_enabled"] and scan_executor is not None:
            scan_executor.shutdown(wait=False, cancel_futures=True)
            scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")

        if not enabled:
            visual_novel_tracker.flush()
        pet.set_visual_novel_mode_enabled(enabled)
        mode_text = "已开启" if enabled else "已关闭"
        chat.append_message("系统", f"视觉小说模式{mode_text}")
        pet.show_comment_bubble(f"视觉小说模式{mode_text}", duration_ms=1800)
        log_heartbeat(
            "vn_mode",
            f"enabled={enabled}, scan_tick={scan_tick_interval_sec}s, submit_min={scan_submit_min_interval_sec}s",
        )

    def _switch_visual_novel_story(filename: str) -> None:
        nonlocal scan_executor
        target_path = visual_novel_library.directory / filename
        if target_path == visual_novel_tracker.path:
            pet.set_visual_novel_story_name(filename)
            return

        pending_meta = state.get("pending_comment_meta")
        if isinstance(pending_meta, dict) and pending_meta.get("kind") == "visual_novel":
            comment_future = state.get("comment_future")
            if comment_future is not None and hasattr(comment_future, "cancel"):
                try:
                    comment_future.cancel()
                except Exception:
                    pass
            state["comment_future"] = None
            state["pending_comment_meta"] = None

        scan_future = state.get("scan_future")
        if scan_future is not None and hasattr(scan_future, "cancel"):
            try:
                scan_future.cancel()
            except Exception:
                pass
        state["scan_future"] = None
        state["scan_started_at"] = None
        state["last_scan_submit_at"] = None
        pet.set_scan_busy(False)
        if state["scan_enabled"] and settings.enable_scan_subprocess:
            _stop_scan_worker()
            _start_scan_worker()
            _drain_scan_results()
        elif state["scan_enabled"] and scan_executor is not None:
            scan_executor.shutdown(wait=False, cancel_futures=True)
            scan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan-worker")

        visual_novel_tracker.switch_story(target_path)
        visual_novel_library.set_active(filename)
        state["vn_last_evaluation_at"] = None
        state["scan_cache_fingerprint"] = None
        state["scan_cache_ocr_hash"] = ""
        state["scan_cache_ocr_text"] = ""
        state["scan_cache_ocr_confidence"] = 0.0
        state["scan_cache_screen_context"] = ""
        state["scan_cache_scene_summary"] = ""
        state["scan_cache_scene_should_comment"] = False
        state["scan_cache_mode"] = "none"
        state["scan_cache_vision_route"] = "none"
        pet.set_visual_novel_story_name(filename)

    def on_visual_novel_story_manager_requested() -> None:
        action, ok = QInputDialog.getItem(
            pet,
            "剧情缓存",
            f"当前缓存：{visual_novel_library.active_name}\n请选择操作：",
            ["新建缓存", "载入缓存", "删除缓存"],
            0,
            False,
        )
        if not ok:
            return
        try:
            if action == "新建缓存":
                name, accepted = QInputDialog.getText(
                    pet,
                    "新建剧情缓存",
                    "输入游戏名或缓存名：",
                )
                if not accepted:
                    return
                path = visual_novel_library.create(name)
                _switch_visual_novel_story(path.name)
                chat.append_message("系统", f"已新建并载入剧情缓存：{path.name}")
                pet.show_comment_bubble(f"已切换缓存：{path.stem}", duration_ms=2200)
                return

            stories = visual_novel_library.list_stories()
            current_index = stories.index(visual_novel_library.active_name)
            filename, accepted = QInputDialog.getItem(
                pet,
                "载入剧情缓存" if action == "载入缓存" else "删除剧情缓存",
                "选择缓存文件：",
                stories,
                current_index,
                False,
            )
            if not accepted:
                return
            if action == "载入缓存":
                _switch_visual_novel_story(filename)
                chat.append_message("系统", f"已载入剧情缓存：{filename}")
                pet.show_comment_bubble(f"已切换缓存：{Path(filename).stem}", duration_ms=2200)
                return

            if len(stories) <= 1:
                raise ValueError("至少需要保留一个剧情缓存")
            answer = QMessageBox.question(
                pet,
                "删除剧情缓存",
                f"确定永久删除“{filename}”吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            if filename == visual_novel_library.active_name:
                replacement = next(item for item in stories if item != filename)
                _switch_visual_novel_story(replacement)
            visual_novel_library.delete(filename)
            chat.append_message("系统", f"已删除剧情缓存：{filename}")
            pet.show_comment_bubble(f"已删除缓存：{Path(filename).stem}", duration_ms=2200)
        except Exception as exc:
            QMessageBox.warning(pet, "剧情缓存操作失败", str(exc))

    def on_tts_mute_toggled(muted: bool) -> None:
        speech.set_muted(muted)
        pet.set_tts_muted(muted)
        if muted:
            chat.append_message("系统", "语音已静音")
            pet.show_comment_bubble("语音已静音", duration_ms=1800)
        else:
            chat.append_message("系统", "语音已开启")
            pet.show_comment_bubble("语音已开启", duration_ms=1800)

    def on_gaze_follow_toggled(enabled: bool) -> None:
        pet.set_gaze_follow_enabled(enabled)
        if enabled:
            chat.append_message("系统", "视线跟随已开启")
            pet.show_comment_bubble("视线跟随已开启", duration_ms=1600)
        else:
            chat.append_message("系统", "视线跟随已关闭")
            pet.show_comment_bubble("视线跟随已关闭", duration_ms=1600)

    def on_tutor_mode_toggled(enabled: bool) -> None:
        dialog.set_tutor_enabled(enabled)
        comment_engine.set_tutor_enabled(enabled)
        pet.set_tutor_mode_enabled(enabled)
        if enabled:
            chat.append_message("系统", "教学模式已开启")
            pet.show_comment_bubble("教学模式已开启", duration_ms=1600)
        else:
            chat.append_message("系统", "教学模式已关闭")
            pet.show_comment_bubble("教学模式已关闭", duration_ms=1600)

    def on_select_scan_region() -> None:
        try:
            qt_screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            if qt_screen is None:
                raise RuntimeError("未找到可用显示器")
            qt_geometry = qt_screen.geometry()
            logical_geometry = (
                int(qt_geometry.x()),
                int(qt_geometry.y()),
                int(qt_geometry.width()),
                int(qt_geometry.height()),
            )
            monitor_index = find_monitor_index(qt_geometry.x(), qt_geometry.y())
            _left, _top, capture_width, capture_height = get_monitor_geometry(monitor_index)
            overlay = RegionSelectOverlay(
                logical_geometry,
                capture_size=(int(capture_width), int(capture_height)),
            )
            state["region_overlay"] = overlay

            def _on_selected(region: tuple[int, int, int, int]) -> None:
                state["scan_monitor_index"] = monitor_index
                state["scan_region"] = region
                state["scan_cache_fingerprint"] = None
                state["scan_cache_ocr_hash"] = ""
                state["scan_cache_ocr_text"] = ""
                state["scan_cache_ocr_confidence"] = 0.0
                state["scan_cache_screen_context"] = ""
                state["scan_cache_scene_summary"] = ""
                state["scan_cache_vision_route"] = "none"
                chat.append_message("系统", f"已设置显示器{monitor_index}扫描区域: {region}")
                pet.show_comment_bubble(
                    f"扫描区域已更新：显示器{monitor_index}，{region[2]}×{region[3]}",
                    duration_ms=2400,
                )
                state["region_overlay"] = None

            def _on_cancelled() -> None:
                chat.append_message("系统", "已取消区域选择")
                state["region_overlay"] = None

            overlay.region_selected.connect(_on_selected)
            overlay.cancelled.connect(_on_cancelled)
            overlay.show_for_selection()
            pet.show_comment_bubble("拖拽框选扫描区域，右键或Esc取消", duration_ms=3500)
        except Exception as exc:
            chat.append_message("系统", f"区域选择失败: {exc}")

    def on_clear_scan_region() -> None:
        state["scan_monitor_index"] = 0
        state["scan_region"] = None
        state["scan_cache_fingerprint"] = None
        state["scan_cache_ocr_hash"] = ""
        state["scan_cache_ocr_text"] = ""
        state["scan_cache_ocr_confidence"] = 0.0
        state["scan_cache_screen_context"] = ""
        state["scan_cache_scene_summary"] = ""
        state["scan_cache_vision_route"] = "none"
        chat.append_message("系统", "已清除扫描区域，恢复整个虚拟桌面扫描")
        pet.show_comment_bubble("已恢复所有屏幕扫描")

    def on_quit_requested() -> None:
        scheduler.stop()
        if visual_novel_tracker is not None:
            visual_novel_tracker.flush()
        pet.set_scan_busy(False)
        _apply_runtime_resource_policy(scan_active=False, force=True)
        future = state.get("scan_future")
        if future is not None and hasattr(future, "cancel"):
            future.cancel()
        comment_future = state.get("comment_future")
        if comment_future is not None:
            comment_future.cancel()
        if settings.enable_scan_subprocess:
            _stop_scan_worker()
        elif scan_executor is not None:
            scan_executor.shutdown(wait=False, cancel_futures=True)
        comment_executor.shutdown(wait=False, cancel_futures=True)
        tts_executor.shutdown(wait=False, cancel_futures=True)
        speech.shutdown()
        chat.set_disable_auto_archive_on_close(True)
        chat.close()
        pet.close()
        if live2d_py_process is not None:
            try:
                live2d_py_process.terminate()
            except Exception:
                pass
        app.quit()

    HWND_TOPMOST = -1
    SWP_NOACTIVATE = 0x0010
    SWP_SHOWWINDOW = 0x0040
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_NOZORDER = 0x0004
    SW_RESTORE = 9
    live2d_hwnd_cache = 0
    last_input_diag_ts = 0.0
    input_diag_interval_sec = 20.0
    last_input_write_ts = 0.0
    last_input_signature = None
    last_live2d_relaunch_ts = 0.0
    last_live2d_applied_rect = None
    live2d_initial_rect_synced = False

    class _WinRect(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def _set_live2d_window_pos(
        user32,
        hwnd: int,
        x: int,
        y: int,
        w: int,
        h: int,
        insert_after: int = HWND_TOPMOST,
    ) -> None:
        user32.SetWindowPos(
            hwnd,
            insert_after,
            int(x),
            int(y),
            int(w),
            int(h),
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )

    def _write_json_state_file(state_path: Path, payload: dict) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(payload, ensure_ascii=True)

        # On Windows, os.replace may fail with WinError 5 when destination is briefly locked.
        # Use unique temp names + short retry; finally fall back to direct overwrite.
        last_exc: Exception | None = None
        for attempt in range(4):
            tmp_path = state_path.with_suffix(f".json.tmp.{os.getpid()}.{int(time.time() * 1000)}.{attempt}")
            try:
                tmp_path.write_text(body, encoding="utf-8")
                os.replace(tmp_path, state_path)
                return
            except Exception as exc:
                last_exc = exc
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                time.sleep(0.006 * (attempt + 1))

        try:
            state_path.write_text(body, encoding="utf-8")
            return
        except Exception as exc:
            last_exc = exc

        if last_exc is not None:
            raise last_exc

    def _write_live2d_target_rect(x: int, y: int, w: int, h: int) -> None:
        try:
            state_path = base_dir / "data" / "live2d_py_target_rect.json"
            payload = {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
            }
            _write_json_state_file(state_path, payload)
        except Exception:
            pass

    def _get_live2d_target_geometry_native(user32, host_hwnd: int) -> tuple[int, int, int, int]:
        # Use Win32 host rect as ground truth to avoid cross-monitor DPI drift.
        _x, _y, model_w_logical, model_h_logical = pet.get_live2d_py_target_geometry()
        logical_host_w = max(1, int(pet.width()))
        logical_host_h = max(1, int(pet.height()))

        rect = _WinRect()
        if host_hwnd and user32.GetWindowRect(host_hwnd, ctypes.byref(rect)):
            native_x = int(rect.left)
            native_y = int(rect.top)
            native_host_w = max(1, int(rect.right - rect.left))
            native_host_h = max(1, int(rect.bottom - rect.top))
            scale_x = float(native_host_w) / float(logical_host_w)
            scale_y = float(native_host_h) / float(logical_host_h)
            native_w = max(1, int(round(float(model_w_logical) * scale_x)))
            native_h = max(1, int(round(float(model_h_logical) * scale_y)))
            return native_x, native_y, native_w, native_h

        # Fallback if host rect temporarily unavailable.
        return int(_x), int(_y), max(1, int(model_w_logical)), max(1, int(model_h_logical))

    def _native_to_logical_geometry(x: int, y: int, w: int, h: int, user32, host_hwnd: int) -> tuple[int, int, int, int]:
        # Convert using host native/logical size ratio instead of global DPR.
        logical_host_w = max(1, int(pet.width()))
        logical_host_h = max(1, int(pet.height()))

        rect = _WinRect()
        if host_hwnd and user32.GetWindowRect(host_hwnd, ctypes.byref(rect)):
            native_host_w = max(1, int(rect.right - rect.left))
            native_host_h = max(1, int(rect.bottom - rect.top))
            inv_x = float(logical_host_w) / float(native_host_w)
            inv_y = float(logical_host_h) / float(native_host_h)
            return (
                int(round(float(x) * inv_x)),
                int(round(float(y) * inv_y)),
                max(1, int(round(float(w) * inv_x))),
                max(1, int(round(float(h) * inv_y))),
            )

        return int(x), int(y), max(1, int(w)), max(1, int(h))

    def _write_live2d_input_state(
        user32,
        x: int,
        y: int,
        w: int,
        h: int,
        follow_enabled: bool,
        drag_active: bool = False,
    ) -> None:
        nonlocal last_input_diag_ts, last_input_write_ts, last_input_signature
        try:
            scan_busy = bool(pet.scan_busy)
            now_ts = time.time()

            # When follow is disabled, stop polling mouse position/buttons to reduce OCR contention.
            # Keep a low-frequency heartbeat so runner can switch to non-follow mode immediately.
            if not follow_enabled:
                follow_off_keepalive = 3.0 if scan_busy else 1.8
                signature = (int(scan_busy), int(follow_enabled), int(drag_active), int(max(1, w)), int(max(1, h)))
                if signature == last_input_signature and (now_ts - last_input_write_ts) < follow_off_keepalive:
                    return

                payload = {
                    "ts": now_ts,
                    "x": int(max(0, min(max(1, w) - 1, w // 2))),
                    "y": int(max(0, min(max(1, h) - 1, h // 2))),
                    "w": int(max(1, w)),
                    "h": int(max(1, h)),
                    "inside": False,
                    "left_down": 0,
                    "right_down": 0,
                    "scan_busy": int(scan_busy),
                    "follow_enabled": 0,
                    "drag_active": int(bool(drag_active)),
                }
                state_path = base_dir / "data" / "live2d_py_input_state.json"
                _write_json_state_file(state_path, payload)
                last_input_write_ts = now_ts
                last_input_signature = signature
                return

            # Keep follow responsive even when OCR is busy; avoid near-stop behavior.
            if scan_busy and (now_ts - last_input_write_ts) < 0.12:
                return

            pt = wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(pt)):
                return
            local_x = int(pt.x - x)
            local_y = int(pt.y - y)
            inside = 0 <= local_x < max(1, w) and 0 <= local_y < max(1, h)
            local_x = max(0, min(max(1, w) - 1, local_x))
            local_y = max(0, min(max(1, h) - 1, local_y))

            left_down = 1 if (user32.GetAsyncKeyState(0x01) & 0x8000) else 0
            right_down = 1 if (user32.GetAsyncKeyState(0x02) & 0x8000) else 0
            min_interval = 0.12 if scan_busy else 0.06
            keepalive_interval = 0.7 if scan_busy else 0.6
            signature = (
                local_x,
                local_y,
                left_down,
                right_down,
                inside,
                int(scan_busy),
                int(follow_enabled),
                int(drag_active),
            )
            changed = signature != last_input_signature
            if (not changed and (now_ts - last_input_write_ts) < keepalive_interval) or (
                changed and (now_ts - last_input_write_ts) < min_interval
            ):
                return

            payload = {
                "ts": now_ts,
                "x": int(local_x),
                "y": int(local_y),
                "w": int(max(1, w)),
                "h": int(max(1, h)),
                "inside": bool(inside),
                "left_down": int(left_down),
                "right_down": int(right_down),
                "scan_busy": int(scan_busy),
                "follow_enabled": int(follow_enabled),
                "drag_active": int(bool(drag_active)),
            }
            state_path = base_dir / "data" / "live2d_py_input_state.json"
            _write_json_state_file(state_path, payload)
            last_input_write_ts = now_ts
            last_input_signature = signature

            if now_ts - last_input_diag_ts >= input_diag_interval_sec:
                print(
                    "[DIAG][MAIN][INPUT_WRITE]",
                    f"inside={inside}",
                    f"local=({local_x},{local_y})",
                    f"wh=({w},{h})",
                    f"left={left_down}",
                    f"right={right_down}",
                    f"scan_busy={scan_busy}",
                    f"follow={follow_enabled}",
                    f"drag={int(bool(drag_active))}",
                )
                last_input_diag_ts = now_ts
        except Exception as exc:
            now_ts = time.time()
            if now_ts - last_input_diag_ts >= input_diag_interval_sec:
                print(f"[DIAG][MAIN][INPUT_WRITE] failed: {exc}")
                last_input_diag_ts = now_ts

    def _is_cursor_near_rect(user32, x: int, y: int, w: int, h: int, threshold_px: int) -> bool:
        try:
            pt = wintypes.POINT()
            if not user32.GetCursorPos(ctypes.byref(pt)):
                return False
            left = int(x)
            top = int(y)
            right = int(x + max(1, w) - 1)
            bottom = int(y + max(1, h) - 1)
            dx = max(left - int(pt.x), 0, int(pt.x) - right)
            dy = max(top - int(pt.y), 0, int(pt.y) - bottom)
            return (dx * dx + dy * dy) <= int(threshold_px * threshold_px)
        except Exception:
            return False

    def _raise_topmost(user32, hwnd: int, above_hwnd: int = 0) -> None:
        if not hwnd:
            return
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
        if above_hwnd:
            user32.SetWindowPos(
                hwnd,
                above_hwnd,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )

    def _resolve_live2d_hwnd(user32) -> int:
        nonlocal live2d_hwnd_cache
        if live2d_hwnd_cache and user32.IsWindow(live2d_hwnd_cache):
            return int(live2d_hwnd_cache)

        pid = live2d_py_process.pid if live2d_py_process is not None else 0
        if pid <= 0:
            return 0

        # Preferred path: exact hwnd reported by renderer process.
        try:
            state_path = base_dir / "data" / "live2d_py_window_state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state_pid = int(state.get("pid", 0))
                state_hwnd = int(state.get("hwnd", 0))
                if state_pid == int(pid) and state_hwnd and user32.IsWindow(state_hwnd):
                    live2d_hwnd_cache = state_hwnd
                    return state_hwnd
        except Exception:
            pass

        hwnd_result = {"value": 0, "area": 0}
        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        rect_type = _WinRect

        def _enum_cb(hwnd, _lparam):
            proc_id = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
            if int(proc_id.value) != int(pid):
                return True
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.GetParent(hwnd):
                return True
            rect = rect_type()
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return True
            width = max(0, int(rect.right - rect.left))
            height = max(0, int(rect.bottom - rect.top))
            area = width * height
            if area <= hwnd_result["area"]:
                return True
            hwnd_result["value"] = int(hwnd)
            hwnd_result["area"] = area
            return True

        user32.EnumWindows(WNDENUMPROC(_enum_cb), 0)
        resolved = int(hwnd_result["value"])
        if resolved:
            live2d_hwnd_cache = resolved
        return resolved

    def _try_sync_live2d_py_windows() -> None:
        nonlocal live2d_py_process, live2d_py_retry_used, live2d_hwnd_cache, last_live2d_relaunch_ts
        nonlocal last_live2d_applied_rect
        nonlocal live2d_initial_rect_synced
        nonlocal runtime_follow_near, runtime_follow_effective, runtime_follow_hold_until_ts, runtime_drag_active
        nonlocal runtime_follow_near_on_streak, runtime_follow_near_off_streak
        nonlocal drag_lock_prev_active, drag_lock_last_rect, drag_lock_last_write_ts, last_zorder_sync_ts
        if not settings.enable_live2d_py:
            return
        if not hasattr(ctypes, "windll"):
            return
        user32 = ctypes.windll.user32
        host_hwnd = int(pet.winId())
        chat_hwnd = int(chat.winId()) if chat.isVisible() else 0
        exp_x, exp_y, exp_w, exp_h = _get_live2d_target_geometry_native(user32, host_hwnd)
        now_ts = time.time()
        runtime_drag_active = bool(pet.is_live2d_py_interacting())
        near_enter_raw = _is_cursor_near_rect(user32, exp_x, exp_y, exp_w, exp_h, follow_enter_distance_px)
        near_exit_raw = _is_cursor_near_rect(user32, exp_x, exp_y, exp_w, exp_h, follow_exit_distance_px)
        if near_enter_raw:
            runtime_follow_hold_until_ts = now_ts + follow_hold_sec
        raw_near = bool(near_enter_raw or near_exit_raw)
        if raw_near:
            runtime_follow_near_on_streak += 1
            runtime_follow_near_off_streak = 0
            if (not runtime_follow_near) and runtime_follow_near_on_streak >= follow_near_enter_frames:
                runtime_follow_near = True
        else:
            runtime_follow_near_off_streak += 1
            runtime_follow_near_on_streak = 0
            if runtime_follow_near and runtime_follow_near_off_streak >= follow_near_exit_frames:
                runtime_follow_near = False

        follow_gate_ok = bool(runtime_follow_near or (now_ts < runtime_follow_hold_until_ts))
        runtime_follow_effective = bool(pet.gaze_follow_enabled and follow_gate_ok and (not runtime_drag_active))
        target_rect = (int(exp_x), int(exp_y), int(exp_w), int(exp_h))

        # Drag lock: avoid cross-process z-order churn while dragging to prevent flicker/disappear.
        if runtime_drag_active:
            if (target_rect != drag_lock_last_rect) and (now_ts - drag_lock_last_write_ts >= 0.05):
                _write_live2d_target_rect(exp_x, exp_y, exp_w, exp_h)
                drag_lock_last_rect = target_rect
                drag_lock_last_write_ts = now_ts
            _write_live2d_input_state(user32, exp_x, exp_y, exp_w, exp_h, False, True)

            # Main process becomes the single geometry writer during drag to avoid cross-process jitter.
            hwnd_drag = _resolve_live2d_hwnd(user32)
            if hwnd_drag:
                try:
                    user32.SetWindowPos(
                        hwnd_drag,
                        0,
                        int(exp_x),
                        int(exp_y),
                        int(exp_w),
                        int(exp_h),
                        SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER,
                    )
                    last_live2d_applied_rect = target_rect
                except Exception:
                    pass
            _apply_runtime_resource_policy(
                scan_active=bool(pet.scan_busy),
                follow_active=False,
                drag_active=True,
                force=False,
            )
            drag_lock_prev_active = True
            return

        force_post_drag_resync = False
        if drag_lock_prev_active:
            force_post_drag_resync = True
            drag_lock_prev_active = False

        _write_live2d_target_rect(exp_x, exp_y, exp_w, exp_h)
        drag_lock_last_rect = target_rect
        drag_lock_last_write_ts = now_ts
        _write_live2d_input_state(user32, exp_x, exp_y, exp_w, exp_h, runtime_follow_effective, False)
        # Cursor-distance driven policy:
        # - near model => follow priority
        # - far from model => idle
        # This avoids tick-driven policy updates that can cause periodic UI stutter.
        _apply_runtime_resource_policy(
            scan_active=False,
            follow_active=bool(runtime_follow_near),
            drag_active=runtime_drag_active,
            force=False,
        )
        hwnd = _resolve_live2d_hwnd(user32)
        if not hwnd:
            process_dead = (live2d_py_process is None) or (live2d_py_process.poll() is not None)
            if process_dead and (now_ts - last_live2d_relaunch_ts >= 3.0):
                try:
                    live2d_py_process = start_live2d_py_process(force_gl_init=True)
                    live2d_py_retry_used = True
                    live2d_hwnd_cache = 0
                    last_live2d_relaunch_ts = now_ts
                    print(f"[STARTUP] Live2D-py auto relaunch started (pid={live2d_py_process.pid})")
                except Exception as exc:
                    print(f"[STARTUP] Live2D-py auto relaunch failed: {exc}")
            if host_hwnd:
                _raise_topmost(user32, host_hwnd)
            return

        try:
            if not user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetWindowPos(
                    hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
                )
        except Exception:
            pass

        # One-shot startup alignment: trust runner's actual rect and sync host to it.
        if not live2d_initial_rect_synced:
            try:
                rect = _WinRect()
                if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    rw = max(1, int(rect.right - rect.left))
                    rh = max(1, int(rect.bottom - rect.top))
                    if rw > 1 and rh > 1:
                        lx, ly, lw, lh = _native_to_logical_geometry(
                            int(rect.left),
                            int(rect.top),
                            rw,
                            rh,
                            user32,
                            host_hwnd,
                        )
                        pet.sync_from_live2d_py_rect(lx, ly, lw, lh)
                        live2d_initial_rect_synced = True
                        last_live2d_applied_rect = (
                            int(rect.left),
                            int(rect.top),
                            int(rw),
                            int(rh),
                        )
                        return
            except Exception:
                pass

        # Single geometry writer: always drive model rect from host side.
        if target_rect != last_live2d_applied_rect:
            try:
                user32.SetWindowPos(
                    hwnd,
                    0,
                    int(exp_x),
                    int(exp_y),
                    int(exp_w),
                    int(exp_h),
                    SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOZORDER,
                )
                last_live2d_applied_rect = target_rect
            except Exception:
                pass

        # Throttle z-order relock to avoid unnecessary flashing; force once after drag end.
        if force_post_drag_resync or (now_ts - last_zorder_sync_ts >= 0.45):
            _raise_topmost(user32, host_hwnd)
            user32.SetWindowPos(
                hwnd,
                host_hwnd or HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
            last_zorder_sync_ts = now_ts

    if state["scan_enabled"] and settings.enable_scan_subprocess:
        _start_scan_worker()

    scheduler = ScanScheduler(scan_tick_interval_sec)
    scheduler.tick.connect(do_auto_comment)
    if state["scan_enabled"]:
        scheduler.start()

    chat.comment_btn.clicked.disconnect()
    chat.comment_btn.clicked.connect(do_manual_comment)

    def on_new_chat_requested() -> None:
        if settings.enable_live2d_py:
            mx, my, mw, _mh = pet.get_live2d_py_target_geometry()
            target_x = max(20, int(mx - chat.width() - 14))
            target_y = max(20, int(my + 24))
            chat.move(target_x, target_y)
        chat.show_and_focus()

    pet.open_chat_requested.connect(on_new_chat_requested)
    pet.auto_comment_requested.connect(do_manual_comment)
    pet.auto_scan_toggled.connect(on_scan_toggle)
    pet.visual_novel_mode_toggled.connect(on_visual_novel_mode_toggled)
    pet.visual_novel_story_manager_requested.connect(on_visual_novel_story_manager_requested)
    pet.tts_mute_toggled.connect(on_tts_mute_toggled)
    pet.gaze_follow_toggled.connect(on_gaze_follow_toggled)
    pet.tutor_mode_toggled.connect(on_tutor_mode_toggled)
    pet.select_scan_region_requested.connect(on_select_scan_region)
    pet.clear_scan_region_requested.connect(on_clear_scan_region)
    pet.quit_requested.connect(on_quit_requested)

    pet.set_auto_scan_enabled(state["scan_enabled"])
    pet.set_visual_novel_mode_enabled(state["visual_novel_mode_enabled"])
    pet.set_visual_novel_story_name(visual_novel_library.active_name)
    pet.set_tts_muted(False)
    pet.set_gaze_follow_enabled(settings.live2d_follow_cursor)
    pet.set_tutor_mode_enabled(settings.enable_tutor_persona)

    pet.show()
    pet.raise_()
    pet.activateWindow()

    def emit_opening_greeting() -> None:
        greeting = dialog.build_opening_greeting().strip()
        if not greeting:
            return
        show_and_speak(greeting, role="桌宠")
        # Start a fresh auto-comment cycle after greeting so timing feels natural.
        state["cycle_memories"] = []
        state["next_comment_at"] = schedule_next_comment(datetime.now())
        log_heartbeat("schedule_reset", f"next_due={state['next_comment_at'].strftime('%H:%M:%S')}")

    QTimer.singleShot(1200, emit_opening_greeting)

    if settings.enable_live2d_py:
        live2d_ui_sync_timer = QTimer()
        live2d_ui_sync_timer.setInterval(90)
        live2d_ui_sync_timer.timeout.connect(_try_sync_live2d_py_windows)
        live2d_ui_sync_timer.start()
        QTimer.singleShot(500, _try_sync_live2d_py_windows)

    app.aboutToQuit.connect(speech.shutdown)
    if visual_novel_tracker is not None:
        app.aboutToQuit.connect(visual_novel_tracker.flush)
    app.aboutToQuit.connect(lambda: live2d_py_process and live2d_py_process.terminate())
    if settings.enable_scan_subprocess:
        app.aboutToQuit.connect(_stop_scan_worker)
    else:
        app.aboutToQuit.connect(lambda: scan_executor and scan_executor.shutdown(wait=False, cancel_futures=True))
    app.aboutToQuit.connect(lambda: comment_executor.shutdown(wait=False, cancel_futures=True))
    app.aboutToQuit.connect(lambda: tts_executor.shutdown(wait=False, cancel_futures=True))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
