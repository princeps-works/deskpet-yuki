"""Minimal test runner used when pytest is not installed.

Discovers ``test_*`` functions in the sibling test modules and runs them.
Only plain-``assert`` tests are supported, which is what this suite uses.

Usage:  python tests/_run_all.py
"""

from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PACKAGE_PARENT = _TESTS_DIR.parent.parent
sys.path.insert(0, str(_PACKAGE_PARENT))
sys.path.insert(0, str(_TESTS_DIR))

MODULES = [
    "test_policy",
    "test_call_site_consistency",
    "test_vn_scan_coupling",
    "test_vn_cache_binding",
    "test_vn_ocr_stability",
    "test_live2d_drag_geometry",
    "test_worker_thread_qt_safety",
    "test_scan_pipeline_locals",
    "test_text_change_detection",
    "test_vn_panel_story_switch",
    "test_vn_text_band_and_chrome",
    "test_vn_ui_mask",
    "test_persona_and_moment_tracks",
    "test_viewer_impressions",
    "test_comment_personalization",
    "test_voicevox_pid_cache",
    "test_speech_sync",
    "test_translation",
    "test_vision_trigger",
    "test_vn_improvements",
    "test_vn_capture",
    "test_capture",
    "test_vision",
    "test_vision_router",
    "test_semantic_attention",
    "test_semantic_comment_candidates",
    "test_memory_store",
    "test_visual_novel",
    "test_ocr_first_pipeline",
    "test_chat_pipeline",
    "test_dialog",
    "test_vn_menu_and_geometry",
    "test_vn_comment_latency",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    passed = 0
    failures: list[str] = []
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: import failed: {type(exc).__name__}: {exc}")
            print(f"[IMPORT-FAIL] {module_name}: {type(exc).__name__}: {exc}")
            continue
        names = [
            name
            for name, value in vars(module).items()
            if name.startswith("test_") and inspect.isfunction(value)
        ]
        for name in sorted(names):
            func = getattr(module, name)
            if inspect.signature(func).parameters:
                continue
            try:
                func()
                passed += 1
            except Exception as exc:
                failures.append(f"{module_name}::{name}: {type(exc).__name__}: {exc}")
                print(f"[FAIL] {module_name}::{name}")
                traceback.print_exc()
    print()
    print(f"passed={passed} failed={len(failures)}")
    for item in failures:
        print(" -", item)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
