"""Consistency tests for functions with multiple parallel call sites.

``_resolve_vn_capture`` is called from three places (main scan pipeline,
subprocess scan pipeline, sidebar OCR probe). A signature change that updates
only some of them fails at runtime inside a background worker, where the error
surfaces only as a scan reset -- which is exactly how a stale ``manual_region=``
call site survived a refactor. This test fails loudly instead.
"""

from __future__ import annotations

import ast
import inspect
import queue
from pathlib import Path

import desktop_pet.main as main_module

_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "main.py"


def _accepted_keywords(func) -> set[str]:
    signature = inspect.signature(func)
    names: set[str] = set()
    for name, parameter in signature.parameters.items():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(name)
    return names


def test_resolve_vn_capture_call_sites_match_signature():
    accepted = _accepted_keywords(main_module._resolve_vn_capture)
    tree = ast.parse(_PROJECT_ROOT.read_text(encoding="utf-8"), filename=str(_PROJECT_ROOT))

    call_sites = 0
    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_target = (isinstance(func, ast.Name) and func.id == "_resolve_vn_capture") or (
            isinstance(func, ast.Attribute) and func.attr == "_resolve_vn_capture"
        )
        if not is_target:
            continue
        call_sites += 1
        for keyword in node.keywords:
            if keyword.arg is None:  # **kwargs
                continue
            if keyword.arg not in accepted:
                problems.append(f"line {node.lineno}: unexpected keyword '{keyword.arg}'")

    assert call_sites >= 3, f"expected at least 3 call sites, found {call_sites}"
    assert not problems, "; ".join(problems)


def test_resolve_vn_capture_requires_ocr_region_not_manual_region():
    parameters = inspect.signature(main_module._resolve_vn_capture).parameters
    assert "ocr_region" in parameters
    assert "manual_region" not in parameters
    # vision_region was dropped when the vision frame became the whole window.
    assert "full_frame" not in parameters


def test_capture_rect_carries_window_title():
    from desktop_pet.vision.game_window import CaptureRect

    rect = CaptureRect(10, 20, 800, 600, 1, "window", "My Game")
    assert rect.title == "My Game"
    # Transformations must preserve the title so logs stay meaningful.
    assert rect.inset(0.1, 0.1).title == "My Game"
    assert rect.expand(5).title == "My Game"
    assert rect.text_box(0.5).title == "My Game"
    clamped = rect.clamp_to((0, 0, 1920, 1080))
    assert clamped is not None and clamped.title == "My Game"


def test_resolver_rejects_own_ui_titles():
    from desktop_pet.vision.game_window import GameWindowResolver

    for title in ("Live2D-py", "视觉小说侧栏", "桌宠聊天", "长期记忆日记"):
        assert GameWindowResolver._is_own_ui_title(title), title
    for title in ("Sakura no Uta", "记事本", "Notepad"):
        assert not GameWindowResolver._is_own_ui_title(title), title


def test_resolver_rejects_codex_as_a_game_target():
    from desktop_pet.vision.game_window import GameWindowResolver

    assert GameWindowResolver._is_unlikely_target("Codex")
    assert GameWindowResolver._is_unlikely_process("Codex.exe")


def test_scan_worker_forwards_runtime_capture_state():
    tasks: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    tasks.put(
        {
            "task_id": 7,
            "scan_monitor_index": 1,
            "visual_novel_mode": True,
            "vision_region": [10, 20, 300, 200],
            "capture_mode": "manual",
            "text_ratio": 0.67,
            "target_title": "My Visual Novel",
        }
    )
    tasks.put(None)
    captured: dict = {}
    original = main_module.run_scan_pipeline_subprocess

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return {"mode": "ocr"}

    main_module.run_scan_pipeline_subprocess = fake_pipeline
    try:
        main_module.run_scan_pipeline_worker_loop(tasks, results)
    finally:
        main_module.run_scan_pipeline_subprocess = original

    assert captured["vision_region"] == (10, 20, 300, 200)
    assert captured["capture_mode"] == "manual"
    assert captured["text_ratio"] == 0.67
    assert captured["target_title"] == "My Visual Novel"
    assert results.get_nowait()["ok"] is True


def test_subprocess_scan_returns_capture_metadata_on_every_path():
    tree = ast.parse(_PROJECT_ROOT.read_text(encoding="utf-8"), filename=str(_PROJECT_ROOT))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_scan_pipeline_subprocess"
    )
    required = {
        "capture_source",
        "capture_title",
        "capture_process_name",
        "capture_hwnd",
        "capture_warning",
        "ocr_chars",
        "ocr_note",
    }
    result_dicts = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert len(result_dicts) >= 3
    for result in result_dicts:
        keys = {
            key.value
            for key in result.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert required <= keys, f"missing result metadata: {sorted(required - keys)}"


def test_scan_worker_prewarms_before_accepting_tasks():
    tasks: queue.Queue = queue.Queue()
    results: queue.Queue = queue.Queue()
    tasks.put(None)
    calls: list[str] = []

    class _Ready:
        def set(self):
            calls.append("ready")

    original = main_module._warm_scan_worker_ocr
    main_module._warm_scan_worker_ocr = lambda: calls.append("warmup")
    try:
        main_module.run_scan_pipeline_worker_loop(tasks, results, _Ready())
    finally:
        main_module._warm_scan_worker_ocr = original

    assert calls == ["warmup", "ready"]


def test_subprocess_vn_scan_uses_the_fast_text_path():
    tree = ast.parse(_PROJECT_ROOT.read_text(encoding="utf-8"), filename=str(_PROJECT_ROOT))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_scan_pipeline_subprocess"
    )
    source = ast.get_source_segment(_PROJECT_ROOT.read_text(encoding="utf-8"), function) or ""
    assert "ocr_text_max_edge" in source
    assert "extract_ocr_result(text_image)" in source
    assert "_ocr_text_variants(text_image)" not in source
    assert "for index, variant in enumerate(_ocr_text_variants(text_image))" not in source


def test_vn_scan_rechecks_pending_text_and_paces_static_frames():
    source_text = _PROJECT_ROOT.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(_PROJECT_ROOT))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "do_auto_comment"
    )
    source = ast.get_source_segment(source_text, function) or ""
    assert "_process_visual_novel_text(result, datetime.now())" in source
    assert "_submit_scan_worker_task" in source
    assert "visual_novel_tracker.has_pending_ocr" in source
    assert 'state["vn_scan_due_at"]' in source
    assert "delay_sec = 0.2" in source
    assert "else 1.0" in source
