"""Guard against use-before-assignment in the scan pipeline.

``run_scan_pipeline`` reports ``ocr_info`` in the dict it returns, including from
the early "nothing changed, reuse the cache" returns near the top. That variable
was only assigned further down, so the first scan of every quiet scene -- which
is exactly when the cache-reuse branch fires -- raised ``UnboundLocalError:
local variable 'ocr_info' referenced before assignment``. The tick handler
catches it and logs a heartbeat ``error``, so the visible symptom was simply
"OCR never produces text" while the pet kept ticking and retrying every second.
"""

from __future__ import annotations

import ast
from pathlib import Path

import desktop_pet.main as main_module

_MAIN = Path(main_module.__file__)
_SOURCE = _MAIN.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _find_function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _first_load_line(func: ast.FunctionDef, name: str) -> int | None:
    """Line of the earliest read of ``name`` inside ``func``."""
    lines = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
    ]
    return min(lines) if lines else None


def _assigned_by_line(func: ast.FunctionDef, name: str, cutoff: int) -> bool:
    """True when ``name`` is assigned somewhere strictly before ``cutoff``."""
    for node in ast.walk(func):
        lineno = getattr(node, "lineno", None)
        if lineno is None or lineno >= cutoff:
            continue
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id == name:
                    return True
    return False


def _assignment_line(func: ast.FunctionDef, name: str) -> int | None:
    lines = []
    for node in ast.walk(func):
        lineno = getattr(node, "lineno", None)
        if lineno is None:
            continue
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if any(isinstance(sub, ast.Name) and sub.id == name for sub in ast.walk(target)):
                lines.append(lineno)
    return min(lines) if lines else None


def test_ocr_info_is_assigned_before_its_first_use():
    """The regression: the cache-reuse return read it first."""
    func = _find_function("run_scan_pipeline")
    first_use = _first_load_line(func, "ocr_info")
    assert first_use is not None, "expected run_scan_pipeline to report ocr_info"
    assert _assigned_by_line(func, "ocr_info", first_use), (
        f"ocr_info is first read on line {first_use} but assigned later; "
        "move the assignment above every early return"
    )


def test_band_sweep_result_is_assigned_before_its_first_use():
    """Same trap, second time: best_band was set only inside ``if run_sweep``.

    The guard that reads it sits outside that block, so any scan that skipped the
    sweep would have raised UnboundLocalError and aborted the whole cycle.
    """
    func = _find_function("run_scan_pipeline")
    first_use = _first_load_line(func, "best_band")
    assert first_use is not None, "expected run_scan_pipeline to use best_band"
    assert _assigned_by_line(func, "best_band", first_use), (
        f"best_band is first read on line {first_use} but assigned later; "
        "initialise it before the conditional that fills it"
    )


def test_cache_reuse_return_comes_after_the_ocr_status_lookup():
    """The cache-reuse return reads ocr_info, so it must not precede it."""
    func = _find_function("run_scan_pipeline")
    assigned_at = _assignment_line(func, "ocr_info")
    assert assigned_at is not None, "ocr_info is never assigned"

    returns_reading = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Return)
        and node.value is not None
        and any(
            isinstance(sub, ast.Name) and sub.id == "ocr_info" and isinstance(sub.ctx, ast.Load)
            for sub in ast.walk(node.value)
        )
    ]
    assert returns_reading, "expected a return that reports ocr_info"
    earliest_return = min(node.lineno for node in returns_reading)
    assert assigned_at < earliest_return, (
        f"ocr_info is assigned on line {assigned_at} but reported from a return "
        f"starting on line {earliest_return}"
    )


def test_reported_ocr_note_keys_are_covered():
    """The result contract still mentions ocr_info where callers read it."""
    func = _find_function("run_scan_pipeline")
    keys = {
        key.value
        for node in ast.walk(func)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "ocr_info" in keys
    assert "ocr_note" in keys
