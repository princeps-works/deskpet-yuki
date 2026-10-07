"""The VOICEVOX process scan must not run while a valid pid is cached.

Finding the engine means enumerating every process on the machine. Measured on
this machine: 386 processes, exe + cmdline each, **1.86s median** (max 1.94s).
It ran on the GUI thread from the Live2D sync loop (a 90ms Qt timer) and,
because the trigger included a flat 20s interval, it fired every 20s whether or
not the cached pid was still valid. That produced the periodic freeze:

    live2d on : 13 stalls >= 2s in 255 ticks (~1 per 20.5s)
    live2d off:  1 stall  >= 2s in 265 ticks

The same flag also made every policy reapply scan, so toggling auto-scan paid the
1.9s synchronously (`force=True`).

Now: the cached pid is validated on every call and the scan only runs as a
fallback when no pid is known. A restarted engine is still caught, because its
pid dies, the cache clears, and the scan runs on that same call.
"""

from __future__ import annotations

import ast
import pathlib

import desktop_pet.main as main_module

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SOURCE = (pathlib.Path(main_module.__file__)).read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_SPEECH = (_ROOT / "audio" / "speech.py").read_text(encoding="utf-8")


def _segment(name: str) -> str:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(_SOURCE, node) or ""
    raise AssertionError(f"function {name} not found")


def test_scan_is_gated_on_having_no_pid():
    source = _segment("_apply_runtime_resource_policy")
    assert "need_full_scan = voicevox_pid_cache <= 0 and (" in source, (
        "the scan must only run while no pid is cached"
    )
    # The flat 20s refresh is what made it fire even with a valid cache.
    assert (
        "need_full_scan = force or voicevox_pid_cache <= 0 or (now_ts - voicevox_policy_last_scan_ts"
        not in source
    )


def test_a_policy_reapply_no_longer_forces_the_scan():
    """``force=True`` is used by the auto-scan toggle and by quit."""
    source = _segment("_apply_runtime_resource_policy")
    scan_clause = source[source.index("need_full_scan") : source.index("if need_full_scan")]
    assert "force" not in scan_clause, (
        "force should reapply the policy, not re-enumerate every process"
    )


def test_engine_restart_still_triggers_a_rescan():
    """A dead pid must clear the cache AND allow the scan immediately."""
    source = _segment("_apply_runtime_resource_policy")
    assert source.count("voicevox_policy_last_scan_ts = 0.0") >= 2, (
        "both invalid-pid paths should release the interval so the rescan is immediate"
    )


def test_cached_pid_is_validated_every_call():
    source = _segment("_apply_runtime_resource_policy")
    assert "proc.is_running()" in source
    assert 'str(proc.name()).lower() == "run.exe"' in source
    assert "_apply_proc_policy_cached(int(voicevox_pid_cache), normal_cls, [])" in source, (
        "the feature the scan exists for must be preserved"
    )


def test_launcher_records_the_pid_it_started():
    assert "process = subprocess.Popen(" in _SPEECH, (
        "the launch must keep the Popen handle to read .pid"
    )
    assert "self._voicevox_engine_pid = int(getattr(process, \"pid\", 0) or 0)" in _SPEECH
    assert "_voicevox_engine_pid = 0" in _SPEECH, "and default it to 0"


def test_launcher_pid_is_exposed_and_consumed():
    assert "def voicevox_engine_pid(self) -> int:" in _SPEECH
    assert 'getattr(speech, "voicevox_engine_pid", 0)' in _SOURCE, (
        "the policy must adopt the pid we launched before falling back to a scan"
    )


def test_launcher_pid_is_initialised_before_the_runtime_probe():
    """``__init__`` probes the engine (and may launch it) at the end."""
    init = _SPEECH[_SPEECH.index("self._azure_synthesizer = None") :]
    init = init[: init.index("self._runtime_ok = self._compute_runtime_ok()")]
    assert "_voicevox_engine_pid = 0" in init, (
        "the field must exist before _compute_runtime_ok() can launch the engine"
    )
