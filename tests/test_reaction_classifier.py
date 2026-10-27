import ast
import json
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.reaction_classifier import LocalReactionClassifier
from desktop_pet.llm.reaction_classifier import selected_filter_model
from desktop_pet.llm.visual_novel import VisualNovelTracker


def test_active_rows_keep_original_age_and_do_not_cross_reload():
    rows = [
        {"id": 1, "branch": "old", "text": "旧分支"},
        {"id": 2, "branch": "new", "text": "已过期背景"},
        {"id": 6, "branch": "new", "text": "现在的反应点"},
        {"id": 7, "branch": "new", "text": "当前对白"},
    ]
    active, full = LocalReactionClassifier.input_pairs(rows)
    assert active == ("现在的反应点", "当前对白")
    assert full == ("已过期背景\n现在的反应点", "当前对白")


def test_missing_local_model_fails_closed_to_legacy_availability():
    with TemporaryDirectory() as tmp:
        classifier = LocalReactionClassifier(Path(tmp))
        assert classifier.warmup() is False
        assert classifier.status.startswith("unavailable:")
        assert classifier.classify([{"id": 1, "text": "测试"}]) is None


def test_model_selection_preserves_old_config_and_explicit_choice_wins():
    assert selected_filter_model(SimpleNamespace(enable_vn_reaction_classifier=False)) == "semantic"
    assert selected_filter_model(SimpleNamespace(enable_vn_reaction_classifier=True)) == "window-linear-v64"
    assert selected_filter_model(SimpleNamespace(enable_vn_reaction_classifier=False,
        vn_reaction_filter_model="target-gru-fusion-r8")) == "target-gru-fusion-r8"


def test_fusion_history_includes_only_displayed_comments_in_the_current_branch():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "story.json")
        tracker.enable_local_reactions(True)
        tracker.observe("燐「我们终于见面了。」")
        tracker.remember_comment("等了这么久，终于见到了。")
        tracker.observe("惠麻「我一直在这里等你。」")
        snapshot, _ = tracker.reserve_evaluation_snapshot()
        rows = snapshot._fusion_classifier_lines
        assert len(rows) == 3 and rows[1]["is_agent"]
        assert all(not r.get("is_agent") for r in snapshot._classifier_lines)
        messages = LocalReactionClassifier.fusion_messages(rows)
        assert [m["speaker"] for m in messages] == ["Speaker_0", "Assistant", "Speaker_1"]
        assert messages[0]["text"] == "我们终于见面了。"
        tracker.mark_reload("读档")
        tracker.observe("燐「这是读档后另一条路线。」")
        snapshot, _ = tracker.reserve_evaluation_snapshot()
        assert len(snapshot._fusion_classifier_lines) == 1


def test_stale_model_warmup_cannot_enable_the_new_selection():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_start_vn_reaction_warmup")
    work, delivered, enabled = [], [], []
    old = SimpleNamespace(warmup=lambda: True, status="ready:old")
    namespace = {"threading": SimpleNamespace(Thread=lambda target, **kw: SimpleNamespace(start=lambda: work.append(target))),
        "speech_ui": SimpleNamespace(ready=SimpleNamespace(emit=delivered.append)),
        "visual_novel_tracker": SimpleNamespace(enable_local_reactions=enabled.append),
        "vn_reaction_classifier": old}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<warmup-test>", "exec"), namespace)
    namespace[node.name](old)
    namespace["vn_reaction_classifier"] = None
    work.pop()()
    delivered.pop()()
    assert not enabled


def test_filter_switch_persists_selection_and_invalidates_pending_comment():
    from unittest.mock import patch
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
             for name in ("_discard_pending_vn_evaluation", "on_vn_filter_model_changed")]
    switch = nodes[1]
    switch.body = [ast.copy_location(ast.Global(names=n.names), n) if isinstance(n, ast.Nonlocal) else n
                   for n in switch.body]
    token, calls, warmed, enabled = object(), [], [], []
    future = SimpleNamespace(cancel=lambda: calls.append("cancel"))
    state = {"vn_cache_token": token, "pending_comment_meta": {"kind": "visual_novel"}, "comment_future": future}
    fresh = object()
    namespace = {"state": state, "vn_filter_model": "semantic", "vn_reaction_classifier": None,
        "FILTER_MODELS": {"target-gru-fusion-r8": "最佳模型"}, "base_dir": Path("unused"),
        "visual_novel_tracker": SimpleNamespace(enable_local_reactions=enabled.append),
        "_make_vn_classifier": lambda _: fresh, "_start_vn_reaction_warmup": warmed.append,
        "refresh_vn_sidebar_state": lambda: calls.append("refresh"),
        "chat": SimpleNamespace(append_message=lambda *args: calls.append("message"))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<switch-test>", "exec"), namespace)
    with patch("dotenv.set_key") as save:
        namespace[switch.name]("target-gru-fusion-r8")
        assert save.call_args.args[1:] == ("VN_REACTION_FILTER_MODEL", "target-gru-fusion-r8")
    assert namespace["vn_filter_model"] == "target-gru-fusion-r8"
    assert namespace["vn_reaction_classifier"] is fresh and warmed == [fresh]
    assert state["vn_cache_token"] is not token and state["pending_comment_meta"] is None
    assert state["comment_future"] is None and "cancel" in calls and enabled == [False]


def test_local_nomination_is_immediate_and_snapshot_uses_latest_branch():
    def forbidden(*args):
        raise AssertionError("Legacy semantic scoring must not run on this path")
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "story.json", similarity_fn=forbidden)
        tracker.enable_local_reactions(True)
        for i in range(6):
            assert tracker.observe(f"旧分支对白第{i}条").accepted
        tracker.mark_reload("读档")
        assert tracker.observe("新分支刚发生了一件事情。").should_evaluate
        snapshot, _ = tracker.reserve_evaluation_snapshot()
        assert len(snapshot._classifier_lines) == 1
        assert snapshot._classifier_lines[0]["text"] == "新分支刚发生了一件事情。"
        payload = snapshot.build_evaluation_payload()
        assert not payload["summary_due"]
        assert payload["reaction_line_ids"] == [tracker.total_lines]
        assert not tracker.has_pending_evaluation
        assert tracker.observe("紧接着出现了新的变化。").should_evaluate


def test_memory_only_local_silence_blocks_stream_and_final_comment():
    events = []
    raw = json.dumps({"should_comment": True, "moment_type": "humor",
                      "reaction_repeats": False, "comment": "不应该发出",
                      "facts": ["剧情记忆继续更新。"], "scene_summary": "新的摘要"})
    class Client:
        def chat_stream(self, user_text, system_prompt, on_text, **kwargs):
            assert "仅整理剧情记忆" in user_text
            on_text(raw)
            return raw
    result = CommentEngine(Client()).evaluate_visual_novel(
        {"local_reaction_class": 0}, on_reaction=events.append)
    assert not events and not result["should_comment"]
    assert result["facts"] == ["剧情记忆继续更新。"]
    assert result["scene_summary"] == "新的摘要"


def _worker():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "evaluate_snapshot")
    namespace = {"time": time, "datetime": datetime}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<worker-test>", "exec"), namespace)
    return namespace


def test_worker_silence_skips_remote_but_summary_due_keeps_memory_call():
    for total, expected_remote in [(3, False), (24, True)]:
        calls = []
        meta = {}
        snapshot = SimpleNamespace(
            _local_reaction_enabled=True, _classifier_lines=[{"id": total, "text": "普通安排"}],
            total_lines=total, _summarized_line_count=0, _summary_batch_size=24,
            build_evaluation_payload=lambda: {"evidence_line_ids": [total]})
        def remote(payload, callback):
            assert payload["local_reaction_class"] == 0
            calls.append(payload)
            return {"should_comment": False, "facts": ["记忆更新"]}
        ns = _worker()
        ns.update(state={"pending_comment_meta": meta}, meta=meta, snapshot=snapshot,
                  vn_reaction_classifier=SimpleNamespace(classify=lambda rows: {"class": 0, "model": "test"}),
                  log_heartbeat=lambda *args: None, remote_evaluation_due=True,
                  personalization_hint="", _current_visual_context=lambda: "",
                  comment_engine=SimpleNamespace(evaluate_visual_novel=remote), on_reaction=lambda _: None)
        result = ns["evaluate_snapshot"]()
        assert bool(calls) is expected_remote
        if expected_remote:
            assert result["facts"] == ["记忆更新"]
        else:
            assert result == {"local_filter_skipped": True}
            assert "vn_last_evaluation_at" not in ns["state"]


def test_deferred_candidate_does_not_reset_remote_clock():
    meta = {}
    ns = _worker()
    previous = datetime(2026, 1, 1)
    ns.update(state={"pending_comment_meta": meta, "vn_last_evaluation_at": previous},
              meta=meta, snapshot=SimpleNamespace(
                  _local_reaction_enabled=True, _classifier_lines=[],
                  total_lines=3, _summarized_line_count=0, _summary_batch_size=24),
              vn_reaction_classifier=SimpleNamespace(classify=lambda rows: {"class": 1, "model": "test"}),
              log_heartbeat=lambda *args: None, remote_evaluation_due=False)
    assert ns["evaluate_snapshot"]() == {"local_filter_deferred": True}
    assert ns["state"]["vn_last_evaluation_at"] is previous


def test_disabled_classifier_uses_legacy_remote_evaluation_without_local_gate():
    class DisconnectedClassifier:
        def classify(self, rows):
            raise AssertionError("Disabled local model must not be called")

    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "story.json")
        assert tracker._local_reaction_enabled is False
    meta = {}
    calls = []
    def remote(payload, callback):
        assert "local_reaction_class" not in payload
        calls.append(payload)
        return {"should_comment": True, "comment": "远程判断的评论"}
    ns = _worker()
    ns.update(state={"pending_comment_meta": meta}, meta=meta,
              snapshot=SimpleNamespace(_local_reaction_enabled=False,
                  build_evaluation_payload=lambda: {"evidence_line_ids": [1]}),
              vn_reaction_classifier=DisconnectedClassifier(),
              log_heartbeat=lambda *args: None, remote_evaluation_due=True,
              personalization_hint="", _current_visual_context=lambda: "",
              comment_engine=SimpleNamespace(evaluate_visual_novel=remote), on_reaction=lambda _: None)
    result = ns["evaluate_snapshot"]()
    assert len(calls) == 1 and result["should_comment"] is True
