import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import numpy as np

from desktop_pet.llm.dialog_manager import DialogManager
from desktop_pet.llm.memory_store import MemoryStore
from desktop_pet.llm.semantic_attention import SemanticAttentionRouter
from desktop_pet.llm.visual_novel import VisualNovelTracker


def diary(text, scope="user"):
    return {"summary": text, "scope": scope, "freshness": "stable", "source_type": "user"}


def fact(value):
    return {"subject": "角色", "predicate": "身份", "object": value, "policy": "replace"}


def test_scope_filters_before_candidate_limit_and_before_embedding():
    class Encoder:
        status = "ready"
        def __init__(self): self.seen = []
        def encode(self, texts):
            self.seen.extend(texts)
            assert not any("禁止越界" in text for text in texts)
            return np.array([[1., 0.] for _ in texts])

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        tracker = VisualNovelTracker(Path(tmp) / "a.json")
        encoder = Encoder()
        manager = DialogManager(None, path, semantic_attention=SemanticAttentionRouter(
            enabled=True, model_path=Path(tmp), encoder=encoder),
            visual_novel_context_provider=tracker.retrieve_for_query)
        scope = manager._current_memory_scope()
        allowed = manager._memory_store.add_diary(diary("当前作品线索", scope))
        for i in range(90):
            manager._memory_store.add_diary(diary(f"禁止越界的作品线索{i}", "story:other"))
        assert [r["id"] for r in manager._memory_store.candidates("作品线索", scopes=(scope,), limit=1)] == [allowed]
        assert manager._rank_memory_candidates("作品线索", top_k=1)[0]["id"] == allowed
        assert "当前作品线索" in encoder.seen


def test_fact_replacement_and_evidence_do_not_cross_scopes():
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.json")
        store.add_diary(diary("角色是人", "story:a"), [fact("人")], source_transcript="A 的原话")
        store.add_diary(diary("角色是狗", "story:b"), [fact("狗")], source_transcript="B 的原话")
        a = next(r for r in store.candidates("角色", scopes=("story:a",)) if r["kind"] == "fact")
        assert a["object"] == "人"
        assert store.read_evidence(a["id"], scopes=("story:b",))["status"] == "not_found"
        assert store.read_evidence(a["id"], scopes=("story:a",))["sources"][0]["transcript"] == "A 的原话"
        assert store.candidates("角色", scopes=()) == []


def test_archive_revision_is_checked_atomically_and_retry_is_idempotent():
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.json")
        old = store.prepare_archive("旧原话", "user", source_key="turn-1")
        new = store.prepare_archive("纠正后的原话", "user", source_key="turn-1")
        assert store.add_diary(diary("旧摘要"), [fact("旧值")], source_transcript="旧原话", archive=old) is None
        written = store.add_diary(diary("新摘要"), [fact("新值")], source_transcript="纠正后的原话", archive=new)
        assert written
        assert store.add_diary(diary("重复摘要"), source_transcript="纠正后的原话", archive=new) == written
        assert len(store.list_diaries()) == 1
        reopened = MemoryStore(store.legacy_path)
        assert reopened.add_diary(diary("晚到的旧摘要"), source_transcript="旧原话", archive=old) is None


def test_revising_committed_source_hides_derived_memory_and_preferences():
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.json")
        old = store.prepare_archive("你: 旧偏好", "user", source_key="turn")
        memory_id = store.add_diary({**diary("旧摘要"), "personalization": [{"topic": "偏好", "preference": "旧偏好"}]},
                                   [fact("旧值")], source_transcript="你: 旧偏好", archive=old)
        store.prepare_archive("你: 新偏好", "user", source_key="turn")
        assert store.candidates("旧") == []
        assert store.read_evidence(memory_id)["status"] == "not_found"
        assert store.personalizations(scopes=("user",)) == []
        assert store.list_diaries()[0]["summary"] == "旧摘要"


def test_switching_story_while_archiving_keeps_original_scope():
    started, release = Event(), Event()
    class Client:
        def chat(self, **kwargs):
            started.set()
            assert release.wait(5)
            return json.dumps({"summary": "A 的剧情记忆，角色在车站相遇。", "facts": [fact("旅人")]})

    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "a.json")
        manager = DialogManager(Client(), Path(tmp) / "memory.json", visual_novel_context_provider=tracker.retrieve_for_query)
        a_scope = manager._current_memory_scope()
        manager.record_session_message("你", "角色在车站相遇")
        transcript = manager.pop_current_session_transcript()
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(manager.archive_transcript, transcript)
            assert started.wait(5)
            tracker.switch_story(Path(tmp) / "b.json")
            manager.start_new_chat()
            release.set()
            assert future.result(timeout=5)
        assert manager._memory_store.candidates("角色", scopes=(a_scope,))
        assert not manager._rank_memory_candidates("角色", top_k=3)


def test_source_revision_during_llm_call_prevents_all_archive_writes():
    started, release = Event(), Event()
    class Client:
        def chat(self, **kwargs):
            started.set()
            assert release.wait(5)
            return json.dumps({"summary": "旧偏好摘要", "facts": [fact("旧值")],
                               "personalization": [{"topic": "评论", "preference": "长评论", "evidence_quote": "我希望评论很长"}]})

    with TemporaryDirectory() as tmp:
        manager = DialogManager(Client(), Path(tmp) / "memory.json")
        transcript = "你: 我希望评论很长"
        snapshot = manager._memory_store.prepare_archive(transcript, "user", source_key="edit")
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(manager.archive_transcript, transcript, archive=snapshot)
            assert started.wait(5)
            manager._memory_store.prepare_archive("你: 改为短评", "user", source_key="edit")
            release.set()
            assert future.result(timeout=5) == ""
        assert manager._memory_store.list_diaries() == []
        assert manager._load_personalization() == []


def test_mixed_story_session_is_split_before_summary_generation():
    class Client:
        def __init__(self): self.inputs = []
        def chat(self, user_text, **kwargs):
            self.inputs.append(user_text)
            assert not ("A剧情" in user_text and "B剧情" in user_text)
            return json.dumps({"summary": user_text, "facts": []})

    with TemporaryDirectory() as tmp:
        client = Client()
        tracker = VisualNovelTracker(Path(tmp) / "a.json")
        manager = DialogManager(client, Path(tmp) / "memory.json", visual_novel_context_provider=tracker.retrieve_for_query)
        a_scope = manager._current_memory_scope()
        manager.record_session_message("你", "A剧情在车站发生")
        tracker.switch_story(Path(tmp) / "b.json")
        b_scope = manager._current_memory_scope()
        manager.record_session_message("你", "B剧情在山洞发生")
        snapshot = manager.pop_archive_snapshot()
        tracker.switch_story(Path(tmp) / "c.json")  # Switch before the background worker even starts.
        assert manager.archive_transcript(**snapshot)
        assert len(client.inputs) == 2
        assert {r["scope"] for r in manager._memory_store.list_diaries()} == {a_scope, b_scope}
        assert manager._rank_memory_candidates("剧情", top_k=5) == []


def test_user_memory_is_shared_but_same_named_works_are_distinct():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        tracker = VisualNovelTracker(root / "first" / "同名.json")
        manager = DialogManager(None, root / "memory.json", visual_novel_context_provider=tracker.retrieve_for_query)
        manager._memory_store.add_diary(diary("用户喜欢草莓"))
        manager.record_session_message("你", "旧作品的秘密线索")
        manager._memory_store.add_diary(diary("旧作品的秘密线索", manager._current_memory_scope()))
        tracker.switch_story(root / "second" / "同名.json")
        assert not manager._build_recent_session_block()
        assert not manager._build_session_recall_block("秘密线索")
        assert all("旧作品" not in r["summary"] for r in manager._rank_memory_candidates("秘密线索", top_k=3))
        assert manager._rank_memory_candidates("草莓", top_k=1)[0]["summary"] == "用户喜欢草莓"


def test_invalidated_source_cannot_return_via_raw_conversation_history():
    class Client:
        def chat(self, **kwargs):
            return json.dumps({"summary": "用户说了旧计划"})

    with TemporaryDirectory() as tmp:
        manager = DialogManager(Client(), Path(tmp) / "memory.json")
        manager.record_session_message("你", "旧计划是明天去车站")
        snapshot = manager.pop_archive_snapshot()
        key = snapshot["evidence"]["batches"][0]["archive"]["key"]
        manager.archive_transcript(**snapshot)
        manager._memory_store.invalidate_archive(key)
        assert manager._build_session_recall_block("旧计划车站") == ""
        assert manager._rank_memory_candidates("旧计划", top_k=3) == []
        assert manager._session_archive.rows()  # Evidence is preserved, not erased.


def test_out_of_order_duplicate_archive_commits_only_once():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        first = MemoryStore(path)
        second = MemoryStore(path)
        snapshot = first.prepare_archive("相同原文", "user", source_key="same")
        def write(store):
            return store.add_diary(diary("同一份摘要"), [fact("值")], source_transcript="相同原文", archive=snapshot)
        with ThreadPoolExecutor(2) as executor:
            ids = list(executor.map(write, (first, second)))
        assert ids[0] == ids[1]
        assert len(first.list_diaries()) == 1


def test_lifecycle_is_rechecked_after_semantic_encoding():
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.json")
        snapshot = store.prepare_archive("旧原话", "user", source_key="pending")
        store.add_diary(diary("旧知识"), source_transcript="旧原话", archive=snapshot)
        class Encoder:
            status = "ready"
            def encode(self, texts):
                if "旧知识" in texts:
                    store.invalidate_archive("pending")
                return np.asarray([[1., 0.] for _ in texts])
        manager = DialogManager(None, store.legacy_path, semantic_attention=SemanticAttentionRouter(
            enabled=True, model_path=Path(tmp), encoder=Encoder()))
        assert manager._rank_memory_candidates("旧知识", top_k=3) == []


def test_source_and_preferences_are_scoped_and_survive_reopen():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        manager = DialogManager(None, path)
        store = manager._memory_store
        snapshot = store.prepare_archive("你: 只在A里多评论", "story:a")
        memory_id = store.add_diary({**diary("A偏好", "story:a"), "personalization": [
            {"topic": "评论", "preference": "多评论", "evidence_quote": "只在A里多评论"}]},
            source_transcript=snapshot["transcript"], archive=snapshot)
        reopened = MemoryStore(path)
        assert reopened.personalizations(scopes=("user",)) == []
        assert reopened.personalizations(scopes=("story:a",))[0]["preference"] == "多评论"
        assert reopened.read_evidence(memory_id, scopes=("story:a",))["sources"][0]["revision"] == 1
        assert reopened.read_evidence(memory_id, scopes=("story:b",))["status"] == "not_found"


def test_explicit_global_preference_shares_only_verified_quote_and_retracts():
    class Client:
        def chat(self, **kwargs):
            return json.dumps({"summary": "作品里有秘密剧情，用户希望所有游戏都少剧透", "personalization": [
                {"topic": "剧透", "preference": "所有游戏都不要剧透", "evidence_quote": "所有游戏都不要剧透", "scope": "user"}]})

    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "a.json")
        manager = DialogManager(Client(), Path(tmp) / "memory.json", visual_novel_context_provider=tracker.retrieve_for_query)
        manager.record_session_message("你", "所有游戏都不要剧透")
        snapshot = manager.pop_archive_snapshot()
        key = snapshot["evidence"]["batches"][0]["archive"]["key"]
        manager.archive_transcript(**snapshot)
        tracker.switch_story(Path(tmp) / "b.json")
        assert "所有游戏都不要剧透" in manager.build_personalization_hint()
        assert "秘密剧情" not in manager._build_long_memory_block("剧透")
        manager._memory_store.invalidate_archive(key)
        assert manager.build_personalization_hint() == ""


def test_ranked_memory_survives_second_context_selection():
    from types import SimpleNamespace
    class Attention:
        def embeddings(self, texts): return None
        def route(self, **kwargs):
            return SimpleNamespace(contexts={}, debug="discarded_all")
    class Client:
        def chat(self, user_text, **kwargs):
            self.prompt = user_text
            return "ok"
    with TemporaryDirectory() as tmp:
        client = Client()
        manager = DialogManager(client, Path(tmp) / "memory.json", semantic_attention=Attention())
        manager._memory_store.add_diary(diary("用户喜欢草莓"))
        manager.reply("我喜欢草莓吗？")
        assert "用户喜欢草莓" in client.prompt


def test_legacy_json_without_scope_does_not_enter_named_work():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"
        path.write_text(json.dumps([{"summary": "未知作品中的角色身份", "source_type": "user"}],
                                   ensure_ascii=False), encoding="utf-8")
        store = MemoryStore(path)
        assert store.list_diaries()[0]["scope"] == "legacy"
        assert store.candidates("角色", scopes=("user", "story:current")) == []
        assert store.candidates("角色", scopes=("user", "legacy"))
