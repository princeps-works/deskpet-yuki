import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from desktop_pet.llm.dialog_manager import DialogManager
from desktop_pet.llm.memory_store import MemoryStore
from desktop_pet.llm.semantic_attention import SemanticAttentionRouter


class _NoopClient:
    def chat(self, user_text: str, system_prompt: str) -> str:
        return "ok"


class _ArchiveClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, user_text: str, system_prompt: str) -> str:
        self.calls += 1
        return json.dumps(
            {
                "summary": "今天哥哥告诉我他开始玩《樱云》，还特意提醒我不要剧透。我会记住这项长期偏好，陪他慢慢阅读故事。",
                "topics": ["樱云", "避免剧透"],
                "freshness": "stable",
                "facts": [
                    {
                        "subject": "哥哥",
                        "predicate": "正在玩",
                        "object": "樱云",
                        "slot": "current_state",
                        "policy": "replace",
                        "importance": 0.75,
                        "confidence": 0.96,
                        "freshness": "volatile",
                        "pinned": False,
                    },
                    {
                        "subject": "哥哥",
                        "predicate": "不喜欢",
                        "object": "剧情剧透",
                        "slot": "user_profile",
                        "policy": "append",
                        "importance": 0.95,
                        "confidence": 0.98,
                        "freshness": "stable",
                        "pinned": True,
                    },
                ],
            },
            ensure_ascii=False,
        )


class _CountingEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def status(self) -> str:
        return "ready"

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "魔裁" in text else [0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _diary(summary: str, timestamp: str = "2026-08-28T12:00:00+08:00") -> dict[str, object]:
    return {
        "schema_version": 3,
        "timestamp": timestamp,
        "created_at": timestamp,
        "summary": summary,
        "source_type": "user",
        "sources": [{"type": "user", "confidence": 0.95}],
        "confidence": 0.95,
        "freshness": "stable",
        "expires_at": "",
        "topics": [],
    }


def test_plot_history_does_not_fill_diary_or_generic_recall_and_is_retained():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        store = MemoryStore(path, diary_limit=20)
        for i in range(20):
            store.add_diary(_diary(f"和哥哥的一轮交流-{i}"))
        for i in range(25):
            plot = {**_diary(f"《作品》剧情进展-{i}"), "source_type": "screen_plot"}
            store.add_diary(plot, [{"subject": "作品", "predicate": "发生", "object": f"剧情事件-{i}", "slot": "knowledge"}])
        assert len(store.list_diaries()) == 20
        assert len(json.loads(path.read_text(encoding="utf-8"))) == 20
        assert all(row["source_type"] != "screen_plot" for row in store.candidates("作品剧情事件"))
        with store._connect() as connection:
            assert connection.execute("SELECT COUNT(*) FROM memory_items WHERE source_type='screen_plot'").fetchone()[0] == 50
        connection.close()
        store.add_diary(_diary("新一轮交流总结"))
        assert len(store.list_diaries()) == 20
        assert store.list_diaries()[-1]["summary"] == "新一轮交流总结"
        assert store.list_diaries()[0]["summary"] == "和哥哥的一轮交流-1"
        # SQLite transaction contexts do not close connections on Windows.
        import gc
        gc.collect()


def test_chat_still_reads_active_story_after_plot_diaries_are_excluded():
    from desktop_pet.llm.visual_novel import VisualNovelTracker
    with TemporaryDirectory() as directory:
        path = Path(directory)
        tracker = VisualNovelTracker(path / "作品A.json", similarities_fn=lambda q, ts: [.8] * len(ts))
        tracker.apply_evaluation({"scene_summary": "萤与焕被树木墙挡住去路。"})
        class Client:
            def chat(self, user_text, system_prompt):
                self.prompt = user_text
                return "测试响应"
        client = Client()
        manager = DialogManager(client, memory_path=path / "diary.json",
                                visual_novel_context_provider=tracker.retrieve_for_query)
        manager.append_plot_memory(story_title="旧作品", digest="无关作品的结局旧记录", facts=["已经通关旧作品。"])
        manager.reply("刚才的剧情发生了什么？")
        assert "视觉小说历史缓存" in client.prompt and "树木墙" in client.prompt
        assert "结局旧记录" not in client.prompt
        assert manager.list_long_memory() == []
        other = VisualNovelTracker(path / "作品B.json")
        other.apply_evaluation({"scene_summary": "小狗在公园里找到了主人。"})
        tracker.switch_story(other.path)
        manager.reply("当前作品的剧情是什么？")
        assert "小狗" in client.prompt and "树木墙" not in client.prompt
        import gc
        gc.collect()


def test_legacy_json_is_migrated_without_being_deleted():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "memory.json"
        path.write_text(json.dumps([_diary("哥哥喜欢草莓")], ensure_ascii=False), encoding="utf-8")

        store = MemoryStore(path)

        assert path.exists()
        assert path.with_suffix(".sqlite3").exists()
        assert store.list_diaries()[0]["summary"] == "哥哥喜欢草莓"


def test_hybrid_retrieval_can_recall_memory_older_than_last_twelve_chunks():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "memory.json"
        entries = [_diary("魔裁的主要角色包括樱羽艾玛和二阶堂希罗")]
        entries.extend(_diary(f"无关的日常记录-{index}") for index in range(140))
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        manager = DialogManager(_NoopClient(), memory_path=path)

        context, debug = manager._retrieve_long_memory_for_query("魔裁的主要角色有哪些？")

        assert "樱羽艾玛" in context
        assert "hybrid_sqlite" in debug


def test_expired_memory_is_not_recalled():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "memory.json"
        expired = _diary("魔裁的旧错误人物信息")
        expired["freshness"] = "volatile"
        expired["expires_at"] = "2020-01-01T00:00:00+08:00"
        path.write_text(json.dumps([expired], ensure_ascii=False), encoding="utf-8")
        manager = DialogManager(_NoopClient(), memory_path=path)

        context, _ = manager._retrieve_long_memory_for_query("魔裁的人物信息")

        assert context == ""


def test_replace_and_pinned_conflict_rules_keep_one_active_value():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "memory.json"
        store = MemoryStore(path)
        store.add_diary(
            _diary("哥哥开始玩樱云"),
            [{"subject": "哥哥", "predicate": "正在玩", "object": "樱云", "slot": "current_state"}],
        )
        store.add_diary(
            _diary("哥哥开始玩魔裁"),
            [{"subject": "哥哥", "predicate": "正在玩", "object": "魔裁", "slot": "current_state"}],
        )
        active = [entry for entry in store.candidates("正在玩") if entry["kind"] == "fact"]
        assert [entry["object"] for entry in active] == ["魔裁"]

        store.add_diary(
            _diary("哥哥明确要求固定无剧透偏好"),
            [{
                "subject": "哥哥",
                "predicate": "允许剧透",
                "object": "否",
                "slot": "user_profile",
                "policy": "pinned",
                "pinned": True,
            }],
        )
        store.add_diary(
            _diary("一次识别误以为可以剧透"),
            [{
                "subject": "哥哥",
                "predicate": "允许剧透",
                "object": "是",
                "slot": "user_profile",
                "policy": "replace",
            }],
        )
        pinned = [
            entry
            for entry in store.candidates("允许剧透")
            if entry["kind"] == "fact" and entry["predicate"] == "允许剧透"
        ]
        assert [entry["object"] for entry in pinned] == ["否"]


def test_archive_extracts_facts_in_same_llm_call_and_hides_them_from_diary_ui():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "memory.json"
        client = _ArchiveClient()
        manager = DialogManager(client, memory_path=path)

        manager.archive_transcript("你: 我开始玩樱云了，而且不要给我剧透。")

        assert client.calls == 1
        facts = [entry for entry in manager._memory_store.candidates("樱云 剧透") if entry["kind"] == "fact"]
        assert {entry["slot"] for entry in facts} == {"current_state", "user_profile"}
        assert next(entry for entry in facts if entry["slot"] == "current_state")["expires_at"]
        assert set(manager.list_long_memory(limit=1)[0]) == {"timestamp", "summary"}
        assert "facts" not in json.loads(path.read_text(encoding="utf-8"))[-1]


def test_memory_embeddings_are_reused_after_restart():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "memory.json"
        path.write_text(json.dumps([_diary("魔裁的主要人物是樱羽艾玛")], ensure_ascii=False), encoding="utf-8")
        first_encoder = _CountingEncoder()
        first_router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            encoder=first_encoder,
        )
        first = DialogManager(_NoopClient(), memory_path=path, semantic_attention=first_router)
        assert first._rank_memory_candidates("魔裁人物", top_k=3)
        assert any(len(call) >= 1 and "樱羽艾玛" in "".join(call) for call in first_encoder.calls)

        second_encoder = _CountingEncoder()
        second_router = SemanticAttentionRouter(
            enabled=True,
            model_path=Path(temp_dir),
            encoder=second_encoder,
        )
        second = DialogManager(_NoopClient(), memory_path=path, semantic_attention=second_router)
        assert second._rank_memory_candidates("魔裁人物", top_k=3)

        assert second_encoder.calls == [["魔裁人物"]]


def test_semantic_recall_finds_old_memory_without_keyword_overlap():
    class Encoder(_CountingEncoder):
        def encode(self, texts):
            self.calls.append(list(texts))
            return np.asarray([[1., 0.] if text in {
                "幽暗走廊让我发抖，希望你多说几句", "惊悚作品陪伴方式"
            } else [0., 1.] for text in texts], dtype=np.float32)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        entries = [_diary("幽暗走廊让我发抖，希望你多说几句")]
        entries.extend(_diary(f"例行记录-{i}") for i in range(150))
        path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        encoder = Encoder()
        router = SemanticAttentionRouter(enabled=True, model_path=Path(directory), encoder=encoder)
        manager = DialogManager(_NoopClient(), path, semantic_attention=router)
        recalled = manager._rank_memory_candidates("惊悚作品陪伴方式", top_k=3)
        assert recalled and recalled[0]["summary"] == entries[0]["summary"]
        assert max(map(len, encoder.calls)) <= 32
        next_encoder = Encoder()
        restarted = DialogManager(_NoopClient(), path, semantic_attention=SemanticAttentionRouter(
            enabled=True, model_path=Path(directory), encoder=next_encoder))
        assert restarted._rank_memory_candidates("惊悚作品陪伴方式", top_k=3)[0]["summary"] == entries[0]["summary"]
        assert next_encoder.calls == [["惊悚作品陪伴方式"]]


def test_superseded_fact_cannot_return_through_any_parent_summary():
    with TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "memory.json")
        fact = {"subject": "用户", "predicate": "夜间语音", "object": "开启", "policy": "replace"}
        old = store.add_diary(_diary("夜间语音开启，喜欢听桌宠说话"), [fact])
        duplicate = store.add_diary(_diary("再次确认夜间语音开启"), [fact])
        store.add_diary(_diary("夜间语音关闭"), [{**fact, "object": "关闭"}])
        recalled = store.candidates("夜间语音开启")
        assert old not in {r["id"] for r in recalled}
        assert duplicate not in {r["id"] for r in recalled}
        assert not any("开启" in r["summary"] for r in recalled)
        assert len(store.list_diaries()) == 3  # Historical diary remains readable.


def test_source_transcript_is_exact_and_survives_diary_retention():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        store = MemoryStore(path, diary_limit=20)
        transcript = "你: 记住这句话：\n  我喜欢安静。\n桌宠: 好。"
        store.add_diary(_diary("用户喜欢安静"), [
            {"subject": "用户", "predicate": "喜欢", "object": "安静"}
        ], source_transcript=transcript)
        fact = next(r for r in store.candidates("安静") if r["kind"] == "fact")
        for i in range(20):
            store.add_diary(_diary(f"后续日记{i}"))
        restarted = MemoryStore(path, diary_limit=20)
        evidence = restarted.read_evidence(fact["id"])
        assert evidence["status"] == "success"
        assert evidence["sources"][0]["transcript"] == transcript


def test_expired_sibling_blocks_full_source_but_keeps_live_fact():
    with TemporaryDirectory() as directory:
        store = MemoryStore(Path(directory) / "memory.json")
        diary = store.add_diary(_diary("用户喜欢安静，临时状态已过期"), [
            {"subject": "用户", "predicate": "喜欢", "object": "安静"},
            {"subject": "用户", "predicate": "临时状态", "object": "旧值", "expires_at": "2020-01-01"},
        ], source_transcript="你: 喜欢安静，临时状态是旧值。")
        rows = store.candidates("安静")
        assert diary not in {r["id"] for r in rows}
        fact = next(r for r in rows if r["kind"] == "fact")
        evidence = store.read_evidence(fact["id"])
        assert evidence["status"] == "success"
        assert evidence["sources"] == []
        assert evidence["source_status"] == "lifecycle_limited"


def test_legacy_database_migration_filters_stale_diary_without_inventing_source():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        store = MemoryStore(path)
        old = store.add_diary(_diary("旧语音开启"), [
            {"subject": "用户", "predicate": "语音", "object": "开启", "policy": "replace"}])
        store.add_diary(_diary("新语音关闭"), [
            {"subject": "用户", "predicate": "语音", "object": "关闭", "policy": "replace"}])
        with store._connect() as db:
            db.execute("DROP TABLE memory_source_links")
            db.execute("DROP TABLE memory_sources")
        db.close()
        migrated = MemoryStore(path)
        rows = migrated.candidates("语音")
        assert old not in {r["id"] for r in rows}
        fact = next(r for r in rows if r["kind"] == "fact")
        assert migrated.read_evidence(fact["id"])["source_status"] == "unavailable"
        assert migrated.read_evidence(old)["status"] == "lifecycle_limited"
        assert len(migrated.list_diaries()) == 2


def test_archive_preserves_source_and_current_memory_reaches_final_prompt():
    class Client(_ArchiveClient):
        def chat(self, user_text, system_prompt):
            self.prompt = user_text
            if "JSON" in system_prompt:
                return super().chat(user_text, system_prompt)
            return "ok"

    with TemporaryDirectory() as directory:
        client = Client()
        manager = DialogManager(client, Path(directory) / "memory.json")
        transcript = "你: 我开始玩樱云了，而且不要给我剧透。"
        manager.archive_transcript(transcript)
        fact = next(r for r in manager._memory_store.candidates("剧透") if r["kind"] == "fact")
        assert manager._memory_store.read_evidence(fact["id"])["sources"][0]["transcript"] == transcript
        store = manager._memory_store
        old_fact = {"subject": "用户", "predicate": "夜间语音", "object": "开启", "policy": "replace"}
        store.add_diary(_diary("夜间语音开启"), [old_fact])
        store.add_diary(_diary("夜间语音关闭"), [{**old_fact, "object": "关闭"}])
        manager.reply("我的夜间语音偏好是什么？")
        assert "夜间语音关闭" in client.prompt and "夜间语音开启" not in client.prompt
        assert "memory_id=" in client.prompt
        assert "夜间语音开启" not in manager._build_long_memory_block()
        manager.build_opening_greeting()
        assert "夜间语音开启" not in client.prompt


def test_unavailable_encoder_keeps_lexical_recall_and_lifecycle_filter():
    class Encoder:
        status = "unavailable"
        def encode(self, texts):
            raise RuntimeError("offline encoder")

    with TemporaryDirectory() as directory:
        manager = DialogManager(_NoopClient(), Path(directory) / "memory.json",
                                semantic_attention=SemanticAttentionRouter(
                                    enabled=True, model_path=Path(directory), encoder=Encoder()))
        manager._memory_store.add_diary(_diary("用户喜欢草莓"))
        expired = manager._memory_store.add_diary({**_diary("用户喜欢草莓旧记录"), "expires_at": "2020-01-01"})
        rows = manager._rank_memory_candidates("草莓", top_k=3)
        assert rows and rows[0]["summary"] == "用户喜欢草莓"
        assert expired not in {r["id"] for r in rows}


def test_recent_records_do_not_displace_better_old_semantic_match():
    class Encoder(_CountingEncoder):
        def encode(self, texts):
            return np.asarray([[1., 0.] if text == "question" else
                               [.6, .8] if text == "旧约定" else
                               [.58, np.sqrt(1 - .58 ** 2)] for text in texts], dtype=np.float32)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "memory.json"
        path.write_text(json.dumps([_diary("旧约定"), *[_diary(f"新记录{i}") for i in range(100)]],
                                   ensure_ascii=False), encoding="utf-8")
        manager = DialogManager(_NoopClient(), path, semantic_attention=SemanticAttentionRouter(
            enabled=True, model_path=Path(directory), encoder=Encoder()))
        assert manager._rank_memory_candidates("question", top_k=1)[0]["summary"] == "旧约定"
