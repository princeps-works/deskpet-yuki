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
