from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from desktop_pet.llm.dialog_manager import DialogManager
from desktop_pet.llm.memory_store import MemoryStore
from desktop_pet.llm.visual_novel import VisualNovelTracker


class Client:
    def chat(self, user_text, system_prompt):
        self.prompt = user_text
        return "测试回复"


def test_reply_preserves_long_user_input_including_final_correction():
    with TemporaryDirectory() as tmp:
        client = Client()
        mgr = DialogManager(client, Path(tmp) / "memory.json")
        message = "此前发生了很多事情。" * 80 + "\n纠正：我问的是妹妹，请不要剧透。"
        mgr.reply(message)
        assert message in client.prompt


def test_fact_freshness_and_expiry_do_not_inherit_diary():
    diary = {"freshness": "volatile", "expires_at": "2020-01-01T00:00:00+08:00"}
    base = {"subject": "用户", "predicate": "生日", "object": "5月2日"}
    for freshness in ("stable", "unknown", None):
        fact = dict(base)
        if freshness is not None:
            fact["freshness"] = freshness
        normalized = MemoryStore._normalize_fact(fact, diary)
        assert normalized["freshness"] == (freshness or "unknown")
        assert normalized["expires_at"] == ""
    volatile = MemoryStore._normalize_fact({**base, "freshness": "volatile"}, diary)
    assert volatile["expires_at"] > diary["expires_at"]
    explicit = "2030-01-01T00:00:00+08:00"
    dated = MemoryStore._normalize_fact({**base, "freshness": "volatile", "expires_at": explicit}, diary)
    assert dated["expires_at"] == explicit


def test_fact_dedup_preserves_negation_and_state_changes():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "story.json",
                                     similarities_fn=lambda query, texts: [.99] * len(texts))
        facts = [
            "小明已经答应明天陪小红一起去车站接她的哥哥。",
            "小明没有答应明天陪小红一起去车站接她的哥哥。",
            "小明后来答应明天陪小红一起去车站接她的哥哥。",
        ]
        tracker.apply_evaluation({"facts": facts})
        assert tracker.facts == facts
        repeated = {"facts": [facts[0]]}
        tracker.apply_evaluation(repeated)
        assert repeated["added_facts"] == []
        assert tracker.facts == facts
        corrected = "小明答应的是后天陪小红一起去车站接她的哥哥。"
        tracker.apply_evaluation({"fact_updates": [{"existing": facts[0], "replacement": corrected}]})
        assert tracker.facts == [corrected, *facts[1:]]


def test_retention_keeps_active_facts_after_parent_diary_is_trimmed():
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.json", diary_limit=20)
        store.add_diary({"summary": "早期偏好", "source_type": "user"}, [
            {"subject": "用户", "predicate": "偏好", "object": "短评", "pinned": True},
            {"subject": "用户", "predicate": "约定", "object": "周末讨论"},
        ])
        for i in range(20):
            store.add_diary({"summary": f"后续对话{i}", "source_type": "user"})
        facts = [r for r in store.candidates("短评") if r["kind"] == "fact"]
        assert {r["object"] for r in facts} == {"短评", "周末讨论"}
        assert len(store.list_diaries()) == 20


def test_personal_question_does_not_inject_story_or_unrelated_recall():
    with TemporaryDirectory() as tmp:
        client = Client()
        tracker = VisualNovelTracker(Path(tmp) / 'story.json')
        tracker.apply_evaluation({'scene_summary': '狒狒正在搜查。'})
        mgr = DialogManager(client, Path(tmp) / "memory.json",
                            visual_novel_context_provider=tracker.retrieve_for_query)
        mgr._memory_store.add_diary({"summary": "仪式传承", "source_type": "user"})
        mgr.record_session_message("你", "刚才的剧情发生了什么？")
        mgr.reply("换个话题，我以前告诉过你我的生日是哪天吗？")
        assert "狒狒" not in client.prompt
        assert "仪式传承" not in client.prompt
        assert "未找到" in client.prompt


def test_story_identity_survives_attention_content_selection():
    class Attention:
        def route(self, **kwargs):
            return SimpleNamespace(contexts={"novel": "剧情摘要: 两人躲藏。"}, debug="test")
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "加缪.json")
        tracker.apply_evaluation({"scene_summary": "两人躲藏。"})
        client = Client()
        mgr = DialogManager(client, Path(tmp) / "memory.json", semantic_attention=Attention(),
                            visual_novel_context_provider=tracker.retrieve_for_query)
        mgr.reply("刚才加缪的剧情发生了什么？")
        assert "当前作品: 《加缪》" in client.prompt
        assert "两人躲藏" in client.prompt


def test_character_evidence_uses_evaluation_snapshot_not_newer_lines():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "story.json")
        tracker._lines = [{"id": i, "text": str(i)} for i in range(1, 13)]
        tracker._total_lines = 12
        tracker.apply_evaluation({"evidence_line_ids": [1, 2, 3],
                                  "character_updates": [{"name": "甲", "confidence": .9}]})
        char = next(iter(tracker.characters.values()))
        assert char["evidence_line_ids"] == [1, 2, 3]
        assert char["last_seen_line"] == 3


def test_low_confidence_rename_does_not_inherit_confirmed_identity():
    with TemporaryDirectory() as tmp:
        tracker = VisualNovelTracker(Path(tmp) / "story.json")
        tracker.apply_evaluation({"character_updates": [{"name": "甲", "confidence": .95}]})
        cid = next(iter(tracker.characters))
        tracker.apply_evaluation({"character_updates": [{"id": cid, "name": "乙", "confidence": .2}]})
        char = tracker.characters[cid]
        assert char["name"] == "甲"
        assert "乙" not in char["aliases"]


def test_repeated_comment_preference_merges_topic_alias_and_correction():
    with TemporaryDirectory() as tmp:
        mgr = DialogManager(Client(), Path(tmp) / "memory.json")
        for topic, preference in [("游戏评论风格", "评论简短直接"), ("游戏评论", "评论简短直接"),
                                  ("评论方式", "评论可以详细一些")]:
            mgr._record_personalization([{"topic": topic, "preference": preference,
                                          "evidence_quote": preference}], "你: " + preference)
        prefs = mgr._load_personalization()
        assert len(prefs) == 1
        assert prefs[0]["preference"] == "评论可以详细一些"


def test_user_profile_subject_aliases_do_not_prevent_replacement():
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp) / "memory.json")
        for subject, value in [("哥哥", "简短"), ("user", "详细")]:
            store.add_diary({"summary": "用户明确修改偏好", "source_type": "user"}, [
                {"subject": subject, "predicate": "评论风格", "object": value,
                 "slot": "user_profile", "policy": "replace"},
            ])
        facts = [r for r in store.candidates("评论") if r["kind"] == "fact"]
        assert [(r["subject"], r["object"]) for r in facts] == [("用户", "详细")]


def test_personal_character_opinion_still_reads_story_context():
    with TemporaryDirectory() as tmp:
        client = Client()
        mgr = DialogManager(client, Path(tmp) / "memory.json",
                            visual_novel_context_provider=lambda q: "剧情摘要: 萤守夜。")
        mgr.reply("你记得我对萤这个角色的看法吗？")
        assert "萤守夜" in client.prompt


def test_chat_carries_displayed_vn_comments_into_reload_followup():
    with TemporaryDirectory() as tmp:
        client = Client()
        queries = []

        def story_context(query):
            queries.append(query)
            return "当前作品: 《加缪》\n剧情摘要: 焕试图救萤。"

        mgr = DialogManager(client, Path(tmp) / "memory.json",
                            visual_novel_context_provider=story_context)
        mgr.record_session_message("桌宠(视觉小说)", "焕还站不稳，我想看他把萤拉回来。")
        mgr.record_session_message("桌宠(自动)", "无关桌面通知")
        mgr.reply("刚才be了，我又重新读档了")
        assert "焕还站不稳" in queries[0]
        assert "焕还站不稳" in client.prompt
        assert "无关桌面通知" in client.prompt
        assert "当前作品: 《加缪》" in client.prompt


def test_auto_comments_are_recent_context_without_recall_keywords():
    with TemporaryDirectory() as tmp:
        client = Client()
        mgr = DialogManager(client, Path(tmp) / "memory.json")
        mgr.record_session_message("你", "我们继续看吧。")
        for i in range(10):
            mgr.record_session_message("桌宠(视觉小说)", f"评论{i}：希望她能平安离开。")
        mgr.reply("嗯，我也是这么想的。")
        assert "评论9" in client.prompt
        assert "评论0" not in client.prompt
        assert "我们继续看吧" in client.prompt


def test_semantic_recall_finds_older_comments_beyond_recent_window():
    class Attention:
        def similarities(self, query, texts):
            self.count = len(texts)
            return [.9 if "那封信" in text else .1 for text in texts]

        def route(self, **kwargs):
            return SimpleNamespace(contexts=kwargs["contexts"], debug="test")

    with TemporaryDirectory() as tmp:
        client = Client()
        attention = Attention()
        mgr = DialogManager(client, Path(tmp) / "memory.json", semantic_attention=attention)
        mgr.record_session_message("桌宠(视觉小说)", "那封信原来是他留的，我之前误会他了。")
        for i in range(40):
            mgr.record_session_message("桌宠(视觉小说)", f"经过第{i}间房屋。")
        mgr.reply("你对寄信人的看法变了吗？")
        assert attention.count == 38
        assert "我之前误会他了" in client.prompt
        assert "经过第39间房屋" in client.prompt
        assert "经过第0间房屋" not in client.prompt
        assert "与本轮相关的较早会话" in client.prompt
