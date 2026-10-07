"""Regression tests for the visual-novel comment, memory and capture changes."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_pet.config.prompts import (
    get_system_screen_comment_prompt,
    get_system_visual_novel_prompt,
)
from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.visual_novel import (
    DedupCalibrator,
    VisualNovelTracker,
    _ngram_jaccard,
)
from desktop_pet.vision.game_window import CaptureRect, GameWindowResolver


def _tracker(tmp: str, **kwargs) -> VisualNovelTracker:
    return VisualNovelTracker(Path(tmp) / "story.json", **kwargs)


# --- fact de-duplication ---------------------------------------------------


def test_ngram_jaccard_separates_related_from_duplicate():
    duplicate = _ngram_jaccard(
        "李梅在双龙馆找到了疑似目标人物，对方害怕被抓且情绪不稳定",
        "李梅在双龙馆找到疑似目标人物，对方害怕且情绪不稳定",
    )
    distinct = _ngram_jaccard(
        "司与所长前往伏仓家，伏仓不在",
        "司与所长前往伏仓家，找到万斋",
    )
    unrelated = _ngram_jaccard(
        "伏仓骗过所长后逃走，几天未归",
        "司与所长前往伏仓家，找到万斋",
    )
    assert duplicate >= 0.75
    assert distinct < 0.75
    assert unrelated < 0.45


def test_fact_dedup_merges_duplicates_and_keeps_distinct():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, dedup_ngram_merge=0.75, dedup_ngram_guard=0.45)
        tracker.apply_evaluation(
            {
                "scene_delta": "场景一",
                "facts": [
                    "万斋否认制作时光机并试图逃跑",
                    "李梅在双龙馆找到疑似目标人物，对方害怕且情绪不稳定",
                ],
            }
        )
        assert len(tracker.facts) == 2

        # Same fact, punctuation variation: must merge rather than append.
        tracker.apply_evaluation(
            {"scene_delta": "场景二", "facts": ["万斋否认制作时光机并试图逃跑。"]}
        )
        assert len(tracker.facts) == 2

        # Different fact: must survive.
        tracker.apply_evaluation(
            {"scene_delta": "场景三", "facts": ["卢仓骗过所长后逃走，几天未归"]}
        )
        assert len(tracker.facts) == 3
        assert len(tracker.scene_log) == 3


def test_apply_evaluation_reports_only_new_facts():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        result = {"facts": ["万斋否认制作时光机并试图逃跑", "李梅在双龙馆找到疑似目标人物"]}
        tracker.apply_evaluation(result)
        assert len(result["added_facts"]) == 2

        repeat = {"facts": ["万斋否认制作时光机并试图逃跑"]}
        tracker.apply_evaluation(repeat)
        assert repeat["added_facts"] == []


def test_dedup_calibration_is_recorded_and_optional():
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "cal.jsonl"
        tracker = _tracker(tmp, calibrator=DedupCalibrator(log_path, enabled=True))
        tracker.apply_evaluation({"facts": ["事实一", "事实二"]})
        assert log_path.is_file()
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2
        assert '"decision"' in lines[0]
        assert '"top_matches"' in lines[1]
        assert '"matched": "事实一"' in lines[1]

        off_path = Path(tmp) / "off.jsonl"
        disabled = _tracker(tmp, calibrator=DedupCalibrator(off_path, enabled=False))
        disabled.apply_evaluation({"facts": ["事实三"]})
        assert not off_path.exists()


def test_semantic_dedup_runs_without_high_ngram_prefilter():
    calls: list[tuple[str, list[str]]] = []

    def similarities(query: str, texts: list[str]) -> list[float]:
        calls.append((query, list(texts)))
        return [0.85 for _ in texts]

    with TemporaryDirectory() as tmp:
        tracker = _tracker(
            tmp,
            similarities_fn=similarities,
            dedup_semantic_merge=0.82,
            dedup_ngram_guard=0.12,
        )
        original = "手机收到工作通知：导演吩咐分镜制作暂停、制作流程有变，请确认，暗示叙述者参与分镜制作"
        rewritten = "叙述者以动画师身份参与分镜制作，受导演指示，工作被喊停（分镜暂停、流程变更）"
        tracker.apply_evaluation({"facts": [original]})
        result = {"facts": [rewritten]}
        tracker.apply_evaluation(result)

        assert len(tracker.facts) == 1
        assert result["added_facts"] == []
        assert calls and calls[-1][1] == [original]


def test_semantic_dedup_requires_lexical_support():
    def high_similarity(_query: str, texts: list[str]) -> list[float]:
        return [0.90 for _ in texts]

    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, similarities_fn=high_similarity, dedup_semantic_merge=0.82)
        tracker.apply_evaluation({"facts": ["晓在第二节车厢遇见一名金发女性"]})
        tracker.apply_evaluation({"facts": ["蒸汽机车完全停稳大约需要五分钟"]})
        assert len(tracker.facts) == 2


def test_fact_context_combines_semantic_relevance_and_recent_tail():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, summary_batch_size=8)
        facts = [f"普通剧情事实{i}" for i in range(16)] + [
            "万斋曾经否认自己制作时光机",
            "最近事实一",
            "最近事实二",
            "最近事实三",
            "最近事实四",
        ]
        tracker.apply_evaluation({"facts": facts})

        def similarities(_query: str, texts: list[str]) -> list[float]:
            return [0.95 if "万斋" in text else 0.10 for text in texts]

        tracker._similarities_fn = similarities
        for index in range(8):
            tracker.observe(f"众人再次讨论万斋和时光机，第{index}句")
        payload = tracker.build_evaluation_payload()

        assert payload is not None
        assert "万斋曾经否认自己制作时光机" in payload["facts"]
        assert payload["facts"][-4:] == ["最近事实一", "最近事实二", "最近事实三", "最近事实四"]
        assert sum(len(item) for item in payload["facts"]) <= 760
        assert payload["story_title"] == "story"


def test_newer_cumulative_summary_may_be_shorter():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        tracker.apply_evaluation({"scene_summary": "一条足够长的主线摘要，用于确认后续更短的摘要不会把它覆盖掉"})
        tracker.apply_evaluation({"scene_summary": "更准确的压缩摘要"})
        assert tracker.scene_summary == "更准确的压缩摘要"


def test_explicit_fact_revision_can_replace_with_shorter_truth():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        old = "晓上车后确认乘客只有自己一人，整列列车似乎空无一人"
        new = "晓所在车厢无人，其他车厢已有乘客"
        tracker.apply_evaluation({"facts": [old]})
        result = {"fact_updates": [{"existing": old, "replacement": new}]}
        tracker.apply_evaluation(result)
        assert tracker.facts == [new]
        assert result["updated_facts"] == [new]


def test_visual_novel_prompt_carries_title_atomic_facts_and_revisions():
    class RevisionClient:
        def __init__(self) -> None:
            self.prompt = ""

        def chat(self, user_text: str, system_prompt: str) -> str:
            self.prompt = user_text
            return (
                '{"should_comment":false,"moment_type":"ordinary","comment":"",'
                '"scene_delta":"","scene_summary":"晓正在乘坐银河号",'
                '"facts":["晓所在车厢空荡"],'
                '"fact_updates":[{"existing":"列车上只有晓一人",'
                '"replacement":"晓所在车厢无人，其他车厢已有乘客"}],'
                '"character_updates":[{"id":"char_001","name":"钟城晓",'
                '"aliases":["晓"],"role":"当前视角人物","confidence":0.95}]}'
            )

    client = RevisionClient()
    result = CommentEngine(client).evaluate_visual_novel(
        {
            "story_title": "星白",
            "scene_summary": "晓登上列车",
            "facts": ["列车上只有晓一人"],
            "characters": [{"id": "char_001", "name": "钟城晓", "aliases": ["晓"]}],
            "recent_dialogue": "其他车厢已经有乘客了。",
        }
    )
    assert "当前作品: 星白" in client.prompt
    assert "每条只描述一个事件、关系或状态" in client.prompt
    assert result["fact_updates"] == [
        {"existing": "列车上只有晓一人", "replacement": "晓所在车厢无人，其他车厢已有乘客"}
    ]
    assert result["character_updates"][0]["id"] == "char_001"


def test_character_table_persists_and_reveals_anonymous_identity_by_id():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        tracker.apply_evaluation(
            {
                "character_updates": [
                    {
                        "id": "",
                        "name": "",
                        "aliases": ["紫发和服女性"],
                        "role": "银河号乘客",
                        "appearance": ["紫发盘发", "和服风服装"],
                        "known_so_far": ["会抽烟"],
                        "confidence": 0.65,
                    }
                ]
            }
        )
        character_id = next(iter(tracker.characters))
        assert tracker.characters[character_id]["status"] == "provisional"

        reloaded = _tracker(tmp)
        reloaded.apply_evaluation(
            {
                "character_updates": [
                    {
                        "id": character_id,
                        "name": "紫苑",
                        "aliases": ["紫发和服女性"],
                        "role": "银河号乘客",
                        "confidence": 0.92,
                    }
                ]
            }
        )
        assert list(reloaded.characters) == [character_id]
        assert reloaded.characters[character_id]["name"] == "紫苑"
        assert reloaded.characters[character_id]["status"] == "confirmed"
        assert "会抽烟" in reloaded.characters[character_id]["known_so_far"]


def test_named_character_updates_merge_without_creating_a_second_record():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        tracker.apply_evaluation(
            {"character_updates": [{"name": "钟城晓", "aliases": ["晓"], "confidence": 0.9}]}
        )
        tracker.apply_evaluation(
            {
                "character_updates": [
                    {
                        "name": "钟城晓",
                        "aliases": ["叙述者"],
                        "known_so_far": ["参与动画分镜制作"],
                        "confidence": 0.95,
                    }
                ]
            }
        )
        assert len(tracker.characters) == 1
        character = next(iter(tracker.characters.values()))
        assert character["aliases"] == ["晓", "叙述者"]
        assert character["known_so_far"] == ["参与动画分镜制作"]


def test_character_context_is_relevant_and_budgeted():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, summary_batch_size=8)
        for index in range(7):
            tracker.apply_evaluation(
                {
                    "character_updates": [
                        {
                            "name": f"角色{index}",
                            "aliases": [f"别名{index}"],
                            "role": "银河号乘客",
                            "known_so_far": ["拥有一段用于测试上下文预算的人物信息"],
                            "confidence": 0.8,
                        }
                    ]
                }
            )
        for index in range(8):
            tracker.observe(f"角色0正在说明自己的经历，第{index}句")
        payload = tracker.build_evaluation_payload()
        assert payload is not None
        assert any(item["name"] == "角色0" for item in payload["characters"])
        assert len(payload["characters"]) <= 5
        assert sum(len(__import__("json").dumps(item, ensure_ascii=False)) for item in payload["characters"]) <= 420


def test_character_name_is_retrievable_without_semantic_encoder():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        tracker.apply_evaluation(
            {
                "character_updates": [
                    {
                        "name": "钟城晓",
                        "aliases": ["晓", "叙述者"],
                        "role": "当前视角人物",
                        "known_so_far": ["参与动画分镜制作"],
                        "confidence": 0.95,
                    }
                ]
            }
        )
        context = tracker.retrieve_for_query("钟城晓是谁？")
        assert "钟城晓" in context
        assert "当前视角人物" in context


def test_scene_log_is_bounded():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, scene_log_limit=3)
        for index in range(6):
            tracker.apply_evaluation({"scene_delta": f"场景片段{index}"})
        assert len(tracker.scene_log) == 3
        assert tracker.scene_log[-1] == "场景片段5"


def test_default_scene_log_keeps_more_than_the_old_eight_entries():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        for index in range(30):
            tracker.apply_evaluation({"scene_delta": f"场景片段{index}"})
        assert len(tracker.scene_log) == 30


# --- evaluation lifecycle --------------------------------------------------


def test_evaluation_cursor_can_be_restored():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, summary_batch_size=8)
        for index in range(8):
            tracker.observe(f"台词第{index}句")
        assert tracker.has_pending_evaluation
        payload = tracker.build_evaluation_payload()
        assert payload is not None and payload["line_count"] == 8
        assert not tracker.has_pending_evaluation

        tracker.restore_evaluation_cursor()
        assert tracker.has_pending_evaluation
        assert tracker.build_evaluation_payload() is not None


def test_payload_exposes_scene_log_and_recent_comments():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, summary_batch_size=8)
        for index in range(8):
            tracker.observe(f"台词第{index}句")
        tracker.apply_evaluation({"scene_delta": "上一段剧情", "facts": ["事实一"]})
        tracker.remember_comment("上一次说过的话")
        for index in range(8):
            tracker.observe(f"后续台词第{index}句")
        payload = tracker.build_evaluation_payload()
        assert payload is not None
        for field in ("scene_summary", "scene_log", "facts", "characters", "recent_comments", "recent_dialogue"):
            assert field in payload
        assert payload["recent_comments"] == ["上一次说过的话"]
        assert payload["scene_log"] == ["上一段剧情"]


def test_recent_comments_survive_reload():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp, recent_comment_limit=6)
        tracker.remember_comment("哥哥，这里好有意思")
        tracker.flush()
        reloaded = _tracker(tmp, recent_comment_limit=6)
        assert reloaded.recent_comments == ["哥哥，这里好有意思"]


def test_observe_tracks_acceptance_for_vision_gate():
    with TemporaryDirectory() as tmp:
        tracker = _tracker(tmp)
        assert tracker.last_observation_accepted is False
        tracker.observe("第一句台词内容")
        assert tracker.last_observation_accepted is True
        # Duplicate lines are rejected and must not look like story progress.
        tracker.observe("第一句台词内容")
        assert tracker.last_observation_accepted is False


# --- capture rects ---------------------------------------------------------


def test_capture_rect_geometry():
    rect = CaptureRect(100, 200, 1600, 900, 1, "window")
    assert rect.as_region() == (100, 200, 1600, 900)
    assert rect.clamp_to((0, 0, 1920, 1080)) is not None
    assert rect.clamp_to((0, 0, 400, 300)) is None

    text = rect.text_box(0.58)
    assert text.top > rect.top and text.height < rect.height
    assert rect.inset(0.1, 0.1).width < rect.width
    assert rect.expand(20).width == rect.width + 40
    assert rect.expand(0) is rect


def test_capture_rect_preserves_identity_through_transforms():
    """hwnd/process/title must survive every rect transform.

    A positional-argument slip in a transform silently blanked hwnd, which
    disabled direct window capture and let OCR read whatever window happened to
    be in front of the game.
    """
    rect = CaptureRect(
        100,
        200,
        1600,
        1000,
        1,
        source="window",
        title="My Game",
        uses_full_screen=True,
        process_name="game.exe",
        hwnd=4242,
    )
    derived = [
        rect.clamp_to((0, 0, 1920, 1200)),
        rect.inset(0.1, 0.1),
        rect.expand(20),
        rect.text_box(0.58),
    ]
    for item in derived:
        assert item is not None
        assert item.hwnd == 4242, f"hwnd lost: {item}"
        assert item.title == "My Game"
        assert item.uses_full_screen is True
        assert item.process_name == "game.exe"


def test_window_resolver_never_raises_and_can_be_configured():
    resolver = GameWindowResolver(excluded_pids={0}, monitor_index=0, cache_ttl_sec=0.0)
    result = resolver.resolve()
    if result is not None:
        assert result.width >= 160 and result.height >= 160
        assert result.monitor_index >= 0
    else:
        assert resolver.last_error
    resolver.invalidate()
    resolver.set_excluded_pids({0})
    resolver.set_monitor_index(1)
    resolver.resolve()


# --- prompts ---------------------------------------------------------------


def test_short_comment_prompt_has_no_contradictory_length_limit():
    short_prompt = get_system_screen_comment_prompt(tutor_enabled=False)
    assert "不超过25字" not in short_prompt


def test_visual_novel_prompt_is_separate_and_allows_silence():
    short_prompt = get_system_screen_comment_prompt(tutor_enabled=False)
    vn_prompt = get_system_visual_novel_prompt(tutor_enabled=False)
    assert vn_prompt and vn_prompt != short_prompt
    assert "沉默" in vn_prompt
