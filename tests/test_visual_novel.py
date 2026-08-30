import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from desktop_pet.main import _crop_visual_novel_scan_image
from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.visual_novel import VisualNovelStoryLibrary, VisualNovelTracker


def _semantic_similarity(left: str, right: str) -> float:
    if right.startswith(("这段", "角色", "剧情")):
        return 0.54 if left.endswith("关键片段。") else 0.42
    return 0.65 if left.endswith("关键片段。") else 0.90


def _recall_similarity(query: str, candidate: str) -> float:
    if candidate.startswith("用户正在询问"):
        return 0.72 if any(word in query for word in ("剧情", "游戏", "台词")) else 0.28
    if candidate.startswith("用户想回忆"):
        return 0.58 if "刚才" in query else 0.30
    if "李梅" in query and "李梅" in candidate:
        return 0.86
    if any(word in query for word in ("剧情", "游戏", "台词")):
        return 0.50
    return 0.26


class _FakeClient:
    def chat(self, user_text: str, system_prompt: str) -> str:
        assert "近期台词" in user_text
        return json.dumps(
            {
                "should_comment": True,
                "moment_type": "twist",
                "comment": "原来真相藏在这里呀，突然都串起来了。",
                "scene_summary": "角色发现了此前被隐瞒的真相。",
                "facts": ["重要真相已经揭露"],
            },
            ensure_ascii=False,
        )


def test_visual_novel_story_library_migrates_legacy_without_deleting_it():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        legacy = root / "visual_novel_story.json"
        legacy_payload = {
            "version": 1,
            "total_lines": 1,
            "summarized_line_count": 0,
            "scene_summary": "旧缓存摘要",
            "facts": [],
            "lines": [{"id": 1, "text": "旧缓存台词"}],
        }
        legacy.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

        library = VisualNovelStoryLibrary(root / "visual_novel_stories", legacy_path=legacy)

        assert library.active_name == "default.json"
        assert json.loads(library.active_path.read_text(encoding="utf-8"))["scene_summary"] == "旧缓存摘要"
        assert legacy.exists()


def test_visual_novel_story_library_creates_selects_and_deletes_caches():
    with TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir) / "stories"
        library = VisualNovelStoryLibrary(directory)
        new_path = library.create("樱云")
        library.set_active(new_path.name)

        restored = VisualNovelStoryLibrary(directory)
        assert restored.active_name == "樱云.json"
        assert set(restored.list_stories()) == {"default.json", "樱云.json"}

        try:
            restored.delete("樱云.json")
            raise AssertionError("active cache deletion must be rejected")
        except ValueError:
            pass
        restored.set_active("default.json")
        restored.delete("樱云.json")
        assert restored.list_stories() == ["default.json"]


def test_visual_novel_tracker_switch_keeps_story_files_isolated():
    with TemporaryDirectory() as temp_dir:
        library = VisualNovelStoryLibrary(Path(temp_dir) / "stories")
        first_path = library.active_path
        second_path = library.create("第二个游戏")
        tracker = VisualNovelTracker(first_path, min_context_similarity=0.0)
        tracker.observe("第一个游戏的剧情。", confidence=0.9)

        tracker.switch_story(second_path)
        tracker.observe("第二个游戏的剧情。", confidence=0.9)
        tracker.flush()

        first_payload = json.loads(first_path.read_text(encoding="utf-8"))
        second_payload = json.loads(second_path.read_text(encoding="utf-8"))
        assert "第一个游戏" in first_payload["lines"][0]["text"]
        assert "第二个游戏" not in first_payload["lines"][0]["text"]
        assert "第二个游戏" in second_payload["lines"][0]["text"]


def test_visual_novel_tracker_deduplicates_and_nominates_semantic_moment():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=_semantic_similarity,
            summary_batch_size=8,
        )
        assert tracker.observe("普通台词一。", confidence=0.9).accepted is True
        assert tracker.observe("普通台词一。", confidence=0.9).accepted is False
        for index in range(2, 7):
            tracker.observe(f"普通台词{index}。", confidence=0.9)
        tracker.observe("关键片段。", confidence=0.9)
        moment = tracker.observe("事件后的普通台词。", confidence=0.9)

        assert moment.should_evaluate is True
        assert moment.reason == "semantic_candidate"
        assert moment.semantic_score < 0.58
        payload = tracker.build_evaluation_payload()
        assert payload is not None
        assert "关键片段" in payload["recent_dialogue"]


def test_visual_novel_tracker_does_not_nominate_flat_dialogue():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=_semantic_similarity,
            summary_batch_size=24,
        )
        observations = [tracker.observe(f"普通对话{index}。") for index in range(12)]

        assert not any(item.reason == "semantic_candidate" for item in observations)


def test_visual_novel_tracker_persists_summary_and_facts():
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "story.json"
        tracker = VisualNovelTracker(path, similarity_fn=_semantic_similarity, summary_batch_size=8)
        for index in range(8):
            tracker.observe(f"第{index + 1}句剧情。", confidence=0.9)
        assert tracker.build_evaluation_payload() is not None
        tracker.apply_evaluation({"scene_summary": "发现真相。", "facts": ["秘密被揭露"]})

        restored = VisualNovelTracker(path, similarity_fn=_semantic_similarity)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["scene_summary"] == "发现真相。"
        assert "秘密被揭露" in payload["facts"]
        restored.flush()


def test_visual_novel_retrieval_is_relevant_and_filters_runtime_logs():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=_recall_similarity,
        )
        tracker.observe("HEARTBEAT scan_busy context_built vn_buffer", confidence=0.8)
        tracker.observe("李梅说她在双龙馆发现了目标人物。", confidence=0.9)
        tracker.apply_evaluation(
            {
                "scene_summary": "众人正在调查帝都出现的异常事件。",
                "facts": ["李梅在双龙馆发现了目标人物"],
            }
        )

        context = tracker.retrieve_for_query("李梅在双龙馆发现了什么？")

        assert "李梅在双龙馆发现了目标人物" in context
        assert "HEARTBEAT" not in context
        assert "scan_busy" not in context


def test_visual_novel_filters_logs_and_gibberish_before_observe():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")

        runtime_log, log_reason = tracker.prepare_ocr_text(
            "[HEARTBEAT 21:57:00] context_built mode=ocr len=354",
            confidence=0.98,
        )
        gibberish, gibberish_reason = tracker.prepare_ocr_text("，，，，12331asd", confidence=0.92)

        assert runtime_log == ""
        assert "log=1" in log_reason
        assert gibberish == ""
        assert "gibberish=1" in gibberish_reason


def test_visual_novel_filter_keeps_dialogue_when_same_frame_contains_log():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")

        prepared, reason = tracker.prepare_ocr_text(
            "[INFO] scan_submit route=ocr\n为什么你直到现在才告诉我真相？",
            confidence=0.88,
        )

        assert prepared == "为什么你直到现在才告诉我真相？"
        assert "log=1" in reason


def test_visual_novel_filter_keeps_low_confidence_semantic_turn():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=lambda _left, _right: 0.05,
        )

        prepared, _reason = tracker.prepare_ocr_text(
            "陌生的天空突然裂开，世界已经完全不同了。",
            confidence=0.18,
        )

        assert prepared == "陌生的天空突然裂开，世界已经完全不同了。"


def test_visual_novel_filter_rejects_below_context_similarity_floor():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=lambda _left, _right: 0.12,
            min_context_similarity=0.18,
        )
        for text in ("第一段剧情。", "角色继续交谈。", "众人准备出发。"):
            tracker.observe(text, confidence=0.9)

        prepared, reason = tracker.prepare_ocr_text("完全无关的识别内容。", confidence=0.9)

        assert prepared == ""
        assert reason == "context_similarity=0.12<0.18"


def test_visual_novel_context_similarity_floor_can_be_disabled():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=lambda _left, _right: 0.01,
            min_context_similarity=0.0,
        )
        for text in ("第一段剧情。", "角色继续交谈。", "众人准备出发。"):
            tracker.observe(text, confidence=0.9)

        prepared, _reason = tracker.prepare_ocr_text("场景突然转移到了陌生世界。", confidence=0.9)

        assert prepared == "场景突然转移到了陌生世界。"


def test_visual_novel_context_floor_ignores_old_runtime_noise():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=lambda _left, _right: 0.01,
            min_context_similarity=0.30,
        )
        for text in (
            "HEARTBEAT scan_busy context_built",
            "[INFO] scan_submit route=ocr mode=vn",
            "vn_buffer semantic_score=0.42 elapsed_ms=20",
        ):
            tracker.observe(text, confidence=0.9)

        prepared, _reason = tracker.prepare_ocr_text("真正的剧情现在开始了。", confidence=0.9)

        assert prepared == "真正的剧情现在开始了。"


def test_visual_novel_retrieval_ignores_unrelated_chat_questions():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(
            Path(temp_dir) / "story.json",
            similarity_fn=_recall_similarity,
        )
        tracker.observe("李梅说她在双龙馆发现了目标人物。", confidence=0.9)
        tracker.apply_evaluation({"scene_summary": "众人正在调查帝都的异常事件。"})

        assert tracker.retrieve_for_query("Python 列表应该怎么排序？") == ""


def test_visual_novel_planner_hint_contains_only_summary_and_facts():
    with TemporaryDirectory() as temp_dir:
        tracker = VisualNovelTracker(Path(temp_dir) / "story.json")
        tracker.observe("这条原始OCR台词不应进入规划提示。", confidence=0.9)
        tracker.apply_evaluation(
            {
                "scene_summary": "风见司穿越到了大正时代。",
                "facts": ["风见司协助所长调查案件"],
            }
        )

        hint = tracker.build_planner_hint()

        assert "风见司穿越到了大正时代" in hint
        assert "风见司协助所长调查案件" in hint
        assert "原始OCR台词" not in hint


def test_visual_novel_llm_call_returns_comment_and_memory_update():
    engine = CommentEngine(_FakeClient())
    result = engine.evaluate_visual_novel(
        {
            "reason": "semantic_candidate",
            "scene_summary": "两人正在调查。",
            "facts": ["某人隐瞒了消息"],
            "recent_dialogue": "原来你一直隐瞒着真相。",
        }
    )

    assert result["should_comment"] is True
    assert result["moment_type"] == "twist"
    assert "真相" in result["comment"]
    assert result["scene_summary"] == "角色发现了此前被隐瞒的真相。"


def test_visual_novel_uses_bottom_dialogue_area_unless_region_is_manual():
    image = Image.new("RGB", (100, 100), "white")
    automatic = _crop_visual_novel_scan_image(image, enabled=True, has_manual_region=False)
    manual = _crop_visual_novel_scan_image(image, enabled=True, has_manual_region=True)

    assert automatic.size == (100, 42)
    assert manual.size == (100, 100)
