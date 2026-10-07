import json
from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.visual_novel import VisualNovelStoryLibrary, VisualNovelTracker


def test_viewer_impressions_follow_the_story_cache_and_can_be_revised():
    with TemporaryDirectory() as temp_dir:
        library = VisualNovelStoryLibrary(Path(temp_dir))
        first = library.create("first")
        second = library.create("second")
        tracker = VisualNovelTracker(first)

        tracker.apply_evaluation({"viewer_notes": ["我仍怀疑他隐瞒了来信。"]})
        assert json.loads(first.read_text(encoding="utf-8"))["viewer_notes"] == ["我仍怀疑他隐瞒了来信。"]

        tracker.switch_story(second)
        assert tracker._viewer_notes == []
        tracker.apply_evaluation({"viewer_notes": ["我觉得她很勇敢。"]})
        tracker.switch_story(first)
        assert tracker._viewer_notes == ["我仍怀疑他隐瞒了来信。"]

        tracker.apply_evaluation({"viewer_notes": ["知道信的来历后，我可能误会了他。"]})
        tracker.switch_story(second)
        tracker.switch_story(first)
        assert tracker._viewer_notes == ["知道信的来历后，我可能误会了他。"]

        library.reset_story("first.json")
        tracker.switch_story(first)
        assert tracker._viewer_notes == []


def test_visual_novel_model_receives_and_can_update_viewer_impressions():
    class Client:
        prompt = ""

        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            self.prompt = user_text
            return json.dumps({
                "should_comment": False,
                "moment_type": "ordinary",
                "comment": "",
                "viewer_notes": ["新线索说明我可能误会了他。"],
            }, ensure_ascii=False)

    client = Client()
    result = CommentEngine(client).evaluate_visual_novel({
        "recent_dialogue": "信是我留下的。",
        "viewer_notes": ["我仍怀疑他隐瞒了来信。"],
    })
    assert "我仍怀疑他隐瞒了来信" in client.prompt
    assert result["viewer_notes"] == ["新线索说明我可能误会了他。"]
    assert result["should_comment"] is False


def test_viewer_impressions_drop_scene_observations_and_unknown_relationships():
    class Client:
        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            return json.dumps({
                "should_comment": False,
                "moment_type": "ordinary",
                "comment": "",
                "viewer_notes": [
                    "列车里的气氛很轻松。",
                    "我暂时还看不出他们是什么关系。",
                    "我还没看清他们是什么关系，只能安静看着。",
                    "他让我有点担心，似乎总在隐瞒重要的事。",
                ],
            }, ensure_ascii=False)

    result = CommentEngine(Client()).evaluate_visual_novel({"recent_dialogue": "他没说信从哪里来。"})
    assert result["viewer_notes"] == ["他让我有点担心，似乎总在隐瞒重要的事。"]


def test_invalid_viewer_notes_do_not_erase_existing_impressions():
    class Client:
        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            return json.dumps({
                "should_comment": False,
                "moment_type": "ordinary",
                "comment": "",
                "viewer_notes": ["当前气氛很轻松。"],
            }, ensure_ascii=False)

    result = CommentEngine(Client()).evaluate_visual_novel({
        "recent_dialogue": "好，出发吧。",
        "viewer_notes": ["我开始相信他是在保护朋友。"],
    })
    assert "viewer_notes" not in result
