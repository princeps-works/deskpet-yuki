import json
from pathlib import Path
from tempfile import TemporaryDirectory

from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.dialog_manager import DialogManager


def test_diary_personalization_accepts_only_user_quotes_and_replaces_corrections():
    class Client:
        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            user_quote = "以后看剧情时少问我问题，多说你自己的感受。"
            if "改成更常问我怎么看" in user_text:
                user_quote = "改成更常问我怎么看。"
            return json.dumps({
                "summary": "今天和哥哥聊了自动评论应该怎样说话，也记下了他的明确要求。",
                "personalization": [
                    {"topic": "剧情评论方式", "preference": user_quote, "evidence_quote": user_quote},
                    {"topic": "编造的偏好", "preference": "喜欢剧透", "evidence_quote": "我喜欢剧透"},
                ],
                "facts": [],
            }, ensure_ascii=False)

    with TemporaryDirectory() as temp_dir:
        manager = DialogManager(Client(), memory_path=Path(temp_dir) / "memory.json")
        manager.archive_transcript("你: 以后看剧情时少问我问题，多说你自己的感受。\n桌宠: 我喜欢剧透")
        hint = manager.build_personalization_hint()
        assert "少问我问题" in hint
        assert "喜欢剧透" not in hint

        manager.archive_transcript("你: 改成更常问我怎么看。")
        stored = json.loads((Path(temp_dir) / "personalization.json").read_text(encoding="utf-8"))
        assert len(stored["preferences"]) == 1
        assert stored["preferences"][0]["preference"] == "改成更常问我怎么看。"


def test_visual_novel_comment_receives_confirmed_user_preference():
    class Client:
        prompt = ""

        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            self.prompt = user_text
            return '{"should_comment":false,"moment_type":"ordinary","comment":""}'

    client = Client()
    CommentEngine(client).evaluate_visual_novel({
        "recent_dialogue": "接下来该去哪？",
        "personalization_hint": "哥哥明确表达的长期互动偏好：少问我问题。",
    })
    assert "少问我问题" in client.prompt


def test_visual_novel_evaluation_failure_reason_clears_after_success():
    class Client:
        responses = [None, "", "not json", '{"should_comment":false,"moment_type":"ordinary","comment":""}']

        def chat(self, **_kwargs):
            return self.responses.pop(0)

    engine = CommentEngine(Client())
    assert engine.evaluate_visual_novel({}) == {}
    assert engine.last_visual_novel_error == "empty_response"
    assert engine.evaluate_visual_novel({}) == {}
    assert engine.last_visual_novel_error == "empty_response"
    assert engine.evaluate_visual_novel({}) == {}
    assert engine.last_visual_novel_error.startswith("invalid_json:")
    assert engine.evaluate_visual_novel({})["should_comment"] is False
    assert engine.last_visual_novel_error == ""


def test_chat_and_screen_comment_receive_confirmed_user_preference():
    class Client:
        prompts = []

        def chat(self, user_text: str, system_prompt: str = "", **_kwargs):
            self.prompts.append(user_text)
            return "收到。"

    with TemporaryDirectory() as temp_dir:
        client = Client()
        manager = DialogManager(client, memory_path=Path(temp_dir) / "memory.json")
        manager._record_personalization(
            [{"topic": "互动方式", "preference": "少问我问题", "evidence_quote": "少问我问题"}],
            "你: 少问我问题",
        )
        hint = manager.build_personalization_hint()
        manager.reply("你好")
        assert hint in client.prompts[-1]

        CommentEngine(client).comment_on_summary("屏幕上显示两人正在交谈。", personalization_hint=hint)
        assert hint in client.prompts[-1]
