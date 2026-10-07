import json

import numpy as np
from contextlib import redirect_stdout
from io import StringIO
import sys
from types import SimpleNamespace
from unittest.mock import patch

from desktop_pet.llm.comment_engine import CommentEngine


class Client:
    def __init__(self, response):
        self.response = response
        self.prompt = ""
        self.calls = 0

    def chat(self, user_text, **kwargs):
        self.prompt = user_text
        self.calls += 1
        return json.dumps(self.response, ensure_ascii=False)


def test_overlap_ranking_keeps_end_of_long_dialogue_and_chronological_history():
    calls = []

    def scores(texts):
        calls.extend(texts[:-4])
        queries = [[1, 0] if "看不清" in query else [0, 1] for query in texts[:-4]]
        return np.array([*queries, [0.9, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]])

    engine = CommentEngine(Client({}), embeddings_fn=scores)
    result = engine._comment_overlap_candidates("普通对白" * 100 + "她说自己看不清", ["躲他", "社团", "打工", "演出"])
    assert result[0]["index"] == 0
    assert any("看不清" in query for query in calls)
    assert all(len(query) <= 200 for query in calls)


def test_high_overlap_does_not_suppress_revision_or_new_detail():
    for comment in ["原来她是看不清啊，我刚才误会了。", "这次他真的留下来陪她练习了。"]:
        client = Client({"should_comment": True, "moment_type": "twist", "comment": comment, "reaction_repeats": False})
        engine = CommentEngine(client, embeddings_fn=lambda texts: np.ones((len(texts), 1), dtype=np.float32))
        result = engine.evaluate_visual_novel({"new_dialogue": "她解释自己的视力很差。", "recent_comments": ["她是不是在躲他？"]})
        assert result["should_comment"] is True
        assert result["comment"] == comment
        assert client.calls == 1
        assert result["comment_overlap_candidates"]
        json.dumps(result)


def test_repeated_reaction_still_updates_memory_in_one_call():
    client = Client({"should_comment": True, "moment_type": "tender", "comment": "他很关心她。", "reaction_repeats": True, "facts": ["两人明天练习。"], "scene_summary": "已经约好练习。"})
    engine = CommentEngine(client, embeddings_fn=lambda texts: np.ones((len(texts), 1)))
    result = engine.evaluate_visual_novel({"new_dialogue": "明天见。", "recent_comments": ["他很关心她。"]})
    assert result["should_comment"] is False
    assert result["facts"] == ["两人明天练习。"]
    assert result["scene_summary"] == "已经约好练习。"
    assert client.calls == 1


def test_missing_or_failed_encoder_restores_existing_prompt_and_decision():
    context = {"new_dialogue": "她解释自己没看清。", "recent_comments": ["她是不是在躲他？"]}
    response = {"should_comment": True, "moment_type": "twist", "comment": "啊，误会她了。"}
    original = Client(response)
    old_result = CommentEngine(original).evaluate_visual_novel(context)

    def broken(texts):
        raise RuntimeError("encoder unavailable")

    for callback in [broken, lambda texts: None, lambda texts: []]:
        client = Client(response)
        result = CommentEngine(client, embeddings_fn=callback).evaluate_visual_novel(context)
        assert client.prompt == original.prompt
        assert result["should_comment"] == old_result["should_comment"]
        assert result["comment_overlap_candidates"] == []
        assert client.calls == 1


def test_native_import_warning_preserves_real_warning_but_allows_preloaded_runtimes():
    from desktop_pet.main import _warn_if_qt_loaded_too_early

    for ready in [False, True]:
        native = SimpleNamespace(__version__="loaded") if ready else None
        output = StringIO()
        with patch.dict(sys.modules, {"PyQt6": SimpleNamespace(), "onnxruntime": native, "torch": native}):
            with redirect_stdout(output):
                _warn_if_qt_loaded_too_early()
        assert bool(output.getvalue()) is not ready


def test_chat_reopening_restores_minimized_window_before_focusing():
    from desktop_pet.ui.chat_panel import ChatPanel

    for minimized in [False, True]:
        calls = []
        panel = SimpleNamespace(
            winId=lambda: 123,
            isMinimized=lambda: minimized,
            showNormal=lambda: calls.append("restore"),
            show=lambda: calls.append("show"),
            raise_=lambda: calls.append("raise"),
            activateWindow=lambda: calls.append("activate"),
            input_line=SimpleNamespace(setFocus=lambda reason: calls.append("focus")),
        )
        with patch("desktop_pet.ui.chat_panel.ctypes.windll", SimpleNamespace(
            user32=SimpleNamespace(IsWindowVisible=lambda hwnd: True)
        ), create=True):
            ChatPanel.show_and_focus(panel)
        assert calls == ["restore" if minimized else "show", "raise", "activate", "focus"]


def test_chat_recovers_native_hidden_startup_without_hiding_visible_windows():
    from desktop_pet.ui.chat_panel import ChatPanel

    for native_visible in [False, True]:
        calls = []
        panel = SimpleNamespace(
            winId=lambda: 123,
            isMinimized=lambda: False,
            show=lambda: calls.append("show"),
            hide=lambda: calls.append("hide"),
            raise_=lambda: calls.append("raise"),
            activateWindow=lambda: calls.append("activate"),
            input_line=SimpleNamespace(setFocus=lambda reason: calls.append("focus")),
        )
        with patch("desktop_pet.ui.chat_panel.ctypes.windll", SimpleNamespace(
            user32=SimpleNamespace(IsWindowVisible=lambda hwnd: native_visible)
        ), create=True):
            ChatPanel.show_and_focus(panel)
        assert calls == (["show"] if native_visible else ["show", "hide", "show"]) + ["raise", "activate", "focus"]
