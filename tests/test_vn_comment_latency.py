import ast
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.client import LLMClient
from desktop_pet.audio.speech import SpeechService, _SpeechTask


def test_reaction_delivered_once_before_memory_and_memory_still_returns():
    events = []
    head = '{"should_comment":true,"moment_type":"humor","reaction_repeats":false,"comment":"这张传单也太逗了。",'
    class Client:
        def chat_stream(self, user_text, system_prompt, on_text, **kwargs):
            for i in range(1, len(head)+1):
                on_text(head[:i])
            assert len(events) == 1
            final = head + '"scene_summary":"正在招募同好会成员。","facts":["惠麻设计了招募传单。"]}'
            on_text(final)
            return final
    result = CommentEngine(Client()).evaluate_visual_novel({}, on_reaction=events.append)
    assert len(events) == 1 and events[0]["comment"] == result["comment"]
    assert result["facts"] == ["惠麻设计了招募传单。"]


def test_incomplete_escaped_comment_and_negative_decisions_never_emit_early():
    partial = '{"should_comment":true,"moment_type":"humor","reaction_repeats":false,"comment":"他说\\\"别急\\\"'
    assert "comment" not in CommentEngine._leading_json_fields(partial)
    for values in [dict(should_comment=False,moment_type="humor",reaction_repeats=False),
                   dict(should_comment=True,moment_type="ordinary",reaction_repeats=False),
                   dict(should_comment=True,moment_type="humor",reaction_repeats=True)]:
        events = []
        raw = json.dumps({**values,"comment":"不应该被发出。","facts":["剧情仍正常更新。"]})
        class Client:
            def chat_stream(self, user_text, system_prompt, on_text, **kwargs):
                on_text(raw)
                return raw
        result = CommentEngine(Client()).evaluate_visual_novel({},on_reaction=events.append)
        assert events == [] and not result["should_comment"]
        assert result["facts"] == ["剧情仍正常更新。"]


def test_client_stream_ignores_reasoning_and_usage_and_closes_connection():
    calls=[]
    class Stream:
        closed=False
        def __iter__(self):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None,reasoning_content="internal"))])
            for text in ["hello", " world"]:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        def close(self): self.closed=True
    stream=Stream()
    client=LLMClient.__new__(LLMClient)
    client.settings=SimpleNamespace(model_name="test")
    client._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:stream)))
    assert client.chat_stream("user","system",calls.append) == "hello world"
    assert calls == ["hello","hello world"] and stream.closed


def test_expired_speech_during_translation_has_no_audio_or_display_callback():
    valid=[True];events=[]
    service=SpeechService.__new__(SpeechService)
    service._generation=service._active_generation=0
    service._muted=False;service._diag=False;service._provider="voicevox"
    service._queue=queue.Queue()
    def translate(text):
        valid[0]=False
        return "こんにちは"
    service._translation_service=SimpleNamespace(translate=translate)
    service._speak_once=lambda *args:events.append("audio")
    service._queue.put(_SpeechTask("旧剧情",on_start=lambda:events.append("display"),is_valid=lambda:valid[0]))
    service._queue.put(None)
    service._run_worker()
    assert events == []


def test_complete_memory_is_kept_when_early_comment_already_emitted_or_expired():
    tree=ast.parse((Path(__file__).resolve().parents[1]/"main.py").read_text(encoding="utf-8"))
    def load(name,env):
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"main.py","exec"),env)
    for emitted in [False,True]:
        applied=[];spoken=[]
        result={"should_comment":True,"comment":"旧剧情的吐槽。","facts":["记忆照常写入。"]}
        meta={"kind":"visual_novel","submitted_at":time.monotonic()-25,"story_path":"story.json","reaction_emitted":emitted}
        state={"comment_future":SimpleNamespace(done=lambda:True,result=lambda:result),"pending_comment_meta":meta}
        env=dict(time=time,datetime=datetime,state=state,visual_novel_tracker=SimpleNamespace(path=Path("story.json"),apply_evaluation=applied.append),
                 _sync_plot_memory=lambda r:None,log_heartbeat=lambda *a:None,_emit_comment=lambda *a,**kw:spoken.append(a))
        load("_publish_comment",env);load("poll_comment_future",env)
        env["poll_comment_future"]()
        assert applied == [result] and spoken == []


def test_fast_reasoning_only_applies_to_vn_callback_and_can_be_rolled_back():
    from unittest.mock import patch
    raw = '{"should_comment":false,"moment_type":"ordinary","reaction_repeats":false,"comment":""}'
    for model, flag, expected in [("deepseek-v4-flash-vision-exp","true",True),
                                   ("deepseek-v4-flash-vision-exp","false",False),
                                   ("other-model","true",False)]:
        calls=[]
        client=LLMClient.__new__(LLMClient)
        client.settings=SimpleNamespace(model_name=model)
        def stream(**kwargs):
            calls.append(kwargs["disable_thinking"])
            kwargs["on_text"](raw)
            return raw
        client.chat_stream=stream
        client.chat=lambda **kw:raw
        with patch.dict("os.environ",{"VN_FAST_REASONING":flag}):
            engine=CommentEngine(client)
            engine.evaluate_visual_novel({},on_reaction=lambda r:None)
            engine.evaluate_visual_novel({})
        assert calls == [expected]


def test_reaction_window_is_fresh_while_older_dialogue_remains_for_memory():
    from tempfile import TemporaryDirectory
    from desktop_pet.llm.visual_novel import VisualNovelTracker
    with TemporaryDirectory() as directory:
        tracker=VisualNovelTracker(Path(directory)/"story.json",summary_batch_size=8)
        for i in range(8):
            tracker.observe(f"第{i}段：这是内容不同的剧情对白。")
        payload=tracker.build_evaluation_payload()
        assert payload is not None
        assert "第0段" in payload["new_dialogue"]
        assert "第0段" not in payload["reaction_dialogue"]
        assert all(f"第{i}段" in payload["reaction_dialogue"] for i in [5,6,7])
