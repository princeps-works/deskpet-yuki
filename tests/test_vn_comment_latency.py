import ast
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
from datetime import timedelta
from desktop_pet.llm.comment_engine import CommentEngine
from desktop_pet.llm.client import LLMClient
from desktop_pet.audio.speech import SpeechService, _SpeechTask
from desktop_pet.main import _fingerprint_diff_ratio, _vn_visual_fingerprint, vision_should_capture


def test_visual_observation_keeps_capture_age_and_marks_changed_dialogue():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_current_visual_context")
    state = {"vision_context": "角色瞪大眼睛，台词是上一段。", "vision_context_at": datetime.now(),
             "vision_context_stable": "黑发角色在室内。",
             "vision_context_dialogue": "旧台词", "scan_cache_ocr_text": "新台词"}
    ns = dict(state=state, datetime=datetime, settings=SimpleNamespace(vision_context_ttl_sec=30),
              _fingerprint_diff_ratio=_fingerprint_diff_ratio)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    current = ns["_current_visual_context"]
    assert current().startswith("较早画面观察")
    state["scan_cache_ocr_text"] = "旧台词"
    assert current() == state["vision_context"]
    state.update(vision_context_fingerprint=b"\x00" * 576,
                 vision_current_fingerprint=b"\x00" * 576,
                 vision_context_full_fingerprint=b"\x00" * 576,
                 vision_current_full_fingerprint=b"\x00" * 576,
                 vision_current_frame_at=datetime.now(), scan_cache_ocr_text="又一段新对白")
    assert current().startswith("持续外观参考")
    assert '瞪大眼睛' not in current()
    state['vision_context_at'] = datetime.now() - timedelta(seconds=31)
    assert current().startswith('持续外观参考')
    state['vision_current_full_fingerprint'] = b'\xff' * 576
    assert current() == ''  # A changed portrait below the dialogue band invalidates reuse.
    state['vision_current_full_fingerprint'] = b'\x00' * 576
    state['vision_context_at'] = datetime.now()
    state["vision_current_fingerprint"] = b"\xff" * 576
    assert current().startswith("较早画面观察")
    assert "瞪大眼睛" not in current() and "上一段" not in current()
    state["scan_cache_ocr_text"] = "旧台词"
    assert current().startswith("较早画面观察")  # CG changes with unchanged text.
    state["vision_context_at"] = datetime.now() - timedelta(seconds=31)
    assert current() == ""


def test_vn_vision_separates_static_description_and_moment_in_one_call():
    from PIL import Image
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_vision_worker")
    calls = []
    class Client:
        def describe_image(self, frame, user_text, system_prompt):
            calls.append(user_text)
            return '{"appearance_environment":"黑发角色，背景是厨房。","moment":"角色微笑，抬起右手。"}'
    ns = dict(json=json, llm_client=Client(), settings=SimpleNamespace(mm_timeout_sec=2, mm_image_max_edge=256,
               ocr_context_max_chars=300, vision_focus_query=""))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    result = ns["_vision_worker"](Image.new("RGB", (64, 64)), "window_direct", "测试对白")
    assert len(calls) == 1 and "不根据台词猜地点" in calls[0]
    assert result["appearance_environment"] == "黑发角色，背景是厨房。"
    assert result["moment"] == "角色微笑，抬起右手。"
    assert result["reason"] == "ok"


def test_art_fingerprint_ignores_dialogue_but_detects_scene_change():
    from PIL import Image, ImageDraw
    frame = Image.new("RGB", (200, 200), "white")
    original = _vn_visual_fingerprint(frame, 0.58)
    ImageDraw.Draw(frame).rectangle((0, 125, 199, 199), fill="black")
    assert _vn_visual_fingerprint(frame, 0.58) == original
    ImageDraw.Draw(frame).rectangle((0, 0, 199, 110), fill="black")
    assert _fingerprint_diff_ratio(_vn_visual_fingerprint(frame, 0.58), original) > 0.03


def test_speaker_portrait_link_is_not_reused_as_stable_appearance():
    from PIL import Image
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text(encoding='utf8'))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_vision_worker')
    class Client:
        def describe_image(self, frame, user_text, system_prompt):
            return json.dumps({'appearance_environment':'两人站在风车前','moment':'',
                'speaker_name':'燐','speaker_object':'B',
                'objects':[{'id':'B','type':'human','position':'右侧','appearance':'金发格纹发箍'}]})
    ns = dict(json=json, llm_client=Client(), settings=SimpleNamespace(mm_timeout_sec=2, mm_image_max_edge=256,
               ocr_context_max_chars=300, vision_focus_query=''))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), ns)
    result = ns['_vision_worker'](Image.new('RGB', (64,64)), 'window_direct', '焕：你好。')
    assert '姓名栏识读「燐」' in result['moment']
    assert '对象B' in result['moment']
    assert '姓名栏' not in result['appearance_environment']


def test_vn_vision_failure_reason_survives_worker_and_poll():
    from PIL import Image
    from unittest.mock import patch
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    ns = dict(json=json, datetime=datetime, llm_client=None,
              settings=SimpleNamespace(mm_timeout_sec=2, mm_image_max_edge=320, ocr_context_max_chars=300, vision_focus_query=""))
    for name in ("_vision_worker", "poll_vision_future"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    with patch("desktop_pet.vision.multimodal.describe_screen_image_compat",
               return_value=SimpleNamespace(summary="", reason="timeout", elapsed_ms=2000)):
        result = ns["_vision_worker"](Image.new("RGB", (64, 64)), "window_direct", "测试对白")
    assert result == {"reason": "timeout", "elapsed_ms": 2000}
    events = []
    ns["state"] = {"vision_future": SimpleNamespace(done=lambda: True, result=lambda: result), "vision_started_at": datetime.now()}
    ns["log_heartbeat"] = lambda *args: events.append(args)
    ns["poll_vision_future"]()
    assert events == [("vision_empty", "reason=timeout, elapsed_ms=2000")]


def test_vn_vision_distinguishes_empty_observation_from_wrong_fields():
    from PIL import Image
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_vision_worker")
    class Client:
        reply = ""
        def describe_image(self, *_args):
            return self.reply
    client = Client()
    ns = dict(json=json, llm_client=client, settings=SimpleNamespace(mm_timeout_sec=2,
              mm_image_max_edge=320, ocr_context_max_chars=300, vision_focus_query=""))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    for text, reason in (("", "empty"), ('{"appearance_environment":"","moment":""}', "empty_observation"),
                         ('{"description":"人物在室内"}', "invalid_fields"), ('[]', "invalid_schema")):
        client.reply = text
        result = ns["_vision_worker"](Image.new("RGB", (64, 64)), "window_direct", "对白")
        assert result["reason"] == reason


def test_art_freshness_updates_without_spending_vision_quota():
    from PIL import Image
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "vision_sampling_step")
    captures = []
    state = dict(scan_enabled=True)
    settings = SimpleNamespace(enable_multimodal_vision=True, enable_mm_screen_comment=True,
                               visual_novel_text_ratio=0.58, vision_change_threshold=0.1, vision_min_gap_sec=6)
    ns = dict(state=state, settings=settings, datetime=datetime, visual_novel_tracker=object(),
              _refresh_vision_exclusions=lambda: None,
              _grab_vision_frame=lambda: (captures.append("capture"), Image.new("RGB", (200, 200)))[1],
              _vn_visual_fingerprint=_vn_visual_fingerprint,
              _vision_quota_state=lambda now: (0, now), _vision_since_last_sec=lambda now: None,
              _vision_rate_per_minute=lambda: 3, _vision_budget_available=lambda now: True,
              vision_should_capture=vision_should_capture, log_heartbeat=lambda *a: None)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    ns["vision_sampling_step"]()
    assert captures == ["capture"]
    assert state["vision_current_fingerprint"]
    assert state.get("vision_future") is None


def test_late_visual_result_does_not_reenter_a_switched_cache():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "poll_vision_future")
    state = dict(vision_future=SimpleNamespace(done=lambda: True, result=lambda: "旧作品的视觉信息"),
                 vision_started_at=datetime.now(), vision_request_cache_token=object(), vn_cache_token=object())
    events = []
    ns = dict(state=state, datetime=datetime, settings=SimpleNamespace(mm_timeout_sec=8),
              log_heartbeat=lambda *a: events.append(a))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    ns["poll_vision_future"]()
    assert state.get("vision_context", "") == ""
    assert events == [("vision_discard", "cache_changed")]


def test_evaluation_uses_visual_result_that_arrives_during_memory_selection():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "evaluate_snapshot")
    meta = {}
    state = {"pending_comment_meta": meta, "vision_context": "旧观察"}
    def build():
        state["vision_context"] = "刚返回的新观察"
        return {"new_dialogue": "新对白", "recent_dialogue": "新对白", "evidence_line_ids": [7, 8]}
    queries = []
    ns = dict(time=time, vn_reaction_classifier=None,
              snapshot=SimpleNamespace(build_evaluation_payload=build, _local_reaction_enabled=False,
              _select_characters_for_context=lambda query, visual_context: (queries.append((query, visual_context)) or [])), state=state, meta=meta,
              personalization_hint="", _current_visual_context=lambda: state["vision_context"],
              log_heartbeat=lambda *a: None, on_reaction=None,
              comment_engine=SimpleNamespace(evaluate_visual_novel=lambda payload, callback: payload))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
    result = ns["evaluate_snapshot"]()
    assert result["visual_context"] == "刚返回的新观察"
    assert result["evidence_line_ids"] == [7, 8]
    assert queries == [('新对白', '刚返回的新观察')]


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
        assert payload["reaction_line_ids"] == [4, 5, 6, 7, 8]
        assert all(f"第{i}段" in payload["reaction_dialogue"] for i in [5,6,7])

def test_scan_worker_receives_explicit_game_title_before_previous_capture():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text(encoding='utf-8'))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_submit_scan_worker_task')
    payload = next(n.value for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
    expr = next(v for k, v in zip(payload.keys, payload.values) if isinstance(k, ast.Constant) and k.value == 'target_title')
    code = compile(ast.Expression(expr), 'main.py', 'eval')
    for saved, configured, previous, expected in [('', '游戏A', 'Steam', '游戏A'), ('游戏B', '游戏A', 'Steam', '游戏B'), ('', '', '游戏A', '游戏A'), ('', '', '', '')]:
        assert eval(code, dict(_persisted_regions={'target_title': saved}, settings=SimpleNamespace(vision_target_title=configured), state={'last_capture_title': previous})) == expected

def test_cache_reset_invalidates_already_queued_comment_even_at_same_path():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text(encoding='utf8'))
    emitted=[]
    state={'visual_novel_mode_enabled':True,'last_comment_at':None,'last_comment_text':''}
    tracker=SimpleNamespace(path=Path('same.json'), total_lines=10)
    env=dict(state=state,visual_novel_tracker=tracker,time=time,datetime=datetime,settings=SimpleNamespace(auto_comment_cooldown_sec=30,comment_similarity_skip_threshold=.9),
             can_emit_comment=lambda *a:True,text_similarity=lambda *a:0,log_heartbeat=lambda *a:None,
             _emit_comment=lambda *a,**kw:emitted.append(kw.get('is_valid')))
    for name in ['_publish_comment','_discard_pending_vn_evaluation']:
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'main.py','exec'),env)
    meta={'kind':'visual_novel','story_path':'same.json','submitted_at':time.monotonic(),'cache_token':None,
          'submitted_line_count':10}
    env['_publish_comment']('当前剧情的反应。',meta)
    assert emitted and emitted[0]()
    env['_discard_pending_vn_evaluation']()
    assert not emitted[0]()


def test_ten_line_expiry_does_not_spend_cooldown_and_new_comment_can_display():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text(encoding='utf8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_publish_comment')
    queued=[];remembered=[]
    tracker=SimpleNamespace(path=Path('story.json'),total_lines=14,
                            remember_comment=lambda text,**kw:remembered.append(text))
    state={'visual_novel_mode_enabled':True,'last_comment_at':None,'last_comment_text':''}
    env=dict(state=state,visual_novel_tracker=tracker,time=time,datetime=datetime,
             settings=SimpleNamespace(auto_comment_cooldown_sec=30,comment_similarity_skip_threshold=.9),
             can_emit_comment=lambda last,*a:last is None,text_similarity=lambda *a:0,
             log_heartbeat=lambda *a:None,_emit_comment=lambda *a,**kw:queued.append(kw))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'main.py','exec'),env)
    meta=dict(kind='visual_novel',story_path='story.json',submitted_at=time.monotonic(),
              cache_token=None,submitted_line_count=14,reaction_line_id=10)
    assert env['_publish_comment']('仍然有效的反应。',meta)
    assert queued[0]['is_valid']()  # Four later lines are allowed.
    assert state['last_comment_at'] is None  # Queued synthesis is not speech.
    tracker.total_lines=15
    assert queued[0]['is_valid']()  # Five later lines are now allowed.
    tracker.total_lines=19
    assert queued[0]['is_valid']()  # Nine later lines are still allowed.
    tracker.total_lines=20
    assert not queued[0]['is_valid']()  # Ten later lines expire it.
    fresh={**meta,'reaction_line_id':20}
    fresh.pop('reaction_emitted',None)
    assert env['_publish_comment']('新的反应。',fresh)
    assert queued[1]['is_valid']()
    queued[1]['on_display']()
    assert state['last_comment_at'] is not None
    assert remembered == ['新的反应。']


def test_reaction_anchor_must_be_in_fresh_window_for_stream_and_full_result():
    for anchor,expected in [(3,False),(8,True),(True,False)]:
        raw=json.dumps(dict(should_comment=True,moment_type='humor',reaction_repeats=False,
                            comment='这一下没忍住。',reaction_line_id=anchor))
        class Client:
            def chat_stream(self,**kwargs):
                kwargs['on_text'](raw)
                return raw
        delivered=[]
        result=CommentEngine(Client()).evaluate_visual_novel(
            {'reaction_line_ids':[8,9],'reaction_dialogue':'[8] 当前的笑点。'},delivered.append)
        assert bool(delivered) is expected
        assert result['should_comment'] is expected


def test_displayed_comment_archives_anchor_and_latency_without_inventing_old_state():
    from tempfile import TemporaryDirectory
    from desktop_pet.llm.visual_novel import VisualNovelTracker
    with TemporaryDirectory() as directory:
        tracker=VisualNovelTracker(Path(directory)/'story.json')
        tracker.observe('第一句。')
        tracker.observe('下一句。')
        assert not [r for r in tracker._archive.rows() if r['kind']=='已显示评论']
        tracker.remember_comment('接住这个点。',evidence={
            'reaction_line_id':1,'submitted_line_count':1,
            'request_to_display_ms':1200,'reaction_ready_ms':500})
        row=next(r for r in tracker._archive.rows() if r['kind']=='已显示评论')
        assert row['payload']['displayed_line_id']==2
        assert row['payload']['reaction_line_id']==1
        assert row['payload']['request_to_display_ms']==1200
        assert row['payload']['reaction_ready_ms']==500


def test_memory_backlog_does_not_force_reaction_to_oldest_batch():
    from tempfile import TemporaryDirectory
    from desktop_pet.llm.visual_novel import VisualNovelTracker
    with TemporaryDirectory() as directory:
        tracker=VisualNovelTracker(Path(directory)/'story.json')
        for i in range(40):
            tracker.observe(f'片段{i}：'+chr(0x4e00+i)*130+'。')
        snapshot,_=tracker.reserve_evaluation_snapshot()
        payload=snapshot.build_evaluation_payload()
        assert payload['line_count']<tracker.total_lines  # Memory is catching up.
        assert payload['reaction_line_ids']==list(range(tracker.total_lines-4,tracker.total_lines+1))
        assert '片段39：' in payload['reaction_dialogue']
        assert '片段0：' not in payload['reaction_dialogue']
        assert '片段0：' in payload['new_dialogue']
