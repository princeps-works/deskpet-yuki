"""Playback readiness must reach the GUI thread without an early display timer."""

import ast
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

from PyQt6.QtCore import QObject, Qt, QCoreApplication, pyqtSignal, pyqtSlot
from desktop_pet.audio.speech import SpeechService, _SpeechTask
import desktop_pet.audio.speech as speech_module


_APP = QCoreApplication.instance() or QCoreApplication([])
_TREE = ast.parse((Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8"))


def _load(name, namespace):
    node = next(n for n in ast.walk(_TREE) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), namespace)
    return namespace[name]


def test_comment_waits_for_playback_and_displays_once_on_gui_thread():
    import time
    calls = []
    tasks = []
    ns = dict(QObject=QObject, Qt=Qt, pyqtSignal=pyqtSignal, pyqtSlot=pyqtSlot,
              threading=threading, time=time)
    bridge = _load("_SpeechUiBridge", ns)()
    ns.update(speech_ui=bridge, speech=SimpleNamespace(is_muted=lambda: False, get_status=lambda: (True, "ok")),
              estimate_bubble_duration_ms=lambda _: 3000,
              chat=SimpleNamespace(append_message=lambda *a: calls.append(("chat", threading.get_ident()))),
              pet=SimpleNamespace(show_comment_bubble=lambda *a, **k: calls.append(("bubble", threading.get_ident()))),
              tts_executor=SimpleNamespace(submit=lambda *args: tasks.append(args)),
              _speak_async_with_callback=lambda *args: None)
    show = _load("show_and_speak", ns)
    show("测试评论", role="桌宠")
    _APP.processEvents()
    assert calls == []  # No 2.2s fallback timer is available in this namespace.
    worker = threading.Thread(target=lambda: (tasks[0][2](), tasks[0][3]()))
    worker.start()
    worker.join()
    assert calls == []
    _APP.processEvents()
    assert calls == [("chat", threading.get_ident()), ("bubble", threading.get_ident())]
    for muted, available in ((True, True), (False, False)):
        calls.clear()
        tasks.clear()
        ns["speech"] = SimpleNamespace(is_muted=lambda: muted, get_status=lambda: (available, ""))
        show("纯文字", role="桌宠")
        assert len(calls) == 2
        assert tasks == []  # Do not wait behind translation warmup when audio is off.


def test_synthesis_failure_displays_text_through_error_callback():
    service = SpeechService.__new__(SpeechService)
    service._generation = 0
    service._active_generation = 0
    service._muted = False
    service._queue = queue.Queue()
    service._translation_service = None
    service._provider = "voicevox"
    service._diag = False
    def fail(*args):
        raise RuntimeError("synthesis failed")
    service._speak_once = fail
    calls = []
    service._queue.put(_SpeechTask("测试", on_start=lambda: calls.append("started"),
                                  on_error=lambda: calls.append("text_only")))
    service._queue.put(None)
    service._run_worker()
    assert calls == ["text_only"]


def test_muted_or_disabled_speech_still_delivers_text():
    for muted in (True, False):
        calls = []
        ns = {"speech": SimpleNamespace(is_muted=lambda: muted, speak=lambda *a, **k: False)}
        speak = _load("_speak_async_with_callback", ns)
        speak("测试", lambda: calls.append("start"), lambda: calls.append("text_only"))
        assert calls == ["text_only"]


def test_replacement_during_synthesis_discards_old_audio_and_callback():
    entered = threading.Event()
    release = threading.Event()
    played = threading.Event()
    callbacks = []
    service = SpeechService.__new__(SpeechService)
    service._runtime_ok = True
    service._muted = False
    service._generation = service._active_generation = 0
    service._lock = threading.Lock()
    service._queue = queue.Queue()
    service._provider = "voicevox"
    service._translation_service = None
    service._diag = False
    service._voicevox_audio_query = lambda text: {"text": text}
    def synth(query):
        if query["text"] == "旧语音":
            entered.set()
            assert release.wait(3)
        return b"audio"
    service._voicevox_synthesis = synth
    original = speech_module.pygame
    music = SimpleNamespace(get_busy=lambda: False, stop=lambda: None,
                            load=lambda _: None, play=lambda: None)
    speech_module.pygame = SimpleNamespace(mixer=SimpleNamespace(get_init=lambda: True, music=music, quit=lambda: None))
    service._worker = threading.Thread(target=service._run_worker)
    service._worker.start()
    try:
        service.speak("旧语音", on_start=lambda: callbacks.append("old"), on_error=lambda: callbacks.append("old_error"))
        assert entered.wait(3)
        service.speak("新语音", on_start=lambda: (callbacks.append("new"), played.set()))
        release.set()
        assert played.wait(3)
        assert callbacks == ["new"]
    finally:
        release.set()
        service.shutdown()
        speech_module.pygame = original


def test_sentence_prefetch_starts_before_previous_playback_finishes():
    service = SpeechService.__new__(SpeechService)
    service._generation = service._active_generation = 0
    service._muted = False
    next_ready = threading.Event()
    parts = []
    callbacks = []
    def prepare(text):
        if text == "お兄ちゃん！":
            next_ready.set()
        return text.encode()
    def play(audio, callback):
        parts.append(audio.decode())
        if len(parts) == 1:
            assert next_ready.wait(3)
        if callback:
            callback()
    service._prepare_voicevox = prepare
    service._play_voicevox = play
    service._speak_voicevox("こんにちは。お兄ちゃん！", lambda: callbacks.append("shown"))
    assert parts == ["こんにちは。", "お兄ちゃん！"]
    assert callbacks == ["shown"]


def test_long_voicevox_clauses_keep_text_order_and_leave_short_sentences_intact():
    text = "リリーが拒否したとき、いつもより真剣に...何年もギターに触れず、それほど絶対的に話すことなく、常に不便だけではないと感じた。"
    parts = SpeechService._voicevox_parts(text)
    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(len(part) >= 12 for part in parts)
    assert parts[0].endswith("、")
    assert SpeechService._voicevox_parts(text, split_clauses=False) == [text]
    assert SpeechService._voicevox_parts("こんにちは。お兄ちゃん！") == ["こんにちは。", "お兄ちゃん！"]
    assert SpeechService._voicevox_parts("あ" * 70) == ["あ" * 70]


def test_translation_warmup_starts_before_gui_and_has_its_own_thread():
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding="utf-8")
    assert source.index('name="tts-warmup"') < source.index("    app = QApplication")
    assert "tts_executor.submit(warmup_tts_translation)" not in source
    assert "tts_translation_warmed.wait()" in source
    assert "tts_translation_warmed.set()" in source


def test_mute_during_audio_query_prevents_synthesis():
    service = SpeechService.__new__(SpeechService)
    service._generation = service._active_generation = 0
    service._muted = False
    called = []
    def query(text):
        service._muted = True
        return {"text": text}
    service._voicevox_audio_query = query
    service._voicevox_synthesis = lambda _: called.append("synthesized")
    assert service._prepare_voicevox("待播报") == b""
    assert called == []
