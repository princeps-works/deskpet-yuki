import io
import json
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import queue
from urllib.error import URLError

from desktop_pet.audio.gpt_sovits import GPTSoVITSClient
from desktop_pet.audio.speech import SpeechService, _SpeechTask
import desktop_pet.audio.gpt_sovits as client_module
import desktop_pet.audio.speech as speech_module


def test_local_startup_is_idempotent_and_does_not_launch_for_remote_service():
    client = GPTSoVITSClient("http://127.0.0.1:9880", "ref.wav", "こんにちは。")
    client.check_ready = lambda: (False, "offline")
    with patch.object(client_module, "os", SimpleNamespace(name="nt")), patch.object(client_module.subprocess, "run") as run:
        client.ensure_local_service()
        assert run.call_count == 1
        assert run.call_args.kwargs["timeout"] == 180
        client.check_ready = lambda: (True, "ok")
        client.ensure_local_service()
        client.base_url = "https://example.org:9880"
        client.check_ready = lambda: (False, "offline")
        client.ensure_local_service()
        assert run.call_count == 1


def test_unavailable_speech_recovers_once_in_background_and_can_shutdown():
    service = SpeechService.__new__(SpeechService)
    service._runtime_ok = False
    service._provider = "gpt_sovits"
    service._enabled = True
    service._diag = False
    service._worker = None
    service._queue = queue.Queue()
    service._lock = threading.Lock()
    service._muted = False
    service._generation = 0
    service._gpt_sovits_status = "offline"
    checked = []
    started = []
    class Stop:
        count = 0
        stopped = False
        def wait(self, seconds):
            assert seconds == 5
            self.count += 1
            return self.stopped or self.count > 3
        def is_set(self):
            return self.stopped
        def set(self):
            self.stopped = True
    service._recovery_stop = Stop()
    def probe():
        checked.append(threading.current_thread().name)
        service._gpt_sovits_status = "ok" if len(checked) == 2 else "offline"
        return len(checked) == 2
    service._compute_runtime_ok = probe
    service._start_worker = lambda: started.append("worker")
    worker = threading.Thread(target=service._watch_gpt_sovits, name="recovery-test")
    worker.start()
    worker.join(2)
    assert not worker.is_alive()
    assert service.get_status() == (True, "ok")
    assert checked == ["recovery-test", "recovery-test"]
    assert started == ["worker"]
    service._runtime_ok = False
    with patch.object(speech_module, "pygame", SimpleNamespace(mixer=SimpleNamespace(get_init=lambda: False))):
        service.shutdown()
    assert service._recovery_stop.is_set()
    assert service._queue.get_nowait() is None


def test_service_loss_marks_offline_and_finishes_current_comment():
    service = SpeechService.__new__(SpeechService)
    service._provider = "gpt_sovits"
    service._runtime_ok = service._enabled = True
    service._diag = service._muted = False
    service._generation = 0
    service._translation_service = None
    service._queue = queue.Queue()
    events = []
    def fail(*args):
        raise URLError("service stopped")
    service._speak_once = fail
    service._queue.put(_SpeechTask("テスト", on_error=lambda: events.append("text_fallback"),
                                   on_end=lambda: events.append("end")))
    service._queue.put(None)
    service._run_worker()
    assert not service.get_status()[0]
    assert "awaiting recovery" in service.get_status()[1]
    assert events == ["text_fallback", "end"]


def test_shutdown_during_recovery_probe_does_not_restart_playback_worker():
    service = SpeechService.__new__(SpeechService)
    service._runtime_ok = False
    service._lock = threading.Lock()
    class Stop:
        stopped = False
        def wait(self, seconds):
            return self.stopped
        def is_set(self):
            return self.stopped
        def set(self):
            self.stopped = True
    service._recovery_stop = Stop()
    def probe():
        service._recovery_stop.set()
        return True
    service._compute_runtime_ok = probe
    service._start_worker = lambda: (_ for _ in ()).throw(AssertionError("restarted after shutdown"))
    service._watch_gpt_sovits()
    assert not service._runtime_ok


class FakeOpener:
    def __init__(self, data):
        self.data = data
        self.request = None

    def open(self, request, timeout):
        self.request = request
        return io.BytesIO(self.data)


def test_gpt_sovits_checks_reference_and_api_identity():
    with TemporaryDirectory() as directory:
        ref = Path(directory) / "ref.wav"
        client = GPTSoVITSClient("http://127.0.0.1:9880", str(ref), "こんにちは。")
        assert not client.check_ready()[0]
        ref.write_bytes(b"reference")
        client._opener = FakeOpener(b'{"paths":{"/unrelated":{}}}')
        assert not client.check_ready()[0]
        client._opener = FakeOpener(b'{"paths":{"/tts":{}}}')
        assert client.check_ready()[0]


def test_gpt_sovits_requests_japanese_with_reference_and_rejects_json():
    client = GPTSoVITSClient("http://127.0.0.1:9880", "ref.wav", "こんにちは。")
    client._opener = FakeOpener(b"RIFF\x00\x00\x00\x00WAVEaudio")
    assert client.synthesize("びっくりした！").startswith(b"RIFF")
    payload = json.loads(client._opener.request.data)
    assert payload["text_lang"] == payload["prompt_lang"] == "all_ja"
    assert payload["ref_audio_path"] == "ref.wav"
    assert payload["prompt_text"] == "こんにちは。"
    assert payload["streaming_mode"] is False
    client._opener = FakeOpener(b'{"error":"failed"}')
    try:
        client.synthesize("テスト")
    except ValueError:
        pass
    else:
        raise AssertionError("non-audio response accepted")


def test_gpt_sovits_does_not_play_or_display_cancelled_synthesis():
    service = SpeechService.__new__(SpeechService)
    service._provider = "gpt_sovits"
    service._diag = False
    service._generation = service._active_generation = 0
    service._muted = False
    events = []
    class Client:
        def synthesize(self, text):
            service._generation += 1
            return b"audio"
    service._gpt_sovits = Client()
    service._play_voicevox = lambda *args: events.append("played")
    service._speak_once("テスト", lambda: events.append("displayed"))
    assert events == []


def test_gpt_sovits_segments_preserve_text_and_short_reaction():
    assert SpeechService._gpt_sovits_parts("びっくりした！") == ["びっくりした！"]
    text = "さっきまで追いかけられていたのに、もう図書室の話をしているなんて、二人ともすごいね。"
    parts = SpeechService._gpt_sovits_parts(text)
    assert len(parts) > 1
    assert "".join(parts) == text
    assert all(len(part) >= 8 for part in parts)
    assert SpeechService._gpt_sovits_parts("あ" * 40) == ["あ" * 40]


def test_gpt_sovits_prefetches_next_segment_and_notifies_only_once():
    service = SpeechService.__new__(SpeechService)
    service._diag = False
    service._generation = service._active_generation = 0
    service._muted = False
    ready = threading.Event()
    played = []
    notified = []
    class Client:
        def synthesize(self, text):
            if text == "次の文章です。":
                ready.set()
            return text.encode()
    service._gpt_sovits = Client()
    def play(audio, callback):
        played.append(audio.decode())
        if callback:
            callback()
            assert ready.wait(2), "next segment not synthesized during playback"
    service._play_voicevox = play
    service._speak_gpt_sovits("最初の文章です。次の文章です。", lambda: notified.append("start"))
    assert played == ["最初の文章です。", "次の文章です。"]
    assert notified == ["start"]
    service._gpt_sovits_segmented = False
    played.clear()
    service._speak_gpt_sovits("最初の文章です。次の文章です。")
    assert played == ["最初の文章です。次の文章です。"]


def test_gpt_sovits_cancellation_during_playback_stops_remaining_segments():
    service = SpeechService.__new__(SpeechService)
    service._diag = False
    service._generation = service._active_generation = 0
    service._muted = False
    service._gpt_sovits = type("Client", (), {"synthesize": lambda self, text: text.encode()})()
    played = []
    def play(audio, callback):
        played.append(audio)
        service._generation += 1
    service._play_voicevox = play
    service._speak_gpt_sovits("一つ目です。二つ目です。三つ目です。")
    assert len(played) == 1
