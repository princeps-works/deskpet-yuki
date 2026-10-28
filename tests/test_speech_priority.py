import queue
import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import patch

from desktop_pet.audio.speech import SpeechService
import desktop_pet.audio.speech as module


def service():
    s = SpeechService.__new__(SpeechService)
    s._runtime_ok = True
    s._muted = s._diag = False
    s._generation = s._active_generation = 0
    s._active_task = s._latest_comment_task = None
    s._lock = threading.Lock()
    s._queue = queue.Queue()
    s._translation_service = None
    s._provider = "gpt_sovits"
    return s


def run(s):
    s._worker = threading.Thread(target=s._run_worker)
    s._worker.start()


def test_unspoken_latest_replaces_synthesis_and_finishes_discarded_callbacks():
    s = service()
    entered, release, played = threading.Event(), threading.Event(), threading.Event()
    starts, ends = [], []
    def speak(text, callback):
        if text == "A":
            entered.set()
            assert release.wait(2)
            assert s._active_task.cancelled.is_set()
            raise CancelledError()
        starts.append(text)
        s._notify_on_start(callback)
        played.set()
    s._speak_once = speak
    with patch.object(module, "pygame", None):
        run(s)
        try:
            s.speak("A", on_end=lambda: ends.append("A"))
            assert entered.wait(2)
            s.speak("B", on_end=lambda: ends.append("B"))
            s.speak("C", on_end=lambda: ends.append("C"))
            release.set()
            assert played.wait(2)
        finally:
            release.set()
            s.shutdown()
    assert starts == ["C"]
    assert sorted(ends) == ["A", "B", "C"]


def test_started_comment_finishes_then_only_latest_pending_comment_plays():
    s = service()
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    order = []
    def speak(text, callback):
        s._notify_on_start(callback)
        order.append(text)
        if text == "A":
            started.set()
            assert release.wait(2)
            assert not s._active_task.cancelled.is_set()
        else:
            done.set()
    s._speak_once = speak
    with patch.object(module, "pygame", None):
        run(s)
        try:
            s.speak("A")
            assert started.wait(2)
            s.speak("B")
            s.speak("C")
            assert order == ["A"]
            release.set()
            assert done.wait(2)
        finally:
            release.set()
            s.shutdown()
    assert order == ["A", "C"]


def test_user_reply_interrupts_started_comment_and_is_protected_from_auto_comments():
    s = service()
    auto_started, user_started, release, done = (threading.Event() for _ in range(4))
    order = []
    def speak(text, callback):
        s._notify_on_start(callback)
        order.append(text)
        if text == "A":
            auto_started.set()
            assert s._active_task.cancelled.wait(2)
            raise CancelledError()
        if text == "user":
            user_started.set()
            assert release.wait(2)
            assert not s._active_task.cancelled.is_set()
        else:
            done.set()
    s._speak_once = speak
    with patch.object(module, "pygame", None):
        run(s)
        try:
            s.speak("A")
            assert auto_started.wait(2)
            s.speak("discard")
            s.speak("user", priority=1)
            assert user_started.wait(2)
            s.speak("B")
            s.speak("C")
            release.set()
            assert done.wait(2)
        finally:
            release.set()
            s.shutdown()
    assert order == ["A", "user", "C"]


def test_user_reply_wins_when_old_comment_was_dequeued_but_not_claimed():
    s = service()
    dequeued, release, done = threading.Event(), threading.Event(), threading.Event()
    order = []
    class GatedQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            item = super().get(block, timeout)
            if block and item is not None and item.text == "A":
                dequeued.set()
                assert release.wait(2)
            return item
    s._queue = GatedQueue()
    def speak(text, callback):
        order.append(text)
        s._notify_on_start(callback)
        done.set()
    s._speak_once = speak
    with patch.object(module, "pygame", None):
        s.speak("A")
        run(s)
        try:
            assert dequeued.wait(2)
            s.speak("user", priority=1)
            release.set()
            assert done.wait(2)
        finally:
            release.set()
            s.shutdown()
    assert order == ["user"]
