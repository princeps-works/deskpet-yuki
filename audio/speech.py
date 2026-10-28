from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import subprocess
import threading
import tempfile
import time
from html import escape
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen

from desktop_pet.config.settings import Settings
from desktop_pet.audio.translation import JapaneseTranslationService, TranslationStatus
from desktop_pet.audio.gpt_sovits import GPTSoVITSClient

try:
    import edge_tts
except Exception:  # pragma: no cover
    edge_tts = None

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None

try:
    import azure.cognitiveservices.speech as speechsdk
except Exception:  # pragma: no cover
    speechsdk = None


@dataclass
class _SpeechTask:
    text: str
    on_start: Optional[Callable[[], None]] = None
    on_error: Optional[Callable[[], None]] = None
    generation: int = 0
    is_valid: Optional[Callable[[], bool]] = None
    on_ready: Optional[Callable[[], bool]] = None
    on_end: Optional[Callable[[], None]] = None
    priority: int = 0
    started: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _end_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def finish(self) -> None:
        with self._end_lock:
            callback, self.on_end = self.on_end, None
        if callback is not None:
            try:
                callback()
            except Exception:
                pass


class SpeechService:
    def __init__(
        self,
        settings: Settings,
        translation_api_fallback: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._enabled = settings.enable_tts
        self._provider = settings.tts_provider.lower()
        self._voice = settings.tts_voice
        self._rate = settings.tts_rate
        self._volume = settings.tts_volume
        self._diag = bool(getattr(settings, "tts_diag_logs", False))
        self._voicevox_clause_streaming = os.getenv("TTS_VOICEVOX_CLAUSE_STREAMING", "true").lower() in {"1", "true", "yes", "on"}
        self._gpt_sovits_segmented = os.getenv("TTS_GPT_SOVITS_SEGMENTED", "true").lower() in {"1", "true", "yes", "on"}
        self._azure_key = settings.tts_azure_key
        self._azure_region = settings.tts_azure_region
        self._azure_endpoint = settings.tts_azure_endpoint
        self._voicevox_base_url = settings.tts_voicevox_base_url.rstrip("/")
        self._voicevox_speaker = settings.tts_voicevox_speaker
        self._voicevox_engine_path = Path(settings.tts_voicevox_engine_path)
        self._enable_voicevox_auto_launch = settings.enable_voicevox_auto_launch
        self._gpt_sovits = None
        self._gpt_sovits_status = ""
        if self._provider == "gpt_sovits":
            self._gpt_sovits = GPTSoVITSClient(
                settings.tts_gpt_sovits_base_url,
                settings.tts_gpt_sovits_reference_audio,
                settings.tts_gpt_sovits_reference_text,
            )
        self._translation_service = None
        if self._provider == "gpt_sovits" or (self._provider == "voicevox" and settings.enable_voicevox_ja_translation):
            self._translation_service = JapaneseTranslationService(
                settings,
                api_fallback=translation_api_fallback,
            )
        self._muted = False
        self._queue: queue.Queue[_SpeechTask | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._active_generation = 0
        self._active_task = None
        self._latest_comment_task = None
        self._azure_synthesizer = None
        self._voicevox_status_detail = ""
        # pid of an engine this process launched, so the CPU-policy code does not
        # have to discover it by enumerating every process on the machine. That
        # scan costs ~1.9s here (386 processes, exe + cmdline each) and used to
        # run on the GUI thread every 20s.
        self._voicevox_engine_pid = 0

        self._recovery_stop = threading.Event()
        self._recovery_worker = None
        if self._enabled and self._gpt_sovits is not None:
            self._gpt_sovits.ensure_local_service()

        self._runtime_ok = self._compute_runtime_ok()
        self._status_message = self._build_status_message()

        if self._runtime_ok:
            self._start_worker()
        if self._enabled and self._gpt_sovits is not None and pygame is not None:
            self._recovery_worker = threading.Thread(target=self._watch_gpt_sovits, daemon=True)
            self._recovery_worker.start()

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if self._provider in {"edge", "voicevox", "gpt_sovits"}:
            pygame.mixer.init()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def _watch_gpt_sovits(self) -> None:
        while not self._recovery_stop.wait(5):
            if self._runtime_ok:
                continue
            ready = self._compute_runtime_ok()
            with self._lock:
                if self._recovery_stop.is_set():
                    return
                if ready:
                    try:
                        self._start_worker()
                    except Exception as exc:
                        self._gpt_sovits_status = f"audio initialization failed: {type(exc).__name__}"
                    else:
                        self._runtime_ok = True
                        self._diag_log("GPT-SoVITS recovered")
                self._status_message = self._build_status_message()

    def _diag_log(self, message: str) -> None:
        if self._diag:
            print(f"[TTS-DIAG] {message}")

    def _compute_runtime_ok(self) -> bool:
        if not self._enabled:
            return False
        if self._provider == "edge":
            return edge_tts is not None and pygame is not None
        if self._provider == "gpt_sovits":
            if pygame is None:
                self._gpt_sovits_status = "pygame not installed"
                return False
            ready, self._gpt_sovits_status = self._gpt_sovits.check_ready()
            if ready:
                # Startup and recovery both finish graph capture before starting playback.
                started = time.perf_counter()
                try:
                    self._gpt_sovits.synthesize(self._gpt_sovits.reference_text)
                except (URLError, TimeoutError, ConnectionError, ValueError) as exc:
                    self._gpt_sovits_status = f"GPT-SoVITS warmup failed: {type(exc).__name__}; awaiting recovery"
                    return False
                print(f"[STARTUP] GPT-SoVITS synthesis warmed (CUDA Graph requested), elapsed={time.perf_counter() - started:.3f}s", flush=True)
            return ready
        if self._provider == "azure":
            has_auth = bool(self._azure_key) and (bool(self._azure_region) or bool(self._azure_endpoint))
            return speechsdk is not None and has_auth
        if self._provider == "voicevox":
            if pygame is None or not self._voicevox_base_url:
                return False
            assets_ok, assets_detail = self._check_voicevox_assets()
            if not assets_ok:
                self._voicevox_status_detail = assets_detail
                return False
            ok, detail = self._check_voicevox_engine()
            if not ok and self._enable_voicevox_auto_launch:
                launched, launch_detail = self._try_launch_voicevox_engine()
                if launched:
                    ok, detail = self._check_voicevox_engine()
                    if ok:
                        detail = f"auto-launched, {detail}"
                    else:
                        detail = f"auto-launch attempted: {launch_detail}; {detail}"
                else:
                    detail = f"{detail}; auto-launch failed: {launch_detail}"
            self._voicevox_status_detail = detail
            return ok
        return False

    def _check_voicevox_assets(self) -> tuple[bool, str]:
        try:
            engine_dir = self._voicevox_engine_path.parent
            model_dir = engine_dir / "model"
            if not model_dir.exists():
                return False, f"voicevox model dir not found: {model_dir}"

            vvm_files = sorted(model_dir.glob("*.vvm"))
            if not vvm_files:
                return False, f"voicevox model files not found under: {model_dir}"

            # Validate a few files to avoid long startup waits when model files are LFS pointers.
            probe_files = vvm_files[: min(3, len(vvm_files))]
            for file_path in probe_files:
                try:
                    size = file_path.stat().st_size
                except Exception:
                    return False, f"voicevox model unreadable: {file_path.name}"

                if size < 1024 * 1024:
                    try:
                        head = file_path.read_text(encoding="utf-8", errors="ignore")[:120]
                    except Exception:
                        head = ""
                    if "git-lfs.github.com/spec/v1" in head:
                        return (
                            False,
                            (
                                "voicevox model appears to be Git LFS pointer; "
                                "run 'git lfs pull' in project root"
                            ),
                        )
                    return False, f"voicevox model file too small: {file_path.name} ({size} bytes)"

            return True, "voicevox assets ok"
        except Exception as exc:
            return False, f"voicevox asset check failed: {exc}"

    def _build_status_message(self) -> str:
        if not self._enabled:
            return "tts disabled"
        if self._provider == "edge":
            if edge_tts is None:
                return "edge_tts not installed"
            if pygame is None:
                return "pygame not installed"
            return "ok (provider=edge)"
        if self._provider == "azure":
            if speechsdk is None:
                return "azure-cognitiveservices-speech not installed"
            if not self._azure_key:
                return "missing TTS_AZURE_KEY"
            if not (self._azure_region or self._azure_endpoint):
                return "missing TTS_AZURE_REGION or TTS_AZURE_ENDPOINT"
            return "ok (provider=azure)"
        if self._provider == "voicevox":
            if pygame is None:
                return "pygame not installed"
            if self._voicevox_status_detail:
                return f"{self._voicevox_status_detail} (provider=voicevox)"
            return "ok (provider=voicevox)"
        if self._provider == "gpt_sovits":
            return self._gpt_sovits_status
        return f"unsupported provider: {self._provider}"

    def _check_voicevox_engine(self) -> tuple[bool, str]:
        version_url = f"{self._voicevox_base_url}/version"
        request = Request(url=version_url, method="GET")
        try:
            with urlopen(request, timeout=5) as response:
                version = response.read().decode("utf-8").strip().strip('"')
                if version:
                    return True, f"ok engine={version}"
                return True, "ok engine=unknown"
        except Exception as exc:
            return False, f"voicevox engine unreachable: {exc}"

    def _try_launch_voicevox_engine(self) -> tuple[bool, str]:
        if not self._voicevox_engine_path.exists():
            return False, f"engine path not found: {self._voicevox_engine_path}"

        try:
            creationflags = 0
            if os.name == "nt":
                creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
                creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

            process = subprocess.Popen(
                [str(self._voicevox_engine_path)],
                cwd=str(self._voicevox_engine_path.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            # Remember it: the policy code can then skip the full process scan.
            try:
                self._voicevox_engine_pid = int(getattr(process, "pid", 0) or 0)
            except Exception:
                self._voicevox_engine_pid = 0

            # Wait briefly for the local API to become ready.
            for _ in range(24):
                ok, _ = self._check_voicevox_engine()
                if ok:
                    return True, "engine started"
                time.sleep(0.5)
            return False, "engine process started but API not ready within timeout"
        except Exception as exc:
            return False, str(exc)

    def get_status(self) -> tuple[bool, str]:
        return self._runtime_ok, self._status_message

    @property
    def voicevox_engine_pid(self) -> int:
        """pid of a VOICEVOX engine this process launched, or 0 if we did not."""
        return int(self._voicevox_engine_pid)

    def warmup_translation(self) -> TranslationStatus:
        if self._translation_service is None:
            return TranslationStatus(True, "translation disabled or not required")
        return self._translation_service.warmup()

    def get_translation_status(self) -> TranslationStatus:
        if self._translation_service is None:
            return TranslationStatus(True, "translation disabled or not required")
        return self._translation_service.get_status()

    def speak(self, text: str, on_start: Optional[Callable[[], None]] = None,
              on_error: Optional[Callable[[], None]] = None,
              is_valid: Optional[Callable[[], bool]] = None,
              on_ready: Optional[Callable[[], bool]] = None,
              on_end: Optional[Callable[[], None]] = None,
              priority: int = 0) -> bool:
        if not self._runtime_ok:
            return False
        if self._muted:
            return False
        cleaned = self._clean_text(text)
        if not cleaned:
            return False
        if is_valid is not None and not is_valid():
            return False
        self._diag_log(f"enqueue provider={self._provider} chars={len(cleaned)}")
        interrupted = None
        with self._lock:
            active = getattr(self, "_active_task", None)
            if priority > 0 and (active is None or active.cancelled.is_set()):
                self._generation += 1  # Also invalidate a lower-priority task already dequeued by the worker.
            if (active is not None and not active.cancelled.is_set()
                    and (priority > active.priority or (priority == active.priority and not active.started))):
                interrupted = self._interrupt_locked()
            removed = self._clear_queue(priority)
            task = _SpeechTask(cleaned, on_start=on_start, on_error=on_error,
                               generation=self._generation, is_valid=is_valid,
                               on_ready=on_ready, on_end=on_end, priority=priority)
            if priority == 0:
                self._latest_comment_task = task
            self._queue.put(task)
        for old in removed + ([interrupted] if interrupted is not None else []):
            old.finish()
        return True

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self.stop_current()
            for task in self._clear_queue():
                task.finish()

    def is_muted(self) -> bool:
        return self._muted

    def stop_current(self) -> None:
        if not self._runtime_ok and getattr(self, "_worker", None) is None:
            return
        with self._lock:
            active_task = self._interrupt_locked()
        if active_task is not None:
            active_task.finish()

    def _interrupt_locked(self):
        self._generation += 1
        active = getattr(self, "_active_task", None)
        if active is not None:
            active.cancelled.set()
        if self._provider in {"edge", "voicevox", "gpt_sovits"}:
            try:
                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except Exception:
                pass
        elif self._provider == "azure":
            try:
                synth = self._get_azure_synthesizer()
                if synth is not None:
                    synth.stop_speaking_async()
            except Exception:
                pass
        return active

    def _clear_queue(self, max_priority: int = 1) -> list:
        removed, keep = [], []
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    keep.append(None)
                    break
                if item.priority <= max_priority:
                    item.cancelled.set()
                    removed.append(item)
                else:
                    keep.append(item)
            except queue.Empty:
                break
        for item in keep:
            self._queue.put(item)
        return removed

    def shutdown(self) -> None:
        recovery_stop = getattr(self, "_recovery_stop", None)
        if recovery_stop is not None:
            with self._lock:
                recovery_stop.set()
        self.stop_current()
        for task in self._clear_queue():
            task.finish()
        self._queue.put(None)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)
        with self._lock:
            if self._provider in {"edge", "voicevox", "gpt_sovits"}:
                try:
                    if pygame.mixer.get_init():
                        pygame.mixer.quit()
                except Exception:
                    pass

    def _run_worker(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            with self._lock:
                latest = getattr(self, "_latest_comment_task", None)
                if task.priority == 0 and latest is not None and task is not latest:
                    task.cancelled.set()
                self._active_task = task
                self._active_generation = task.generation
                self._active_valid = task.is_valid
                self._active_on_ready = task.on_ready
            if not self._is_current():
                task.cancelled.set()
                with self._lock:
                    if self._active_task is task:
                        self._active_task = None
                task.finish()
                continue
            started_ts = time.perf_counter()
            try:
                prepared_text = task.text
                if self._translation_service is not None:
                    translation_started = time.perf_counter()
                    prepared_text = self._translation_service.translate(task.text)
                    if not self._is_current():
                        continue
                    if not prepared_text:
                        self._diag_log("skip voicevox: translation unavailable or unsafe")
                        self._notify_on_start(task.on_error or task.on_start)
                        continue
                    self._diag_log(
                        f"translation source_chars={len(task.text)} ja_chars={len(prepared_text)} elapsed={time.perf_counter() - translation_started:.3f}s"
                    )
                self._speak_once(prepared_text, task.on_start)
                self._diag_log(f"done provider={self._provider} total={time.perf_counter() - started_ts:.3f}s")
            except Exception as exc:
                # Do not break app flow on TTS failures.
                if self._provider == "gpt_sovits" and isinstance(exc, (URLError, TimeoutError, ConnectionError)):
                    self._runtime_ok = False
                    self._gpt_sovits_status = f"GPT-SoVITS request failed: {type(exc).__name__}; awaiting recovery"
                    self._status_message = self._build_status_message()
                self._diag_log(f"failed provider={self._provider} reason={type(exc).__name__} total={time.perf_counter() - started_ts:.3f}s")
                if self._is_current():
                    self._notify_on_start(task.on_error or task.on_start)
            finally:
                task.finish()
                with self._lock:
                    if self._active_task is task:
                        self._active_task = None

    def _is_current(self) -> bool:
        valid = getattr(self, "_active_valid", None)
        task = getattr(self, "_active_task", None)
        return (self._active_generation == self._generation and not self._muted
                and (task is None or not task.cancelled.is_set())
                and (valid is None or valid()))

    def _speak_once(self, text: str, on_start: Optional[Callable[[], None]] = None) -> None:
        if self._provider == "edge":
            self._speak_edge(text, on_start=on_start)
            return
        if self._provider == "voicevox":
            self._speak_voicevox(text, on_start=on_start)
            return
        if self._provider == "gpt_sovits":
            self._speak_gpt_sovits(text, on_start)
            return
        if self._provider == "azure":
            self._speak_azure(text, on_start=on_start)

    def _notify_on_start(self, on_start: Optional[Callable[[], None]]) -> None:
        self._active_valid = None  # Once playback begins, finish the whole comment.
        task = getattr(self, "_active_task", None)
        if task is not None:
            task.started = True
        if on_start is None:
            return
        try:
            on_start()
        except Exception:
            pass

    def _await_playback_ready(self) -> bool:
        ready = getattr(self, "_active_on_ready", None)
        self._active_on_ready = None
        return self._is_current() and (ready is None or ready() is not False) and self._is_current()

    def _speak_edge(self, text: str, on_start: Optional[Callable[[], None]] = None) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix="pet_tts_", suffix=".mp3")
        os.close(fd)
        try:
            asyncio.run(self._synthesize_to_file(text, tmp_path))
            if not self._await_playback_ready():
                return
            with self._lock:
                if not self._is_current():
                    return
                if not pygame.mixer.get_init():
                    raise RuntimeError("audio mixer unavailable")
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                self._notify_on_start(on_start)

            while True:
                with self._lock:
                    busy = pygame.mixer.music.get_busy() if pygame.mixer.get_init() else False
                if not busy:
                    break
                threading.Event().wait(0.05)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _gpt_sovits_parts(text: str) -> list[str]:
        output = []
        for sentence in re.split(r"(?<=[。！？!?])\s*", text):
            part = sentence.strip()
            while len(part) > 28:
                boundaries = [match.end() for match in re.finditer(r"[、，,；;]|…{2,}|\.{3,}", part)
                              if 8 <= match.end() <= len(part) - 8]
                if not boundaries:
                    break
                split = min(boundaries, key=lambda index: abs(index - 24))
                output.append(part[:split].strip())
                part = part[split:].strip()
            if part:
                output.append(part)
        return output

    def _speak_gpt_sovits(self, text: str, on_start=None) -> None:
        parts = (self._gpt_sovits_parts(text) if getattr(self, "_gpt_sovits_segmented", True)
                 else [text])
        def prepare(part):
            if not self._is_current():
                return b""
            started = time.perf_counter()
            task = getattr(self, "_active_task", None)
            audio = (self._gpt_sovits.synthesize(part, cancel_event=task.cancelled, is_valid=self._is_current)
                     if task is not None else self._gpt_sovits.synthesize(part))
            self._diag_log(f"gpt_sovits synthesis={time.perf_counter() - started:.3f}s chars={len(part)}")
            return audio
        if not parts:
            return
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpt-sovits-synthesis") as executor:
            pending = executor.submit(prepare, parts[0])
            for index in range(len(parts)):
                audio = pending.result()
                if not self._is_current() or not audio:
                    return
                if index + 1 < len(parts):
                    pending = executor.submit(prepare, parts[index + 1])
                self._play_voicevox(audio, on_start if index == 0 else None)

    @staticmethod
    def _voicevox_parts(text: str, *, split_clauses: bool = True) -> list[str]:
        output = []
        for sentence in re.split(r"(?<=[。！？!?])\s*", text):
            if not sentence:
                continue
            pending = [sentence]
            while pending:
                part = pending.pop()
                boundaries = [match.end() for match in re.finditer(r"[、，,；;]", part)
                              if 12 <= match.end() <= len(part) - 12]
                if split_clauses and len(part) > 48 and boundaries:
                    split = min(boundaries, key=lambda index: abs(index - len(part) / 2))
                    pending.extend([part[split:], part[:split]])
                else:
                    output.append(part)
        return output

    def _speak_voicevox(self, text: str, on_start: Optional[Callable[[], None]] = None) -> None:
        # Keep sentence boundaries; synthesise one sentence ahead while playing.
        parts = self._voicevox_parts(text, split_clauses=getattr(self, "_voicevox_clause_streaming", True))
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicevox-synthesis") as executor:
            pending = executor.submit(self._prepare_voicevox, parts[0])
            for index in range(len(parts)):
                wav_bytes = pending.result()
                if not self._is_current() or not wav_bytes:
                    return
                if index + 1 < len(parts):
                    pending = executor.submit(self._prepare_voicevox, parts[index + 1])
                self._play_voicevox(wav_bytes, on_start if index == 0 else None)

    def _prepare_voicevox(self, text: str) -> bytes:
        if not self._is_current():
            return b""
        t0 = time.perf_counter()
        query = self._voicevox_audio_query(text)
        t1 = time.perf_counter()
        if not self._is_current():
            return b""
        wav_bytes = self._voicevox_synthesis(query)
        self._diag_log(f"voicevox query={t1 - t0:.3f}s synthesis={time.perf_counter() - t1:.3f}s")
        return wav_bytes

    def _play_voicevox(self, wav_bytes: bytes, on_start=None) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix="pet_tts_", suffix=".wav")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(wav_bytes)

            if not self._await_playback_ready():
                return
            with self._lock:
                if not self._is_current():
                    return
                if not pygame.mixer.get_init():
                    raise RuntimeError("audio mixer unavailable")
                t3 = time.perf_counter()
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                self._notify_on_start(on_start)
                t4 = time.perf_counter()
                self._diag_log(f"voicevox load+play={(t4 - t3):.3f}s")

            while True:
                with self._lock:
                    busy = pygame.mixer.music.get_busy() if pygame.mixer.get_init() else False
                if not busy:
                    break
                threading.Event().wait(0.05)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _voicevox_audio_query(self, text: str) -> dict:
        params = urlencode({"text": text, "speaker": self._voicevox_speaker})
        url = f"{self._voicevox_base_url}/audio_query?{params}"
        request = Request(url=url, method="POST", data=b"")
        with urlopen(request, timeout=20) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _voicevox_synthesis(self, query: dict) -> bytes:
        params = urlencode({"speaker": self._voicevox_speaker})
        url = f"{self._voicevox_base_url}/synthesis?{params}"
        body = json.dumps(query).encode("utf-8")
        request = Request(
            url=url,
            method="POST",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            return response.read()

    def _get_azure_synthesizer(self):
        if speechsdk is None:
            return None
        if self._azure_synthesizer is not None:
            return self._azure_synthesizer

        speech_config = None
        if self._azure_endpoint:
            speech_config = speechsdk.SpeechConfig(subscription=self._azure_key, endpoint=self._azure_endpoint)
        elif self._azure_region:
            speech_config = speechsdk.SpeechConfig(subscription=self._azure_key, region=self._azure_region)

        if speech_config is None:
            return None

        speech_config.speech_synthesis_voice_name = self._voice
        self._azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        return self._azure_synthesizer

    def _speak_azure(self, text: str, on_start: Optional[Callable[[], None]] = None) -> None:
        synth = self._get_azure_synthesizer()
        if synth is None:
            return
        ssml = self._build_azure_ssml(text)
        self._notify_on_start(on_start)
        result = synth.speak_ssml_async(ssml).get()
        if speechsdk is None:
            return
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = ""
            if result.reason == speechsdk.ResultReason.Canceled:
                cancel = speechsdk.SpeechSynthesisCancellationDetails(result)
                details = f" canceled: {cancel.reason}"
                if cancel.error_details:
                    details += f" ({cancel.error_details})"
            raise RuntimeError(f"azure tts failed: {result.reason}{details}")

    def _build_azure_ssml(self, text: str) -> str:
        safe_text = escape(text)
        rate = self._normalize_edge_percent(self._rate)
        volume = self._normalize_edge_percent(self._volume)
        prosody_attrs = []
        if rate is not None:
            prosody_attrs.append(f'rate="{rate}"')
        if volume is not None:
            prosody_attrs.append(f'volume="{volume}"')
        prosody = " ".join(prosody_attrs)
        if prosody:
            return (
                "<speak version=\"1.0\" xml:lang=\"zh-CN\">"
                f"<voice name=\"{escape(self._voice)}\"><prosody {prosody}>{safe_text}</prosody></voice>"
                "</speak>"
            )
        return (
            "<speak version=\"1.0\" xml:lang=\"zh-CN\">"
            f"<voice name=\"{escape(self._voice)}\">{safe_text}</voice>"
            "</speak>"
        )

    @staticmethod
    def _normalize_edge_percent(value: str) -> str | None:
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+%", text):
            return text if text.startswith(("+", "-")) else f"+{text}"
        return None

    async def _synthesize_to_file(self, text: str, output_path: str) -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=self._voice,
            rate=self._rate,
            volume=self._volume,
        )
        await communicate.save(output_path)

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"https?://\S+", "", cleaned)
        cleaned = re.sub(r"`{1,3}.*?`{1,3}", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Avoid very long speech blocks.
        return cleaned[:160]
