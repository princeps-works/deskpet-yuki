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
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from desktop_pet.config.settings import Settings

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


class SpeechService:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.enable_tts
        self._provider = settings.tts_provider.lower()
        self._voice = settings.tts_voice
        self._rate = settings.tts_rate
        self._volume = settings.tts_volume
        self._azure_key = settings.tts_azure_key
        self._azure_region = settings.tts_azure_region
        self._azure_endpoint = settings.tts_azure_endpoint
        self._voicevox_base_url = settings.tts_voicevox_base_url.rstrip("/")
        self._voicevox_speaker = settings.tts_voicevox_speaker
        self._voicevox_engine_path = Path(settings.tts_voicevox_engine_path)
        self._enable_voicevox_auto_launch = settings.enable_voicevox_auto_launch
        self._muted = False
        self._queue: queue.Queue[_SpeechTask | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._azure_synthesizer = None
        self._voicevox_status_detail = ""

        self._runtime_ok = self._compute_runtime_ok()
        self._status_message = self._build_status_message()

        if self._runtime_ok:
            if self._provider in {"edge", "voicevox"}:
                pygame.mixer.init()
            self._worker = threading.Thread(target=self._run_worker, daemon=True)
            self._worker.start()

    def _compute_runtime_ok(self) -> bool:
        if not self._enabled:
            return False
        if self._provider == "edge":
            return edge_tts is not None and pygame is not None
        if self._provider == "azure":
            has_auth = bool(self._azure_key) and (bool(self._azure_region) or bool(self._azure_endpoint))
            return speechsdk is not None and has_auth
        if self._provider == "voicevox":
            if pygame is None or not self._voicevox_base_url:
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

            subprocess.Popen(
                [str(self._voicevox_engine_path)],
                cwd=str(self._voicevox_engine_path.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )

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

    def speak(self, text: str) -> None:
        if not self._runtime_ok:
            return
        if self._muted:
            return
        cleaned = self._clean_text(text)
        if not cleaned:
            return
        self.stop_current()
        self._clear_queue()
        self._queue.put(_SpeechTask(cleaned))

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if muted:
            self.stop_current()
            self._clear_queue()

    def is_muted(self) -> bool:
        return self._muted

    def stop_current(self) -> None:
        if not self._runtime_ok:
            return
        with self._lock:
            if self._provider == "edge":
                try:
                    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                except Exception:
                    pass
            elif self._provider == "voicevox":
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

    def _clear_queue(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
                if item is None:
                    self._queue.put(None)
                    break
            except queue.Empty:
                break

    def shutdown(self) -> None:
        if not self._runtime_ok:
            return
        self.stop_current()
        self._queue.put(None)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)
        with self._lock:
            if self._provider in {"edge", "voicevox"}:
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
            try:
                self._speak_once(task.text)
            except Exception:
                # Do not break app flow on TTS failures.
                pass

    def _speak_once(self, text: str) -> None:
        if self._provider == "edge":
            self._speak_edge(text)
            return
        if self._provider == "voicevox":
            self._speak_voicevox(text)
            return
        if self._provider == "azure":
            self._speak_azure(text)

    def _speak_edge(self, text: str) -> None:
        fd, tmp_path = tempfile.mkstemp(prefix="pet_tts_", suffix=".mp3")
        os.close(fd)
        try:
            asyncio.run(self._synthesize_to_file(text, tmp_path))
            with self._lock:
                if not pygame.mixer.get_init():
                    return
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

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

    def _speak_voicevox(self, text: str) -> None:
        query = self._voicevox_audio_query(text)
        wav_bytes = self._voicevox_synthesis(query)

        fd, tmp_path = tempfile.mkstemp(prefix="pet_tts_", suffix=".wav")
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(wav_bytes)

            with self._lock:
                if not pygame.mixer.get_init():
                    return
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()

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

    def _speak_azure(self, text: str) -> None:
        synth = self._get_azure_synthesizer()
        if synth is None:
            return
        ssml = self._build_azure_ssml(text)
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
