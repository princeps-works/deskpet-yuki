from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, ProxyHandler, build_opener


class GPTSoVITSClient:
    def __init__(self, base_url: str, reference_audio: str, reference_text: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self._opener = build_opener(ProxyHandler({}))

    def ensure_local_service(self) -> None:
        endpoint = urlparse(self.base_url)
        if os.name != "nt" or endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != 9880:
            return
        if self.check_ready()[0]:
            return
        script = Path(__file__).resolve().parent.parent / "scripts/start_gpt_sovits.ps1"
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                check=True, timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[STARTUP] GPT-SoVITS auto-start failed: {type(exc).__name__}")

    def check_ready(self) -> tuple[bool, str]:
        if not Path(self.reference_audio).is_file() or not self.reference_text.strip():
            return False, "GPT-SoVITS reference audio or text missing"
        try:
            with self._opener.open(f"{self.base_url}/openapi.json", timeout=3) as response:
                schema = json.load(response)
            if "/tts" not in schema.get("paths", {}):
                return False, "endpoint is not a GPT-SoVITS TTS API"
            return True, "ok (provider=gpt_sovits)"
        except Exception as exc:
            return False, f"GPT-SoVITS service unavailable: {type(exc).__name__}; run scripts/start_gpt_sovits.ps1"

    def synthesize(self, text: str) -> bytes:
        payload = {
            "text": text, "text_lang": "all_ja",
            "ref_audio_path": self.reference_audio,
            "prompt_text": self.reference_text, "prompt_lang": "all_ja",
            "text_split_method": "cut5", "batch_size": 1,
            "parallel_infer": True, "streaming_mode": False,
            "use_cuda_graph": False,
            "media_type": "wav", "seed": 42,
        }
        request = Request(f"{self.base_url}/tts", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        with self._opener.open(request, timeout=45) as response:
            audio = response.read()
        if not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            raise ValueError("GPT-SoVITS returned invalid WAV audio")
        return audio
