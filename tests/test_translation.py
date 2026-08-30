from pathlib import Path
import queue
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from desktop_pet.audio.translation import (
    JapaneseTranslationService,
    contains_japanese_kana,
    is_safe_japanese_tts_text,
)
from desktop_pet.audio.speech import SpeechService, _SpeechTask


def _settings(temp_dir: str, *, provider: str, api_fallback: bool = False):
    return SimpleNamespace(
        tts_translation_provider=provider,
        tts_translation_model_id="facebook/m2m100_418M",
        tts_translation_model_path=str(Path(temp_dir) / "missing-model"),
        tts_translation_device="cpu",
        tts_translation_compute_type="int8",
        tts_translation_cache_path=str(Path(temp_dir) / "translation-cache.json"),
        tts_translation_cache_size=20,
        tts_translation_max_chars=220,
        enable_tts_translation_api_fallback=api_fallback,
        tts_skip_on_translation_failure=True,
        tts_diag_logs=False,
    )


def test_japanese_language_guard():
    assert contains_japanese_kana("こんにちは、お兄ちゃん。")
    assert is_safe_japanese_tts_text("こんにちは、お兄ちゃん。")
    assert not is_safe_japanese_tts_text("哥哥正在写代码", source_had_han=True)


def test_missing_local_model_never_returns_raw_chinese():
    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(_settings(temp_dir, provider="local"))
        assert service.translate("哥哥正在写代码") == ""


def test_api_provider_and_cache():
    calls: list[str] = []

    def translate_api(text: str) -> str:
        calls.append(text)
        return "お兄ちゃんはコードを書いています。"

    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(
            _settings(temp_dir, provider="api"),
            api_fallback=translate_api,
        )
        first = service.translate("哥哥正在写代码")
        second = service.translate("哥哥正在写代码")
        assert first == "お兄ちゃんはコードを書いています。"
        assert second == first
        assert len(calls) == 1


def test_existing_japanese_bypasses_translation():
    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(_settings(temp_dir, provider="api"))
        assert service.translate("もう準備できたよ。") == "もう準備できたよ。"


class _FakeTranslation:
    def __init__(self, result: str) -> None:
        self.result = result

    def translate(self, _text: str) -> str:
        return self.result


def _speech_worker_with_translation(result: str):
    service = SpeechService.__new__(SpeechService)
    service._provider = "voicevox"
    service._translation_service = _FakeTranslation(result)
    service._diag = False
    service._queue = queue.Queue()
    spoken: list[str] = []
    service._speak_once = lambda text, _on_start=None: spoken.append(text)
    return service, spoken


def test_speech_worker_uses_translated_text_only():
    service, spoken = _speech_worker_with_translation("お兄ちゃん、休んでね。")
    service._queue.put(_SpeechTask("哥哥要休息哦"))
    service._queue.put(None)
    service._run_worker()
    assert spoken == ["お兄ちゃん、休んでね。"]


def test_speech_worker_skips_audio_when_translation_fails():
    service, spoken = _speech_worker_with_translation("")
    callbacks: list[str] = []
    service._queue.put(_SpeechTask("哥哥要休息哦", on_start=lambda: callbacks.append("shown")))
    service._queue.put(None)
    service._run_worker()
    assert not spoken
    assert callbacks == ["shown"]
