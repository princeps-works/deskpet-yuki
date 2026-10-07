from pathlib import Path
import io
import json
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


class _IndexResponse:
    def __init__(self, text):
        self.body = json.dumps({"choices": [{"message": {"content": text}}]}).encode()

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return io.BytesIO(self.body)


def test_index_request_and_persistent_cache():
    with TemporaryDirectory() as temp_dir:
        settings = _settings(temp_dir, provider="index")
        service = JapaneseTranslationService(settings)
        response = _IndexResponse("びっくりした！\n本当に怖かった。")
        service._index_opener = response
        service._translate_local = lambda _: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
        assert service.translate("吓死我了") == "びっくりした！ 本当に怖かった。"
        payload = json.loads(response.request.data)
        assert payload["model"] == "Index-Translate-35B-A3B"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert response.timeout == 3.0
        restored = JapaneseTranslationService(settings)
        restored._translate_index = lambda _: (_ for _ in ()).throw(AssertionError("cache miss"))
        assert restored.translate("吓死我了") == "びっくりした！ 本当に怖かった。"


def test_index_failure_falls_back_without_poisoning_cache():
    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(_settings(temp_dir, provider="index"))
        class Timeout:
            def open(self, request, timeout):
                raise TimeoutError("offline")
        service._index_opener = Timeout()
        calls = []
        service._translate_local = lambda text: calls.append(text) or "怖かった！"
        assert service.translate("吓死我了") == "怖かった！"
        assert len(calls) == 1
        service._index_opener = _IndexResponse("びっくりした！")
        assert service.translate("吓死我了") == "びっくりした！"
        assert len(calls) == 1


def test_index_invalid_output_and_local_failure_never_leak_chinese():
    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(_settings(temp_dir, provider="index"))
        service._index_opener = _IndexResponse("吓死我了")
        service._translate_local = lambda _: ""
        assert service.translate("吓死我了") == ""


def test_translation_cache_isolates_provider_model_and_rules():
    import desktop_pet.audio.translation as module
    with TemporaryDirectory() as temp_dir:
        settings = _settings(temp_dir, provider="api")
        api = JapaneseTranslationService(settings, api_fallback=lambda _: "こんにちは。")
        api.translate("你好")
        local_settings = _settings(temp_dir, provider="local")
        local = JapaneseTranslationService(local_settings)
        assert local._get_cached("你好") == ""
        settings.model_name = "different-model"
        assert JapaneseTranslationService(settings)._get_cached("你好") == ""
        index = JapaneseTranslationService(_settings(temp_dir, provider="index"))
        index._put_cached("你好", "こんにちは。")
        original = module._INDEX_PROMPT
        try:
            module._INDEX_PROMPT = original + "変更"
            assert index._get_cached("你好") == ""
        finally:
            module._INDEX_PROMPT = original


def test_legacy_cache_is_not_reused():
    with TemporaryDirectory() as temp_dir:
        settings = _settings(temp_dir, provider="index")
        Path(settings.tts_translation_cache_path).write_text(json.dumps({
            "version": 1, "entries": [["old-source-only-hash", "こんにちは。"]]
        }), encoding="utf-8")
        assert not JapaneseTranslationService(settings)._cache


def test_settings_accept_index_translation_channel():
    import os
    from desktop_pet.config.settings import load_settings
    previous = os.environ.get("TTS_TRANSLATION_PROVIDER")
    try:
        for provider in ("index", "index_local"):
            os.environ["TTS_TRANSLATION_PROVIDER"] = provider
            with TemporaryDirectory() as temp_dir:
                assert load_settings(Path(temp_dir)).tts_translation_provider == provider
    finally:
        if previous is None:
            os.environ.pop("TTS_TRANSLATION_PROVIDER", None)
        else:
            os.environ["TTS_TRANSLATION_PROVIDER"] = previous


def test_index_local_uses_loopback_and_isolated_cache():
    with TemporaryDirectory() as temp_dir:
        local = JapaneseTranslationService(_settings(temp_dir, provider="index_local"))
        response = _IndexResponse("びっくりした！")
        local._index_opener = response
        assert local.translate("吓死我了") == "びっくりした！"
        assert response.request.full_url == "http://127.0.0.1:8768/v1/chat/completions"
        assert json.loads(response.request.data)["model"] == "Index-Translate-2B-Q5_K_M"
        online = JapaneseTranslationService(_settings(temp_dir, provider="index"))
        assert online._get_cached("吓死我了") == ""


def test_index_local_failure_uses_m2m100_without_network_fallback():
    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(_settings(temp_dir, provider="index_local"))
        service._index_opener = _IndexResponse("")
        service._translate_local = lambda _: "怖かった！"
        service._translate_api = lambda _: (_ for _ in ()).throw(AssertionError("unexpected network call"))
        assert service.translate("吓死我了") == "怖かった！"
        assert not service._get_cached("吓死我了")


def test_translation_startup_never_loads_m2m100():
    with TemporaryDirectory() as temp_dir:
        for provider in ("local", "index", "index_local"):
            service = JapaneseTranslationService(_settings(temp_dir, provider=provider))
            service._ensure_local_backend = lambda: (_ for _ in ()).throw(AssertionError("startup loaded M2M100"))
            service._translate_local = lambda _: (_ for _ in ()).throw(AssertionError("startup translated text"))
            service.warmup()
            assert service._translator is None
            assert not service._local_init_attempted


def test_local_model_load_is_deferred_until_actual_translation():
    with TemporaryDirectory() as temp_dir:
        service = JapaneseTranslationService(_settings(temp_dir, provider="local"))
        model_path = Path(temp_dir) / "missing-model"
        model_path.mkdir()
        (model_path / "model.bin").touch()
        (model_path / "tokenizer").mkdir()
        calls = []
        service._translate_local = lambda text: calls.append(text) or "こんにちは。"
        assert service.warmup().ready
        assert not calls
        assert service.translate("你好") == "こんにちは。"
        assert calls == ["你好"]


class _FakeTranslation:
    def __init__(self, result: str) -> None:
        self.result = result

    def translate(self, _text: str) -> str:
        return self.result


def _speech_worker_with_translation(result: str):
    service = SpeechService.__new__(SpeechService)
    service._generation = 0
    service._active_generation = 0
    service._muted = False
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
