from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from desktop_pet.config.settings import Settings


_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


@dataclass(frozen=True)
class TranslationStatus:
    ready: bool
    detail: str


def _normalize_source(text: str, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    patterns = (
        r"\[[^\[\]]*\]",
        r"\([^\(\)]*\)",
        r"【[^【】]*】",
        r"（[^（）]*）",
    )
    for _ in range(4):
        previous = cleaned
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)
        if cleaned == previous:
            break
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[: max(1, int(max_chars))]


def _normalize_translation(text: str, max_chars: int) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("[离线回声]"):
        return ""
    cleaned = cleaned.replace("\r", "\n").split("\n", 1)[0].strip()
    cleaned = re.sub(r"^(?:日语|日文|翻译|訳文)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" \t\"'“”‘’「」『』")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[: max(1, int(max_chars))]


def contains_japanese_kana(text: str) -> bool:
    return bool(_KANA_RE.search(str(text or "")))


def is_safe_japanese_tts_text(text: str, *, source_had_han: bool = True) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    if contains_japanese_kana(cleaned):
        return True
    # Latin text, digits, and punctuation are safe to pass through. When the
    # source contained Chinese Han characters, a Han-only result is ambiguous
    # and is rejected so raw Chinese can never reach a Japanese-only voice.
    if source_had_han and _HAN_RE.search(cleaned):
        return False
    return True


class JapaneseTranslationService:
    """Persistent Chinese-to-Japanese translation with local-first routing."""

    def __init__(
        self,
        settings: Settings,
        api_fallback: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._provider = str(settings.tts_translation_provider or "local").lower()
        self._model_id = str(settings.tts_translation_model_id)
        self._model_path = Path(settings.tts_translation_model_path)
        self._device = str(settings.tts_translation_device or "auto").lower()
        self._compute_type = str(settings.tts_translation_compute_type or "auto").lower()
        self._cache_path = Path(settings.tts_translation_cache_path)
        self._cache_size = max(0, int(settings.tts_translation_cache_size))
        self._max_chars = max(40, int(settings.tts_translation_max_chars))
        self._api_fallback_enabled = bool(settings.enable_tts_translation_api_fallback)
        self._skip_on_failure = bool(settings.tts_skip_on_translation_failure)
        self._diag = bool(getattr(settings, "tts_diag_logs", False))
        self._api_fallback = api_fallback

        self._lock = threading.RLock()
        self._translator = None
        self._tokenizer = None
        self._local_init_attempted = False
        self._status = TranslationStatus(False, "not initialized")
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._load_cache()

    def _diag_log(self, message: str) -> None:
        if self._diag:
            print(f"[TTS-TRANSLATE] {message}")

    @staticmethod
    def _cache_key(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip().casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if self._cache_size <= 0 or not self._cache_path.exists():
            return
        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))
            entries = payload.get("entries", []) if isinstance(payload, dict) else []
            for item in entries[-self._cache_size :]:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                key, value = str(item[0]), str(item[1])
                if key and value:
                    self._cache[key] = value
        except Exception as exc:
            self._diag_log(f"cache load failed: {exc}")

    def _save_cache(self) -> None:
        if self._cache_size <= 0:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            payload = {"version": 1, "entries": list(self._cache.items())[-self._cache_size :]}
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(self._cache_path)
        except Exception as exc:
            self._diag_log(f"cache save failed: {exc}")

    def _get_cached(self, source: str) -> str:
        key = self._cache_key(source)
        with self._lock:
            value = self._cache.pop(key, "")
            if value:
                self._cache[key] = value
            return value

    def _put_cached(self, source: str, translated: str) -> None:
        if self._cache_size <= 0:
            return
        key = self._cache_key(source)
        with self._lock:
            self._cache.pop(key, None)
            self._cache[key] = translated
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            self._save_cache()

    def _ensure_local_backend(self) -> bool:
        with self._lock:
            if self._translator is not None and self._tokenizer is not None:
                return True
            if self._local_init_attempted:
                return False
            self._local_init_attempted = True

            model_bin = self._model_path / "model.bin"
            tokenizer_path = self._model_path / "tokenizer"
            if not model_bin.exists() or not tokenizer_path.exists():
                self._status = TranslationStatus(
                    False,
                    f"local model missing: run scripts/setup_local_translation.py ({self._model_path})",
                )
                return False

            try:
                import ctranslate2
                from transformers import AutoTokenizer
            except Exception as exc:
                self._status = TranslationStatus(False, f"local translation dependency missing: {exc}")
                return False

            requested_device = self._device
            if requested_device == "auto":
                requested_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            requested_compute = self._compute_type
            if requested_device == "cpu" and requested_compute in {
                "float16",
                "bfloat16",
                "int8_float16",
                "int8_bfloat16",
            }:
                requested_compute = "int8"

            try:
                translator = ctranslate2.Translator(
                    str(self._model_path),
                    device=requested_device,
                    compute_type=requested_compute,
                    inter_threads=1,
                )
            except Exception as primary_exc:
                if requested_device != "cuda":
                    self._status = TranslationStatus(False, f"local model load failed: {primary_exc}")
                    return False
                try:
                    translator = ctranslate2.Translator(
                        str(self._model_path),
                        device="cpu",
                        compute_type="int8",
                        inter_threads=1,
                    )
                    requested_device = "cpu"
                    requested_compute = "int8"
                except Exception as fallback_exc:
                    self._status = TranslationStatus(
                        False,
                        f"CUDA load failed: {primary_exc}; CPU fallback failed: {fallback_exc}",
                    )
                    return False

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    local_files_only=True,
                )
                tokenizer.src_lang = "zh"
            except Exception as exc:
                self._status = TranslationStatus(False, f"tokenizer load failed: {exc}")
                return False

            self._translator = translator
            self._tokenizer = tokenizer
            self._status = TranslationStatus(
                True,
                f"ok model={self._model_id} device={requested_device} compute={requested_compute}",
            )
            self._diag_log(self._status.detail)
            return True

    def _translate_local(self, source: str) -> str:
        if not self._ensure_local_backend():
            return ""
        with self._lock:
            tokenizer = self._tokenizer
            translator = self._translator
            tokenizer.src_lang = "zh"
            source_ids = tokenizer.encode(source)
            source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
            target_prefix = [tokenizer.lang_code_to_token["ja"]]
            results = translator.translate_batch(
                [source_tokens],
                target_prefix=[target_prefix],
                beam_size=2,
                max_input_length=256,
                max_decoding_length=256,
                repetition_penalty=1.1,
            )
            target_tokens = list(results[0].hypotheses[0])
            if target_tokens and target_tokens[0] == target_prefix[0]:
                target_tokens = target_tokens[1:]
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            return str(tokenizer.decode(target_ids, skip_special_tokens=True)).strip()

    def _translate_api(self, source: str) -> str:
        if self._api_fallback is None:
            return ""
        try:
            return str(self._api_fallback(source) or "").strip()
        except Exception as exc:
            self._diag_log(f"API fallback failed: {exc}")
            return ""

    def translate(self, text: str) -> str:
        source = _normalize_source(text, self._max_chars)
        if not source:
            return ""
        source_had_han = bool(_HAN_RE.search(source))
        if contains_japanese_kana(source):
            return source

        cached = self._get_cached(source)
        if cached and is_safe_japanese_tts_text(cached, source_had_han=source_had_han):
            self._diag_log("cache hit")
            return cached

        translated = ""
        if self._provider == "local":
            try:
                translated = self._translate_local(source)
            except Exception as exc:
                self._diag_log(f"local translation failed: {exc}")
            if not translated and self._api_fallback_enabled:
                translated = self._translate_api(source)
        elif self._provider == "api":
            translated = self._translate_api(source)
        elif self._provider == "none":
            translated = source

        translated = _normalize_translation(translated, self._max_chars)
        if not is_safe_japanese_tts_text(translated, source_had_han=source_had_han):
            self._diag_log("translation rejected: Japanese kana missing or source text leaked")
            return "" if self._skip_on_failure else source

        self._put_cached(source, translated)
        return translated

    def warmup(self) -> TranslationStatus:
        if self._provider != "local":
            ready = self._provider == "api" and self._api_fallback is not None
            self._status = TranslationStatus(ready, f"provider={self._provider}")
            return self._status
        if self._ensure_local_backend():
            try:
                self._translate_local("你好")
            except Exception as exc:
                self._status = TranslationStatus(False, f"warmup failed: {exc}")
        return self._status

    def get_status(self) -> TranslationStatus:
        return self._status
