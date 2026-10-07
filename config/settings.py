from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


@dataclass
class Settings:
    api_key: str
    base_url: str
    model_name: str
    vision_model_name: str
    enable_multimodal_vision: bool
    enable_mm_compat_mode: bool
    enable_mm_screen_comment: bool
    mm_timeout_sec: float
    mm_failure_threshold: int
    mm_cooldown_sec: int
    mm_image_max_edge: int
    mm_output_max_tokens: int
    mm_auto_min_interval_sec: int
    vision_capture_mode: str
    vision_window_inset_lr: float
    vision_window_inset_tb: float
    vision_window_margin_px: int
    vision_window_monitor_index: int
    vision_min_interval_sec: int
    vision_shots_per_minute: float
    vision_burst_cap: float
    vision_min_gap_sec: float
    vision_context_ttl_sec: float
    vision_change_threshold: float
    vision_focus_query: str
    vision_target_title: str
    vision_budget_per_hour: int
    visual_novel_text_ratio: float
    dedup_ngram_merge: float
    dedup_ngram_guard: float
    dedup_semantic_merge: float
    dedup_calibration_enabled: bool
    dedup_calibration_path: str
    visual_novel_max_facts: int
    enable_ocr_first_routing: bool
    ocr_only_min_chars: int
    ocr_only_min_confidence: float
    ocr_hybrid_min_chars: int
    ocr_context_max_chars: int
    enable_auto_comment_heartbeat: bool
    chat_show_system_messages: bool
    chat_show_session_debug_marker: bool
    enable_chat_multimodal: bool
    chat_screen_context_max_chars: int
    enable_visual_novel_mode: bool
    visual_novel_min_context_similarity: float
    enable_semantic_attention: bool
    enable_semantic_comment_candidates: bool
    semantic_attention_model_path: str
    semantic_attention_top_k: int
    semantic_attention_min_score: float
    semantic_attention_max_length: int
    semantic_attention_cache_size: int
    semantic_attention_cpu_threads: int
    web_search_soft_deadline_sec: float
    web_search_hard_deadline_sec: float
    web_search_circuit_failure_threshold: int
    web_search_circuit_cooldown_sec: int
    web_search_max_results: int
    web_search_context_max_chars: int
    baidu_ai_search_api_key: str
    long_memory_limit: int
    long_memory_context_window: int
    enable_tutor_persona: bool
    enable_tts: bool
    tts_provider: str
    tts_voice: str
    tts_rate: str
    tts_volume: str
    tts_azure_key: str
    tts_azure_region: str
    tts_azure_endpoint: str
    tts_voicevox_base_url: str
    tts_voicevox_speaker: int
    tts_voicevox_engine_path: str
    enable_voicevox_auto_launch: bool
    enable_voicevox_ja_translation: bool
    tts_translation_provider: str
    tts_translation_model_id: str
    tts_translation_model_path: str
    tts_translation_device: str
    tts_translation_compute_type: str
    tts_translation_cache_path: str
    tts_translation_cache_size: int
    tts_translation_max_chars: int
    enable_tts_translation_api_fallback: bool
    tts_skip_on_translation_failure: bool
    webengine_gpu_mode: str
    enable_scan_subprocess: bool
    scan_monitor_index: int
    scan_region: Optional[Tuple[int, int, int, int]]
    scan_tick_interval_sec: int
    scan_submit_min_interval_sec: int
    scan_busy_timeout_sec: int
    auto_start_scan: bool
    scan_adaptive_max_interval_sec: float
    ocr_fallback_min_interval_sec: float
    ocr_fallback_min_quiet_sec: float
    ocr_text_change_threshold: float
    ocr_text_ink_change_threshold: float
    vn_text_band_bottom_margin: float
    vn_text_band_mid_top_ratio: float
    ocr_cpu_threads: int
    ocr_cpu_affinity_count: int
    ocr_max_edge: int
    ocr_text_max_edge: int
    resource_policy_reapply_min_sec: float
    memory_recency_window_sec: int
    memory_min_weight: float
    enable_live2d: bool
    enable_live2d_py: bool
    live2d_py_window_width: int
    live2d_py_window_height: int
    live2d_model_json: str
    live2d_follow_cursor: bool
    live2d_follow_activate_distance_px: int
    live2d_model_scale: float
    live2d_idle_group: str
    screen_comment_memory_limit: int
    screen_comment_memory_weight: float
    auto_comment_style_weights: str
    emotion_keywords_stressed: str
    emotion_keywords_positive: str
    emotion_keywords_focused: str
    comment_similarity_skip_threshold: float
    enable_comment_api_understanding: bool
    screen_scan_interval_sec: int
    auto_comment_cooldown_sec: int
    pet_image_path: Path


def _parse_scan_region(value: str) -> Optional[Tuple[int, int, int, int]]:
    text = value.strip()
    if not text:
        return None

    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("SCAN_REGION must be 'left,top,width,height'")

    left, top, width, height = [int(x) for x in parts]
    if width <= 0 or height <= 0:
        raise ValueError("SCAN_REGION width/height must be positive")
    return left, top, width, height


def _parse_ratio(value: str, default: float) -> float:
    """Parse a 0..0.45 ratio used for window insets."""
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(0.0, min(0.45, parsed))


_VISION_CAPTURE_MODES = {"window", "manual"}


def _resolve_live2d_model_path(base_dir: Path, raw_value: str, default_path: Path) -> str:
    text = (raw_value or "").strip()
    if not text:
        return str(default_path)

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    # Relative paths are resolved from project directory for stable behavior.
    return str((base_dir / candidate).resolve())


def _resolve_project_path(base_dir: Path, raw_value: str, default_path: Path) -> str:
    text = (raw_value or "").strip()
    if not text:
        return str(default_path)

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    return str((base_dir / candidate).resolve())


def load_settings(base_dir: Path) -> Settings:
    env_path = base_dir / ".env"
    if not env_path.exists():
        env_path = base_dir / ".env.example"
    load_dotenv(env_path, override=True)

    api_key = (
        os.getenv("N1N_API_KEY")
        or os.getenv("\ufeffN1N_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("\ufeffDEEPSEEK_API_KEY", "")
    )
    base_url = os.getenv("N1N_BASE_URL", "https://api.n1n.ai/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o")
    vision_model_name = os.getenv("VISION_MODEL_NAME", "").strip() or model_name
    enable_multimodal_vision = os.getenv("ENABLE_MULTIMODAL_VISION", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_mm_compat_mode = os.getenv("ENABLE_MM_COMPAT_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_mm_screen_comment = os.getenv("ENABLE_MM_SCREEN_COMMENT", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mm_timeout_sec = float(os.getenv("MM_TIMEOUT_SEC", "5.0"))
    mm_timeout_sec = max(1.0, min(30.0, mm_timeout_sec))
    mm_failure_threshold = int(os.getenv("MM_FAILURE_THRESHOLD", "3"))
    mm_failure_threshold = max(1, min(12, mm_failure_threshold))
    mm_cooldown_sec = int(os.getenv("MM_COOLDOWN_SEC", "120"))
    mm_cooldown_sec = max(10, min(3600, mm_cooldown_sec))
    mm_image_max_edge = int(os.getenv("MM_IMAGE_MAX_EDGE", "1280"))
    mm_image_max_edge = max(320, min(2048, mm_image_max_edge))
    mm_output_max_tokens = int(os.getenv("MM_OUTPUT_MAX_TOKENS", "220"))
    mm_output_max_tokens = max(64, min(1024, mm_output_max_tokens))
    mm_auto_min_interval_sec = int(os.getenv("MM_AUTO_MIN_INTERVAL_SEC", "300"))
    mm_auto_min_interval_sec = max(0, min(3600, mm_auto_min_interval_sec))
    vision_capture_mode = os.getenv("VISION_CAPTURE_MODE", "window").strip().lower()
    if vision_capture_mode not in _VISION_CAPTURE_MODES:
        vision_capture_mode = "window"
    vision_window_inset_lr = _parse_ratio(os.getenv("VISION_WINDOW_INSET_LR", "0.0"), 0.0)
    vision_window_inset_tb = _parse_ratio(os.getenv("VISION_WINDOW_INSET_TB", "0.0"), 0.0)
    vision_window_margin_px = int(os.getenv("VISION_WINDOW_MARGIN_PX", "0"))
    vision_window_margin_px = max(-200, min(400, vision_window_margin_px))
    vision_window_monitor_index = int(os.getenv("VISION_WINDOW_MONITOR_INDEX", "0"))
    vision_window_monitor_index = max(0, vision_window_monitor_index)
    # Vision pacing is a token bucket, not a hard interval: a big scene change
    # fires as soon as quota allows instead of waiting out a clock. Refill rate
    # gives the average (shots_per_minute / 60 per second) and burst_cap bounds
    # how many can be saved up, so a quiet spell cannot be cashed in as a sudden
    # burst of requests.
    vision_shots_per_minute = float(os.getenv("VISION_SHOTS_PER_MINUTE", "3"))
    vision_shots_per_minute = max(0.0, min(60.0, vision_shots_per_minute))
    vision_burst_cap = float(os.getenv("VISION_BURST_CAP", "3"))
    vision_burst_cap = max(1.0, min(10.0, vision_burst_cap))
    # Minimum spacing between two vision requests. The token bucket caps the
    # total (shots_per_minute) but nothing about it stops the saved burst from
    # being spent in the first seconds of an opening movie; this floor spreads
    # the samples across the scene. Set 0 to disable.
    vision_min_gap_sec = float(os.getenv("VISION_MIN_GAP_SEC", "6"))
    vision_min_gap_sec = max(0.0, min(300.0, vision_min_gap_sec))
    # Legacy knob, kept only so an old .env does not crash. No longer a gate.
    vision_min_interval_sec = int(os.getenv("VISION_MIN_INTERVAL_SEC", "0"))
    vision_min_interval_sec = max(0, min(600, vision_min_interval_sec))
    # How long a captured description stays valid as context for a comment.
    vision_context_ttl_sec = float(os.getenv("VISION_CONTEXT_TTL_SEC", "30"))
    vision_context_ttl_sec = max(5.0, min(300.0, vision_context_ttl_sec))
    vision_change_threshold = float(os.getenv("VISION_CHANGE_THRESHOLD", "0.10"))
    vision_change_threshold = max(0.0, min(1.0, vision_change_threshold))
    vision_focus_query = os.getenv(
        "VISION_FOCUS_QUERY",
        "主角表情与情绪、立绘或服装变化、是否出现CG画面、场景地点与时间",
    ).strip()
    # Pin the capture to a specific window by title substring. Without this the
    # resolver picks the topmost foreign window, which is often a browser or a
    # tool window rather than the game.
    vision_target_title = os.getenv("VISION_TARGET_TITLE", "").strip()
    # Optional hard ceiling on top of the token bucket (0 = unlimited).
    vision_budget_per_hour = int(os.getenv("VISION_BUDGET_PER_HOUR", "0"))
    vision_budget_per_hour = max(0, min(3600, vision_budget_per_hour))
    visual_novel_text_ratio = float(os.getenv("VN_TEXT_REGION_RATIO", "0.58"))
    visual_novel_text_ratio = max(0.2, min(0.95, visual_novel_text_ratio))
    dedup_ngram_merge = float(os.getenv("DEDUP_NGRAM_MERGE", "0.75"))
    dedup_ngram_merge = max(0.3, min(1.0, dedup_ngram_merge))
    dedup_ngram_guard = float(os.getenv("DEDUP_NGRAM_GUARD", "0.12"))
    dedup_ngram_guard = max(0.05, min(1.0, dedup_ngram_guard))
    dedup_semantic_merge = float(os.getenv("DEDUP_SEMANTIC_MERGE", "0.82"))
    dedup_semantic_merge = max(0.5, min(1.0, dedup_semantic_merge))
    dedup_calibration_enabled = os.getenv("DEDUP_CALIBRATION_ENABLED", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    dedup_calibration_path = _resolve_project_path(
        base_dir,
        os.getenv("DEDUP_CALIBRATION_PATH", ""),
        base_dir / "data" / "dedup_calibration.jsonl",
    )
    visual_novel_max_facts = int(os.getenv("VN_MAX_FACTS", "60"))
    visual_novel_max_facts = max(24, min(400, visual_novel_max_facts))
    enable_ocr_first_routing = os.getenv("ENABLE_OCR_FIRST_ROUTING", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ocr_only_min_chars = int(os.getenv("OCR_ONLY_MIN_CHARS", "120"))
    ocr_only_min_chars = max(1, min(10000, ocr_only_min_chars))
    ocr_only_min_confidence = float(os.getenv("OCR_ONLY_MIN_CONFIDENCE", "0.75"))
    ocr_only_min_confidence = max(0.0, min(1.0, ocr_only_min_confidence))
    ocr_hybrid_min_chars = int(os.getenv("OCR_HYBRID_MIN_CHARS", "30"))
    ocr_hybrid_min_chars = max(1, min(ocr_only_min_chars, ocr_hybrid_min_chars))
    ocr_context_max_chars = int(os.getenv("OCR_CONTEXT_MAX_CHARS", "1200"))
    ocr_context_max_chars = max(200, min(6000, ocr_context_max_chars))
    enable_auto_comment_heartbeat = os.getenv("ENABLE_AUTO_COMMENT_HEARTBEAT", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    chat_show_system_messages = os.getenv("CHAT_SHOW_SYSTEM_MESSAGES", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    chat_show_session_debug_marker = os.getenv("CHAT_SHOW_SESSION_DEBUG_MARKER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_chat_multimodal = os.getenv("ENABLE_CHAT_MULTIMODAL", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    chat_screen_context_max_chars = int(os.getenv("CHAT_SCREEN_CONTEXT_MAX_CHARS", "1600"))
    chat_screen_context_max_chars = max(200, min(6000, chat_screen_context_max_chars))
    enable_visual_novel_mode = os.getenv("ENABLE_VISUAL_NOVEL_MODE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    visual_novel_min_context_similarity = float(
        os.getenv("VN_MIN_CONTEXT_SIMILARITY", "0.30")
    )
    visual_novel_min_context_similarity = max(0.0, min(1.0, visual_novel_min_context_similarity))
    enable_semantic_attention = os.getenv("ENABLE_SEMANTIC_ATTENTION", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    default_semantic_attention_model_path = base_dir / "assets" / "models" / "bge-small-zh-v1.5-onnx"
    enable_semantic_comment_candidates = os.getenv("ENABLE_SEMANTIC_COMMENT_CANDIDATES", "true").lower() in {
        "1", "true", "yes", "on",
    }
    semantic_attention_model_path = _resolve_project_path(
        base_dir,
        os.getenv("SEMANTIC_ATTENTION_MODEL_PATH", ""),
        default_semantic_attention_model_path,
    )
    semantic_attention_top_k = int(os.getenv("SEMANTIC_ATTENTION_TOP_K", "8"))
    semantic_attention_top_k = max(2, min(32, semantic_attention_top_k))
    semantic_attention_min_score = float(os.getenv("SEMANTIC_ATTENTION_MIN_SCORE", "0.34"))
    semantic_attention_min_score = max(0.0, min(1.0, semantic_attention_min_score))
    semantic_attention_max_length = int(os.getenv("SEMANTIC_ATTENTION_MAX_LENGTH", "256"))
    semantic_attention_max_length = max(32, min(512, semantic_attention_max_length))
    semantic_attention_cache_size = int(os.getenv("SEMANTIC_ATTENTION_CACHE_SIZE", "512"))
    semantic_attention_cache_size = max(0, min(10000, semantic_attention_cache_size))
    semantic_attention_cpu_threads = int(os.getenv("SEMANTIC_ATTENTION_CPU_THREADS", "2"))
    semantic_attention_cpu_threads = max(1, min(16, semantic_attention_cpu_threads))
    web_search_soft_deadline_sec = float(os.getenv("WEB_SEARCH_SOFT_DEADLINE_SEC", "3.0"))
    web_search_soft_deadline_sec = max(0.5, min(10.0, web_search_soft_deadline_sec))
    web_search_hard_deadline_sec = float(os.getenv("WEB_SEARCH_HARD_DEADLINE_SEC", "8.0"))
    web_search_hard_deadline_sec = max(
        web_search_soft_deadline_sec,
        min(30.0, web_search_hard_deadline_sec),
    )
    web_search_circuit_failure_threshold = int(os.getenv("WEB_SEARCH_CIRCUIT_FAILURE_THRESHOLD", "3"))
    web_search_circuit_failure_threshold = max(1, min(10, web_search_circuit_failure_threshold))
    web_search_circuit_cooldown_sec = int(os.getenv("WEB_SEARCH_CIRCUIT_COOLDOWN_SEC", "600"))
    web_search_circuit_cooldown_sec = max(10, min(3600, web_search_circuit_cooldown_sec))
    web_search_max_results = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
    web_search_max_results = max(1, min(5, web_search_max_results))
    web_search_context_max_chars = int(os.getenv("WEB_SEARCH_CONTEXT_MAX_CHARS", "900"))
    web_search_context_max_chars = max(300, min(3000, web_search_context_max_chars))
    baidu_ai_search_api_key = os.getenv("BAIDU_AI_SEARCH_API_KEY", "").strip()
    long_memory_limit = int(os.getenv("LONG_MEMORY_LIMIT", "360"))
    long_memory_limit = max(20, min(2000, long_memory_limit))
    long_memory_context_window = int(os.getenv("LONG_MEMORY_CONTEXT_WINDOW", "30"))
    long_memory_context_window = max(1, min(long_memory_limit, long_memory_context_window))
    enable_tutor_persona = os.getenv("ENABLE_TUTOR_PERSONA", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_tts = os.getenv("ENABLE_TTS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    tts_provider = os.getenv("TTS_PROVIDER", "edge")
    tts_voice = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    tts_rate = os.getenv("TTS_RATE", "+0%")
    tts_volume = os.getenv("TTS_VOLUME", "+0%")
    tts_azure_key = os.getenv("TTS_AZURE_KEY") or os.getenv("AZURE_SPEECH_KEY", "")
    tts_azure_region = os.getenv("TTS_AZURE_REGION") or os.getenv("AZURE_SPEECH_REGION", "")
    tts_azure_endpoint = os.getenv("TTS_AZURE_ENDPOINT", "")
    tts_voicevox_base_url = os.getenv("TTS_VOICEVOX_BASE_URL", "http://127.0.0.1:50021")
    tts_voicevox_speaker = int(os.getenv("TTS_VOICEVOX_SPEAKER", "1"))
    default_voicevox_engine_path = base_dir / "VOICEVOX" / "VOICEVOX" / "vv-engine" / "run.exe"
    tts_voicevox_engine_path = _resolve_project_path(
        base_dir,
        os.getenv("TTS_VOICEVOX_ENGINE_PATH", ""),
        default_voicevox_engine_path,
    )
    enable_voicevox_auto_launch = os.getenv("ENABLE_VOICEVOX_AUTO_LAUNCH", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_voicevox_ja_translation = os.getenv("ENABLE_VOICEVOX_JA_TRANSLATION", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    tts_translation_provider = os.getenv("TTS_TRANSLATION_PROVIDER", "local").strip().lower()
    if tts_translation_provider not in {"local", "index", "index_local", "api", "none"}:
        tts_translation_provider = "local"
    tts_translation_model_id = os.getenv(
        "TTS_TRANSLATION_MODEL_ID",
        "facebook/m2m100_418M",
    ).strip() or "facebook/m2m100_418M"
    default_translation_model_path = base_dir / "assets" / "models" / "m2m100_418M_ct2"
    tts_translation_model_path = _resolve_project_path(
        base_dir,
        os.getenv("TTS_TRANSLATION_MODEL_PATH", ""),
        default_translation_model_path,
    )
    tts_translation_device = os.getenv("TTS_TRANSLATION_DEVICE", "cuda").strip().lower()
    if tts_translation_device not in {"auto", "cpu", "cuda"}:
        tts_translation_device = "auto"
    tts_translation_compute_type = os.getenv(
        "TTS_TRANSLATION_COMPUTE_TYPE",
        "int8_float16",
    ).strip().lower()
    allowed_translation_compute_types = {
        "auto",
        "default",
        "int8",
        "int8_float32",
        "int8_float16",
        "int8_bfloat16",
        "int16",
        "float16",
        "bfloat16",
        "float32",
    }
    if tts_translation_compute_type not in allowed_translation_compute_types:
        tts_translation_compute_type = "auto"
    default_translation_cache_path = base_dir / "data" / "tts_translation_cache.json"
    tts_translation_cache_path = _resolve_project_path(
        base_dir,
        os.getenv("TTS_TRANSLATION_CACHE_PATH", ""),
        default_translation_cache_path,
    )
    tts_translation_cache_size = int(os.getenv("TTS_TRANSLATION_CACHE_SIZE", "2000"))
    tts_translation_cache_size = max(0, min(20000, tts_translation_cache_size))
    tts_translation_max_chars = int(os.getenv("TTS_TRANSLATION_MAX_CHARS", "220"))
    tts_translation_max_chars = max(40, min(1000, tts_translation_max_chars))
    enable_tts_translation_api_fallback = os.getenv(
        "ENABLE_TTS_TRANSLATION_API_FALLBACK",
        "false",
    ).lower() in {"1", "true", "yes", "on"}
    tts_skip_on_translation_failure = os.getenv(
        "TTS_SKIP_ON_TRANSLATION_FAILURE",
        "true",
    ).lower() in {"1", "true", "yes", "on"}
    webengine_gpu_mode = os.getenv("WEBENGINE_GPU_MODE", "gpu").strip().lower()
    if webengine_gpu_mode not in {"gpu", "software", "auto"}:
        webengine_gpu_mode = "gpu"
    enable_scan_subprocess = os.getenv("ENABLE_SCAN_SUBPROCESS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # 0 means the complete virtual desktop; positive values select an MSS display.
    scan_monitor_index = int(os.getenv("SCAN_MONITOR_INDEX", "0"))
    scan_monitor_index = max(0, scan_monitor_index)
    scan_region = _parse_scan_region(os.getenv("SCAN_REGION", ""))
    scan_tick_interval_sec = int(os.getenv("SCAN_TICK_INTERVAL_SEC", "0"))
    scan_tick_interval_sec = max(0, scan_tick_interval_sec)
    scan_submit_min_interval_sec = int(os.getenv("SCAN_SUBMIT_MIN_INTERVAL_SEC", "0"))
    scan_submit_min_interval_sec = max(0, scan_submit_min_interval_sec)
    scan_busy_timeout_sec = int(os.getenv("SCAN_BUSY_TIMEOUT_SEC", "12"))
    scan_busy_timeout_sec = max(5, scan_busy_timeout_sec)
    # Start with auto-scan already on. The toggle begins every launch switched
    # off, and enabling visual-novel mode via config used to leave the scan off
    # with it, so nothing was read until the user found the button.
    auto_start_scan = os.getenv("AUTO_START_SCAN", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Upper bound for the load-adaptive scan interval. The interval normally
    # stretches to ~1.35x the OCR duration; in visual-novel mode that makes the
    # scan feel slow, so it is capped much lower.
    scan_adaptive_max_interval_sec = float(os.getenv("SCAN_ADAPTIVE_MAX_INTERVAL_SEC", "1.0"))
    scan_adaptive_max_interval_sec = max(1.0, min(90.0, scan_adaptive_max_interval_sec))
    # A full-band OCR sweep costs several OCR passes (~17s measured). Rate-limit
    # it so one unreadable frame cannot stall the scan loop.
    ocr_fallback_min_interval_sec = float(os.getenv("OCR_FALLBACK_MIN_INTERVAL_SEC", "15.0"))
    ocr_fallback_min_interval_sec = max(0.0, min(600.0, ocr_fallback_min_interval_sec))
    # How long the dialogue box must stay unreadable before the band sweep is
    # worth paying for. One blank read is normal (line changes, scene
    # transitions, a frame caught mid-fade); sweeping on it cost a ~10s recovery
    # every time the box blinked out.
    ocr_fallback_min_quiet_sec = float(os.getenv("OCR_FALLBACK_MIN_QUIET_SEC", "5.0"))
    ocr_fallback_min_quiet_sec = max(0.0, min(120.0, ocr_fallback_min_quiet_sec))
    # How much the OCR crop may change before we re-run OCR. Text areas are
    # small and static between lines, so a tight threshold lets most ticks skip
    # the ~4.5s OCR pass entirely.
    ocr_text_change_threshold = float(os.getenv("OCR_TEXT_CHANGE_THRESHOLD", "0.008"))
    ocr_text_change_threshold = max(0.0, min(1.0, ocr_text_change_threshold))
    # Threshold for the ink-based dialogue-change comparison, which replaced the
    # coarse brightness fingerprint in the text-change gates. Measured on this
    # game's 1979x369 box: two different short lines score ~0.57, a name-plate
    # change ~0.85, while identical frames and pure background brightness shifts
    # score 0.00. 0.12 sits far below every real change and far above the noise.
    ocr_text_ink_change_threshold = float(
        os.getenv("OCR_TEXT_INK_CHANGE_THRESHOLD", "0.12")
    )
    ocr_text_ink_change_threshold = max(0.0, min(1.0, ocr_text_ink_change_threshold))
    # Fraction of the window height ignored at the bottom of the automatic
    # dialogue band. Measured on 青空下的加缪 (2560x1600): dialogue ends at ~91%
    # of the height, the SAVE/LOAD/CONFIG bar sits at 93-95%, and everything
    # below 95.5% is a black letterbox. Including the bar made OCR return the
    # dialogue and the chrome glued together ("...没什么变化。 SAVELOADCONFIGSKIP").
    vn_text_band_bottom_margin = float(
        os.getenv("VN_TEXT_BAND_BOTTOM_MARGIN", "0.08")
    )
    vn_text_band_bottom_margin = max(0.0, min(0.40, vn_text_band_bottom_margin))
    # Where the retry band above the dialogue box starts, as a fraction of the
    # window height. The app assumes the text sits in the lower band; when that
    # reads nothing the sweep retries this strip above it, which is where games
    # with a centred text window put their dialogue. The two are disjoint, so no
    # OCR pass ever re-reads the same pixels.
    vn_text_band_mid_top_ratio = float(
        os.getenv("VN_TEXT_BAND_MID_TOP_RATIO", "0.30")
    )
    vn_text_band_mid_top_ratio = max(0.0, min(0.90, vn_text_band_mid_top_ratio))
    ocr_cpu_threads = int(os.getenv("OCR_CPU_THREADS", "0"))
    ocr_cpu_threads = max(0, min(64, ocr_cpu_threads))
    ocr_cpu_affinity_count = int(os.getenv("OCR_CPU_AFFINITY_COUNT", "0"))
    ocr_cpu_affinity_count = max(0, min(64, ocr_cpu_affinity_count))
    ocr_max_edge = int(os.getenv("OCR_MAX_EDGE", "0"))
    ocr_max_edge = max(0, min(4096, ocr_max_edge))
    # Resolution cap for the visual-novel text box. Measured on a real 2560x672
    # dialogue crop (RapidOCR, 4 threads): 640-800px reads the same 27 chars in
    # ~1.75s, while 1600px costs ~4.5s for no extra characters. Higher is
    # strictly slower here, so this stays modest.
    ocr_text_max_edge = int(os.getenv("OCR_TEXT_MAX_EDGE", "800"))
    ocr_text_max_edge = max(0, min(4096, ocr_text_max_edge))
    resource_policy_reapply_min_sec = float(os.getenv("RESOURCE_POLICY_REAPPLY_MIN_SEC", "4.0"))
    resource_policy_reapply_min_sec = max(0.2, min(20.0, resource_policy_reapply_min_sec))
    memory_recency_window_sec = int(os.getenv("MEMORY_RECENCY_WINDOW_SEC", "0"))
    memory_recency_window_sec = max(0, memory_recency_window_sec)
    memory_min_weight = float(os.getenv("MEMORY_MIN_WEIGHT", "0.15"))
    memory_min_weight = max(0.0, min(1.0, memory_min_weight))
    default_live2d_model = base_dir / "皮套" / "mao_pro_zh" / "runtime" / "mao_pro.model3.json"
    if not default_live2d_model.exists():
        legacy_default_live2d_model = (
            base_dir.parent.parent / "皮套" / "mao_pro_zh" / "runtime" / "mao_pro.model3.json"
        )
        default_live2d_model = legacy_default_live2d_model
    live2d_model_json = _resolve_live2d_model_path(
        base_dir,
        os.getenv("LIVE2D_MODEL_JSON", ""),
        default_live2d_model,
    )
    enable_live2d = os.getenv("ENABLE_LIVE2D", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    enable_live2d_py = os.getenv("ENABLE_LIVE2D_PY", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Treat ENABLE_LIVE2D as a global master switch for all Live2D backends.
    if not enable_live2d:
        enable_live2d_py = False
    live2d_py_window_width = int(os.getenv("LIVE2D_PY_WINDOW_WIDTH", "280"))
    live2d_py_window_width = max(180, live2d_py_window_width)
    live2d_py_window_height = int(os.getenv("LIVE2D_PY_WINDOW_HEIGHT", "430"))
    live2d_py_window_height = max(220, live2d_py_window_height)
    live2d_follow_cursor = os.getenv("LIVE2D_FOLLOW_CURSOR", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    live2d_follow_activate_distance_px = int(os.getenv("LIVE2D_FOLLOW_ACTIVATE_DISTANCE_PX", "220"))
    live2d_follow_activate_distance_px = max(40, min(2000, live2d_follow_activate_distance_px))
    live2d_model_scale = float(os.getenv("LIVE2D_MODEL_SCALE", "1.0"))
    live2d_model_scale = max(0.2, min(3.0, live2d_model_scale))
    live2d_idle_group = os.getenv("LIVE2D_IDLE_GROUP", "Idle").strip() or "Idle"
    screen_comment_memory_limit = int(os.getenv("SCREEN_COMMENT_MEMORY_LIMIT", "3"))
    screen_comment_memory_limit = max(0, screen_comment_memory_limit)
    screen_comment_memory_weight = float(os.getenv("SCREEN_COMMENT_MEMORY_WEIGHT", "0.2"))
    screen_comment_memory_weight = max(0.0, min(1.0, screen_comment_memory_weight))
    auto_comment_style_weights = os.getenv(
        "AUTO_COMMENT_STYLE_WEIGHTS",
        "陪伴评论:1.0,轻松提问:1.2,俏皮打趣:1.0,温柔锐评:0.8,行动建议:1.0",
    )
    emotion_keywords_stressed = os.getenv(
        "EMOTION_KEYWORDS_STRESSED",
        "烦,崩溃,压力,焦虑,累,卡住,不会,好难,deadline,bug,报错,错误,失败,加班,熬夜,头疼,麻了",
    )
    emotion_keywords_positive = os.getenv(
        "EMOTION_KEYWORDS_POSITIVE",
        "哈哈,开心,搞定,完成,顺利,不错,太好了,舒服,进步,通过,成功,耶,轻松,满意",
    )
    emotion_keywords_focused = os.getenv(
        "EMOTION_KEYWORDS_FOCUSED",
        "学习,复习,写作业,刷题,阅读,写代码,调试,文档,论文,做题,专注,计划,总结,记笔记",
    )
    comment_similarity_skip_threshold = float(os.getenv("COMMENT_SIMILARITY_SKIP_THRESHOLD", "0.86"))
    comment_similarity_skip_threshold = max(0.0, min(1.0, comment_similarity_skip_threshold))
    enable_comment_api_understanding = os.getenv("ENABLE_COMMENT_API_UNDERSTANDING", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    scan_interval = int(os.getenv("SCREEN_SCAN_INTERVAL_SEC", "45"))
    cooldown = int(os.getenv("AUTO_COMMENT_COOLDOWN_SEC", "60"))

    candidate_1 = base_dir / "assets" / "model" / "pet.png"
    candidate_2 = base_dir.parent / "ui" / "pet.png"
    pet_image_path = candidate_1 if candidate_1.exists() else candidate_2

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        vision_model_name=vision_model_name,
        enable_multimodal_vision=enable_multimodal_vision,
        enable_mm_compat_mode=enable_mm_compat_mode,
        enable_mm_screen_comment=enable_mm_screen_comment,
        mm_timeout_sec=mm_timeout_sec,
        mm_failure_threshold=mm_failure_threshold,
        mm_cooldown_sec=mm_cooldown_sec,
        mm_image_max_edge=mm_image_max_edge,
        mm_output_max_tokens=mm_output_max_tokens,
        mm_auto_min_interval_sec=mm_auto_min_interval_sec,
        vision_capture_mode=vision_capture_mode,
        vision_window_inset_lr=vision_window_inset_lr,
        vision_window_inset_tb=vision_window_inset_tb,
        vision_window_margin_px=vision_window_margin_px,
        vision_window_monitor_index=vision_window_monitor_index,
        vision_min_interval_sec=vision_min_interval_sec,
        vision_shots_per_minute=vision_shots_per_minute,
        vision_burst_cap=vision_burst_cap,
        vision_min_gap_sec=vision_min_gap_sec,
        vision_context_ttl_sec=vision_context_ttl_sec,
        vision_change_threshold=vision_change_threshold,
        vision_focus_query=vision_focus_query,
        vision_target_title=vision_target_title,
        vision_budget_per_hour=vision_budget_per_hour,
        visual_novel_text_ratio=visual_novel_text_ratio,
        dedup_ngram_merge=dedup_ngram_merge,
        dedup_ngram_guard=dedup_ngram_guard,
        dedup_semantic_merge=dedup_semantic_merge,
        dedup_calibration_enabled=dedup_calibration_enabled,
        dedup_calibration_path=dedup_calibration_path,
        visual_novel_max_facts=visual_novel_max_facts,
        enable_ocr_first_routing=enable_ocr_first_routing,
        ocr_only_min_chars=ocr_only_min_chars,
        ocr_only_min_confidence=ocr_only_min_confidence,
        ocr_hybrid_min_chars=ocr_hybrid_min_chars,
        ocr_context_max_chars=ocr_context_max_chars,
        enable_auto_comment_heartbeat=enable_auto_comment_heartbeat,
        chat_show_system_messages=chat_show_system_messages,
        chat_show_session_debug_marker=chat_show_session_debug_marker,
        enable_chat_multimodal=enable_chat_multimodal,
        chat_screen_context_max_chars=chat_screen_context_max_chars,
        enable_visual_novel_mode=enable_visual_novel_mode,
        visual_novel_min_context_similarity=visual_novel_min_context_similarity,
        enable_semantic_attention=enable_semantic_attention,
        enable_semantic_comment_candidates=enable_semantic_comment_candidates,
        semantic_attention_model_path=semantic_attention_model_path,
        semantic_attention_top_k=semantic_attention_top_k,
        semantic_attention_min_score=semantic_attention_min_score,
        semantic_attention_max_length=semantic_attention_max_length,
        semantic_attention_cache_size=semantic_attention_cache_size,
        semantic_attention_cpu_threads=semantic_attention_cpu_threads,
        web_search_soft_deadline_sec=web_search_soft_deadline_sec,
        web_search_hard_deadline_sec=web_search_hard_deadline_sec,
        web_search_circuit_failure_threshold=web_search_circuit_failure_threshold,
        web_search_circuit_cooldown_sec=web_search_circuit_cooldown_sec,
        web_search_max_results=web_search_max_results,
        web_search_context_max_chars=web_search_context_max_chars,
        baidu_ai_search_api_key=baidu_ai_search_api_key,
        long_memory_limit=long_memory_limit,
        long_memory_context_window=long_memory_context_window,
        enable_tutor_persona=enable_tutor_persona,
        enable_tts=enable_tts,
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        tts_rate=tts_rate,
        tts_volume=tts_volume,
        tts_azure_key=tts_azure_key,
        tts_azure_region=tts_azure_region,
        tts_azure_endpoint=tts_azure_endpoint,
        tts_voicevox_base_url=tts_voicevox_base_url,
        tts_voicevox_speaker=tts_voicevox_speaker,
        tts_voicevox_engine_path=tts_voicevox_engine_path,
        enable_voicevox_auto_launch=enable_voicevox_auto_launch,
        enable_voicevox_ja_translation=enable_voicevox_ja_translation,
        tts_translation_provider=tts_translation_provider,
        tts_translation_model_id=tts_translation_model_id,
        tts_translation_model_path=tts_translation_model_path,
        tts_translation_device=tts_translation_device,
        tts_translation_compute_type=tts_translation_compute_type,
        tts_translation_cache_path=tts_translation_cache_path,
        tts_translation_cache_size=tts_translation_cache_size,
        tts_translation_max_chars=tts_translation_max_chars,
        enable_tts_translation_api_fallback=enable_tts_translation_api_fallback,
        tts_skip_on_translation_failure=tts_skip_on_translation_failure,
        webengine_gpu_mode=webengine_gpu_mode,
        enable_scan_subprocess=enable_scan_subprocess,
        scan_monitor_index=scan_monitor_index,
        scan_region=scan_region,
        scan_tick_interval_sec=scan_tick_interval_sec,
        scan_submit_min_interval_sec=scan_submit_min_interval_sec,
        scan_busy_timeout_sec=scan_busy_timeout_sec,
        auto_start_scan=auto_start_scan,
        scan_adaptive_max_interval_sec=scan_adaptive_max_interval_sec,
        ocr_fallback_min_interval_sec=ocr_fallback_min_interval_sec,
        ocr_fallback_min_quiet_sec=ocr_fallback_min_quiet_sec,
        ocr_text_change_threshold=ocr_text_change_threshold,
        ocr_text_ink_change_threshold=ocr_text_ink_change_threshold,
        vn_text_band_bottom_margin=vn_text_band_bottom_margin,
        vn_text_band_mid_top_ratio=vn_text_band_mid_top_ratio,
        ocr_cpu_threads=ocr_cpu_threads,
        ocr_cpu_affinity_count=ocr_cpu_affinity_count,
        ocr_max_edge=ocr_max_edge,
        ocr_text_max_edge=ocr_text_max_edge,
        resource_policy_reapply_min_sec=resource_policy_reapply_min_sec,
        memory_recency_window_sec=memory_recency_window_sec,
        memory_min_weight=memory_min_weight,
        enable_live2d=enable_live2d,
        enable_live2d_py=enable_live2d_py,
        live2d_py_window_width=live2d_py_window_width,
        live2d_py_window_height=live2d_py_window_height,
        live2d_model_json=live2d_model_json,
        live2d_follow_cursor=live2d_follow_cursor,
        live2d_follow_activate_distance_px=live2d_follow_activate_distance_px,
        live2d_model_scale=live2d_model_scale,
        live2d_idle_group=live2d_idle_group,
        screen_comment_memory_limit=screen_comment_memory_limit,
        screen_comment_memory_weight=screen_comment_memory_weight,
        auto_comment_style_weights=auto_comment_style_weights,
        emotion_keywords_stressed=emotion_keywords_stressed,
        emotion_keywords_positive=emotion_keywords_positive,
        emotion_keywords_focused=emotion_keywords_focused,
        comment_similarity_skip_threshold=comment_similarity_skip_threshold,
        enable_comment_api_understanding=enable_comment_api_understanding,
        screen_scan_interval_sec=scan_interval,
        auto_comment_cooldown_sec=cooldown,
        pet_image_path=pet_image_path,
    )
