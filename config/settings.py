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
    base_dir: Path
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
    enable_auto_comment_heartbeat: bool
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
    webengine_gpu_mode: str
    enable_scan_subprocess: bool
    scan_monitor_index: int
    scan_region: Optional[Tuple[int, int, int, int]]
    scan_tick_interval_sec: int
    scan_submit_min_interval_sec: int
    scan_busy_timeout_sec: int
    ocr_cpu_threads: int
    ocr_cpu_affinity_count: int
    ocr_max_edge: int
    resource_policy_hard_after_sec: float
    resource_policy_release_grace_sec: float
    resource_policy_reapply_min_sec: float
    memory_recency_window_sec: int
    memory_min_weight: float
    enable_live2d: bool
    enable_live2d_py: bool
    enable_live2d_py_poc: bool
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


def _resolve_live2d_model_path(base_dir: Path, raw_value: str, default_path: Path) -> str:
    text = (raw_value or "").strip()
    if not text:
        return str(default_path)

    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    # Relative paths are resolved from project directory for stable behavior.
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
    vision_model_name = os.getenv("VISION_MODEL_NAME", model_name)
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
    enable_auto_comment_heartbeat = os.getenv("ENABLE_AUTO_COMMENT_HEARTBEAT", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
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
    tts_voicevox_engine_path = os.getenv("TTS_VOICEVOX_ENGINE_PATH", str(default_voicevox_engine_path))
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
    webengine_gpu_mode = os.getenv("WEBENGINE_GPU_MODE", "gpu").strip().lower()
    if webengine_gpu_mode not in {"gpu", "software", "auto"}:
        webengine_gpu_mode = "gpu"
    enable_scan_subprocess = os.getenv("ENABLE_SCAN_SUBPROCESS", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    scan_monitor_index = int(os.getenv("SCAN_MONITOR_INDEX", "1"))
    scan_monitor_index = max(1, scan_monitor_index)
    scan_region = _parse_scan_region(os.getenv("SCAN_REGION", ""))
    scan_tick_interval_sec = int(os.getenv("SCAN_TICK_INTERVAL_SEC", "0"))
    scan_tick_interval_sec = max(0, scan_tick_interval_sec)
    scan_submit_min_interval_sec = int(os.getenv("SCAN_SUBMIT_MIN_INTERVAL_SEC", "0"))
    scan_submit_min_interval_sec = max(0, scan_submit_min_interval_sec)
    scan_busy_timeout_sec = int(os.getenv("SCAN_BUSY_TIMEOUT_SEC", "12"))
    scan_busy_timeout_sec = max(5, scan_busy_timeout_sec)
    ocr_cpu_threads = int(os.getenv("OCR_CPU_THREADS", "0"))
    ocr_cpu_threads = max(0, min(64, ocr_cpu_threads))
    ocr_cpu_affinity_count = int(os.getenv("OCR_CPU_AFFINITY_COUNT", "0"))
    ocr_cpu_affinity_count = max(0, min(64, ocr_cpu_affinity_count))
    ocr_max_edge = int(os.getenv("OCR_MAX_EDGE", "0"))
    ocr_max_edge = max(0, min(4096, ocr_max_edge))
    resource_policy_hard_after_sec = float(os.getenv("RESOURCE_POLICY_HARD_AFTER_SEC", "2.5"))
    resource_policy_hard_after_sec = max(0.5, min(30.0, resource_policy_hard_after_sec))
    resource_policy_release_grace_sec = float(os.getenv("RESOURCE_POLICY_RELEASE_GRACE_SEC", "1.2"))
    resource_policy_release_grace_sec = max(0.0, min(10.0, resource_policy_release_grace_sec))
    resource_policy_reapply_min_sec = float(os.getenv("RESOURCE_POLICY_REAPPLY_MIN_SEC", "4.0"))
    resource_policy_reapply_min_sec = max(0.2, min(20.0, resource_policy_reapply_min_sec))
    memory_recency_window_sec = int(os.getenv("MEMORY_RECENCY_WINDOW_SEC", "0"))
    memory_recency_window_sec = max(0, memory_recency_window_sec)
    memory_min_weight = float(os.getenv("MEMORY_MIN_WEIGHT", "0.15"))
    memory_min_weight = max(0.0, min(1.0, memory_min_weight))
    default_live2d_model = (
        base_dir.parent.parent / "皮套" / "mao_pro_zh" / "runtime" / "mao_pro.model3.json"
    )
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
    enable_live2d_py_poc = os.getenv("ENABLE_LIVE2D_PY_POC", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Backward compatibility for previous PoC flag name.
    if enable_live2d_py_poc:
        enable_live2d_py = True
    # Treat ENABLE_LIVE2D as a global master switch for all Live2D backends.
    if not enable_live2d:
        enable_live2d_py = False
        enable_live2d_py_poc = False
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
    scan_interval = int(os.getenv("SCREEN_SCAN_INTERVAL_SEC", "45"))
    cooldown = int(os.getenv("AUTO_COMMENT_COOLDOWN_SEC", "60"))

    candidate_1 = base_dir / "assets" / "model" / "pet.png"
    candidate_2 = base_dir.parent / "ui" / "pet.png"
    pet_image_path = candidate_1 if candidate_1.exists() else candidate_2

    return Settings(
        base_dir=base_dir,
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
        enable_auto_comment_heartbeat=enable_auto_comment_heartbeat,
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
        webengine_gpu_mode=webengine_gpu_mode,
        enable_scan_subprocess=enable_scan_subprocess,
        scan_monitor_index=scan_monitor_index,
        scan_region=scan_region,
        scan_tick_interval_sec=scan_tick_interval_sec,
        scan_submit_min_interval_sec=scan_submit_min_interval_sec,
        scan_busy_timeout_sec=scan_busy_timeout_sec,
        ocr_cpu_threads=ocr_cpu_threads,
        ocr_cpu_affinity_count=ocr_cpu_affinity_count,
        ocr_max_edge=ocr_max_edge,
        resource_policy_hard_after_sec=resource_policy_hard_after_sec,
        resource_policy_release_grace_sec=resource_policy_release_grace_sec,
        resource_policy_reapply_min_sec=resource_policy_reapply_min_sec,
        memory_recency_window_sec=memory_recency_window_sec,
        memory_min_weight=memory_min_weight,
        enable_live2d=enable_live2d,
        enable_live2d_py=enable_live2d_py,
        enable_live2d_py_poc=enable_live2d_py_poc,
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
        screen_scan_interval_sec=scan_interval,
        auto_comment_cooldown_sec=cooldown,
        pet_image_path=pet_image_path,
    )
