# Desktop Pet MVP Skeleton

## Environment Setup

Recommended environment:

- Windows 10/11
- Python 3.10 to 3.12
- Network access for LLM API and optional online TTS provider

Setup steps:

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Initialize config file

   Copy .env.example to .env, then edit values.

3. Fill minimum required config

   - N1N_API_KEY
   - N1N_BASE_URL
   - MODEL_NAME

4. Run app

   ```bash
   python main.py
   ```

## Config Overview

Full parameter definitions are in .env.example. This README keeps only setup-level guidance.

Key categories:

1. LLM and API
- API key, base URL, text model, vision model.

2. Multimodal and OCR
- Vision fallback, timeout/cooldown, OCR load controls.

3. Runtime behavior
- Heartbeat logs, tutor mode, chat debug display switches.

4. TTS
- edge, azure, voicevox provider configs.

5. Live2D and renderer
- Model path, live2d-py mode, follow behavior, debug log switches.

6. Scheduler and resource policy
- Scan cadence, busy timeout, CPU policy tuning.

7. Memory and auto comment
- Memory weight, style weights, emotion keyword dictionaries.

## Multimodal Notes

- If vision analysis fails or times out, the app can fall back to OCR path in compatibility mode.
- For low-end devices, prioritize region scan + larger submit interval.

## Live2D Notes

- Install dependency first: `pip install PyQt6-WebEngine`.
- If Live2D backend is unavailable, app falls back to static `pet.png` rendering.
- `LIVE2D_INPUT_DIAG=0` by default. Set to `1` only when debugging input-follow snapshots.
- `LIVE2D_FILTER_MOTION_NOISE_LOG=1` by default. It suppresses noisy `can't start motion` lines when actions still execute normally.
- If you want full raw native logs for deep troubleshooting, set `LIVE2D_FILTER_MOTION_NOISE_LOG=0`.

## TTS Notes

- edge is the easiest default provider.
- Use Azure or VOICEVOX when you need specific voice quality or local engine control.

## VOICEVOX Install

- Install VOICEVOX app or engine into any local folder.
- Ensure engine is running and listening on a local port (default `50021`).
- Set env:
  - `TTS_PROVIDER=voicevox`
  - `TTS_VOICEVOX_BASE_URL=http://127.0.0.1:50021`
  - `TTS_VOICEVOX_SPEAKER=1`
   - `TTS_VOICEVOX_ENGINE_PATH=VOICEVOX/VOICEVOX/vv-engine/run.exe`

Notes:

- TTS_VOICEVOX_ENGINE_PATH supports both absolute path and project-relative path.
- Project-relative path is resolved from the project root.
- ENABLE_VOICEVOX_AUTO_LAUNCH=true allows auto-launch when API is unreachable.

### Offline Chinese-to-Japanese Translation

VOICEVOX translation is centralized inside `SpeechService`. Chinese text is never sent directly to a Japanese-only speaker when translation fails.

Install the conversion dependencies and prepare the local model once:

```powershell
python -m pip install "transformers>=4.48,<5" huggingface-hub sentencepiece ctranslate2
python scripts/setup_local_translation.py
```

The setup command downloads `facebook/m2m100_418M`, converts it to CTranslate2 `int8_float16`, and stores the runtime model under `assets/models/m2m100_418M_ct2`. Subsequent translation is fully local. The translated short sentences are persisted in an LRU cache under `data/tts_translation_cache.json`.

Recommended settings:

```env
ENABLE_VOICEVOX_JA_TRANSLATION=true
TTS_TRANSLATION_PROVIDER=local
TTS_TRANSLATION_MODEL_PATH=assets/models/m2m100_418M_ct2
TTS_TRANSLATION_DEVICE=cuda
TTS_TRANSLATION_COMPUTE_TYPE=int8_float16
TTS_TRANSLATION_CACHE_SIZE=2000
ENABLE_TTS_TRANSLATION_API_FALLBACK=false
TTS_SKIP_ON_TRANSLATION_FAILURE=true
```

### Local Semantic Attention

Manual chat can use a local Chinese embedding model to rank recent dialogue, long-term memory,
web results, OCR, and vision descriptions before the main LLM request. The ranking changes which
chunks are sent and how much text is retained; its scores are not merely prompt annotations.

Prepare the quantized ONNX model once:

```powershell
python scripts/setup_semantic_attention.py
```

The model is stored under `assets/models/bge-small-zh-v1.5-onnx`. Runtime inference uses the
already-required `transformers`, `onnxruntime`, and `numpy` packages; PyTorch is not required.
If the model is absent or cannot load, chat automatically keeps the previous truncation behavior.

Recommended semantic-attention settings:

```env
ENABLE_SEMANTIC_ATTENTION=true
SEMANTIC_ATTENTION_MODEL_PATH=assets/models/bge-small-zh-v1.5-onnx
SEMANTIC_ATTENTION_TOP_K=8
SEMANTIC_ATTENTION_MIN_SCORE=0.34
SEMANTIC_ATTENTION_MAX_LENGTH=256
SEMANTIC_ATTENTION_CACHE_SIZE=512
SEMANTIC_ATTENTION_CPU_THREADS=2
```

## Chat Display Switches

- CHAT_SHOW_SYSTEM_MESSAGES=false:
   hide role=系统 lines in chat window for immersion.
- CHAT_SHOW_SESSION_DEBUG_MARKER=false:
   show session markers like 桌宠(自动)[S2] only when debugging archive cycle behavior.
- ENABLE_CHAT_MULTIMODAL=false:
   initial state of the chat panel's multimodal button. When enabled, each manual message reads the current scan region before replying.
- CHAT_SCREEN_CONTEXT_MAX_CHARS=1600:
   maximum screen/OCR context fused into one chat request. Text-rich screens still use the OCR-first token-saving route.
- SCAN_MONITOR_INDEX=0:
   scan the complete virtual desktop by default. UI region selection automatically targets the display under the cursor and converts Qt logical pixels to physical capture pixels.

## Current Capabilities

- Floating desktop pet window (drag, resize, close)
- Chat panel (manual conversation)
- Basic model client abstraction (fallback to local echo)
- Pluggable modules for screen capture, OCR, policy, and Live2D mapping

## Next Steps

- Wire `vision.capture` + `vision.scene_analyzer` to timed auto-comment
- Replace placeholder Live2D driver with actual runtime bridge
- Add system tray controls and settings panel

## Env Quick Reference

Copy the following presets into your `.env` and adjust paths/keys as needed.

### 1) Low-End Device / Smooth First

```env
ENABLE_MM_SCREEN_COMMENT=false
ENABLE_SCAN_SUBPROCESS=true
SCAN_REGION=200,120,1200,700
SCAN_TICK_INTERVAL_SEC=10
SCAN_SUBMIT_MIN_INTERVAL_SEC=15
SCAN_BUSY_TIMEOUT_SEC=10
OCR_CPU_THREADS=2
OCR_CPU_AFFINITY_COUNT=2
OCR_MAX_EDGE=960
SCREEN_SCAN_INTERVAL_SEC=60
AUTO_COMMENT_COOLDOWN_SEC=80
LIVE2D_FOLLOW_CURSOR=true
LIVE2D_FOLLOW_ACTIVATE_DISTANCE_PX=180
```

### 2) Balanced Daily Use (Recommended)

```env
ENABLE_MULTIMODAL_VISION=true
ENABLE_MM_COMPAT_MODE=true
ENABLE_MM_SCREEN_COMMENT=true
ENABLE_OCR_FIRST_ROUTING=true
MM_TIMEOUT_SEC=5.0
MM_FAILURE_THRESHOLD=3
MM_COOLDOWN_SEC=120
MM_OUTPUT_MAX_TOKENS=220
MM_AUTO_MIN_INTERVAL_SEC=180
OCR_ONLY_MIN_CHARS=120
OCR_ONLY_MIN_CONFIDENCE=0.75
OCR_HYBRID_MIN_CHARS=30
OCR_CONTEXT_MAX_CHARS=1200
SCAN_TICK_INTERVAL_SEC=8
SCAN_SUBMIT_MIN_INTERVAL_SEC=12
SCREEN_SCAN_INTERVAL_SEC=45
AUTO_COMMENT_COOLDOWN_SEC=60
```

### 3) Rich and Varied Auto Comments

```env
AUTO_COMMENT_STYLE_WEIGHTS=陪伴评论:0.9,轻松提问:1.3,俏皮打趣:1.3,温柔锐评:0.7,行动建议:1.1
EMOTION_KEYWORDS_STRESSED=烦,崩溃,压力,焦虑,累,卡住,不会,好难,报错,失败,加班,熬夜,头疼
EMOTION_KEYWORDS_POSITIVE=哈哈,开心,搞定,完成,顺利,不错,太好了,进步,通过,成功,耶
EMOTION_KEYWORDS_FOCUSED=学习,复习,刷题,阅读,写代码,调试,文档,论文,专注,计划,总结
SCREEN_COMMENT_MEMORY_LIMIT=4
SCREEN_COMMENT_MEMORY_WEIGHT=0.25
COMMENT_SIMILARITY_SKIP_THRESHOLD=0.84
```

### 4) Quiet Mode / Minimal Disturbance

```env
ENABLE_TTS=false
ENABLE_AUTO_COMMENT_HEARTBEAT=false
SCREEN_SCAN_INTERVAL_SEC=90
AUTO_COMMENT_COOLDOWN_SEC=120
AUTO_COMMENT_STYLE_WEIGHTS=陪伴评论:1.2,轻松提问:0.8,俏皮打趣:0.5,温柔锐评:0.4,行动建议:0.9
```

### Notes

- `SCAN_REGION` is one of the highest-impact settings for smoothness.
- If stutter appears, first increase `SCAN_SUBMIT_MIN_INTERVAL_SEC` and `SCREEN_SCAN_INTERVAL_SEC`.
- If CPU spikes remain high, tune `OCR_CPU_THREADS`, `OCR_CPU_AFFINITY_COUNT`, and `OCR_MAX_EDGE`.
- `OCR_CPU_THREADS` is applied directly to RapidOCR init (`intra/inter op threads`); restart app after changing it.
- You can try GPU OCR path with `OCR_USE_DML=true` (Windows) or `OCR_USE_CUDA=true` when runtime supports it.
- For unstable vision APIs, keep `ENABLE_MM_COMPAT_MODE=true` to preserve OCR fallback.
- OCR-first routing uses local OCR for text-rich screens, combines compact OCR with vision for mixed screens, and uses vision alone when OCR is sparse.
- `MM_AUTO_MIN_INTERVAL_SEC` limits automatic screenshot uploads independently from the local scan interval; manual screen comments bypass this interval.
- `MM_OUTPUT_MAX_TOKENS` and `OCR_CONTEXT_MAX_CHARS` cap the two variable-size parts of a hybrid vision request.
