# Desktop Pet MVP Skeleton

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill your API key.
3. Run:
   ```bash
   python main.py
   ```

## Config Overview

All detailed parameter definitions, defaults, value ranges, and examples are centralized in:

- `.env.example`

`README` only keeps category-level guidance to stay clean.

Adjustable env categories:

1. LLM / API
- API keys, base URL, text model, vision model.

2. Multimodal Scan Pipeline
- Vision enable switches, compatibility fallback, timeout/failure/cooldown, image resize edge.

3. Runtime Behavior
- Heartbeat log switch, tutor persona startup switch.

4. TTS
- Provider selection, voice/rate/volume, Azure settings, VOICEVOX settings.

5. Live2D / Render
- WebEngine mode, Live2D backend switches, model path, follow behavior, scale and idle group.

6. Scan Target and Scheduler
- Monitor/region selection, scan tick interval, submit interval, busy timeout, subprocess mode.

7. Windows Resource Policy
- OCR hard mode threshold, recovery grace, minimum policy reapply interval.

8. Memory and Comment Policy
- Memory recency/weight, long-memory injection, similarity skip threshold, scan/comment cadence.

9. Auto Comment Style and Emotion Tuning
- Style base weights and emotion keyword dictionaries.
- Auto comments jointly consider: scan memory, recent dialog, and long-memory hints.

## Multimodal Notes

- If vision analysis fails or times out, the app can fall back to OCR path in compatibility mode.
- For low-end devices, prioritize region scan + larger submit interval.

## Live2D Notes

- Install dependency first: `pip install PyQt6-WebEngine`.
- If Live2D backend is unavailable, app falls back to static `pet.png` rendering.

## TTS Notes

- `edge` is the easiest default provider.
- Use Azure/VOICEVOX when you need specific voice quality or local engine control.

## VOICEVOX Install on G Drive

- Install VOICEVOX app or engine into a folder on G drive, for example `G:\VOICEVOX`.
- Ensure engine is running and listening on a local port (default `50021`).
- Keep your project on any drive; only VOICEVOX main files need to be on G drive.
- Set env:
  - `TTS_PROVIDER=voicevox`
  - `TTS_VOICEVOX_BASE_URL=http://127.0.0.1:50021`
  - `TTS_VOICEVOX_SPEAKER=1`

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
MM_TIMEOUT_SEC=5.0
MM_FAILURE_THRESHOLD=3
MM_COOLDOWN_SEC=120
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
- For unstable vision APIs, keep `ENABLE_MM_COMPAT_MODE=true` to preserve OCR fallback.
