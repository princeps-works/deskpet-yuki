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

## Multimodal Vision Branch

- `ENABLE_MULTIMODAL_VISION=true` to enable screenshot vision understanding.
- `VISION_MODEL_NAME` can be set to a model that supports image input.
- If vision analysis fails, the app automatically falls back to OCR-only behavior.

## Scan Target Configuration

- `SCAN_MONITOR_INDEX=1` means capture display #1 (external monitor can be 2, 3, ...).
- `SCAN_REGION=left,top,width,height` limits scanning to a rectangle on the selected monitor.
- Leave `SCAN_REGION=` empty to scan the whole selected monitor.
- `SCAN_TICK_INTERVAL_SEC=0` controls high-frequency scan ticks for memory accumulation.
   - `0` means auto-calc from `SCREEN_SCAN_INTERVAL_SEC`.
   - positive integer means fixed tick interval in seconds (e.g. `8`).
- `SCAN_SUBMIT_MIN_INTERVAL_SEC=0` controls minimum interval between scan task submissions.
   - `0` means auto-calc.
   - positive integer limits OCR submit frequency (recommended `12` to `25` when subprocess mode is enabled).
- `SCAN_BUSY_TIMEOUT_SEC=12` controls how long a scan task can remain busy before force-resetting scan workers.
- In subprocess mode, near-identical frames can reuse previous OCR analysis to keep sample cadence high while reducing CPU load.
- `MEMORY_RECENCY_WINDOW_SEC=0` controls weighted-memory recency window.
   - `0` means auto-calc from comment interval.
   - positive integer means custom recency window in seconds.
- `MEMORY_MIN_WEIGHT=0.15` controls the minimum retained weight for older memory items (`0.0` to `1.0`).
- `ENABLE_SCAN_SUBPROCESS=true` runs auto scan pipeline in a dedicated subprocess to reduce UI/Live2D contention.
- `SCREEN_COMMENT_MEMORY_LIMIT=3` controls how many long-memory entries are included in screen comments.
   - `0` disables long-memory hint for screen comments.
- `SCREEN_COMMENT_MEMORY_WEIGHT=0.2` controls suggested long-memory influence in screen-comment prompting (`0.0` to `1.0`).
- `COMMENT_SIMILARITY_SKIP_THRESHOLD=0.86` skips emitting auto comments that are too similar to the previous one.

## Live2D Model Display

- Install dependency first: `pip install PyQt6-WebEngine`.
- `WEBENGINE_GPU_MODE=gpu` controls WebEngine render mode for Live2D:
   - `gpu`: prefer GPU acceleration (recommended default).
   - `software`: force software rendering for compatibility testing.
   - `auto`: do not inject extra Chromium flags.
- `ENABLE_LIVE2D=true` enables WebEngine-based Live2D rendering.
- `LIVE2D_MODEL_JSON` points to your `*.model3.json` file.
   - Example: `G:\py\桌宠\皮套\mao_pro_zh\runtime\mao_pro.model3.json`
- `ENABLE_LIVE2D_PY=true` enables integrated `live2d-py` renderer process mode.
   - In this mode, WebEngine Live2D rendering is automatically disabled to avoid dual-render contention.
   - The main pet host window becomes a transparent top control bar and is auto-docked with the Live2D window.
   - Chat panel switches to frameless overlay style and follows the Live2D window position.
- `LIVE2D_FOLLOW_CURSOR=true` enables gaze follow (mouse tracking).
- `LIVE2D_MODEL_SCALE=1.0` controls model scale (recommended `0.8` to `1.4`).
- `LIVE2D_IDLE_GROUP=Idle` sets preferred idle motion group name.
- `ENABLE_LIVE2D_PY_POC=true` is kept as a backward-compatible alias for `ENABLE_LIVE2D_PY=true`.
- `LIVE2D_PY_WINDOW_WIDTH` / `LIVE2D_PY_WINDOW_HEIGHT` controls PoC renderer window size.
   - Default is reduced to half-size (`280x430`) for lighter load.
- Interaction:
   - Left-click on model triggers tap reaction (head/body hit-area attempt).
   - Mouse move updates model focus direction when follow is enabled.
- Fallback behavior:
   - If WebEngine dependency is missing or model path is invalid, app falls back to static `pet.png` rendering.

## Speech Output

- `ENABLE_TTS=true` enables text-to-speech playback for pet replies.
- `TTS_PROVIDER=edge` uses Edge TTS (local synthesis + local playback).
- `TTS_PROVIDER=azure` uses Azure Speech service.
- `TTS_PROVIDER=voicevox` uses local VOICEVOX Engine HTTP API.
- `TTS_VOICE=zh-CN-XiaoxiaoNeural` selects voice for both providers.
- `TTS_RATE` and `TTS_VOLUME` support percent format like `+0%`, `-10%`, `+20%`.
- Azure required vars when provider is `azure`:
   - `TTS_AZURE_KEY`
   - `TTS_AZURE_REGION` (or `TTS_AZURE_ENDPOINT`)
- VOICEVOX required vars when provider is `voicevox`:
   - `TTS_VOICEVOX_BASE_URL` (default `http://127.0.0.1:50021`)
   - `TTS_VOICEVOX_SPEAKER` (style/speaker id)
   - `ENABLE_VOICEVOX_AUTO_LAUNCH` (default `true`, auto-starts local engine if port is down)
   - `TTS_VOICEVOX_ENGINE_PATH` (path to `run.exe`)
   - `ENABLE_VOICEVOX_JA_TRANSLATION` (default `true`, translates Chinese text to Japanese before synthesis)

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
