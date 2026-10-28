# Windows 换电脑安装：桌宠与本地日语语音

以下步骤复现本项目已验证的组合：Windows x64、Python 3.10、PyTorch / torchaudio 2.5.1＋CUDA 12.4、FlashAttention 2.7.4.post1、tsumugi v2Pro Plus。测试显卡为 RTX 3060 Laptop 6 GB。使用匹配的预编译包，不需要在本机编译 FlashAttention。

两个 Python 环境分开安装：项目根目录的 `.venv` 运行桌宠，`tools/gpt_sovits/.venv` 运行语音服务。不要把语音环境的锁定依赖装进桌宠环境。

## 1. 下载项目和 Git LFS 模型

先安装 Git、Git LFS、Python 3.10 x64 和能运行 CUDA 12.4 版 PyTorch 的 NVIDIA 驱动。以下命令在 PowerShell 中执行。

```powershell
git lfs install
New-Item -ItemType Directory -Force D:/apps | Out-Null
Set-Location D:/apps
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/princeps-works/deskpet-yuki.git desktop_pet
Remove-Item Env:/GIT_LFS_SKIP_SMUDGE
Set-Location ./desktop_pet
git lfs pull --include="assets/models/vn-reaction-v64/**,assets/models/gpt-sovits-pretrained/**,VOICE_model/GPT_weights_v2ProPlus/tsumugi/**"
```

克隆目录请保留为 **`desktop_pet`**，主程序按这个包名导入。上面的定向 LFS 下载包含两个本地筛选器共用的编码器，以及本地语音资产，避免同时下载其它语音引擎的大文件。普通 Git 文件会包含两个筛选器的分类头、配置和分词器。

本次新增的语音资产约 1.4 GB：

| 位置 | 内容 |
| --- | --- |
| `VOICE_model/GPT_weights_v2ProPlus/tsumugi` | GPT / SoVITS 声线权重、8 段参考音频、对应文本 |
| `assets/models/gpt-sovits-pretrained/chinese-roberta-wwm-ext-large` | BERT 权重、配置、分词器 |
| `assets/models/gpt-sovits-pretrained/chinese-hubert-base` | CNHuBERT 权重与预处理配置 |
| `assets/models/gpt-sovits-pretrained/sv` | v2Pro Plus 声纹特征模型 |
| `assets/models/gpt-sovits-pretrained/fast_langdetect` | 语言检测模型 |

文件路径、大小与 SHA-256 见 [`config/speech_model_manifest.json`](../config/speech_model_manifest.json)。若权重文件实际是以 `version https://git-lfs.github.com/spec/v1` 开头的短文本，说明尚未下载 LFS 实体文件，请重新执行 `git lfs pull`。

## 2. 安装桌宠环境

在 `desktop_pet` 根目录运行：

```powershell
py -3.10 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
```

编辑本机 `.env`，填写自己的 API 地址、密钥和模型。真实 `.env` 不上传 GitHub，已有配置不要用示例文件覆盖。

本地日语语音可使用以下设置，路径相对于项目根目录解析：

```dotenv
ENABLE_TTS=true
TTS_PROVIDER=gpt_sovits
TTS_GPT_SOVITS_BASE_URL=http://127.0.0.1:9880
TTS_GPT_SOVITS_REFERENCE_AUDIO=VOICE_model/GPT_weights_v2ProPlus/tsumugi/reference_01.wav
TTS_GPT_SOVITS_REFERENCE_TEXT=あなたが知っているから、ここにあるんです。
TTS_GPT_SOVITS_SEGMENTED=true
TTS_TRANSLATION_PROVIDER=index
```

`index` 是联网的中译日通道。若选择其它翻译通道，按其原有配置另行安装翻译模型或服务；本语音模型包包含发声所需模型。不要直接复制旧电脑 `.env` 中的磁盘绝对路径。

## 3. 安装固定版本的语音源码与独立环境

在项目根目录运行：

```powershell
New-Item -ItemType Directory -Force tools | Out-Null
git clone https://github.com/RVC-Boss/GPT-SoVITS.git tools/gpt_sovits
git -C tools/gpt_sovits checkout e7cd61ec3dcc9a34089f50ed169e3103a16ad675
py -3.10 -m venv tools/gpt_sovits/.venv
$ttsPython = Join-Path (Get-Location) 'tools/gpt_sovits/.venv/Scripts/python.exe'
& $ttsPython -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
& $ttsPython -m pip install -r requirements-gpt-sovits.txt
```

请使用固定源码提交；取消补丁针对该提交生成。已有 `tools/gpt_sovits` 时，先检查其中的本地修改，不要用强制检出覆盖自己的文件。

安装 FFmpeg / FFprobe 到该版本运行时要求的目录，下载地址来自 [GPT-SoVITS 上游安装说明](https://github.com/RVC-Boss/GPT-SoVITS/blob/e7cd61ec3dcc9a34089f50ed169e3103a16ad675/README.md)：

```powershell
New-Item -ItemType Directory -Force tools/gpt_sovits/runtime | Out-Null
Invoke-WebRequest 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe' -OutFile tools/gpt_sovits/runtime/ffmpeg.exe
Invoke-WebRequest 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffprobe.exe' -OutFile tools/gpt_sovits/runtime/ffprobe.exe
```

## 4. 安装匹配的 FlashAttention

此包精确匹配 **Python 3.10 / Windows x64 / torch 2.5.1 / cu124**。其它 Python、PyTorch 或 CUDA 构建不能直接套用这个 wheel。

```powershell
$wheelUrl = 'https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl'
$wheelPath = Join-Path $env:TEMP 'flash_attn-2.7.4.post1+cu124torch2.5.1cxx11abiFALSE-cp310-cp310-win_amd64.whl'
Invoke-WebRequest $wheelUrl -OutFile $wheelPath
if ((Get-FileHash $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ad593bd27ae121d521e0734768bb02ef282945adf1a8bf1f23cb110a2f38d19c') { throw 'FlashAttention wheel checksum mismatch' }
& $ttsPython -m pip install --no-deps $wheelPath
& $ttsPython -c "import torch,flash_attn; print(torch.__version__,torch.version.cuda,flash_attn.__version__); print(callable(flash_attn.flash_attn_with_kvcache))"
```

预期输出包含 `2.5.1+cu124`、`12.4`、`2.7.4.post1` 和 `True`。`--no-deps` 用于保留已安装的 PyTorch 版本。FlashAttention 缺失时，上游可回退到普通 CUDA Graph；是否加载成功以服务日志中的后端名称为准。

## 5. 准备模型并启动

```powershell
& $ttsPython scripts/prepare_gpt_sovits.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_gpt_sovits.ps1
./.venv/Scripts/python.exe main.py
```

准备脚本会幂等应用 Windows `jieba` 兼容补丁和按请求取消补丁，并把仓库中的基础模型接入语音运行目录。优先创建硬链接；文件系统不支持时复制。已经存在的运行模型保留原样，不在每次启动时重复制约 1 GB 数据。

也可以在已激活桌宠 `.venv` 的终端运行 `pet.bat`。启动器只等待 API 就绪；桌宠随后用实际配置的参考文本完成一次静默合成预热，之后才启动播音线程。因此复用已运行的服务、直接运行 `main.py` 或服务断线恢复，都要经过客户端预热。

检查以下证据：

- `data/gpt_sovits_server.log`：成功加载 `flash_attn_varlen_cuda_graph Backend`，且没有加速不可用或图捕获失败提示。
- 桌宠启动输出：`GPT-SoVITS synthesis warmed (CUDA Graph requested)`。
- `http://127.0.0.1:9880/openapi.json`：同时包含 `/tts`、`/cancel_tts`。

播报规则：尚未开口的自动评论只保留最新；已经开口的自动评论正常说完，后续自动评论等待且只保留最新；用户聊天回复可优先打断自动评论。取消针对独立请求 ID，在语义生成步骤之间停止旧任务，不重载声线模型或加速器。一个正在执行的 GPU 步骤需要先结束，不能保证零毫秒取消。

## 6. 旧电脑更新与换机校验

已有环境更新时：

```powershell
git pull
git lfs pull --include="assets/models/vn-reaction-v64/**,assets/models/gpt-sovits-pretrained/**,VOICE_model/GPT_weights_v2ProPlus/tsumugi/**"
$ttsPython = Join-Path (Get-Location) 'tools/gpt_sovits/.venv/Scripts/python.exe'
& $ttsPython -m pip install -r requirements-gpt-sovits.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop_gpt_sovits.ps1
& $ttsPython scripts/prepare_gpt_sovits.py
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_gpt_sovits.ps1
```

按第 4 节补装 FlashAttention，然后重新打开桌宠。旧服务没有 `/cancel_tts` 时必须重启一次；仅更新源码不会替换已加载的服务。停止脚本只处理由本项目记录并核对配置路径的语音进程。若 9880 被其它服务占用，先确认占用来源，启动器不会强行替换它。

验收时先核对文件、依赖导入、API 和启动日志即可，不必重新训练筛选器或重复合成性能评测。两种本地筛选器可在视觉小说菜单第四项「模型设置」里切换，选择会保存在本机 `.env`。
