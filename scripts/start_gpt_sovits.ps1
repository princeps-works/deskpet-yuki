$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = Join-Path $root 'tools/gpt_sovits'
$python = Join-Path $runtime '.venv/Scripts/python.exe'
$config = Join-Path $root 'config/gpt_sovits_tsumugi.yaml'
if (!(Test-Path -LiteralPath $python)) { throw 'GPT-SoVITS environment is missing.' }
function Test-TtsReady {
    try {
        $schema = Invoke-RestMethod 'http://127.0.0.1:9880/openapi.json' -TimeoutSec 2
        return $null -ne $schema.paths.'/tts'
    } catch { return $false }
}
if (Test-TtsReady) {
    Write-Output 'GPT-SoVITS is already running on 127.0.0.1:9880.'
    return
}
if (Get-NetTCPConnection -LocalPort 9880 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 9880 is occupied; refusing to replace the existing service.'
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:NO_PROXY = '*'
$runtimeConfig = Join-Path $runtime 'GPT_SoVITS/configs/desktop_pet_tsumugi.yaml'
Copy-Item -LiteralPath $config -Destination $runtimeConfig -Force
$arguments = @('api_v2.py', '-a', '127.0.0.1', '-p', '9880', '-c', "`"$runtimeConfig`"")
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $runtime `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $root 'data/gpt_sovits_server.log') `
    -RedirectStandardError (Join-Path $root 'data/gpt_sovits_server.err.log') -PassThru
$process.Id | Set-Content (Join-Path $root 'data/gpt_sovits_server.pid')
for ($attempt = 0; $attempt -lt 45; $attempt++) {
    if ($process.HasExited) { throw 'GPT-SoVITS exited. See data/gpt_sovits_server.err.log.' }
    if (Test-TtsReady) {
        Write-Output 'Models loaded; warming up reference audio and synthesis (first installation may take about a minute).'
        $reference = (Get-Content -LiteralPath (Join-Path $root 'VOICE_model/GPT_weights_v2ProPlus/tsumugi/references.json') -Encoding UTF8 | ConvertFrom-Json)[0]
        $payload = @{
            text = $reference.text; text_lang = 'all_ja'; prompt_lang = 'all_ja'
            ref_audio_path = $reference.audio; prompt_text = $reference.text
            batch_size = 1; parallel_infer = $true; use_cuda_graph = $false
            streaming_mode = $false; media_type = 'wav'; seed = 42
        } | ConvertTo-Json
        $response = Invoke-WebRequest 'http://127.0.0.1:9880/tts' -Method Post -UseBasicParsing `
            -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($payload)) -TimeoutSec 90
        if ($response.StatusCode -ne 200) { throw 'GPT-SoVITS synthesis warmup failed.' }
        Write-Output "GPT-SoVITS ready (tsumugi v2Pro Plus, PID $($process.Id))."
        return
    }
    Start-Sleep -Seconds 1
}
throw 'GPT-SoVITS startup timed out; see data/gpt_sovits_server.err.log.'
