$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = Join-Path $root 'tools/gpt_sovits'
$python = Join-Path $runtime '.venv/Scripts/python.exe'
$config = Join-Path $root 'config/gpt_sovits_tsumugi.yaml'
if (!(Test-Path -LiteralPath $python)) { throw 'GPT-SoVITS environment is missing.' }
& $python (Join-Path $root 'scripts/prepare_gpt_sovits.py')
if ($LASTEXITCODE -ne 0) { throw 'GPT-SoVITS cancellation patch could not be prepared.' }
function Test-TtsReady {
    try {
        $schema = Invoke-RestMethod 'http://127.0.0.1:9880/openapi.json' -TimeoutSec 2
        return $null -ne $schema.paths.'/tts' -and $null -ne $schema.paths.'/cancel_tts'
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
        Write-Output "GPT-SoVITS API ready (tsumugi v2Pro Plus, PID $($process.Id)); the desktop client warms CUDA Graph before playback starts."
        return
    }
    Start-Sleep -Seconds 1
}
throw 'GPT-SoVITS startup timed out; see data/gpt_sovits_server.err.log.'
