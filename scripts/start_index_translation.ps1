$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$runtime = Join-Path $root 'tools/llama_cpp/b11429-cuda12.4/llama-server.exe'
$model = Join-Path $root 'assets/models/index_translate_2b/Index-Translate-2B.Q5_K_M.gguf'
if (!(Test-Path -LiteralPath $runtime) -or !(Test-Path -LiteralPath $model)) {
    throw 'Index runtime or Q5_K_M model is missing.'
}
try {
    $existing = Invoke-RestMethod 'http://127.0.0.1:8768/v1/models' -TimeoutSec 2
} catch { $existing = $null }
if ($existing) {
    if ($existing.data.id -contains 'Index-Translate-2B-Q5_K_M') {
        Write-Output 'Index translation service is already running on 127.0.0.1:8768.'
        return
    }
    throw 'Port 8768 belongs to a different model; refusing to replace it.'
}
if (Get-NetTCPConnection -LocalPort 8768 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8768 is occupied; refusing to start another service.'
}
$log = Join-Path $root 'data/index_translation_server.log'
$errorLog = Join-Path $root 'data/index_translation_server.err.log'
$arguments = @('-m', "`"$model`"", '--host', '127.0.0.1', '--port', '8768',
    '--alias', 'Index-Translate-2B-Q5_K_M', '-c', '2048', '-np', '1',
    '-ngl', '99', '-b', '256', '-ub', '128', '-t', '4', '--flash-attn', 'on',
    '--jinja', '--reasoning', 'off')
$process = Start-Process -FilePath $runtime -ArgumentList $arguments -WorkingDirectory $root `
    -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errorLog -PassThru
for ($attempt = 0; $attempt -lt 45; $attempt++) {
    if ($process.HasExited) { throw "Index exited. See $errorLog" }
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8768/health' -TimeoutSec 1
        if ($health.status -eq 'ok') {
            Write-Output "Index ready on 127.0.0.1:8768 (PID $($process.Id))."
            return
        }
    } catch { }
    Start-Sleep -Seconds 1
}
throw "Index is still loading (PID $($process.Id)). See $errorLog"
