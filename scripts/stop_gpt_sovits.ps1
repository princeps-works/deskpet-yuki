$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pidFile = Join-Path $root 'data/gpt_sovits_server.pid'
if (!(Test-Path -LiteralPath $pidFile)) { return }
$servicePid = [int](Get-Content -LiteralPath $pidFile)
$config = (Resolve-Path (Join-Path $root 'tools/gpt_sovits/GPT_SoVITS/configs/desktop_pet_tsumugi.yaml')).Path
$processes = Get-CimInstance Win32_Process | Where-Object {
    ($_.ProcessId -eq $servicePid -or $_.ParentProcessId -eq $servicePid) -and
    $_.CommandLine -and $_.CommandLine.Contains($config)
}
$processes | ForEach-Object { Stop-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
Write-Output 'Stopped the GPT-SoVITS service launched by this project.'
