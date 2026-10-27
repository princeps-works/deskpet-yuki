@echo off
set "SCRIPT_DIR=%~dp0"
findstr /R /C:"^TTS_PROVIDER=gpt_sovits$" "%SCRIPT_DIR%.env" >nul 2>&1
if not errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start_gpt_sovits.ps1"
    findstr /R /C:"^TTS_TRANSLATION_PROVIDER=index_local$" "%SCRIPT_DIR%.env" >nul 2>&1
    if not errorlevel 1 powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start_index_translation.ps1"
)
python "%SCRIPT_DIR%main.py" %*
