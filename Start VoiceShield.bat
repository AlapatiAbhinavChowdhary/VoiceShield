@echo off
setlocal
cd /d "%~dp0"
if not exist "voiceshield\.venv\Scripts\python.exe" (
    echo VoiceShield's Python environment is missing.
    echo Follow the setup instructions in voiceshield\README.md first.
    pause
    exit /b 1
)
"voiceshield\.venv\Scripts\python.exe" "voiceshield\launch.py"
if errorlevel 1 pause
