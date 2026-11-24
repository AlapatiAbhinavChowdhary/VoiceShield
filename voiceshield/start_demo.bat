@echo off
rem VoiceShield demo: double-click to check the laptop, start the server and open the browser.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment .venv is missing. See README.md, "Set up".
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\preflight.py
if errorlevel 1 (
  echo.
  echo Fix the FAIL lines above, then run start_demo.bat again.
  pause
  exit /b 1
)
echo.
".venv\Scripts\python.exe" launch.py
pause
