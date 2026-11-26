@echo off
rem ======================================================================
rem  VoiceShield - double-click this file to start the whole application.
rem  It checks the laptop, starts the server and opens the browser at
rem  http://127.0.0.1:8000 . Keep this window open during the demo.
rem  To stop VoiceShield: close this window or press Ctrl+C in it.
rem ======================================================================
title VoiceShield
cd /d "%~dp0voiceshield"
if not exist "start_demo.bat" (
  echo Could not find the voiceshield folder next to this file.
  pause
  exit /b 1
)
call "%~dp0voiceshield\start_demo.bat"
