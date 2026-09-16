@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Launch.ps1" -Action uninstall
set "result=%errorlevel%"
if not "%result%"=="0" pause
exit /b %result%
