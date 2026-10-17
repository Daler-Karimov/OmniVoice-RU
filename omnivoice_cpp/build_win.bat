@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0..\venv\Scripts\python.exe" (
    "%~dp0..\venv\Scripts\python.exe" "%~dp0build.py" %*
) else (
    python "%~dp0build.py" %*
)

if errorlevel 1 (
    echo.
    echo [ERROR] Build script failed with exit code %ERRORLEVEL%.
)
