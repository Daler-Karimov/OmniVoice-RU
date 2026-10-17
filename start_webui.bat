@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo               OmniVoice WebUI Launcher
echo         (PyTorch FP16 + GGUF BF16 / Q8_0 / Q4_K_M)
echo ========================================================
echo.

if not exist "%~dp0venv\Scripts\python.exe" (
    echo [INFO] Virtual environment 'venv' not found. Creating venv...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Make sure Python 3.10+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo [INFO] Installing PyTorch with CUDA support...
    "%~dp0venv\Scripts\pip.exe" install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
    echo [INFO] Installing OmniVoice dependencies...
    "%~dp0venv\Scripts\pip.exe" install -e .
)

echo Activating virtual environment...
call "%~dp0venv\Scripts\activate.bat"

echo.
echo Starting OmniVoice Web UI on http://127.0.0.1:7860 ...
echo Models available: PyTorch FP16, GGUF BF16, GGUF Q8_0, GGUF Q4_K_M

python webui.py

if errorlevel 1 (
    echo.
    echo [ERROR] OmniVoice stopped with error code %ERRORLEVEL%.
    pause
)
