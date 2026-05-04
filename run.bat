@echo off
setlocal
cd /d "%~dp0"

echo.
echo  ============================================
echo   Deepfake Detection System
echo  ============================================
echo.

REM ── Activate virtual environment ─────────────────────────────────────────────
if not defined VIRTUAL_ENV (
    if exist "venv\Scripts\activate.bat" (
        call "venv\Scripts\activate.bat"
        echo  [OK] Virtual environment activated
    ) else (
        echo  [ERROR] No virtual environment found at venv\
        echo.
        echo         First-time setup:
        echo           python -m venv venv
        echo           venv\Scripts\activate
        echo           pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo  [OK] Virtual environment already active
)

echo.
echo  Starting API...

REM ── Open browser after 3 s (PowerShell delay, runs in background) ────────────
start /b powershell -NoProfile -Command "Start-Sleep 3; Start-Process '%~dp0frontend\index.html'"

echo  API is running on http://localhost:8000
echo  Opening frontend in 3 seconds...
echo.
echo  System ready! Press Ctrl+C to stop.
echo  ============================================
echo.

REM ── Start FastAPI — blocks here, showing live logs ───────────────────────────
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

echo.
echo  Server stopped.
pause
