@echo off
REM Launch FraudGuard on Windows: backend (port 8000) + website (port 5173).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup-windows.bat first.
  pause
  exit /b 1
)
start "FraudGuard backend - keep open" cmd /k ".venv\Scripts\python -m uvicorn backend.app:app --port 8000"
start "FraudGuard website - keep open" cmd /k "cd frontend && npm run dev"
echo Starting... the browser will open in a few seconds.
timeout /t 10 /nobreak >nul
start http://localhost:5173
