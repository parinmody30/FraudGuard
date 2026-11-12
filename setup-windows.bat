@echo off
REM One-time setup for Windows: installs Python and website packages.
cd /d "%~dp0"
echo === Creating Python environment ===
python -m venv .venv || goto :error
echo === Installing Python packages (a few minutes) ===
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt || goto :error
echo === Installing website packages ===
cd frontend
call npm install || goto :error
cd ..
if not exist .env copy .env.example .env >nul
echo.
echo Setup complete. Double-click start-windows.bat to launch the website.
pause
exit /b 0
:error
echo.
echo Setup FAILED. Copy the error above and ask for help.
pause
exit /b 1
