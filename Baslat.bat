@echo off
setlocal
cd /d "%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  echo [1/4] Creating local Python environment...
  python -m venv .venv || goto :error
)
"%VENV_PY%" -c "import torch, fastapi, uvicorn, pandas, sklearn" >nul 2>nul
if errorlevel 1 (
  echo [2/4] Installing PyTorch and API dependencies. First install may take a few minutes...
  "%VENV_PY%" -m pip install --upgrade pip || goto :error
  "%VENV_PY%" -m pip install -r requirements.txt || goto :error
)
echo [3/4] Starting the Iris prediction API...
start "Iris Lab API" /D "%~dp0" "%VENV_PY%" -m uvicorn api:app --host 127.0.0.1 --port 8000
echo Waiting for the API to become ready...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ready=$false; for($i=0;$i -lt 90;$i++){try{$r=Invoke-RestMethod -TimeoutSec 2 'http://127.0.0.1:8000/api/health';if($r.status -eq 'ok'){$ready=$true;break}}catch{};Start-Sleep -Seconds 2};if(-not $ready){exit 1}"
if errorlevel 1 (
  echo API did not respond. Check the Iris Lab API window for details.
  pause
  exit /b 1
)
echo [4/4] Opening the dashboard...
start "" "%~dp0index.html"
exit /b 0
:error
echo Startup failed. Check that Python 3.10 or newer is installed and available as 'python'.
pause
exit /b 1
