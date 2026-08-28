@echo off
setlocal
title Neural Architecture Designer v1.0
cd /d "%~dp0"

echo ============================================================
echo   Neural Architecture Designer v1.0
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo Python was not found.
    echo Install Python 3.10+ and enable "Add Python to PATH".
    pause
    exit /b 1
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating local virtual environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)
set "VPY=.venv\Scripts\python.exe"

echo Checking required packages...
"%VPY%" -c "import matplotlib, PIL" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)

echo Starting Neural Architecture Designer...
"%VPY%" neural_architecture_designer.py
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo The application stopped with an error.
pause
exit /b 1
