@echo off
setlocal
title Build Neural Architecture Designer v1.0 EXE
cd /d "%~dp0"

echo ============================================================
echo   Build Neural Architecture Designer v1.0
echo   Output: NeuralArchitectureDesigner.exe in THIS folder
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

"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
"%VPY%" -m pip install -r requirements-build.txt
if errorlevel 1 goto :fail

echo.
echo Checking Matplotlib export backends before build...
"%VPY%" -c "import matplotlib.backends.backend_agg; import matplotlib.backends.backend_svg; import matplotlib.backends.backend_pdf; import matplotlib.backends.backend_ps; import matplotlib.backends.backend_tkagg; import PIL.JpegImagePlugin; import PIL.PngImagePlugin; print('Export backend preflight: OK')"
if errorlevel 1 goto :fail

echo.
echo Building single-file Windows executable...
if exist ".pyinstaller_build" rmdir /s /q ".pyinstaller_build"
if exist "NeuralArchitectureDesigner.exe" del /q "NeuralArchitectureDesigner.exe"

"%VPY%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "NeuralArchitectureDesigner" ^
  --distpath "%CD%" ^
  --workpath "%CD%\.pyinstaller_build\work" ^
  --specpath "%CD%\.pyinstaller_build" ^
  --add-data "%CD%\assets;assets" ^
  --hidden-import matplotlib.backends.backend_agg ^
  --hidden-import matplotlib.backends.backend_svg ^
  --hidden-import matplotlib.backends.backend_pdf ^
  --hidden-import matplotlib.backends.backend_ps ^
  --hidden-import matplotlib.backends.backend_tkagg ^
  --hidden-import matplotlib.backends._backend_tk ^
  --hidden-import PIL.JpegImagePlugin ^
  --hidden-import PIL.PngImagePlugin ^
  neural_architecture_designer.py
if errorlevel 1 goto :fail

if exist ".pyinstaller_build" rmdir /s /q ".pyinstaller_build"

echo.
echo ============================================================
echo Build complete.
echo The EXE is here:
echo   %CD%\NeuralArchitectureDesigner.exe
echo.
echo Export backends included: PNG/JPG/SVG/PDF/EPS
echo ============================================================
pause
exit /b 0

:fail
echo.
echo Build failed. See the messages above.
pause
exit /b 1
