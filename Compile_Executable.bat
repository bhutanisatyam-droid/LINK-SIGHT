@echo off
title LinkSight - Executable Compiler [SIH 2026]
cd /d "%~dp0"
cls
echo ==============================================================================
echo   SMART INDIA HACKATHON 2026  --  PROBLEM STATEMENT ID: 26169
echo   ISRO / DEPARTMENT OF SPACE (DOS)
echo   COMPILER: LinkSight_FSOC_ATP_Terminal.exe Standalone Builder
echo ==============================================================================
echo.
echo [*] Checking Python environment...

set PYTHON_CMD=python
py -3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.11
)

echo [*] Using: %PYTHON_CMD%
echo [*] Ensuring PyInstaller is available...
%PYTHON_CMD% -m pip install pyinstaller --quiet

echo.
echo [*] Compiling Standalone Executable (LinkSight_FSOC_ATP_Terminal.exe)...
echo.

%PYTHON_CMD% -m PyInstaller --noconfirm --onedir --windowed --name="LinkSight_FSOC_ATP_Terminal" --add-data="models;models" --add-data="assets;assets" --hidden-import="scipy.spatial.transform._rotation_groups" --hidden-import="onnxruntime" --hidden-import="filterpy" --hidden-import="cv2" --hidden-import="PySide6" main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==============================================================================
    echo [ERROR] Build failed! Check the log messages above.
    echo ==============================================================================
    pause
    exit /b %ERRORLEVEL%
)

if not exist "dist\LinkSight_FSOC_ATP_Terminal\models" (
    xcopy /E /I /Y "models" "dist\LinkSight_FSOC_ATP_Terminal\models" >nul
)
if not exist "dist\LinkSight_FSOC_ATP_Terminal\assets" (
    xcopy /E /I /Y "assets" "dist\LinkSight_FSOC_ATP_Terminal\assets" >nul
)

echo.
echo ==============================================================================
echo  BUILD SUCCESSFUL!
echo  Executable Location: dist\LinkSight_FSOC_ATP_Terminal\LinkSight_FSOC_ATP_Terminal.exe
echo ==============================================================================
echo.
pause
