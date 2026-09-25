@echo off
title ISRO FSOC ATP Virtual Tracking Console
cd /d "%~dp0"
cls
echo ====================================================================
echo  ISRO / DOS FSOC ATP COARSE TRACKING TERMINAL - RESEARCH PROTOTYPE
echo ====================================================================
echo [*] Launching LinkSight FSOC Virtual Tracking System...
echo.
py -3.11 main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FALLBACK] Trying system python...
    python main.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%.
    pause
)
