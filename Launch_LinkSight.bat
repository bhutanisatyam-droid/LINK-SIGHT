@echo off
title LinkSight // ISRO FSOC ATP Coarse Tracking Terminal [SIH 2026]
cd /d "%~dp0"
cls
echo ==============================================================================
echo   SMART INDIA HACKATHON 2026  --  PROBLEM STATEMENT ID: 26169
echo   ISRO / DEPARTMENT OF SPACE (DOS)
echo   PROJECT: LinkSight - AI-Assisted FSOC ATP Virtual Tracking Terminal
echo ==============================================================================
echo.
echo [*] Initializing LinkSight Virtual Tracking Engine...
echo.

py -3.11 main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [*] Attempting fallback to system python...
    python main.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start application. Please check Python installation and dependencies:
    echo        pip install -r requirements.txt
    pause
)
