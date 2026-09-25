@echo off
title Push LinkSight to GitHub [SIH 2026]
cd /d "%~dp0"
cls
echo ==============================================================================
echo   LINK-SIGHT // SMART INDIA HACKATHON 2026
echo   Pushing to: https://github.com/bhutanisatyam-droid/LINK-SIGHT.git
echo ==============================================================================
echo.
echo [*] Checking git status...
git status
echo.
echo [*] Pushing commit to origin main...
echo.

git push -u origin main

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ==============================================================================
    echo [NOTE] If GitHub asks for authentication, please sign in via browser/token.
    echo ==============================================================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ==============================================================================
echo   SUCCESSFULLY PUSHED TO GITHUB!
echo   Repository: https://github.com/bhutanisatyam-droid/LINK-SIGHT
echo ==============================================================================
echo.
pause
