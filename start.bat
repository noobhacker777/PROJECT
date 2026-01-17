@echo off
REM ============================================================================
REM HOLO SKU Detection API - Windows Installer
REM Run this batch file to setup the environment and start the server
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo   HOLO SKU Detection API - Windows Setup
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

python --version
echo.

REM Run setup script
echo Step 1: Installing dependencies...
echo.
python setup.py install

if errorlevel 1 (
    echo.
    echo Error: Setup failed
    pause
    exit /b 1
)

echo.
echo Step 2: Setup complete! Starting Flask server...
echo.
echo ============================================================================
echo   API Server Starting
echo ============================================================================
echo.
echo Open in browser: http://localhost:5002
echo API endpoint:    http://localhost:5002/scan?image=IMG_1445.jpeg
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python run_app.py

pause
