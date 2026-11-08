@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo   TELECARE NLP PROTOTYPE - PROJECT REVIEW 2
echo ======================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    echo Please ensure Python is installed and added to PATH.
    pause
    exit /b 1
)

python -c "import sys; print('Python:', sys.version)"
echo.
echo Starting Telecare Prototype server at http://127.0.0.1:8000 ...
echo.

:: Launch Microsoft Edge directly via Windows protocol
start "" microsoft-edge:http://127.0.0.1:8000

python backend\server.py

if errorlevel 1 (
    echo.
    echo Server exited with an error.
    pause
)
endlocal
