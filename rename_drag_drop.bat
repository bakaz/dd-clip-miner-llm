@echo off
chcp 65001 >nul
setlocal

set "SCRIPT=%~dp0scripts\rename_drag_drop.py"

if "%~1"=="" (
    echo Drag one or more video files onto this .bat file.
    echo.
    echo Default output format:
    echo   【StreamerName】歌曲名-歌手名-250101.mp4
    pause
    exit /b 1
)

py -3 "%SCRIPT%" %*
if errorlevel 9009 (
    python "%SCRIPT%" %*
)

echo.
pause
