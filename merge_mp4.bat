@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  MP4 Merge Tool
echo ========================================

:: Check ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] ffmpeg not found
    echo.
    set /p INSTALL="Install ffmpeg via winget? (Y/N): "
    if /i "!INSTALL!"=="Y" (
        echo.
        echo Installing ffmpeg...
        winget install Gyan.FFmpeg --accept-package-agreements
        if errorlevel 1 (
            echo.
            echo Installation failed. Please install manually:
            echo   https://www.gyan.dev/ffmpeg/builds/
            echo.
            pause
            exit /b 1
        )
        echo Done! Please re-run this script.
        pause
        exit /b 0
    ) else (
        echo.
        echo Please install ffmpeg first:
        echo   winget install Gyan.FFmpeg
        echo.
        pause
        exit /b 1
    )
)

if "%~2"=="" (
    echo.
    echo Usage: Drag two MP4 files onto this script
    echo.
    pause
    exit /b 1
)

set "FILE1=%~1"
set "FILE2=%~2"
set "OUTPUT=%~dp1%~n1_merged%~x1"

echo.
echo Input 1: %FILE1%
echo Input 2: %FILE2%
echo Output:  %OUTPUT%
echo.

:: Create concat list
set "LIST=%TEMP%\concat_%RANDOM%.txt"
echo file '%FILE1%' > "%LIST%"
echo file '%FILE2%' >> "%LIST%"

:: Try stream copy first (fastest)
echo Trying stream copy...
ffmpeg -y -f concat -safe 0 -i "%LIST%" -c copy "%OUTPUT%" 2>nul
if not errorlevel 1 (
    echo.
    echo Success!
    echo %OUTPUT%
    del "%LIST%" 2>nul
    pause
    exit /b 0
)

:: Fallback to re-encode
echo Stream copy failed, re-encoding...
ffmpeg -y -f concat -safe 0 -i "%LIST%" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 192k "%OUTPUT%" 2>nul
if not errorlevel 1 (
    echo.
    echo Success!
    echo %OUTPUT%
) else (
    echo.
    echo ERROR: Merge failed
    echo Make sure both files are valid MP4
)

del "%LIST%" 2>nul
pause
