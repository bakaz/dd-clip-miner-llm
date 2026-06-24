@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo  Song Recut Merge Tool
echo ========================================

set "CONTEXT=%~dp0merge_recut_context.json"
set "WORK_DIR=%~dp0"

call "%~dp0_resolve_env.bat"
if errorlevel 1 exit /b 1

if not exist "%CONTEXT%" (
    echo.
    echo ERROR: merge_recut_context.json not found next to this script.
    echo Please use the merge_mp4.bat copied into a dd-clip-miner-llm output folder.
    echo.
    pause
    exit /b 1
)

if "%~2"=="" (
    echo.
    echo Usage: Drag two exported MP4 or MP3 song clips onto this script.
    echo.
    pause
    exit /b 1
)

if not "%~3"=="" (
    echo.
    echo ERROR: Please drag exactly two files.
    echo.
    pause
    exit /b 1
)

set "EXT1=%~x1"
set "EXT2=%~x2"

if /i not "%EXT1%"==".mp4" if /i not "%EXT1%"==".mp3" (
    echo.
    echo ERROR: Unsupported first file type: %EXT1%
    echo Only .mp4 and .mp3 are supported.
    echo.
    pause
    exit /b 1
)

if /i not "%EXT2%"==".mp4" if /i not "%EXT2%"==".mp3" (
    echo.
    echo ERROR: Unsupported second file type: %EXT2%
    echo Only .mp4 and .mp3 are supported.
    echo.
    pause
    exit /b 1
)

if /i not "%EXT1%"=="%EXT2%" (
    echo.
    echo ERROR: Please drag two files with the same extension.
    echo Use MP4+MP4 to output MP4, or MP3+MP3 to output MP3.
    echo.
    pause
    exit /b 1
)

echo.
echo Input 1: %~1
echo Input 2: %~2
echo Context: %CONTEXT%
echo RunRoot: %RUN_ROOT%
echo Python: %PYTHON_CMD%
echo Project: %PROJECT_ROOT%
echo WorkDir: %WORK_DIR%
echo.

pushd "%WORK_DIR%"
%PYTHON_CMD% -m dd_clip_miner_llm post-merge --context "%CONTEXT%" "%~1" "%~2"
set "RESULT=!ERRORLEVEL!"
popd

if not "!RESULT!"=="0" (
    echo.
    echo ERROR: Recut merge failed.
    echo.
    pause
    exit /b !RESULT!
)

echo.
pause