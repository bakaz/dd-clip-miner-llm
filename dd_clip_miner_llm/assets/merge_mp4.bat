@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo  Song Recut Merge Tool
echo ========================================

set "CONTEXT=%~dp0merge_recut_context.json"
set "PYTHON_EXE={python_exe}"
set "PROJECT_ROOT={project_root}"
set "WORK_DIR=%~dp0"

if not exist "%PROJECT_ROOT%\dd_clip_miner_llm\__init__.py" (
    set "PROJECT_ROOT=%~dp0..\..\..\..\..\..\.."
)

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
echo Python: %PYTHON_EXE%
echo Project: %PROJECT_ROOT%
echo WorkDir: %WORK_DIR%
echo.

set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

if exist "%PYTHON_EXE%" (
    pushd "%WORK_DIR%"
    "%PYTHON_EXE%" -m dd_clip_miner_llm post-merge --context "%CONTEXT%" "%~1" "%~2"
    set "RESULT=!ERRORLEVEL!"
    popd
) else (
    pushd "%WORK_DIR%"
    py -3 -m dd_clip_miner_llm post-merge --context "%CONTEXT%" "%~1" "%~2"
    set "RESULT=!ERRORLEVEL!"
    if "!RESULT!"=="9009" (
        python -m dd_clip_miner_llm post-merge --context "%CONTEXT%" "%~1" "%~2"
        set "RESULT=!ERRORLEVEL!"
    )
    popd
)

if not "!RESULT!"=="0" (
    echo.
    echo ERROR: Recut merge failed.
    echo.
    pause
    exit /b !RESULT!
)

echo.
pause
