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

if "%~1"=="" (
    echo.
    echo Usage: Drag two or more exported MP4 or MP3 song clips onto this script.
    echo.
    pause
    exit /b 1
)

set "FILE_COUNT=0"
set "EXPECTED_EXT="
set "MERGE_ARGS="

:collect_files
if "%~1"=="" goto files_collected
set /a FILE_COUNT+=1
set "CURRENT_EXT=%~x1"
if /i not "%CURRENT_EXT%"==".mp4" if /i not "%CURRENT_EXT%"==".mp3" (
    echo.
    echo ERROR: Unsupported file type: %CURRENT_EXT%
    echo Only .mp4 and .mp3 are supported.
    echo.
    pause
    exit /b 1
)
if not defined EXPECTED_EXT (
    set "EXPECTED_EXT=%CURRENT_EXT%"
) else if /i not "%CURRENT_EXT%"=="%EXPECTED_EXT%" (
    echo.
    echo ERROR: Please drag files with the same extension.
    echo Use MP4+MP4 to output MP4, or MP3+MP3 to output MP3.
    echo.
    pause
    exit /b 1
)
set "MERGE_ARGS=!MERGE_ARGS! "%~1""
echo Input !FILE_COUNT!: %~1
shift
goto collect_files

:files_collected
if !FILE_COUNT! LSS 2 (
    echo.
    echo ERROR: Please drag at least two files.
    echo.
    pause
    exit /b 1
)

echo.
echo Context: %CONTEXT%
echo RunRoot: %RUN_ROOT%
echo Python: %PYTHON_CMD%
echo Project: %PROJECT_ROOT%
echo WorkDir: %WORK_DIR%
echo.

pushd "%WORK_DIR%"
%PYTHON_CMD% -m dd_clip_miner_llm post-merge --context "%CONTEXT%" !MERGE_ARGS!
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