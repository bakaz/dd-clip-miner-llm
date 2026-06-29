@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo  Source and SUS Cleanup Tool
echo ========================================

set "CONTEXT=%~dp0merge_recut_context.json"
set "WORK_DIR=%~dp0"

call "%~dp0_resolve_env.bat"
if errorlevel 1 exit /b 1

if not exist "%CONTEXT%" (
    echo.
    echo ERROR: merge_recut_context.json not found next to this script.
    echo Please use cleanup_source.bat in a dd-clip-miner-llm song export folder.
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
%PYTHON_CMD% -m dd_clip_miner_llm cleanup-source --context "%CONTEXT%"
set "RESULT=!ERRORLEVEL!"
popd

if not "!RESULT!"=="0" (
    echo.
    echo ERROR: Cleanup failed or was cancelled.
    echo.
    pause
    exit /b !RESULT!
)

echo.
pause