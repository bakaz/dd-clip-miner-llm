@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo  Manual Cut Tool
echo ========================================

set "CONTEXT=%~dp0manual_cut_context.json"
set "WORK_DIR=%~dp0"

call "%~dp0_resolve_env.bat"
if errorlevel 1 exit /b 1

if not exist "%CONTEXT%" (
    echo.
    echo ERROR: manual_cut_context.json not found next to this script.
    echo Please use this script in a dd-clip-miner-llm output folder.
    echo.
    pause
    exit /b 1
)

echo.
echo Source video will be read from context.
echo.

set /p "START_TIME=Enter start time (e.g. 10:30 or 630): "
if "%START_TIME%"=="" (
    echo ERROR: Start time is required.
    pause
    exit /b 1
)

set /p "END_TIME=Enter end time (e.g. 15:45 or 945): "
if "%END_TIME%"=="" (
    echo ERROR: End time is required.
    pause
    exit /b 1
)

set /p "FILENAME=Enter output filename (without extension, or press Enter for auto): "

echo.
echo Start: %START_TIME%
echo End: %END_TIME%
echo Filename: %FILENAME%
echo Context: %CONTEXT%
echo.

pushd "%WORK_DIR%"
%PYTHON_CMD% -m dd_clip_miner_llm manual-cut-context --context "%CONTEXT%" --start "%START_TIME%" --end "%END_TIME%" --filename "%FILENAME%"
set "RESULT=!ERRORLEVEL!"
popd

if not "!RESULT!"=="0" (
    echo.
    echo ERROR: Manual cut failed.
    echo.
    pause
    exit /b !RESULT!
)

echo.
echo Done! Check the output in this folder.
pause