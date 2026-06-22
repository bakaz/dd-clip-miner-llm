@echo off
chcp 65001 >nul
setlocal

echo ========================================
echo  Manual Cut Tool
echo ========================================

set "CONTEXT=%~dp0manual_cut_context.json"
set "PYTHON_EXE={python_exe}"
set "PROJECT_ROOT={project_root}"
set "WORK_DIR=%~dp0"

if not exist "%PROJECT_ROOT%\dd_clip_miner_llm\__init__.py" (
    set "PROJECT_ROOT=%~dp0..\..\..\..\..\..\.."
)

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

set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"

if exist "%PYTHON_EXE%" (
    pushd "%WORK_DIR%"
    "%PYTHON_EXE%" -m dd_clip_miner_llm manual-cut-context --context "%CONTEXT%" --start "%START_TIME%" --end "%END_TIME%" --filename "%FILENAME%"
    set "RESULT=%ERRORLEVEL%"
    popd
) else (
    pushd "%WORK_DIR%"
    py -3 -m dd_clip_miner_llm manual-cut-context --context "%CONTEXT%" --start "%START_TIME%" --end "%END_TIME%" --filename "%FILENAME%"
    set "RESULT=%ERRORLEVEL%"
    if "%RESULT%"=="9009" (
        python -m dd_clip_miner_llm manual-cut-context --context "%CONTEXT%" --start "%START_TIME%" --end "%END_TIME%" --filename "%FILENAME%"
        set "RESULT=%ERRORLEVEL%"
    )
    popd
)

if not "%RESULT%"=="0" (
    echo.
    echo ERROR: Manual cut failed.
    echo.
    pause
    exit /b %RESULT%
)

echo.
echo Done! Check the output in this folder.
pause
