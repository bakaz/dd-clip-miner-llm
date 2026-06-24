@echo off
setlocal enabledelayedexpansion

if not defined WORK_DIR set "WORK_DIR=%~dp0"

set "RUN_ROOT="
set "SCAN=%WORK_DIR%"
for /L %%i in (1,1,10) do (
    if exist "!SCAN!03_clips\" (
        set "RUN_ROOT=!SCAN!"
        goto found_run
    )
    for %%p in ("!SCAN!..") do set "SCAN=%%~fp\"
)
:found_run
if not defined RUN_ROOT set "RUN_ROOT=%WORK_DIR%"

set "PROJECT_ROOT="
if exist "!RUN_ROOT!_tools\miner\dd_clip_miner_llm\__init__.py" (
    set "PROJECT_ROOT=!RUN_ROOT!_tools\miner"
    goto found_project
)

set "SCAN=%WORK_DIR%"
for /L %%i in (1,1,12) do (
    if exist "!SCAN!dd_clip_miner_llm\__init__.py" (
        set "PROJECT_ROOT=!SCAN!"
        goto found_project
    )
    for %%p in ("!SCAN!..") do set "SCAN=%%~fp\"
)
:found_project

if not defined PROJECT_ROOT (
    echo.
    echo ERROR: Could not find dd-clip-miner-llm code.
    echo Looked for RUN_ROOT\_tools\miner or a parent folder with dd_clip_miner_llm.
    echo.
    exit /b 1
)

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto found_python
)
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto found_python
)

echo.
echo ERROR: Python not found. Install Python 3 and ensure py or python is on PATH.
echo.
exit /b 1

:found_python
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
endlocal & set "RUN_ROOT=%RUN_ROOT%" & set "PROJECT_ROOT=%PROJECT_ROOT%" & set "PYTHON_CMD=%PYTHON_CMD%" & set "PYTHONPATH=%PYTHONPATH%"
exit /b 0