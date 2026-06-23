# setup_cut_copy_task.ps1
# 创建 Windows 计划任务，用于 dd-clip-miner-llm cut-copy 工作流
# 需要以管理员权限运行
#
# 用法:
#   .\scripts\setup_cut_copy_task.ps1
#   .\scripts\setup_cut_copy_task.ps1 -Mode legacy -ConfPath "C:\path\to\cut_copy.conf"
#   .\scripts\setup_cut_copy_task.ps1 -RepeatMinutes 30 -ExecutionTimeLimitHours 12

param(
    [string]$ConfPath = "",  # empty = auto-detect from config.yaml
    [string]$TaskName = "DDClipMiner-CutCopy",
    [int]$RepeatMinutes = 15,
    [string]$PythonExe = "",  # empty = auto-detect
    [string]$ProjectRoot = "",  # empty = auto-detect from script location
    [ValidateSet("batch", "legacy")]
    [string]$Mode = "batch",
    [int]$StartupDelayMinutes = 5,
    [int]$ExecutionTimeLimitHours = 8
)

$ErrorActionPreference = "Stop"

# --- Admin check ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Administrator privileges required." -ForegroundColor Red
    Write-Host "Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    exit 1
}

# --- Auto-detect ProjectRoot ---
if (-not $ProjectRoot) {
    # Script is in scripts/ subfolder; project root is parent
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Host "ERROR: Project directory not found: $ProjectRoot" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " DDClipMiner Cut-Copy Task Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Task Name:    $TaskName"
Write-Host "  Project Root: $ProjectRoot"
Write-Host "  Mode:         $Mode"
Write-Host ""

# --- Auto-detect Python ---
$pythonPath = $null

if ($PythonExe) {
    # User specified a Python path
    if (Test-Path -LiteralPath $PythonExe) {
        $pythonPath = [System.IO.Path]::GetFullPath($PythonExe)
    } else {
        # Try to find in PATH
        $found = Get-Command $PythonExe -ErrorAction SilentlyContinue
        if ($found) {
            $pythonPath = $found.Source
        }
    }
} else {
    # Auto-detect: prefer .venv\Scripts\python.exe
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $pythonPath = [System.IO.Path]::GetFullPath($venvPython)
        Write-Host "  Python (venv): $pythonPath" -ForegroundColor Green
    } else {
        # Fallback to system Python
        $found = Get-Command "python" -ErrorAction SilentlyContinue
        if ($found) {
            $pythonPath = $found.Source
            Write-Host "  Python (system): $pythonPath" -ForegroundColor Yellow
        }
    }
}

if (-not $pythonPath) {
    Write-Host "ERROR: Python not found. Ensure .venv\Scripts\python.exe exists or python is in PATH." -ForegroundColor Red
    exit 1
}

# --- Check dd_clip_miner_llm is importable ---
Write-Host "  Checking dd_clip_miner_llm module..." -ForegroundColor Gray
$moduleCheck = & $pythonPath -c "import dd_clip_miner_llm; print('OK')" 2>&1
if ($LASTEXITCODE -ne 0 -or ($moduleCheck -is [string] -and $moduleCheck.Trim() -ne "OK")) {
    Write-Host "ERROR: Cannot import dd_clip_miner_llm module." -ForegroundColor Red
    Write-Host "Run: pip install -e ." -ForegroundColor Yellow
    Write-Host "Module check output: $moduleCheck" -ForegroundColor Gray
    exit 1
}
Write-Host "  dd_clip_miner_llm module: OK" -ForegroundColor Green

# --- Config file handling ---
$configYamlPath = Join-Path $ProjectRoot "config.yaml"
$confFullPath = $null
$sourcePath = $null

if ($Mode -eq "batch") {
    # batch mode: needs config.yaml
    if (-not (Test-Path -LiteralPath $configYamlPath)) {
        Write-Host "WARNING: config.yaml not found in project root." -ForegroundColor Yellow
        Write-Host "  batch-run will use defaults; cut-copy post-processing may need --cut-copy-conf." -ForegroundColor Yellow
    } else {
        Write-Host "  config.yaml: $configYamlPath" -ForegroundColor Green
    }

    # If ConfPath specified, resolve to absolute
    if ($ConfPath) {
        $confFullPath = [System.IO.Path]::GetFullPath($ConfPath, $ProjectRoot)
        if (-not (Test-Path -LiteralPath $confFullPath)) {
            Write-Host "WARNING: Specified config not found: $confFullPath" -ForegroundColor Yellow
            Write-Host "  batch-run will auto-detect cut_copy.conf_path from config.yaml." -ForegroundColor Yellow
            $confFullPath = $null
        } else {
            Write-Host "  cut_copy conf: $confFullPath" -ForegroundColor Green
        }
    }
} else {
    # legacy mode: needs cut_copy.conf
    if (-not $ConfPath) {
        $ConfPath = Join-Path $ProjectRoot "cut_copy.conf"
    }
    $confFullPath = [System.IO.Path]::GetFullPath($ConfPath, $ProjectRoot)

    if (-not (Test-Path -LiteralPath $confFullPath)) {
        Write-Host "ERROR: cut_copy config not found: $confFullPath" -ForegroundColor Red
        Write-Host "Create the config file, or use -Mode batch to auto-detect from config.yaml." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  cut_copy conf: $confFullPath" -ForegroundColor Green
}

# --- SMB connectivity check ---
# If cut_copy.conf exists, try to extract source.path and check accessibility
if ($confFullPath -and (Test-Path -LiteralPath $confFullPath)) {
    try {
        # Use Python to parse YAML (no external PS module needed)
        $sourcePathRaw = & $pythonPath -c @"
import yaml, sys
with open(r'$($confFullPath -replace "'", "''")', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
p = cfg.get('source', {}).get('path', '')
if p: print(p)
"@ 2>&1
        if ($LASTEXITCODE -eq 0 -and $sourcePathRaw -and $sourcePathRaw.Trim()) {
            $sourcePath = $sourcePathRaw.Trim()
            # Resolve relative paths against project root
            if (-not [System.IO.Path]::IsPathRooted($sourcePath)) {
                $sourcePath = [System.IO.Path]::GetFullPath($sourcePath, $ProjectRoot)
            }

            Write-Host ""
            Write-Host "  Checking SMB source path: $sourcePath" -ForegroundColor Gray

            if (Test-Path -LiteralPath $sourcePath -PathType Container) {
                Write-Host "  SMB source path: accessible" -ForegroundColor Green
            } else {
                Write-Host "  WARNING: SMB source path not accessible: $sourcePath" -ForegroundColor Yellow
                Write-Host "  Ensure network share is mapped. Task runs under SYSTEM account." -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "  WARNING: Could not parse cut_copy config: $_" -ForegroundColor Yellow
    }
}

Write-Host ""

# --- Build command arguments ---
$workingDir = $ProjectRoot

if ($Mode -eq "batch") {
    # batch mode: python -m dd_clip_miner_llm batch-run . --config config.yaml [--cut-copy-conf conf]
    $cmdArgs = '-m dd_clip_miner_llm batch-run . --result-root runs/batch'
    if (Test-Path -LiteralPath $configYamlPath) {
        $cmdArgs += " --config `"$configYamlPath`""
    }
    if ($confFullPath) {
        $cmdArgs += " --cut-copy-conf `"$confFullPath`""
    }
} else {
    # legacy mode: python -m dd_clip_miner_llm cut-copy --conf conf_path
    $cmdArgs = "-m dd_clip_miner_llm cut-copy --conf `"$confFullPath`""
}

Write-Host "  Command: $pythonPath $cmdArgs" -ForegroundColor Gray
Write-Host "  Work Dir: $workingDir" -ForegroundColor Gray
Write-Host ""

# --- Create scheduled task ---

# Trigger 1: at system boot (delayed N minutes for network readiness)
$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerBoot.Delay = "PT${StartupDelayMinutes}M"

# Trigger 2: every N minutes (catch-all in case boot trigger is missed)
$triggerRepeat = New-ScheduledTaskTrigger `
    -Once `
    -At ([DateTime]::Today) `
    -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 9999)

# Task settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionTimeLimitHours) `
    -MultipleInstances IgnoreNew

# Action
$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument $cmdArgs `
    -WorkingDirectory $workingDir

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Task '$TaskName' already exists, updating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Register task
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($triggerBoot, $triggerRepeat) `
    -Settings $settings `
    -RunLevel Highest `
    -Description "DDTV recording auto-process: scan -> clip -> copy to SMB -> shutdown" `
    -Force

Write-Host ""
Write-Host "Scheduled task '$TaskName' created!" -ForegroundColor Green
Write-Host ""

# --- Summary ---
$logPath = Join-Path $ProjectRoot "cut_copy.log"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Mode:           $Mode"
Write-Host "  Python:         $pythonPath"
Write-Host "  Working Dir:    $workingDir"
if ($confFullPath) {
    Write-Host "  Config File:    $confFullPath"
}
Write-Host "  Log File:       $logPath"
Write-Host "  Startup Delay:  $StartupDelayMinutes minutes"
Write-Host "  Time Limit:     $ExecutionTimeLimitHours hours"
Write-Host "  Repeat Every:   $RepeatMinutes minutes"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
if ($Mode -eq "batch") {
    Write-Host "  1. Ensure config.yaml is configured (with cut_copy section)"
    Write-Host "  2. Ensure cut_copy.conf exists and is configured"
} else {
    Write-Host "  1. Edit config file: $confFullPath"
}
Write-Host "  3. Configure DDTV room shell command to send WOL to wake this machine"
Write-Host "  4. Ensure this machine can access DDTV output directory (SMB)"
Write-Host "  5. Ensure this machine can access target SMB share"
Write-Host ""

Write-Host "Management:" -ForegroundColor Cyan
Write-Host "  View task:  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Run now:    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Delete:     Unregister-ScheduledTask -TaskName '$TaskName'"
Write-Host "  View logs:  Get-Content '$logPath' -Tail 50"
