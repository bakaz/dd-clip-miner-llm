# setup_cut_copy_task.ps1
# 创建 Windows 计划任务，用于 dd-clip-miner-llm 录播自动处理
# 需要以管理员权限运行
#
# batch 模式（默认）:
#   batch-run --config config.yaml
#   cut_copy 后处理由 config.yaml 的 cut_copy.enabled 自动触发，无需 --cut-copy-conf
#
# batch 模式扫描目录: config.yaml -> cut_copy.conf_path -> cut_copy.conf -> source.path
# -InputRoot 可选手动覆盖
#
# 用法:
#   .\scripts\setup_cut_copy_task.ps1
#   .\scripts\setup_cut_copy_task.ps1 -ConfPath "D:\opencode\dd-clip-miner-llm\config.yaml"
#   .\scripts\setup_cut_copy_task.ps1 -InputRoot "\\nas\ddtv\recordings"
#   .\scripts\setup_cut_copy_task.ps1 -Mode legacy -ConfPath "C:\path\to\cut_copy.conf"
#   .\scripts\setup_cut_copy_task.ps1 -RepeatMinutes 30 -ExecutionTimeLimitHours 12

param(
    [string]$ConfPath = "",   # batch: config.yaml; legacy: cut_copy.conf
    [string]$InputRoot = "",  # batch mode: 覆盖 cut_copy.conf 的 source.path
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

function Resolve-AgainstBase {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$BasePath
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Get-BatchConfigChain {
    param(
        [Parameter(Mandatory=$true)][string]$PythonPath,
        [Parameter(Mandatory=$true)][string]$ConfigYamlPath
    )

    $resolverScript = Join-Path $PSScriptRoot "resolve_batch_config.py"
    if (-not (Test-Path -LiteralPath $resolverScript)) {
        throw "Resolver script not found: $resolverScript"
    }

    $raw = & $PythonPath $resolverScript $ConfigYamlPath 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        $detail = if ($raw) { ($raw | Out-String).Trim() } else { "(no output)" }
        throw "Failed to resolve config.yaml -> cut_copy.conf (python exit=$LASTEXITCODE): $detail"
    }

    if ($raw -is [System.Array]) {
        $raw = ($raw | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1)
    }
    return ($raw | ConvertFrom-Json)
}

# --- Admin check ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "ERROR: Administrator privileges required." -ForegroundColor Red
    Write-Host "Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    exit 1
}

# --- Auto-detect ProjectRoot ---
if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Host "ERROR: Project directory not found: $ProjectRoot" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " DDClipMiner Scheduled Task Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Task Name:    $TaskName"
Write-Host "  Project Root: $ProjectRoot"
Write-Host "  Mode:         $Mode"
Write-Host ""

# --- Auto-detect Python ---
$pythonPath = $null

if ($PythonExe) {
    if (Test-Path -LiteralPath $PythonExe) {
        $pythonPath = [System.IO.Path]::GetFullPath($PythonExe)
    } else {
        $found = Get-Command $PythonExe -ErrorAction SilentlyContinue
        if ($found) {
            $pythonPath = $found.Source
        }
    }
} else {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        $pythonPath = [System.IO.Path]::GetFullPath($venvPython)
        Write-Host "  Python (venv): $pythonPath" -ForegroundColor Green
    } else {
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
$defaultConfigYaml = Join-Path $ProjectRoot "config.yaml"
$defaultCutCopyConf = Join-Path $ProjectRoot "cut_copy.conf"
$configYamlPath = $null
$cutCopyConfPath = $null
$batchInputRoot = $null
$detectedInputRoot = $null

if ($Mode -eq "batch") {
    if ($ConfPath) {
        $configYamlPath = Resolve-AgainstBase -Path $ConfPath -BasePath $ProjectRoot
    } else {
        $configYamlPath = $defaultConfigYaml
    }

    if (-not (Test-Path -LiteralPath $configYamlPath)) {
        Write-Host "ERROR: config.yaml not found: $configYamlPath" -ForegroundColor Red
        exit 1
    }
    Write-Host "  config.yaml: $configYamlPath" -ForegroundColor Green

    # config.yaml -> cut_copy.conf_path -> cut_copy.conf -> source.path
    try {
        $batchConfig = Get-BatchConfigChain -PythonPath $pythonPath -ConfigYamlPath $configYamlPath
        if ($batchConfig.cut_copy_conf) {
            $cutCopyConfPath = [string]$batchConfig.cut_copy_conf
        }
        if ($batchConfig.source_path) {
            $detectedInputRoot = [string]$batchConfig.source_path
        }
        if ($batchConfig.enabled) {
            Write-Host "  cut_copy post-process: enabled (auto via config.yaml)" -ForegroundColor Green
        } else {
            Write-Host "  cut_copy post-process: disabled in config.yaml" -ForegroundColor Yellow
        }
        if ($cutCopyConfPath) {
            if (Test-Path -LiteralPath $cutCopyConfPath) {
                Write-Host "  cut_copy.conf: $cutCopyConfPath" -ForegroundColor Green
            } else {
                Write-Host "  WARNING: cut_copy.conf not found: $cutCopyConfPath" -ForegroundColor Yellow
            }
        }
        if ($batchConfig.error) {
            Write-Host "  WARNING: $($batchConfig.error)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  WARNING: Could not resolve config.yaml -> cut_copy.conf: $_" -ForegroundColor Yellow
    }

    $batchInputRoot = $InputRoot
    if (-not $batchInputRoot -and $detectedInputRoot) {
        $batchInputRoot = $detectedInputRoot
    }
    if (-not $batchInputRoot) {
        Write-Host "ERROR: Batch input root not found." -ForegroundColor Red
        Write-Host "Set cut_copy.conf source.path, or pass -InputRoot to override." -ForegroundColor Yellow
        exit 1
    }
    if (-not [System.IO.Path]::IsPathRooted($batchInputRoot)) {
        $batchInputRoot = Resolve-AgainstBase -Path $batchInputRoot -BasePath $ProjectRoot
    }

    Write-Host ""
    Write-Host "  Batch input (source.path): $batchInputRoot" -ForegroundColor Green
    Write-Host "  Checking batch input path..." -ForegroundColor Gray
    if (Test-Path -LiteralPath $batchInputRoot -PathType Container) {
        Write-Host "  Batch input path: accessible" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Batch input path not accessible: $batchInputRoot" -ForegroundColor Yellow
        Write-Host "  Ensure network share is reachable when the task runs." -ForegroundColor Yellow
    }
} else {
    if (-not $ConfPath) {
        $ConfPath = $defaultCutCopyConf
    }
    $cutCopyConfPath = Resolve-AgainstBase -Path $ConfPath -BasePath $ProjectRoot

    if (-not (Test-Path -LiteralPath $cutCopyConfPath)) {
        Write-Host "ERROR: cut_copy config not found: $cutCopyConfPath" -ForegroundColor Red
        exit 1
    }
    Write-Host "  cut_copy conf: $cutCopyConfPath" -ForegroundColor Green

    try {
        $sourcePathRaw = & $pythonPath -c @"
import yaml
with open(r'$($cutCopyConfPath -replace "'", "''")', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}
p = cfg.get('source', {}).get('path', '')
if p: print(p)
"@ 2>&1
        if ($LASTEXITCODE -eq 0 -and $sourcePathRaw -and $sourcePathRaw.Trim()) {
            $batchInputRoot = $sourcePathRaw.Trim()
            if (-not [System.IO.Path]::IsPathRooted($batchInputRoot)) {
                $batchInputRoot = Resolve-AgainstBase -Path $batchInputRoot -BasePath $ProjectRoot
            }

            Write-Host ""
            Write-Host "  Checking SMB source path: $batchInputRoot" -ForegroundColor Gray
            if (Test-Path -LiteralPath $batchInputRoot -PathType Container) {
                Write-Host "  SMB source path: accessible" -ForegroundColor Green
            } else {
                Write-Host "  WARNING: SMB source path not accessible: $batchInputRoot" -ForegroundColor Yellow
            }
        }
    } catch {
        Write-Host "  WARNING: Could not parse cut_copy config: $_" -ForegroundColor Yellow
    }
}

Write-Host ""

# --- Build command arguments ---
$workingDir = $ProjectRoot
$taskArgumentString = [string]::Empty

if ($Mode -eq "batch") {
    # cut_copy 后处理由 cli.py 根据 config.yaml 的 cut_copy 段自动触发
    $batchInputRoot = [string]$batchInputRoot
    $configYamlPath = [string]$configYamlPath
    $taskArgumentString = '-m dd_clip_miner_llm batch-run "{0}" --result-root runs/batch --work-root runs/batch --config "{1}"' -f `
        $batchInputRoot, $configYamlPath
    Write-Host "  Batch Input:  $batchInputRoot" -ForegroundColor Green
} elseif ($Mode -eq "legacy") {
    if (-not $cutCopyConfPath) {
        Write-Host "ERROR: cut_copy config path is empty for legacy mode." -ForegroundColor Red
        exit 1
    }
    $taskArgumentString = '-m dd_clip_miner_llm cut-copy --conf "{0}"' -f ([string]$cutCopyConfPath)
} else {
    Write-Host "ERROR: Unknown mode '$Mode'. Use batch or legacy." -ForegroundColor Red
    exit 1
}

if ([string]::IsNullOrWhiteSpace($taskArgumentString)) {
    Write-Host "ERROR: Failed to build scheduled task command (taskArgumentString is empty)." -ForegroundColor Red
    Write-Host "  Mode=$Mode InputRoot=$InputRoot batchInputRoot=$batchInputRoot configYamlPath=$configYamlPath" -ForegroundColor Gray
    exit 1
}

Write-Host "  Command: $pythonPath $taskArgumentString" -ForegroundColor Gray
Write-Host "  Work Dir: $workingDir" -ForegroundColor Gray
Write-Host ""

# --- Create scheduled task ---

$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerBoot.Delay = "PT${StartupDelayMinutes}M"

$triggerRepeat = New-ScheduledTaskTrigger `
    -Once `
    -At ([DateTime]::Today) `
    -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 9999)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionTimeLimitHours) `
    -MultipleInstances IgnoreNew

$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument $taskArgumentString `
    -WorkingDirectory $workingDir

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Task '$TaskName' already exists, updating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($triggerBoot, $triggerRepeat) `
    -Settings $settings `
    -RunLevel Highest `
    -Description "DDTV recording auto-process: batch-run -> cut_copy post-process" `
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
if ($Mode -eq "batch") {
    Write-Host "  Config File:    $configYamlPath"
    Write-Host "  Batch Input:    $batchInputRoot"
    if ($cutCopyConfPath) {
        Write-Host "  Cut-Copy Conf:  $cutCopyConfPath (auto-triggered)"
    }
} else {
    Write-Host "  Cut-Copy Conf:  $cutCopyConfPath"
}
Write-Host "  Log File:       $logPath"
Write-Host "  Startup Delay:  $StartupDelayMinutes minutes"
Write-Host "  Time Limit:     $ExecutionTimeLimitHours hours"
Write-Host "  Repeat Every:   $RepeatMinutes minutes"
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
if ($Mode -eq "batch") {
    Write-Host "  1. Ensure config.yaml has cut_copy.enabled: true and conf_path set"
    Write-Host "  2. Ensure cut_copy.conf has source.path, destination, and behavior configured"
} else {
    Write-Host "  1. Edit cut_copy config: $cutCopyConfPath"
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