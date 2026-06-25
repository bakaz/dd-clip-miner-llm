# setup_cut_copy_task.ps1
# 创建 Windows 计划任务，用于 dd-clip-miner-llm 录播自动处理
# 需要以管理员权限运行
#
# 任务通过 scripts/run_cut_copy_task.ps1 启动：
#   - 默认在用户登录时触发（SMB 凭据可用）
#   - 登录后等待 SMB/UNC 路径就绪，再执行 batch-run
#
# 计划任务以 cut_copy.conf 为入口（batch 工作流配置）:
#   source.path            -> batch-run 扫描目录
#   processing.config_path -> batch-run --config
#   同一文件               -> batch-run --cut-copy-conf（批后归档）
#
# 注意: 这是 batch-run + 批后处理，不是独立的 cut-copy CLI 命令。
#
# 用法:
#   .\scripts\setup_cut_copy_task.ps1
#   .\scripts\setup_cut_copy_task.ps1 -ConfPath "D:\opencode\dd-clip-miner-llm\config\local\cut_copy.conf"
#   .\scripts\setup_cut_copy_task.ps1 -TriggerProfile logon   # 默认
#   .\scripts\setup_cut_copy_task.ps1 -TriggerProfile wol    # WOL 唤醒、无人登录场景
#   .\scripts\setup_cut_copy_task.ps1 -InputRoot "\\nas\ddtv\recordings"
#   .\scripts\setup_cut_copy_task.ps1 -Mode legacy -ConfPath "C:\path\to\cut_copy.conf"
#   .\scripts\setup_cut_copy_task.ps1 -RepeatMinutes 30 -NetworkWaitMinutes 60

param(
    [string]$ConfPath = "",   # cut_copy.conf（batch 工作流配置）
    [string]$InputRoot = "",  # 覆盖 cut_copy.conf 的 source.path
    [string]$TaskName = "DDClipMiner-CutCopy",
    [int]$RepeatMinutes = 15,
    [string]$PythonExe = "",  # empty = auto-detect
    [string]$ProjectRoot = "",  # empty = auto-detect from script location
    [ValidateSet("batch", "legacy")]
    [string]$Mode = "batch",   # batch=推荐；legacy=独立 cut-copy CLI（非计划任务默认）
    [ValidateSet("logon", "wol", "repeat")]
    [string]$TriggerProfile = "logon",
    [int]$StartupDelayMinutes = 2,
    [int]$NetworkWaitMinutes = 45,
    [int]$NetworkPollSeconds = 30,
    [int]$ExecutionTimeLimitHours = 8,
    [string]$TaskUser = "",     # empty = current user
    [string]$TaskPassword = ""  # empty = prompt; required for unattended WOL/logon tasks
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

function Get-TaskPassword {
    param(
        [string]$UserName,
        [string]$PlainPassword
    )

    if ($PlainPassword) {
        return $PlainPassword
    }

    Write-Host ""
    Write-Host "Scheduled task will run as: $UserName" -ForegroundColor Cyan
    Write-Host "Enter the Windows password for unattended runs (WOL/boot without logon)." -ForegroundColor Yellow
    Write-Host "Press Enter to skip (fine for logon-triggered tasks)." -ForegroundColor Yellow
    $secure = Read-Host "Password" -AsSecureString
    if ($null -eq $secure -or $secure.Length -eq 0) {
        return ""
    }

    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function New-TaskTriggers {
    param(
        [Parameter(Mandatory = $true)][string]$Profile,
        [Parameter(Mandatory = $true)][int]$RepeatMinutes,
        [Parameter(Mandatory = $true)][int]$StartupDelayMinutes,
        [Parameter(Mandatory = $true)][string]$TaskUser
    )

    $triggers = @()

    switch ($Profile) {
        "logon" {
            # Primary: run when the target user logs in (SMB credentials available).
            $triggers += New-ScheduledTaskTrigger -AtLogOn -User $TaskUser
        }
        "wol" {
            # Fallback for wake-on-LAN without interactive logon.
            $bootTrigger = New-ScheduledTaskTrigger -AtStartup
            $bootTrigger.Delay = "PT${StartupDelayMinutes}M"
            $triggers += $bootTrigger
            $triggers += New-ScheduledTaskTrigger -AtLogOn -User $TaskUser
        }
        "repeat" {
            # Periodic polling only (no logon/startup).
        }
    }

    if ($Profile -in @("wol", "repeat")) {
        $repeatTrigger = New-ScheduledTaskTrigger `
            -Once `
            -At ([DateTime]::Today) `
            -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
            -RepetitionDuration (New-TimeSpan -Days 9999)
        $triggers += $repeatTrigger
    }

    return $triggers
}

function Get-CutCopyWorkflow {
    param(
        [Parameter(Mandatory=$true)][string]$PythonPath,
        [Parameter(Mandatory=$true)][string]$CutCopyConfPath,
        [Parameter(Mandatory=$true)][string]$BasePath
    )

    $escapedConf = $CutCopyConfPath.Replace("'", "''")
    $raw = & $PythonPath -c @"
import json
from pathlib import Path
from dd_clip_miner_llm.cut_copy import load_cut_copy_config
cfg = load_cut_copy_config(r'$escapedConf')
base = Path(r'$($BasePath.Replace("'", "''"))')
pipeline = str(cfg.get('processing', {}).get('config_path', '') or '')
if pipeline and not Path(pipeline).is_absolute():
    pipeline = str((base / pipeline).resolve())
print(json.dumps({
    'source_path': str(cfg.get('source', {}).get('path', '') or ''),
    'destination_path': str(cfg.get('destination', {}).get('path', '') or ''),
    'pipeline_config': pipeline,
}, ensure_ascii=False))
"@ 2>&1

    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        $detail = if ($raw) { ($raw | Out-String).Trim() } else { "(no output)" }
        throw "Failed to load cut_copy conf (python exit=$LASTEXITCODE): $detail"
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
Write-Host "  Mode:            $Mode"
Write-Host "  Trigger Profile: $TriggerProfile"
Write-Host "  Network Wait:    $NetworkWaitMinutes min (poll every $NetworkPollSeconds s)"
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

# --- Config file handling (cut_copy.conf drives batch-run) ---
$defaultCutCopyConf = Join-Path $ProjectRoot "config\local\cut_copy.conf"
$cutCopyConfPath = $null
$pipelineConfigPath = $null
$batchInputRoot = $null
$detectedInputRoot = $null

if ($ConfPath) {
    $cutCopyConfPath = Resolve-AgainstBase -Path $ConfPath -BasePath $ProjectRoot
    if ($ConfPath -match 'main\.yaml$') {
        Write-Host "WARNING: -ConfPath should be cut_copy.conf, not main.yaml." -ForegroundColor Yellow
        Write-Host "         Use: config\local\cut_copy.conf" -ForegroundColor Yellow
    }
} else {
    $cutCopyConfPath = $defaultCutCopyConf
}

if (-not (Test-Path -LiteralPath $cutCopyConfPath)) {
    Write-Host "ERROR: cut_copy conf not found: $cutCopyConfPath" -ForegroundColor Red
    exit 1
}
Write-Host "  cut_copy.conf: $cutCopyConfPath" -ForegroundColor Green

try {
    $workflow = Get-CutCopyWorkflow -PythonPath $pythonPath -CutCopyConfPath $cutCopyConfPath -BasePath $ProjectRoot
    $detectedInputRoot = [string]$workflow.source_path
    $pipelineConfigPath = [string]$workflow.pipeline_config
    if ($pipelineConfigPath) {
        Write-Host "  pipeline --config: $pipelineConfigPath" -ForegroundColor Green
    }
} catch {
    Write-Host "ERROR: Could not load cut_copy conf: $_" -ForegroundColor Red
    exit 1
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
if (-not $pipelineConfigPath) {
    Write-Host "ERROR: processing.config_path is empty in cut_copy.conf." -ForegroundColor Red
    exit 1
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

Write-Host ""

# --- Build launcher command ---
$workingDir = $ProjectRoot
$launcherScript = Join-Path $PSScriptRoot "run_cut_copy_task.ps1"
if (-not (Test-Path -LiteralPath $launcherScript)) {
    Write-Host "ERROR: Launcher script not found: $launcherScript" -ForegroundColor Red
    exit 1
}

if ($Mode -eq "legacy") {
    Write-Host "ERROR: Legacy cut-copy CLI mode is not for scheduled tasks." -ForegroundColor Red
    Write-Host "Use default -Mode batch with cut_copy.conf (runs batch-run + post-process)." -ForegroundColor Yellow
    exit 1
}

$launcherArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $launcherScript,
    "-CutCopyConf", $cutCopyConfPath,
    "-ProjectRoot", $ProjectRoot,
    "-PythonExe", $pythonPath,
    "-NetworkWaitMinutes", $NetworkWaitMinutes,
    "-NetworkPollSeconds", $NetworkPollSeconds
)

Write-Host "  Batch Input:  $batchInputRoot" -ForegroundColor Green
if ($InputRoot) {
    $launcherArgs += @("-InputRoot", $batchInputRoot)
}

$taskArgumentString = $launcherArgs -join ' '
Write-Host "  Launcher: powershell.exe $taskArgumentString" -ForegroundColor Gray
Write-Host "  Work Dir: $workingDir" -ForegroundColor Gray
Write-Host ""

# --- Create scheduled task ---

$taskUser = if ($TaskUser) { $TaskUser } else { "$env:USERDOMAIN\$env:USERNAME" }

$taskTriggers = New-TaskTriggers `
    -Profile $TriggerProfile `
    -RepeatMinutes $RepeatMinutes `
    -StartupDelayMinutes $StartupDelayMinutes `
    -TaskUser $taskUser

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionTimeLimitHours) `
    -MultipleInstances IgnoreNew
$settings.RunOnlyIfNetworkAvailable = $true

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $taskArgumentString `
    -WorkingDirectory $workingDir

$needsStoredPassword = ($TriggerProfile -eq "wol")
if ($needsStoredPassword) {
    $taskPassword = Get-TaskPassword -UserName $taskUser -PlainPassword $TaskPassword
} else {
    $taskPassword = ""
}

if ($taskPassword) {
    $principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Password -RunLevel Limited
} else {
    $principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Task '$TaskName' already exists, updating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

if ($taskPassword) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $taskTriggers `
        -Settings $settings `
        -Principal $principal `
        -Description "DDTV recording: wait for SMB, batch-run via cut_copy.conf" `
        -User $taskUser `
        -Password $taskPassword `
        -Force
} else {
    Write-Host "WARNING: No task password supplied; registering task without stored credentials." -ForegroundColor Yellow
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $taskTriggers `
        -Settings $settings `
        -Principal $principal `
        -Description "DDTV recording: wait for SMB, batch-run via cut_copy.conf" `
        -Force
}

Write-Host ""
Write-Host "Scheduled task '$TaskName' created!" -ForegroundColor Green
Write-Host ""

# --- Summary ---
$logPath = Join-Path $ProjectRoot "cut_copy.log"
$taskLogPath = Join-Path $ProjectRoot "cut_copy_task.log"

Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Mode:            $Mode"
Write-Host "  Run As:          $taskUser"
Write-Host "  Trigger Profile: $TriggerProfile"
Write-Host "  Python:          $pythonPath"
Write-Host "  Working Dir:     $workingDir"
Write-Host "  Cut-Copy Conf:   $cutCopyConfPath"
Write-Host "  Pipeline Config: $pipelineConfigPath"
Write-Host "  Batch Input:     $batchInputRoot"
Write-Host "  Launcher Log:    $taskLogPath"
Write-Host "  Cut-Copy Log:    $logPath"
Write-Host "  Startup Delay:   $StartupDelayMinutes minutes (wol profile only; logon profile ignores this)"
Write-Host "  Network Wait:    $NetworkWaitMinutes minutes per run"
Write-Host "  Time Limit:      $ExecutionTimeLimitHours hours"
if ($TriggerProfile -in @("wol", "repeat")) {
    Write-Host "  Repeat Every:    $RepeatMinutes minutes"
}
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Edit config/local/cut_copy.conf (source, destination, processing.config_path, behavior)"
Write-Host "  2. processing.config_path should point to config/local/main.yaml"
Write-Host "  3. Configure DDTV room shell command to send WOL to wake this machine"
Write-Host "  4. Ensure this machine can access DDTV output directory (SMB)"
Write-Host "  5. Ensure this machine can access target SMB share"
Write-Host "  6. Check launcher log after WOL: Get-Content '$taskLogPath' -Tail 50"
Write-Host ""

Write-Host "Management:" -ForegroundColor Cyan
Write-Host "  View task:  Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Run now:    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  Delete:     Unregister-ScheduledTask -TaskName '$TaskName'"
Write-Host "  View logs:  Get-Content '$taskLogPath' -Tail 50"
Write-Host "              Get-Content '$logPath' -Tail 50"