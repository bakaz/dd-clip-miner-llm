# setup_cut_copy_task.ps1
# 创建 Windows 计划任务，用于 dd-clip-miner-llm cut-copy 工作流
# 需要以管理员权限运行
#
# 用法:
#   .\scripts\setup_cut_copy_task.ps1
#   .\scripts\setup_cut_copy_task.ps1 -ConfPath "C:\path\to\cut_copy.conf" -RepeatMinutes 30

param(
    [string]$ConfPath = "cut_copy.conf",
    [string]$TaskName = "DDClipMiner-CutCopy",
    [int]$RepeatMinutes = 15,
    [string]$PythonExe = "python",
    [string]$ProjectRoot = $PSScriptRoot + "\.."
)

$ErrorActionPreference = "Stop"

# 检查管理员权限
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "错误: 需要管理员权限运行此脚本。" -ForegroundColor Red
    Write-Host "请右键 PowerShell -> 以管理员身份运行" -ForegroundColor Yellow
    exit 1
}

# 解析配置文件绝对路径
$confFullPath = Resolve-Path -LiteralPath $ConfPath -ErrorAction SilentlyContinue
if (-not $confFullPath) {
    Write-Host "警告: 配置文件 '$ConfPath' 不存在，将使用相对路径。" -ForegroundColor Yellow
    $confFullPath = $ConfPath
}

# 检查 Python
$pythonPath = (Get-Command $PythonExe -ErrorAction SilentlyContinue).Source
if (-not $pythonPath) {
    Write-Host "错误: 找不到 Python ($PythonExe)。请确保 Python 在 PATH 中。" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " DDClipMiner Cut-Copy 计划任务设置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  任务名称: $TaskName"
Write-Host "  Python:   $pythonPath"
Write-Host "  配置文件: $confFullPath"
Write-Host "  重复间隔: 每 $RepeatMinutes 分钟"
Write-Host "  项目目录: $ProjectRoot"
Write-Host ""

# 创建操作
$action = New-ScheduledTaskAction `
    -Execute $pythonPath `
    -Argument "-m dd_clip_miner_llm cut-copy --conf `"$confFullPath`"" `
    -WorkingDirectory $ProjectRoot

# 触发器 1: 系统启动时（延迟 2 分钟，等网络就绪）
$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerBoot.Delay = "PT2M"

# 触发器 2: 每 N 分钟重复（兜底，防止错过启动触发器）
$triggerRepeat = New-ScheduledTaskTrigger `
    -Once `
    -At ([DateTime]::Today) `
    -RepetitionInterval (New-TimeSpan -Minutes $RepeatMinutes) `
    -RepetitionDuration (New-TimeSpan -Days 9999)

# 任务设置
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -MultipleInstances IgnoreNew

# 检查是否已存在
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "任务 '$TaskName' 已存在，将更新..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# 注册任务
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger @($triggerBoot, $triggerRepeat) `
    -Settings $settings `
    -RunLevel Highest `
    -Description "DDTV 录播自动处理: 扫描 -> 切片 -> 复制到 SMB -> 关机" `
    -Force

Write-Host ""
Write-Host "计划任务 '$TaskName' 已创建！" -ForegroundColor Green
Write-Host ""
Write-Host "后续步骤:" -ForegroundColor Yellow
Write-Host "  1. 编辑配置文件: $confFullPath"
Write-Host "  2. 配置 DDTV 房间 Shell 命令发送 WOL 唤醒本机"
Write-Host "  3. 确保本机可以访问 DDTV 输出目录 (SMB)"
Write-Host "  4. 确保本机可以访问目标 SMB 共享"
Write-Host ""
Write-Host "管理:" -ForegroundColor Cyan
Write-Host "  查看任务: Get-ScheduledTask -TaskName '$TaskName'"
Write-Host "  手动触发: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  删除任务: Unregister-ScheduledTask -TaskName '$TaskName'"
Write-Host "  查看日志: Get-Content cut_copy.log -Tail 50"
