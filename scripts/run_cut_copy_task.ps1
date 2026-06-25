# run_cut_copy_task.ps1
# Scheduled-task launcher: delegates to Python cut-copy-task (SMB readiness + batch-run).
#
# Path probing runs entirely in Python (scandir / write probe), avoiding PowerShell
# encoding issues with non-ASCII UNC paths (e.g. 綾音Aya).
#
# Exit codes:
#   0 = skipped (paths not ready within wait window) or batch-run succeeded
#   >0 = batch-run failed

param(
    [Parameter(Mandatory = $true)]
    [string]$CutCopyConf,
    [string]$InputRoot = "",
    [string]$ProjectRoot = "",
    [string]$PythonExe = "",
    [int]$NetworkWaitMinutes = 45,
    [int]$NetworkPollSeconds = 30,
    [string]$LogFile = "cut_copy_task.log"
)

$ErrorActionPreference = "Stop"

function Resolve-AgainstBase {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BasePath
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

$cutCopyConfPath = Resolve-AgainstBase -Path $CutCopyConf -BasePath $ProjectRoot
if (-not (Test-Path -LiteralPath $cutCopyConfPath)) {
    throw "cut_copy conf not found: $cutCopyConfPath"
}

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
    } else {
        $found = Get-Command "python" -ErrorAction SilentlyContinue
        if ($found) {
            $pythonPath = $found.Source
        }
    }
}

if (-not $pythonPath) {
    throw "Python not found. Ensure .venv\Scripts\python.exe exists or pass -PythonExe."
}

$pythonArgs = @(
    "-m", "dd_clip_miner_llm", "cut-copy-task",
    "--conf", $cutCopyConfPath,
    "--project-root", $ProjectRoot,
    "--network-wait-minutes", $NetworkWaitMinutes,
    "--network-poll-seconds", $NetworkPollSeconds,
    "--log-file", $LogFile
)

if ($InputRoot) {
    $pythonArgs += @("--input-root", $InputRoot)
}

& $pythonPath @pythonArgs
exit $(if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 })