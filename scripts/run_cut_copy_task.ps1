# run_cut_copy_task.ps1
# Scheduled-task launcher: wait for SMB/UNC paths, then run batch-run.
#
# Driven by cut_copy.conf (batch workflow conf):
#   source.path              -> batch-run input root
#   processing.config_path   -> batch-run --config
#   (same file)              -> batch-run --cut-copy-conf
#
# This is batch-run post-processing, NOT the standalone "cut-copy" CLI command.
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

function Write-TaskLog {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [string]$Level = "INFO"
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] [$Level] $Message"
    Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Test-RemotePathReady {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Label = "path"
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $true
    }

    try {
        if (Test-Path -LiteralPath $Path -PathType Container) {
            return $true
        }
    } catch {
        Write-TaskLog "Probe failed for ${Label}: $Path ($($_.Exception.Message))" "WARN"
        return $false
    }

    return $false
}

function Get-TaskPathsFromCutCopyConf {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$CutCopyConfPath,
        [Parameter(Mandatory = $true)][string]$BasePath,
        [string]$InputRootOverride = ""
    )

    $escapedConf = $CutCopyConfPath.Replace("'", "''")
    $raw = & $PythonPath -c @"
import json
from dd_clip_miner_llm.cut_copy import load_cut_copy_config
cfg = load_cut_copy_config(r'$escapedConf')
print(json.dumps({
    'source_path': str(cfg.get('source', {}).get('path', '') or ''),
    'destination_path': str(cfg.get('destination', {}).get('path', '') or ''),
    'pipeline_config': str(cfg.get('processing', {}).get('config_path', '') or ''),
}, ensure_ascii=False))
"@ 2>&1

    if ($LASTEXITCODE -ne 0 -or -not $raw) {
        $detail = if ($raw) { ($raw | Out-String).Trim() } else { "(no output)" }
        throw "Failed to load cut_copy conf (python exit=$LASTEXITCODE): $detail"
    }

    if ($raw -is [System.Array]) {
        $raw = ($raw | Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1)
    }

    $payload = $raw | ConvertFrom-Json

    $sourcePath = [string]$InputRootOverride
    if (-not $sourcePath) {
        $sourcePath = [string]$payload.source_path
    }
    if (-not $sourcePath) {
        throw "Batch input root is empty. Set source.path in cut_copy conf or pass -InputRoot."
    }

    $pipelineConfig = [string]$payload.pipeline_config
    if (-not $pipelineConfig) {
        throw "processing.config_path is empty in cut_copy conf."
    }
    $pipelineConfig = (Resolve-AgainstBase -Path $pipelineConfig -BasePath $BasePath).ToString()

    return [pscustomobject]@{
        SourcePath = $sourcePath
        DestinationPath = [string]$payload.destination_path
        PipelineConfig = $pipelineConfig
        CutCopyConf = $CutCopyConfPath
    }
}

function Wait-ForNetworkPaths {
    param(
        [Parameter(Mandatory = $true)][string[]]$Paths,
        [Parameter(Mandatory = $true)][int]$WaitMinutes,
        [Parameter(Mandatory = $true)][int]$PollSeconds
    )

    $deadline = (Get-Date).AddMinutes($WaitMinutes)
    $attempt = 0

    while ((Get-Date) -lt $deadline) {
        $attempt++
        $ready = $true
        $status = @()

        foreach ($entry in $Paths) {
            if ([string]::IsNullOrWhiteSpace($entry)) {
                continue
            }

            $parts = $entry -split "\|", 2
            $label = $parts[0]
            $path = $parts[1]
            $ok = Test-RemotePathReady -Path $path -Label $label
            $status += "${label}=$ok"
            if (-not $ok) {
                $ready = $false
            }
        }

        if ($ready) {
            Write-TaskLog "Network paths ready on attempt ${attempt}: $($status -join ', ')"
            return $true
        }

        $remaining = [math]::Max(0, [int]($deadline - (Get-Date)).TotalMinutes)
        Write-TaskLog "Waiting for network paths (attempt ${attempt}, ~${remaining} min left): $($status -join ', ')"
        Start-Sleep -Seconds $PollSeconds
    }

    return $false
}

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$script:LogPath = Resolve-AgainstBase -Path $LogFile -BasePath $ProjectRoot

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

Write-TaskLog "Launcher started (user=$env:USERDOMAIN\$env:USERNAME, pid=$PID)"
Write-TaskLog "CutCopyConf=$cutCopyConfPath Python=$pythonPath ProjectRoot=$ProjectRoot"

$taskPaths = Get-TaskPathsFromCutCopyConf `
    -PythonPath $pythonPath `
    -CutCopyConfPath $cutCopyConfPath `
    -BasePath $ProjectRoot `
    -InputRootOverride $InputRoot

$watchList = @("source|$($taskPaths.SourcePath)")
if ($taskPaths.DestinationPath) {
    $watchList += "destination|$($taskPaths.DestinationPath)"
}

Write-TaskLog "Pipeline config: $($taskPaths.PipelineConfig)"
Write-TaskLog "Watching paths: $($watchList -join '; ')"

if (-not (Wait-ForNetworkPaths -Paths $watchList -WaitMinutes $NetworkWaitMinutes -PollSeconds $NetworkPollSeconds)) {
    Write-TaskLog "Network paths not ready within ${NetworkWaitMinutes} minutes; skipping this run." "WARN"
    exit 0
}

$pythonArgs = @(
    "-m", "dd_clip_miner_llm", "batch-run", $taskPaths.SourcePath,
    "--result-root", "runs/batch",
    "--work-root", "runs/batch",
    "--config", $taskPaths.PipelineConfig,
    "--cut-copy-conf", $taskPaths.CutCopyConf
)

Write-TaskLog ("Starting batch-run: {0} {1}" -f $pythonPath, ($pythonArgs -join ' '))
$startedAt = Get-Date

& $pythonPath @pythonArgs
$exitCode = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } else { 0 }
$elapsed = [int]((Get-Date) - $startedAt).TotalSeconds

if ($exitCode -eq 0) {
    Write-TaskLog "batch-run finished successfully in ${elapsed}s"
} else {
    Write-TaskLog "batch-run failed with exit code $exitCode after ${elapsed}s" "ERROR"
}

exit $exitCode