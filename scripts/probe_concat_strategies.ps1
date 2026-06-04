param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDir,

    [string]$OutputRoot = "runs\concat_strategy_probe"
)

$ErrorActionPreference = "Stop"

function Get-MediaDuration {
    param([Parameter(Mandatory = $true)][string]$Path)

    $raw = & ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 $Path
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return [double]::Parse(($raw | Select-Object -First 1), [Globalization.CultureInfo]::InvariantCulture)
}

function Get-VideoSignature {
    param([Parameter(Mandatory = $true)][string]$Path)

    $raw = & ffprobe -v error `
        -select_streams v:0 `
        -show_entries stream=codec_name,codec_tag_string,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,start_time,duration `
        -of json `
        $Path
    if ($LASTEXITCODE -ne 0) {
        return [pscustomobject]@{ file = $Path; error = "ffprobe failed" }
    }

    $json = $raw | ConvertFrom-Json
    $stream = $json.streams | Select-Object -First 1
    return [pscustomobject]@{
        file = $Path
        codec = $stream.codec_name
        tag = $stream.codec_tag_string
        profile = $stream.profile
        width = $stream.width
        height = $stream.height
        pix_fmt = $stream.pix_fmt
        r_frame_rate = $stream.r_frame_rate
        avg_frame_rate = $stream.avg_frame_rate
        time_base = $stream.time_base
        start_time = $stream.start_time
        duration = $stream.duration
    }
}

function Write-ConcatList {
    param(
        [Parameter(Mandatory = $true)][string[]]$Files,
        [Parameter(Mandatory = $true)][string]$Path
    )

    $lines = foreach ($file in $Files) {
        "file '$($file.Replace("'", "'\''"))'"
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($Path, [string[]]$lines, $encoding)
}

function Invoke-LoggedFFmpeg {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [string]$OutputFile
    )

    $log = Join-Path $RunDir "$Name.log"
    if ($OutputFile) {
        Remove-Item -LiteralPath $OutputFile -Force -ErrorAction SilentlyContinue
    }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & ffmpeg @Args 2>&1 | Out-File -LiteralPath $log -Encoding UTF8
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        $sw.Stop()
    }

    $duration = $null
    $bytes = $null
    if ($OutputFile -and (Test-Path -LiteralPath $OutputFile)) {
        $duration = Get-MediaDuration -Path $OutputFile
        $bytes = (Get-Item -LiteralPath $OutputFile).Length
    }

    [pscustomobject]@{
        name = $Name
        exit_code = $exitCode
        elapsed_sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        duration_sec = if ($duration -ne $null) { [math]::Round($duration, 3) } else { $null }
        bytes = $bytes
        log = $log
        output = $OutputFile
    }
}

function Invoke-CopyHealthCheck {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$InputFile
    )

    $log = Join-Path $RunDir "$Name.health.log"
    $args = @(
        "-hide_banner",
        "-v", "warning",
        "-i", $InputFile,
        "-map", "0:v:0",
        "-c", "copy",
        "-bsf:v", "h264_mp4toannexb",
        "-f", "null",
        "NUL"
    )

    $sw = [Diagnostics.Stopwatch]::StartNew()
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & ffmpeg @args 2>&1 | Out-File -LiteralPath $log -Encoding UTF8
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        $sw.Stop()
    }

    $text = Get-Content -LiteralPath $log -Raw -ErrorAction SilentlyContinue
    [pscustomobject]@{
        name = $Name
        exit_code = $exitCode
        elapsed_sec = [math]::Round($sw.Elapsed.TotalSeconds, 3)
        suspicious = [bool]($text -match "Invalid NAL|bitstream filters|Invalid data|missing picture")
        log = $log
    }
}

$resolvedSource = (Resolve-Path -LiteralPath $SourceDir).Path
$files = Get-ChildItem -LiteralPath $resolvedSource -Filter "*_fix.mp4" -File |
    Sort-Object Name |
    Select-Object -ExpandProperty FullName

if ($files.Count -lt 2) {
    throw "Need at least two *_fix.mp4 files in $resolvedSource"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path (Resolve-Path ".").Path (Join-Path $OutputRoot $stamp)
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$durations = foreach ($file in $files) {
    [pscustomobject]@{
        file = $file
        duration_sec = [math]::Round((Get-MediaDuration -Path $file), 3)
        bytes = (Get-Item -LiteralPath $file).Length
    }
}
$expectedDuration = ($durations | Measure-Object -Property duration_sec -Sum).Sum
$durations | Export-Csv -LiteralPath (Join-Path $RunDir "inputs.csv") -NoTypeInformation -Encoding UTF8
$signatures = foreach ($file in $files) {
    Get-VideoSignature -Path $file
}
$signatures | Export-Csv -LiteralPath (Join-Path $RunDir "video_signatures.csv") -NoTypeInformation -Encoding UTF8

$health = for ($i = 0; $i -lt $files.Count; $i++) {
    Invoke-CopyHealthCheck -Name ("part{0:D2}" -f $i) -InputFile $files[$i]
}
$health | Export-Csv -LiteralPath (Join-Path $RunDir "copy_health.csv") -NoTypeInformation -Encoding UTF8

$results = New-Object System.Collections.Generic.List[object]

$listAll = Join-Path $RunDir "concat_all.txt"
Write-ConcatList -Files $files -Path $listAll

$directOut = Join-Path $RunDir "direct_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "direct_copy" -OutputFile $directOut -Args @(
    "-hide_banner", "-y",
    "-f", "concat", "-safe", "0", "-i", $listAll,
    "-c", "copy",
    "-movflags", "+faststart",
    $directOut
)))

$discardOut = Join-Path $RunDir "direct_discardcorrupt_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "direct_discardcorrupt_copy" -OutputFile $discardOut -Args @(
    "-hide_banner", "-y",
    "-fflags", "+discardcorrupt",
    "-err_detect", "ignore_err",
    "-f", "concat", "-safe", "0", "-i", $listAll,
    "-c", "copy",
    "-movflags", "+faststart",
    $discardOut
)))

for ($i = 0; $i -lt ($files.Count - 1); $i++) {
    $pair = @($files[$i], $files[$i + 1])
    $pairList = Join-Path $RunDir ("concat_pair_{0:D2}_{1:D2}.txt" -f $i, ($i + 1))
    Write-ConcatList -Files $pair -Path $pairList
    $pairOut = Join-Path $RunDir ("concat_pair_{0:D2}_{1:D2}.mp4" -f $i, ($i + 1))
    $results.Add((Invoke-LoggedFFmpeg -Name ("concat_pair_{0:D2}_{1:D2}" -f $i, ($i + 1)) -OutputFile $pairOut -Args @(
        "-hide_banner", "-y",
        "-f", "concat", "-safe", "0", "-i", $pairList,
        "-c", "copy",
        "-movflags", "+faststart",
        $pairOut
    )))
}

$allRemuxDir = Join-Path $RunDir "all_remuxed_parts"
New-Item -ItemType Directory -Force -Path $allRemuxDir | Out-Null
$allRemuxFiles = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $files.Count; $i++) {
    $remux = Join-Path $allRemuxDir ("part{0:D2}.mp4" -f $i)
    $results.Add((Invoke-LoggedFFmpeg -Name ("remux_part{0:D2}_copy" -f $i) -OutputFile $remux -Args @(
        "-hide_banner", "-y",
        "-fflags", "+genpts+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", $files[$i],
        "-map", "0",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        $remux
    )))
    $allRemuxFiles.Add($remux)
}

$allRemuxList = Join-Path $RunDir "concat_all_remuxed_copy.txt"
Write-ConcatList -Files $allRemuxFiles.ToArray() -Path $allRemuxList
$allRemuxOut = Join-Path $RunDir "concat_all_remuxed_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "concat_all_remuxed_copy" -OutputFile $allRemuxOut -Args @(
    "-hide_banner", "-y",
    "-f", "concat", "-safe", "0", "-i", $allRemuxList,
    "-c", "copy",
    "-movflags", "+faststart",
    $allRemuxOut
)))

$fixedDir = Join-Path $RunDir "fixed_parts"
New-Item -ItemType Directory -Force -Path $fixedDir | Out-Null

$fixed2Copy = Join-Path $fixedDir "part01_copy_discardcorrupt.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "fix_part01_copy_discardcorrupt" -OutputFile $fixed2Copy -Args @(
    "-hide_banner", "-y",
    "-fflags", "+discardcorrupt",
    "-err_detect", "ignore_err",
    "-i", $files[1],
    "-map", "0",
    "-c", "copy",
    "-movflags", "+faststart",
    $fixed2Copy
)))

$filesFixed2Copy = [string[]]$files.Clone()
$filesFixed2Copy[1] = $fixed2Copy
$listFixed2Copy = Join-Path $RunDir "concat_fixed_part01_copy.txt"
Write-ConcatList -Files $filesFixed2Copy -Path $listFixed2Copy
$fixed2CopyOut = Join-Path $RunDir "concat_fixed_part01_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "concat_fixed_part01_copy" -OutputFile $fixed2CopyOut -Args @(
    "-hide_banner", "-y",
    "-f", "concat", "-safe", "0", "-i", $listFixed2Copy,
    "-c", "copy",
    "-movflags", "+faststart",
    $fixed2CopyOut
)))

$fixed2Reencode = Join-Path $fixedDir "part01_reencode.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "fix_part01_reencode" -OutputFile $fixed2Reencode -Args @(
    "-hide_banner", "-y",
    "-fflags", "+discardcorrupt",
    "-err_detect", "ignore_err",
    "-i", $files[1],
    "-map", "0:v:0",
    "-map", "0:a:0?",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "192k",
    "-movflags", "+faststart",
    $fixed2Reencode
)))

$filesFixed2Reencode = [string[]]$files.Clone()
$filesFixed2Reencode[1] = $fixed2Reencode
$listFixed2Reencode = Join-Path $RunDir "concat_fixed_part01_reencode.txt"
Write-ConcatList -Files $filesFixed2Reencode -Path $listFixed2Reencode
$fixed2ReencodeOut = Join-Path $RunDir "concat_fixed_part01_reencode.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "concat_fixed_part01_reencode" -OutputFile $fixed2ReencodeOut -Args @(
    "-hide_banner", "-y",
    "-f", "concat", "-safe", "0", "-i", $listFixed2Reencode,
    "-c", "copy",
    "-movflags", "+faststart",
    $fixed2ReencodeOut
)))

$beforeBad = Join-Path $fixedDir "part01_before_bad_copy.mp4"
$afterBad = Join-Path $fixedDir "part01_after_bad_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "split_part01_before_100s_copy" -OutputFile $beforeBad -Args @(
    "-hide_banner", "-y",
    "-i", $files[1],
    "-t", "100",
    "-map", "0",
    "-c", "copy",
    "-movflags", "+faststart",
    $beforeBad
)))
$results.Add((Invoke-LoggedFFmpeg -Name "split_part01_after_100s_copy" -OutputFile $afterBad -Args @(
    "-hide_banner", "-y",
    "-ss", "100",
    "-i", $files[1],
    "-map", "0",
    "-c", "copy",
    "-movflags", "+faststart",
    $afterBad
)))

$filesSplitPart01 = @($files[0], $beforeBad, $afterBad) + $files[2..($files.Count - 1)]
$listSplitPart01 = Join-Path $RunDir "concat_split_part01_copy.txt"
Write-ConcatList -Files $filesSplitPart01 -Path $listSplitPart01
$splitPart01Out = Join-Path $RunDir "concat_split_part01_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "concat_split_part01_copy" -OutputFile $splitPart01Out -Args @(
    "-hide_banner", "-y",
    "-f", "concat", "-safe", "0", "-i", $listSplitPart01,
    "-c", "copy",
    "-movflags", "+faststart",
    $splitPart01Out
)))

$tsDir = Join-Path $RunDir "ts_parts"
New-Item -ItemType Directory -Force -Path $tsDir | Out-Null
$tsFiles = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $files.Count; $i++) {
    $ts = Join-Path $tsDir ("part{0:D2}.ts" -f $i)
    $results.Add((Invoke-LoggedFFmpeg -Name ("make_ts_part{0:D2}" -f $i) -OutputFile $ts -Args @(
        "-hide_banner", "-y",
        "-fflags", "+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", $files[$i],
        "-map", "0",
        "-c", "copy",
        "-bsf:v", "h264_mp4toannexb",
        "-f", "mpegts",
        $ts
    )))
    $tsFiles.Add($ts)
}

$concatProtocolInput = "concat:" + (($tsFiles.ToArray()) -join "|")
$tsOut = Join-Path $RunDir "concat_ts_intermediate_copy.mp4"
$results.Add((Invoke-LoggedFFmpeg -Name "concat_ts_intermediate_copy" -OutputFile $tsOut -Args @(
    "-hide_banner", "-y",
    "-i", $concatProtocolInput,
    "-c", "copy",
    "-bsf:a", "aac_adtstoasc",
    "-movflags", "+faststart",
    $tsOut
)))

$results | ForEach-Object {
    $delta = $null
    if ($_.duration_sec -ne $null) {
        $delta = [math]::Round($expectedDuration - $_.duration_sec, 3)
    }
    $_ | Add-Member -NotePropertyName expected_sec -NotePropertyValue ([math]::Round($expectedDuration, 3)) -Force
    $_ | Add-Member -NotePropertyName missing_sec -NotePropertyValue $delta -Force
    $_
} | Export-Csv -LiteralPath (Join-Path $RunDir "results.csv") -NoTypeInformation -Encoding UTF8

Write-Host "Probe finished: $RunDir"
Write-Host "Expected duration: $([math]::Round($expectedDuration, 3)) sec"
Write-Host "Results:"
Import-Csv -LiteralPath (Join-Path $RunDir "results.csv") |
    Select-Object name, exit_code, elapsed_sec, duration_sec, missing_sec, bytes |
    Format-Table -AutoSize
Write-Host "Health:"
$health | Select-Object name, exit_code, suspicious, elapsed_sec | Format-Table -AutoSize
