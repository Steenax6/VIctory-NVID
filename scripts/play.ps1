# ztrx launcher: runs the game for normal play (no scripted presses, no capture).
param(
    [Parameter(Mandatory = $true)][string]$Game,
    [string]$Cache = "$env:LOCALAPPDATA\ztrx\cache",
    [string]$Data = "$env:LOCALAPPDATA\ztrx\data",
    [string]$Exe,
    # Leaves the faster settings (README) as the environment has them.
    [switch]$Plain,
    # Holds A through the opening scene (VICTORY_AUTO_SKIP=1): the title
    # skips a scene while A is held.
    [switch]$SkipIntro
)
# Resolved here, not as a param default: Windows PowerShell 5.1 leaves $PSScriptRoot
# empty in param defaults of scripts that use [Parameter()].
if (-not $Exe) { $Exe = Join-Path $PSScriptRoot "..\target\release\victory.exe" }
if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Error "victory.exe not found at $Exe - build it with: cargo build --release -p victory"
    exit 1
}
if (-not (Test-Path -LiteralPath $Game)) {
    Write-Error "Game not found: $Game"
    exit 1
}
foreach ($v in "VICTORY_KEYS", "VICTORY_CAPTURE", "VICTORY_CAPTURE_FRAMES", "VICTORY_DUMP_TARGETS") {
    Remove-Item "Env:$v" -ErrorAction SilentlyContinue
}
# The faster settings (README, "Faster settings"): on unless the caller set
# them or asked for -Plain.
if (-not $Plain) {
    foreach ($v in "VICTORY_ASYNC_SUBMIT", "VICTORY_QUEUE2", "VICTORY_INDIRECT_WINDOW", "VICTORY_TEXTURE_NO_WAIT", "VICTORY_PLACED_TEX") {
        if (-not (Test-Path "Env:$v")) { Set-Item "Env:$v" "1" }
    }
}
if ($SkipIntro -and -not (Test-Path "Env:VICTORY_AUTO_SKIP")) { $env:VICTORY_AUTO_SKIP = "1" }
New-Item -ItemType Directory -Force $Cache, $Data | Out-Null
$env:VICTORY_CACHE_DIR = $Cache
$env:VICTORY_DATA_DIR = $Data
# "$_" turns stderr lines (ErrorRecords in 5.1) into plain text for the log.
& $Exe run $Game --execute 2>&1 | ForEach-Object { "$_" } | Out-File "$env:TEMP\ztrx.log" -Encoding utf8
Write-Host "Exited with $LASTEXITCODE; log: $env:TEMP\ztrx.log"
