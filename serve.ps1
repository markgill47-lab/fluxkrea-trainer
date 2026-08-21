<#
.SYNOPSIS
    Start the FluxKrea daemon on this machine.

.DESCRIPTION
    Runs the daemon in this window, so its lifetime is yours: it does not
    belong to an editor, an agent session, or anything else that might go
    away. Training runs take hours, and a daemon that dies mid-run leaves
    an interrupted job and a trainer process holding VRAM.

    Output stays in this window and is also appended to logs/daemon.log,
    because "it crashed" is not a diagnosis and a window that closed took
    the traceback with it.

    Stop it with Ctrl+C. The window stays open on a crash so you can read
    what happened.

.PARAMETER Port
    Override the configured port. Rarely needed - the port lives in
    config.toml, and this is for running a second daemon temporarily.

.PARAMETER NoLog
    Do not write logs/daemon.log. Console only.

.EXAMPLE
    .\serve.ps1
    Start on the configured port, logging to logs/daemon.log.

.EXAMPLE
    .\serve.ps1 -Port 8480
    Start a second daemon on another port, for a test.
#>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [switch]$NoLog
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Keep the window open on a failure - but only when a person is looking at
# it. Double-clicked, that stops the traceback vanishing; called from a
# script or a pipeline, a Read-Host would hang forever waiting for an Enter
# nobody is there to press.
function Wait-IfWatched {
    if (-not [Console]::IsInputRedirected) { Read-Host "Press Enter to close" | Out-Null }
}

# --- the interpreter -------------------------------------------------------
# The project venv, not whatever python happens to be on PATH: this package
# is installed in editable mode there, and a system python will either not
# find it or find a different copy.
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host "No virtual environment at .venv" -ForegroundColor Red
    Write-Host ""
    Write-Host "Create one and install the package:" -ForegroundColor Yellow
    Write-Host "    uv venv"
    Write-Host "    uv pip install -e `".[dev,daemon]`""
    Write-Host ""
    Wait-IfWatched
    exit 1
}

# --- is one already running? -----------------------------------------------
# Two daemons on one port is a confusing failure: the second exits with an
# address-in-use error that reads like a bug in the app.
$configuredPort = if ($Port -gt 0) { $Port } else {
    try {
        [int](& $python -c "from fluxkrea.core.config import load; print(load().daemon.port)")
    } catch { 8471 }
}

$listening = Get-NetTCPConnection -LocalPort $configuredPort -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    $owner = (Get-Process -Id $listening[0].OwningProcess -ErrorAction SilentlyContinue)
    Write-Host "Something is already listening on port $configuredPort" -ForegroundColor Yellow
    if ($owner) { Write-Host "  $($owner.ProcessName) (pid $($owner.Id))" }
    Write-Host ""
    Write-Host "If that is a FluxKrea daemon, it is already serving:" -ForegroundColor Yellow
    Write-Host "    http://localhost:$configuredPort"
    Write-Host "Otherwise stop it, or run this with -Port <other>."
    Write-Host ""
    Wait-IfWatched
    exit 1
}

# --- go --------------------------------------------------------------------
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host ""
Write-Host "FluxKrea daemon" -ForegroundColor Cyan
Write-Host "  project   $root"
Write-Host "  url       http://localhost:$configuredPort"
Write-Host "  started   $stamp"

$arguments = @('-m', 'fluxkrea.cli', 'serve')
if ($Port -gt 0) { $arguments += @('--port', "$Port") }

$writer = $null
if (-not $NoLog) {
    $logDir = Join-Path $root 'logs'
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
    $log = Join-Path $logDir 'daemon.log'
    Write-Host "  log       $log"

    # A StreamWriter rather than Tee-Object: Tee-Object has no -Encoding in
    # PowerShell 5.1 and writes UTF-16, which mixed with a UTF-8 header
    # produced a file that reads as spaced-out gibberish. AutoFlush so a
    # crash three hours in still leaves the last lines on disk, and no BOM
    # so anything can read it.
    $writer = [System.IO.StreamWriter]::new($log, $true, [System.Text.UTF8Encoding]::new($false))
    $writer.AutoFlush = $true
    $writer.WriteLine("")
    $writer.WriteLine("=== daemon started $stamp ===")
}
Write-Host "  stop      Ctrl+C"
Write-Host ""

# From here on a line on stderr is output, not a failure. PowerShell 5.1
# wraps a native command's stderr in ErrorRecord objects, and with
# ErrorActionPreference 'Stop' the first uvicorn startup line - which goes
# to stderr - terminated this script before the daemon finished booting.
$ErrorActionPreference = 'Continue'

try {
    if ($NoLog) {
        & $python @arguments
    } else {
        # 2>&1 because uvicorn and every traceback go to stderr - capturing
        # stdout alone got the banner and none of the reasons anything
        # stopped. "$_" flattens each item to a string: PowerShell 5.1 wraps
        # a native command's stderr in ErrorRecord objects.
        & $python @arguments 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            $writer.WriteLine($line)
        }
    }
    $code = $LASTEXITCODE
} finally {
    if ($writer) { $writer.Dispose() }
}

Write-Host ""
if ($code -eq 0) {
    Write-Host "Daemon stopped." -ForegroundColor Cyan
} else {
    Write-Host "Daemon exited with code $code." -ForegroundColor Red
    if (-not $NoLog) { Write-Host "The output above is also in $log" }
    Write-Host ""
    Wait-IfWatched
}
exit $code
