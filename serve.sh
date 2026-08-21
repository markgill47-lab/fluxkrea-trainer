#!/usr/bin/env bash
# Start the FluxKrea daemon on this machine.
#
# The Linux twin of serve.ps1, for a fleet node driven over SSH rather than
# sat in front of. For a node that should come back after a reboot, prefer
# the systemd unit in deploy/ - this is for a terminal you are watching.
#
# Output goes to the terminal and to logs/daemon.log. Stop with Ctrl+C.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

# The project venv, not whatever python is on PATH: the package is
# installed there in editable mode, and a system python finds a different
# copy or none at all.
python="$root/.venv/bin/python"
[ -x "$python" ] || python="$root/.venv/Scripts/python.exe"   # a venv made on Windows
if [ ! -x "$python" ]; then
    echo "No virtual environment at .venv" >&2
    echo >&2
    echo "Create one and install the package:" >&2
    echo "    uv venv" >&2
    echo "    uv pip install -e \".[dev,daemon]\"" >&2
    exit 1
fi

port="${1:-}"
if [ -z "$port" ]; then
    port="$("$python" -c 'from fluxkrea.core.config import load; print(load().daemon.port)' 2>/dev/null || echo 8471)"
fi

# Two daemons on one port fail with an address-in-use error that reads like
# a bug in the app rather than one already being up.
if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo "Something is already listening on port $port." >&2
    echo "If that is a FluxKrea daemon it is already serving: http://localhost:$port" >&2
    exit 1
fi

mkdir -p logs
log="$root/logs/daemon.log"

echo
# One directory for config, data and state when .fluxkrea/ is present -
# see the note in serve.ps1. Linux has no package redirection, but the two
# scripts stay interchangeable (rule 4).
if [ -z "${FLUXKREA_HOME:-}" ] && [ -d "$root/.fluxkrea" ]; then
    export FLUXKREA_HOME="$root/.fluxkrea"
fi

echo "FluxKrea daemon"
echo "  project   $root"
[ -n "${FLUXKREA_HOME:-}" ] && echo "  home      $FLUXKREA_HOME"
echo "  url       http://localhost:$port"
echo "  log       $log"
echo "  stop      Ctrl+C"
echo
printf '\n=== daemon started %s ===\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$log"

# 2>&1 because uvicorn and tracebacks go to stderr; tee rather than a
# redirect so the terminal stays live and a crash three hours in is still
# readable tomorrow.
exec "$python" -m fluxkrea.cli serve 2>&1 | tee -a "$log"
