@echo off
REM Double-clickable wrapper around serve.ps1.
REM
REM -ExecutionPolicy Bypass applies to this invocation only; it does not
REM change the machine's policy. Without it a default Windows install
REM refuses to run an unsigned local script, which reads as the daemon
REM being broken rather than PowerShell declining to start it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1" %*
