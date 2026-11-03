param([string]$PythonExe = "python")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path -LiteralPath (Join-Path $Root "VERIFY_PUBLIC.py") -PathType Leaf)) { throw "VERIFY_SCRIPT_MISSING" }
& $PythonExe -I -B (Join-Path $Root "VERIFY_PUBLIC.py")
if ($LASTEXITCODE -ne 0) { throw "STORED_OUTPUT_VERIFICATION_FAILED" }
Write-Host "STORED_OUTPUT_VERIFICATION_EXIT=0"
