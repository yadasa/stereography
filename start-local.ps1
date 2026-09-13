$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
  throw "Node.js 18 or newer is required. Install Node.js, then rerun this script."
}

$WorkerScript = Join-Path $PSScriptRoot "start-worker.ps1"
$FrontendScript = Join-Path $PSScriptRoot "start-frontend.ps1"
Start-Process powershell.exe -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy", "Bypass",
  "-File", $WorkerScript
)
Start-Process powershell.exe -ArgumentList @(
  "-NoExit",
  "-ExecutionPolicy", "Bypass",
  "-File", $FrontendScript
)

Write-Host "Opening Stereo Depth Lab at http://127.0.0.1:4173" -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:4173"
