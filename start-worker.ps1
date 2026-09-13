param(
  [switch]$Reinstall,
  [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
  py -3.11 -m venv .venv
  $Reinstall = $true
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if ($Reinstall) {
  & $Python -m pip install --upgrade pip
  & $Python -m pip install -r worker\requirements.txt
}

Write-Host ""
Write-Host "Stereo Depth Lab worker" -ForegroundColor Cyan
Write-Host "Local endpoint: http://127.0.0.1:$Port"
Write-Host "Your source video and generated files stay under $HOME\.stereo-lab."
Write-Host "Leave this window open while using the private site."
Write-Host ""

& $Python -m uvicorn worker.main:app --host 127.0.0.1 --port $Port

