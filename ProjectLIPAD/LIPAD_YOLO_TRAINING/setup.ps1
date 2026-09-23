# LiPAD YOLO Training — local virtual environment
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

Write-Host "Creating venv in $Root\.venv ..."
python -m venv "$Root\.venv"

Write-Host "Installing dependencies ..."
& "$Root\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Root\.venv\Scripts\pip.exe" install -r "$Root\requirements.txt"

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
