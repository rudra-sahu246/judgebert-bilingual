param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$model = Join-Path $PSScriptRoot 'out/demo/model.json'
if (!(Test-Path $python) -or !(Test-Path $model)) {
    throw 'Set up the environment and train the demo first. See README.md.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use."
}
$env:LMP_API_TOKEN = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
$env:LMP_MODEL_PATH = $model
$env:LMP_ALLOW_SYNTHETIC = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
Start-Process -FilePath $python -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -ArgumentList '-m','uvicorn','service:create_app','--factory','--host','127.0.0.1','--port',$Port,'--limit-concurrency','16' `
    -RedirectStandardOutput (Join-Path $PSScriptRoot 'out/service.log') `
    -RedirectStandardError (Join-Path $PSScriptRoot 'out/service-error.log')
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'out/local-api-token.txt') -Value $env:LMP_API_TOKEN
Write-Output "Starting http://127.0.0.1:$Port. Token saved to out/local-api-token.txt."
