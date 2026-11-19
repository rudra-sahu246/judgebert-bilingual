param([int]$Port = 8000, [string]$ModelPath)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$model = if ($ModelPath) { $ModelPath } else {
    'out/synthetic-v2/model.json', 'out/synthetic/model.json', 'out/demo/model.json' |
        ForEach-Object { Join-Path $PSScriptRoot $_ } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (!(Test-Path $python) -or !$model -or !(Test-Path -LiteralPath $model)) {
    throw 'Set up the environment and train the demo first. See README.md.'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use."
}
$env:LMP_API_TOKEN = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
$env:LMP_MODEL_PATH = (Resolve-Path -LiteralPath $model).Path
$env:LMP_ALLOW_SYNTHETIC = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
$suffix = if ($Port -eq 8000) { '' } else { "-$Port" }
Start-Process -FilePath $python -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
    -ArgumentList '-m','uvicorn','service:create_app','--factory','--host','127.0.0.1','--port',$Port,'--limit-concurrency','16' `
    -RedirectStandardOutput (Join-Path $PSScriptRoot "out/service$suffix.log") `
    -RedirectStandardError (Join-Path $PSScriptRoot "out/service-error$suffix.log")
$tokenPath = "out/local-api-token$suffix.txt"
Set-Content -LiteralPath (Join-Path $PSScriptRoot $tokenPath) -Value $env:LMP_API_TOKEN
Write-Output "Starting http://127.0.0.1:$Port. Token saved to $tokenPath."
