param([ValidateRange(1,8)][int]$Trials = 3)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:LMP_RUN_INTEGRATION = '1'

function Run-Python {
    & $python @args
    if ($LASTEXITCODE -ne 0) { throw "Python failed with exit code $LASTEXITCODE" }
}

Push-Location $PSScriptRoot
try {
    if (!(Test-Path out/demo/model.json)) {
        Run-Python lmp.py train data/demo.jsonl --demo --out out/demo
    }
    Run-Python experiment.py --trials $Trials --out out/synthetic
    Run-Python experiment.py --trials $Trials --varied --incumbent out/synthetic --out out/synthetic-v2
    Run-Python evaluate_independent.py --out out/synthetic-v2
    Run-Python export_results.py --out out/synthetic-v2
    Run-Python package_run.py --out out/synthetic-v2
    Run-Python -m unittest -v
} finally {
    Pop-Location
}
