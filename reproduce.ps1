param(
    [ValidateRange(1,8)][int]$Trials = 3,
    [string]$OutputRoot = ("out/reproductions/" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
)

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
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "Choose a fresh output directory to preserve earlier checkpoints."
    }
    $initial = Join-Path $OutputRoot 'initial'
    $expanded = Join-Path $OutputRoot 'expanded'
    if (!(Test-Path out/demo/model.json)) {
        Run-Python lmp.py train data/demo.jsonl --demo --out out/demo
    }
    Run-Python experiment.py --trials $Trials --out $initial
    Run-Python experiment.py --trials $Trials --varied --incumbent $initial --out $expanded
    Run-Python evaluate_independent.py --out $expanded
    Run-Python diagnostics.py --out $expanded
    Run-Python export_results.py --out $expanded
    Run-Python package_run.py --out $expanded
    $env:LMP_TEST_MODEL = Join-Path $expanded 'model.json'
    Run-Python -m unittest -v
} finally {
    Pop-Location
}
