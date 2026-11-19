param([string]$Image = 'lmp-service:local')

$ErrorActionPreference = 'Stop'
$demo = (Resolve-Path (Join-Path $PSScriptRoot 'out/demo')).Path
$data = (Resolve-Path (Join-Path $PSScriptRoot 'data')).Path

docker run --rm --network none --read-only --cap-drop=ALL --security-opt=no-new-privileges `
    --tmpfs /tmp --mount "type=bind,source=$demo,target=/models,readonly" `
    --mount "type=bind,source=$data,target=/fixtures,readonly" $Image `
    python lmp.py score /models/model.json /fixtures/sanity.jsonl --allow-demo --local-only --out /tmp/scores.json
if ($LASTEXITCODE -ne 0) { throw 'Offline container scoring failed' }

# Use a temporary test token. No network listener is exposed by this check.
$testToken = [guid]::NewGuid().ToString('N')
$startup = docker run --rm --network none --read-only --cap-drop=ALL --security-opt=no-new-privileges `
    --tmpfs /tmp --mount "type=bind,source=$demo,target=/models,readonly" `
    --env "LMP_API_TOKEN=$testToken" $Image 2>&1
$exitCode = $LASTEXITCODE
if ($exitCode -eq 0 -or ($startup | Out-String) -notmatch 'A research model is required') {
    throw "The service did not reject the demo artifact as expected: $startup"
}
Write-Output 'Offline scoring passed. The service rejected demo weights.'
