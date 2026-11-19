param([string]$BaseImage = 'lmp-service:local', [string]$Image = 'lmp-service:synthetic')

$ErrorActionPreference = 'Stop'
$model = (Resolve-Path (Join-Path $PSScriptRoot 'out/synthetic-v2')).Path
$token = [guid]::NewGuid().ToString('N')
$recipe = "FROM $BaseImage`nCOPY lmp.py service.py /app/`nENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
$recipe | docker build --network none --pull=false --file - --tag $Image $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Cached image refresh failed' }
$container = docker run --detach --rm --network none --read-only --cap-drop=ALL `
    --cpus=1 --memory=2g --security-opt=no-new-privileges --tmpfs /tmp `
    --mount "type=bind,source=$model,target=/models,readonly" `
    --env "LMP_API_TOKEN=$token" --env LMP_ALLOW_SYNTHETIC=1 $Image
if ($LASTEXITCODE -ne 0) { throw 'Container launch failed' }
try {
    $probe = @'
import json
import os
import time
from urllib.request import Request, urlopen

base = "http://127.0.0.1:8000"
for attempt in range(25):
    try:
        with urlopen(base + "/healthz", timeout=2) as response:
            assert json.load(response)["status"] == "ready"
        break
    except OSError:
        if attempt == 24:
            raise
        time.sleep(1)

pairs = [
    {"language": "en", "original": "The tenant must give notice in 10 days.",
     "simplified": "The tenant may give notice in 20 days."},
    {"language": "hi", "original": "किराया सोमवार को देना होगा।",
     "simplified": "सोमवार को किराया दें।", "context": "यह मासिक किराया है।"},
]
request = Request(base + "/v1/score", data=json.dumps({"pairs": pairs}).encode(),
                  headers={"Content-Type": "application/json",
                           "Authorization": "Bearer " + os.environ["LMP_API_TOKEN"]})
with urlopen(request, timeout=10) as response:
    output = json.load(response)
assert output["model_origin"] == "synthetic"
assert [item["mode"] for item in output["predictions"]] == ["sentence_only", "with_context"]
assert all(1 <= item["score"] <= 10 for item in output["predictions"])
assert len(output["predictions"][0]["evidence_all"]) >= 2
print("Offline container HTTP scoring passed in English and Hindi")
'@
    $probe | docker exec -i $container python -
    if ($LASTEXITCODE -ne 0) {
        docker logs $container
        throw 'Container HTTP verification failed'
    }
} finally {
    docker stop $container | Out-Null
}
