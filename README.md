# Legal Meaning Preservation in English and Hindi

## Automated synthetic workflow

The main workflow uses synthetic supervision and requires zero human annotation. With the environment and encoder cache installed, `./reproduce.ps1` runs both experiments, evaluates the selected checkpoint, exports the research materials and runs the offline tests. Reproductions go into a new directory under `out/reproductions` to preserve existing checkpoints.

```powershell
./reproduce.ps1
./start-local.ps1
```

`out/synthetic` holds the initial run. `out/synthetic-v2` holds the expanded run and its comparison with the incumbent on the same validation partition. The launcher prefers the expanded run when present. The trained scores represent performance against generated legal transformations and have not been validated against human legal judgments.

See [Synthetic methodology](SYNTHETIC.md) for the model architecture, generation rules, selection criteria, and interpretation of results. The older small demo and human-research commands below are optional workflows.

The reproducibility archive contains the generated data, numerical heads, tokenizer configuration, environment versions, reports and file hashes. The encoder weights are supplied separately by the pinned model cache. For a GitHub submission, commit the source and fixture files, and attach `research-run.zip` to the submission or release.

The CLI compares an original lease clause with a simplified version. It uses a frozen multilingual MiniLM encoder and a Ridge regression head. The two heads differ only in whether the input includes surrounding legal context.

## Run the demo

Use Python 3.12. The first run downloads the encoder. The included fictional leases and synthetic scores exercise the pipeline.

```powershell
python -m uv venv --python 3.12
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt --torch-backend cpu
.\.venv\Scripts\python -m unittest -v
.\.venv\Scripts\python lmp.py train data/demo.jsonl --demo --out out/demo
.\.venv\Scripts\python lmp.py score out/demo/model.json data/sanity.jsonl --allow-demo --out out/demo/sanity.json
```

Install uv with `python -m pip install uv` if needed. The demo results are for smoke checks. Some sanity examples also appear in the demo corpus, so their demo predictions cannot serve as independent evidence.

## Collect human ratings

Prepare at least 150 English and 150 Hindi pairs from one legal domain. Use several independent source documents so both languages are represented in the training and test sets. Give translations, paraphrase variants, or other versions of the same source document the same source ID. Keep the source citation separately if the source ID is opaque.

Write UTF-8 JSONL with one object per line:

```json
{"id":"en-001","language":"en","source":"lease-001","original":"The tenant must pay within 10 days.","simplified":"The tenant must pay within 20 days.","context":"The notice period starts on receipt."}
```

Ask at least two qualified readers to rate each pair independently, with the same supplied context. Use 1 for unrelated or fundamentally changed meaning, 2–6 for partial preservation, and 7–10 for substantially preserved meaning. Ask reviewers to consider obligations, permissions, parties, dates, amounts, conditions and exceptions. Pilot the rubric on a separate set before collecting the final ratings. Each reviewer returns UTF-8 JSONL records such as:

```json
{"id":"en-001","annotator":"reviewer-a","score":3}
```

Reviewers can use the terminal annotation tool. Run the same command to resume, and use a separate output file for each reviewer. A bilingual reviewer can repeat the command with the other language and the same file.

```powershell
.\.venv\Scripts\python -X utf8 annotate.py data/pairs.jsonl --reviewer reviewer-a --language hi --out data/reviewer-a.jsonl
```

```powershell
.\.venv\Scripts\python lmp.py collect data/pairs.jsonl data/reviewer-a.jsonl data/reviewer-b.jsonl --out out/rated.jsonl
.\.venv\Scripts\python lmp.py train out/rated.jsonl --out out/research
.\.venv\Scripts\python lmp.py score out/research/model.json data/sanity.jsonl --out out/research/sanity.json
```

## Read the results

- `report.json` records the train and test source IDs, interval Krippendorff alpha by language, and Pearson, Spearman, MAE and RMSE. Undefined correlations are JSON null.
- `predictions.json` retains each test pair, its individual ratings and mean, all four predictions, and optional text evidence. Baselines are Jaccard word overlap and multilingual embedding cosine similarity, calibrated on training data.
- Compare both encoder heads on the same held-out pairs. The transfer scores fit a head using only the other language's training rows. Compare each transfer result with joint training on the same target test rows.
- `score` accepts JSONL containing `original`, `simplified`, and optional `context`. It clips scores to 1–10 and rounds to the nearest integer for the three decision bands, with halves rounded up. Any expected decision in the input is checked in the output.
- Review at least 15 held-out or independent challenge pairs. Record which clause, number, party, or qualifier changed and whether the predicted decision and evidence reflect that change. The optional evidence flags match surface patterns and can fire on valid paraphrases or miss changes. The challenge file covers identical and unrelated inputs as well as obligations, dates, entities, conditions, and exceptions.

The 300 human-rated pairs and the resulting research metrics require actual independent human ratings.

### Candidate sources and rating coverage

SIMPLE-LAW provides original and simplified legal passages, and MILDSum provides English judgments with summaries in English and Hindi. IL-TUR covers several Indian legal NLP tasks. These resources are useful starting points for data sourcing, but their published tasks do not establish the pair-level English and Hindi legal preservation ratings required by this project. Review the corpus terms and match the source domain before adopting any examples.

- [SIMPLE-LAW paper](https://aclanthology.org/2026.lrec-1.45/)
- [MILDSum paper](https://arxiv.org/abs/2310.18600)
- [IL-TUR paper](https://aclanthology.org/2024.acl-long.618/)

For the lease use case, collect source clauses and related context from distinct documents with permission to use them. Assign each source a stable ID shared across languages and variants. Ask separate reviewers to score a mix of faithful rewrites and changes to deadlines, amounts, parties, obligations, exceptions and conditions. Keep the release documents separate before any model development and record the source references with the annotation handoff.

## Approve a release candidate

Set aside a separate human-rated release set with source IDs absent from the entire development corpus. Include at least 30 pairs per language, including at least 15 with mean ratings of 6 or lower and 10 with mean ratings of 7 or higher.

Before distributing either set to reviewers, run `python audit_corpus.py data/pairs.jsonl data/release-pairs.jsonl`. It reports coverage and flags exact duplicates, shared source IDs, and repeated originals assigned to different sources. Review paraphrased or translated duplicates separately and keep them in the same source group.

```powershell
.\.venv\Scripts\python lmp.py collect data/release-pairs.jsonl data/release-ratings.jsonl --release-set --out out/release-data.jsonl
.\.venv\Scripts\python lmp.py release out/research/model.json out/release-data.jsonl --local-only --out out/release/model.json
```

Release policy version 2 requires **both scoring modes in each language** to achieve RMSE at most 2, Pearson correlation at least 0.5, and false acceptance at most 10% on pairs rated 6 or lower. The gate uses the same human judgments for both modes. Treat these as engineering acceptance criteria and report the underlying sample counts alongside the metrics. Treat every failing candidate as a development result and preserve the independence of the final release sample. Successful artifacts record the policy version, results, and fingerprints of the development and release datasets.

## Serve an approved model

### Local research dashboard

The FastAPI service also serves a browser dashboard at `http://127.0.0.1:8000/`. It reads the completed experiment report, baseline and model metrics, dataset audits, seed/transfer diagnostics, trial history and challenge error analysis. The clause evaluator calls the same local `/v1/score` API. The dashboard labels the included results as synthetic and keeps the API token in the browser tab only.

On macOS or Linux, first install the service dependencies and train the demo model if needed:

```bash
python -m pip install -r requirements-service.txt
python lmp.py train data/demo.jsonl --demo --out out/demo
```

Then start the local server. Set `LMP_RESULTS_DIR` to the completed expanded experiment directory when it is not in one of the default result locations:

```bash
export LMP_API_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
export LMP_MODEL_PATH="out/demo/model.json"
export LMP_ALLOW_SYNTHETIC=1
export LMP_RESULTS_DIR="out/reproductions/mac-run/expanded"
python -m uvicorn service:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`, select **Evaluate clauses**, expand **Local API token**, and enter the same value used for `LMP_API_TOKEN`. The scoring form sends the clause pair to the local service and displays the score, decision band and possible text-change markers. To show another run, change `LMP_RESULTS_DIR` and restart the server. The Windows `start-local.ps1` launcher works with the same dashboard; copy its generated token from `out/local-api-token.txt` into the evaluator.

The synthetic demo is only for local research and presentation. It is not a release-approved model and does not represent human legal judgments.

For a lightweight local run with synthetic weights, run `./start-local.ps1` after training the demo. This runs directly on Windows with one inference thread and uses the cached encoder. It saves the API token in `out/local-api-token.txt`. API responses include `model_origin: synthetic` for this model. The demo has known false positives on unrelated passages.

Install `requirements-service.txt`. Set `LMP_API_TOKEN` to a random secret with at least 32 ASCII characters. Set `LMP_MODEL_PATH` if the release artifact is outside `out/release/model.json`. The service loads the encoder from the local cache at startup.

```powershell
.\.venv\Scripts\python -m uvicorn service:create_app --factory --host 127.0.0.1 --port 8000
```

`GET /healthz` confirms readiness. Send `POST /v1/score` with `Authorization: Bearer <token>` and `Content-Type: application/json`:

```json
{"pairs":[{"id":"case-1","language":"en","original":"Rent is due on Monday.","simplified":"Pay rent by Monday.","context":"This clause applies to monthly rent."}]}
```

Requests accept up to 16 pairs and 256 KiB. Each pair must fit in 256 model tokens, including context and separators. Longer inputs return 422 so a condition or exception is never silently cut off. Each process runs one inference batch at a time and responds with 503 and `Retry-After` when busy. Place the service behind a TLS gateway with request timeouts and rate limits for remote access. Use a read-only model directory and cache for serving.

Install `requirements-dev.txt` and run `python -m unittest -v` to exercise the CLI logic, release policy, and HTTP contract.

After training the demo, set `$env:LMP_RUN_INTEGRATION = '1'` and run `python -m unittest -v` to exercise bilingual HTTP inference against the cached encoder. The integration test injects the demo scorer in its test process. The deployed service still requires an approved artifact.

### Container deployment

Build the image with network access so it can cache the pinned encoder revision. The running container uses that cache offline and runs as an unprivileged user. Mount the directory containing the approved `model.json` as read-only and supply the token through the environment.

```powershell
docker build -t lmp-service:local .
$releaseModel = (Resolve-Path out/release).Path
docker run --rm --cpus=1 --memory=2g --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e LMP_API_TOKEN --mount "type=bind,source=$releaseModel,target=/models,readonly" -p 127.0.0.1:8000:8000 lmp-service:local
```

Deploy a new model by mounting a new release directory and replacing the container. Check `/healthz` before routing traffic. To roll back, restart with the previous approved directory and image.

After building and training the demo, run `./smoke-container.ps1` in PowerShell. This checks offline inference inside a read-only container and confirms that the HTTP service refuses to start with demo weights.

To verify the trained synthetic model with a cached image, run `./verify-container.ps1`. It reuses the encoder layers and checks authenticated English and Hindi scoring in a temporary container with networking disabled, one CPU and a 2 GB memory limit. It removes the test container on exit. For a fresh synthetic deployment, build the main Dockerfile, mount the directory containing the trained `model.json`, and pass `-e LMP_ALLOW_SYNTHETIC=1` when starting the container. The API marks its responses as synthetic.

### Example evidence review

| Challenge cases | Manual reading | Evidence behavior |
| --- | --- | --- |
| en-1 and hi-1 | The deadline changes from 10 to 20 days. | Extracts both numbers. |
| en-2 and hi-2 | Permission becomes an obligation. | Extracts the modal change in both languages. |
| en-3 and hi-3 | The restriction on shared areas disappears. | Flags the missing exception marker. |
| en-4 and hi-4 | Both versions require notice for early departure. | Flags a changed condition marker even though the condition survives. |
| en-5 and hi-5 | The deposit still comes back after inspection. | Returns no flag. |
| en-6 and hi-6 | The duty to pay shifts between parties. | Misses the change in responsible party. |
| en-7 and hi-7 | The due date stays the same. | Returns no flag. |
| en-8 and hi-8 | The weekday restriction disappears. | Flags the missing scope marker. |
