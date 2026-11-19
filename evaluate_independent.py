"""Score the grammar-diverse challenge using the already selected checkpoint."""

import argparse
from pathlib import Path

import torch

from experiment import MODES, decisions, error_analysis, summaries
from lmp import Scorer, read_rows, write_json
from synthetic import audit

CHALLENGE = Path(__file__).parent / "data" / "independent.jsonl"


def run(root):
    torch.set_num_threads(2)
    rows = read_rows(CHALLENGE)
    splits = {name: read_rows(root / "data" / f"{name}.jsonl")
              for name in ("train", "validation", "test")}
    write_json(root / "independent_audit.json", audit({**splits, "independent": rows}))
    scorer = Scorer(root / "model.json", allow_demo=True, local_only=True)
    scores = {}
    for mode in MODES:
        payload = rows if mode == "with_context" else [{**row, "context": ""} for row in rows]
        scores[mode] = [item["score"] for item in scorer.predict(payload)]
    import numpy as np

    write_json(root / "independent.json", {
        "synthetic": True, "trial": scorer.artifact["trial"],
        "classification": {mode: decisions(rows, scores[mode]) for mode in MODES},
        "regression": summaries(rows, {mode: np.asarray(scores[mode]) for mode in MODES}),
    })
    write_json(root / "independent_errors.json", {
        mode: error_analysis(rows, scores[mode]) for mode in MODES})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/synthetic"))
    run(parser.parse_args().out)
