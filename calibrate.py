"""Evaluate a validation-only decision threshold without changing model scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment import MODES
from lmp import Scorer, read_rows, write_json


def rates(scores, labels, threshold):
    accepted = np.asarray(scores) >= threshold
    preserved = np.asarray(labels) >= 7
    return {"false_acceptance": float(accepted[~preserved].mean()),
            "false_rejection": float((~accepted[preserved]).mean())}


def calibrate(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels)
    baseline = rates(scores, labels, 6.5)
    candidates = sorted(set(np.append(scores, 6.5)))
    def objective(threshold):
        result = rates(scores, labels, threshold)
        return (result["false_rejection"], result["false_acceptance"], abs(threshold - 6.5))
    return float(min((t for t in candidates
                      if rates(scores, labels, t)["false_acceptance"] <= baseline["false_acceptance"]),
                     key=objective))


def run(root):
    torch.set_num_threads(2)
    scorer = Scorer(root / "model.json", allow_demo=True, local_only=True)
    validation = read_rows(root / "data/validation.jsonl")
    labels = [row["score"] for row in validation]
    independent = json.loads((root / "independent_errors.json").read_text(encoding="utf-8"))
    report = {}
    for mode in MODES:
        inputs = validation if mode == "with_context" else [
            {**row, "context": ""} for row in validation]
        scores = [row["score"] for row in scorer.predict(inputs)]
        threshold = calibrate(scores, labels)
        held_scores = [row["predicted_score"] for row in independent[mode]]
        held_labels = [row["expected_score"] for row in independent[mode]]
        report[mode] = {
            "threshold": threshold,
            "validation_before": rates(scores, labels, 6.5),
            "validation_after": rates(scores, labels, threshold),
            "challenge_before": rates(held_scores, held_labels, 6.5),
            "challenge_after": rates(held_scores, held_labels, threshold),
        }
    write_json(root / "threshold-experiment.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/synthetic-v2"))
    run(parser.parse_args().out)
