"""Measure seed stability and cross-language transfer without selecting a model."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from experiment import MODES, encode, fit, predicted, summaries, targets
from lmp import MODEL, MODEL_REVISION, read_rows, write_json


def run(root):
    torch.set_num_threads(2)
    train = json.loads((root / "data/selected_train.json").read_text(encoding="utf-8"))
    valid = read_rows(root / "data/validation.jsonl")
    test = read_rows(root / "data/test.jsonl")
    encoder = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu", local_files_only=True)
    encoder.max_seq_length = 256
    cache = {}
    print("Encoding selected training, validation and internal test data", flush=True)
    tx, vx, ex = (encode(rows, encoder, cache) for rows in (train, valid, test))
    train_y, valid_y = targets(train), targets(valid)
    seeds = {}
    for seed in (41, 42, 43):
        print(f"Fitting seed {seed} on the same training split", flush=True)
        heads = {mode: fit(tx[mode], train_y, vx[mode], valid_y, seed)[0] for mode in MODES}
        seeds[seed] = summaries(test, predicted(ex, test, heads, {}))
    stability = {}
    for mode in MODES:
        stability[mode] = {}
        for lang in ("all", "en", "hi"):
            values = [result[mode][lang]["rmse"] for result in seeds.values()]
            stability[mode][lang] = {"mean_rmse": mean(values), "sample_sd": stdev(values)}
    transfer = {}
    for source, target in (("en", "hi"), ("hi", "en")):
        train_mask, val_mask, test_mask = (
            np.asarray([item["language"] == language for item in rows])
            for rows, language in ((train, source), (valid, source), (test, target)))
        heads = {mode: fit(tx[mode][train_mask], train_y[train_mask],
                           vx[mode][val_mask], valid_y[val_mask], 42)[0] for mode in MODES}
        target_rows = [item for item in test if item["language"] == target]
        target_x = {mode: ex[mode][test_mask] for mode in MODES}
        transfer[f"{source}_to_{target}"] = {
            mode: by_lang["all"] for mode, by_lang in summaries(
                target_rows, predicted(target_x, target_rows, heads, {})).items()
        }
    write_json(root / "diagnostics.json", {"synthetic": True, "seed_runs": seeds,
                                          "stability": stability, "language_transfer": transfer})
    print("Saved seed stability and transfer diagnostics", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/synthetic-v2"))
    run(parser.parse_args().out)
