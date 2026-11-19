"""Train and evaluate synthetic scoring heads with a bounded development loop."""

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

from lmp import (MODEL, MODEL_REVISION, head_prediction, legal_features, metrics, overlap,
                 pair_inputs, prediction, text_changes, write_json)
from synthetic import audit, generate, write_dataset

MODES = ("sentence_only", "with_context")


def encode(rows, encoder, cache):
    originals, simplified, pairs, contextual = pair_inputs(rows, encoder)
    unique = dict.fromkeys(originals + simplified + pairs + contextual)
    missing = [value for value in unique if value not in cache]
    if missing:
        cache.update(zip(missing, encoder.encode(missing, batch_size=24, show_progress_bar=False)))
    left, right = (np.asarray([cache[value] for value in texts])
                   for texts in (originals, simplified))
    cosine = np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))
    result = {
        "word_overlap": np.asarray([[overlap(a, b)] for a, b in zip(originals, simplified)]),
        "embedding_similarity": cosine[:, None],
    }
    for mode, texts in zip(MODES, (pairs, contextual)):
        result[mode] = np.asarray([
            np.r_[cache[text], legal_features(row, mode)] for text, row in zip(texts, rows)
        ], dtype=np.float32)
    return result


def targets(rows):
    return np.asarray([row["score"] for row in rows], dtype=np.float32)


def fit(train, y, valid, expected, seed):
    torch.manual_seed(seed)
    x, y, xv, yv = (torch.as_tensor(value, dtype=torch.float32)
                    for value in (train, y, valid, expected))
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.ReLU(),
                                torch.nn.Dropout(.12), torch.nn.Linear(32, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    best_loss, best, stale = float("inf"), None, 0
    history = []
    for _ in range(60):
        model.train()
        for indices in torch.randperm(len(x)).split(48):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(model(x[indices]).flatten(), y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            loss = float(torch.nn.functional.mse_loss(model(xv).flatten().clamp(1, 10), yv))
        history.append(loss)
        if loss < best_loss - .0001:
            best_loss, best, stale = loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best)
    return {"hidden_weights": model[0].weight.tolist(), "hidden_bias": model[0].bias.tolist(),
            "weights": model[3].weight[0].tolist(), "intercept": float(model[3].bias.item())}, history


def predicted(matrix, rows, heads, baselines):
    result = {name: np.clip(model.predict(matrix[name]), 1, 10)
              for name, model in baselines.items()}
    for mode in MODES:
        result[mode] = np.asarray([
            10. if row["original"] == row["simplified"] else
            prediction(head_prediction(vector[:384], row, mode, heads[mode]))["score"]
            for vector, row in zip(matrix[mode], rows)
        ])
    return result


def summaries(rows, values):
    y = targets(rows)
    result = {}
    for name, scores in values.items():
        result[name] = {}
        for lang in ("all", "en", "hi"):
            mask = np.asarray([lang in ("all", row["language"]) for row in rows])
            result[name][lang] = metrics(y[mask], scores[mask])
    return result


def decisions(rows, values):
    buckets = defaultdict(list)
    for row, score in zip(rows, values):
        expected = row["score"] >= 7
        actual = prediction(score)["decision"] == "Preserved"
        for scope in ("all", row["language"], f"category/{row['category']}"):
            buckets[scope].append((expected, actual))
    result = {}
    for scope, comparisons in buckets.items():
        expected, actual = zip(*comparisons)
        tn, fp, fn, tp = confusion_matrix(expected, actual, labels=[False, True]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected, actual, average="binary", zero_division=0)
        n = len(comparisons)
        result[scope] = {
            "n": n, "accuracy": float((tn + tp) / n), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "false_acceptance": float(fp / (fp + tn)) if fp + tn else None,
            "false_rejection": float(fn / (fn + tp)) if fn + tp else None,
        }
    return result


def error_analysis(rows, scores):
    examples = []
    for row, score in zip(rows, scores):
        observed = prediction(score)
        examples.append({**row, "expected_score": row["score"], "predicted_score": observed["score"],
                         "predicted_decision": observed["decision"], "evidence": text_changes(row),
                         "correct": observed["decision"] == prediction(row["score"])["decision"],
                         "absolute_error": abs(float(score) - row["score"])})
    return sorted(examples, key=lambda item: (-item["absolute_error"], item["id"]))


def markdown(report):
    lines = ["# Synthetic evaluation", "", "Scores are generated labels. No human judgments were used.",
             "", f"Selected trial {report['selected_trial']}.", "",
             "| Internal test method | Language | Pearson | RMSE | MAE | Spearman |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, groups in report["internal_test"].items():
        for lang, results in groups.items():
            cells = [f"{results[key]:.3f}" if results[key] is not None else "—"
                     for key in ("pearson", "rmse", "mae", "spearman")]
            lines.append(f"| {name} | {lang} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Independent challenge", "",
                  "| Mode | Scope | Accuracy | F1 | False acceptance | False rejection |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for mode, groups in report["challenge"].items():
        for scope, results in groups.items():
            cells = [f"{results[key]:.1%}" if results[key] is not None else "—"
                     for key in ("accuracy", "f1", "false_acceptance", "false_rejection")]
            lines.append(f"| {mode} | {scope} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(root, trials, *, varied=False, incumbent=None):
    torch.set_num_threads(2)
    root = Path(root)
    datasets = write_dataset(root / "data", varied=varied)
    train = list(datasets["train"])
    valid = datasets["validation"]
    encoder = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu", local_files_only=True)
    encoder.max_seq_length = 256
    cache = {}
    print("Encoding validation pairs", flush=True)
    valid_x = encode(valid, encoder, cache)
    best = None
    weak = set()
    all_trials = []
    if incumbent:
        old = json.loads((incumbent / "model.json").read_text(encoding="utf-8"))
        old_train = json.loads((incumbent / "data/selected_train.json").read_text(encoding="utf-8"))
        old_x = encode(old_train, encoder, cache)
        old_baselines = {name: Ridge(alpha=.01).fit(old_x[name], targets(old_train))
                         for name in ("word_overlap", "embedding_similarity")}
        old_metrics = summaries(valid, predicted(valid_x, valid, old["heads"], old_baselines))
        initial_losses = [old_metrics[mode][lang]["rmse"]
                          for mode in MODES for lang in ("en", "hi")]
        best = (float(np.mean(initial_losses)), initial_losses,
                {**old, "trial": "incumbent"}, old_baselines, old_train)
        write_json(root / "incumbent_validation.json", old_metrics)
    for trial in range(trials):
        print(f"Training trial {trial + 1}/{trials}", flush=True)
        if trial:
            train.extend(generate("train", extra_round=trial, categories=weak))
        train_x = encode(train, encoder, cache)
        heads, histories = {}, {}
        baselines = {name: Ridge(alpha=.01).fit(train_x[name], targets(train))
                     for name in ("word_overlap", "embedding_similarity")}
        for mode in MODES:
            heads[mode], histories[mode] = fit(
                train_x[mode], targets(train), valid_x[mode], targets(valid), 42 + trial)
        values = predicted(valid_x, valid, heads, baselines)
        validation = summaries(valid, values)
        losses = [validation[mode][lang]["rmse"] for mode in MODES for lang in ("en", "hi")]
        objective = float(np.mean(losses))
        accepted = best is None or (objective < best[0] - .001 and all(
            new <= old + .05 for new, old in zip(losses, best[1])))
        artifact = {"demo": True, "rating_origin": "synthetic", "encoder": MODEL,
                    "revision": MODEL_REVISION, "max_seq_length": 256, "heads": heads,
                    "trial": trial, "seed": 42 + trial}
        info = {"trial": trial, "training_pairs": len(train), "validation": validation,
                "accepted": accepted, "epoch_validation_mse": histories}
        all_trials.append(info)
        write_json(root / "trials" / f"{trial}.json", {**info, "model": artifact})
        if accepted:
            best = (objective, losses, artifact, baselines, list(train))
        by_category = defaultdict(list)
        for row, score in zip(valid, values["with_context"]):
            by_category[row["category"]].append(abs(float(score) - row["score"]))
        weak = set(sorted(by_category, key=lambda key: -np.mean(by_category[key]))[:5])
        if trial >= 2 and not accepted and not all_trials[-2]["accepted"]:
            break
    _, _, artifact, baselines, selected_train = best
    artifact["dataset_sha256"] = sha256((root / "data/train.jsonl").read_bytes()).hexdigest()
    artifact["selected_training_sha256"] = sha256(
        repr(selected_train).encode("utf-8")).hexdigest()
    print("Evaluating fixed checkpoint on internal test and challenge", flush=True)
    test, challenge = datasets["test"], datasets["challenge"]
    test_scores = predicted(encode(test, encoder, cache), test, artifact["heads"], baselines)
    challenge_scores = predicted(
        encode(challenge, encoder, cache), challenge, artifact["heads"], baselines)
    report = {"synthetic": True, "selected_trial": artifact["trial"], "trials": all_trials,
              "internal_test": summaries(test, test_scores),
              "challenge": {mode: decisions(challenge, challenge_scores[mode]) for mode in MODES},
              "sanity": {mode: {key: value for key, value in decisions(test, test_scores[mode]).items()
                                if key in {f"category/{category}" for category in
                                           ("identical", "unrelated", "contradiction",
                                            "obligation_to_permission", "deadline", "exception")}}
                         for mode in MODES}}
    write_json(root / "report.json", report)
    write_json(root / "error_analysis.json", error_analysis(challenge, challenge_scores["with_context"]))
    (root / "evaluation.md").write_text(markdown(report), encoding="utf-8")
    write_json(root / "model.json", artifact)
    write_json(root / "data" / "selected_train.json", selected_train)
    print(f"Saved trial {artifact['trial']} and evaluation to {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/synthetic")
    parser.add_argument("--trials", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--varied", action="store_true")
    parser.add_argument("--incumbent", type=Path)
    args = parser.parse_args()
    run(args.out, args.trials, varied=args.varied, incumbent=args.incumbent)
