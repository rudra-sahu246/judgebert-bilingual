"""Train and evaluate a bilingual legal meaning preservation scorer."""

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import mean
from tempfile import NamedTemporaryFile
import unicodedata

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
RELEASE_POLICY_VERSION = 2
LANGUAGES = ("en", "hi")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            delete=False) as handle:
        staged = Path(handle.name)
        handle.write(content)
    try:
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def validate(rows, demo=False, *, minimum_per_language=None, require_balance=True):
    counts = Counter()
    ids = set()
    for row in rows:
        for key in ("id", "original", "simplified", "context", "source", "language"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing {key} in {row.get('id', 'unnamed row')}")
        if row["id"] in ids or row["language"] not in LANGUAGES:
            raise ValueError(f"Duplicate ID or invalid language: {row['id']}")
        ids.add(row["id"])
        counts[row["language"]] += 1
        expected = "synthetic" if demo else "human"
        if row.get("rating_origin") != expected:
            raise ValueError(f"{row['id']} must have {expected} ratings")
        ratings = row.get("ratings")
        if not isinstance(ratings, dict) or len(ratings) < 2 or any(
            type(v) is not int or not 1 <= v <= 10 for v in ratings.values()
        ):
            raise ValueError(f"{row['id']} needs at least two integer ratings from 1 to 10")
    minimum = minimum_per_language if minimum_per_language is not None else (2 if demo else 150)
    if min(counts[lang] for lang in LANGUAGES) < minimum:
        raise ValueError(f"Use at least {minimum} pairs per language")
    if require_balance and counts["en"] != counts["hi"]:
        raise ValueError("Use the same number of pairs per language")


def agreement(rows):
    """Interval Krippendorff alpha, weighting each item's coincidences by 1/(n-1)."""
    ratings = [list(row["ratings"].values()) for row in rows]
    all_ratings = [rating for item in ratings for rating in item]
    n = len(all_ratings)
    if n < 2:
        return None
    observed = sum(sum((a-b)**2 for a in item for b in item) / (len(item)-1)
                   for item in ratings) / n
    center = mean(all_ratings)
    expected = 2 * sum((x-center)**2 for x in all_ratings) / (n-1)
    return 1 - observed / expected if expected else None


def split_rows(rows, seed):
    groups = sorted({row["source"] for row in rows})
    rng = random.Random(seed)
    count = max(1, round(len(groups) * .25))
    for _ in range(500):
        test_sources = set(rng.sample(groups, count))
        train = [row for row in rows if row["source"] not in test_sources]
        test = [row for row in rows if row["source"] in test_sources]
        if all(sum(row["language"] == lang for row in part) >= 2
               for part in (train, test) for lang in LANGUAGES):
            return train, test
    raise ValueError("Could not hold out source documents with at least two pairs per language in each split")


def words(value):
    normalized = unicodedata.normalize("NFC", value).casefold()
    return set("".join(c if unicodedata.category(c)[0] in "LMN" else " "
                       for c in normalized).split())


def overlap(original, simplified):
    a, b = words(original), words(simplified)
    return len(a & b) / len(a | b) if a or b else 0.0


def prediction(value):
    score = round(min(10., max(1., float(value))), 3)
    band = int(Decimal(str(score)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return {"score": score, "decision": "Not Preserved" if band == 1 else
            "Partially Preserved" if band <= 6 else "Preserved"}


def evidence(row):
    changes = text_changes(row)
    return changes[0] if changes else None


LEGAL_PATTERNS = {
    "number_or_date": r"\d+(?:[./-]\d+)*",
    "date": r"\d{1,4}[./-]\d{1,2}(?:[./-]\d{1,4})?",
    "exception": r"unless|except|सिवाय|को छोड़कर",
    "condition": r"if|provided that|after approval|यदि|अगर|बशर्ते|अनुरोध मिलने पर|स्वीकृति मिलने पर|स्वीकृति के बाद",
    "obligation_or_permission": r"must|shall|may|should|होगा|होगी|होंगे|चाहिए|सकता है|सकती है",
    "negation": r"not|never|नहीं",
    "entity": r"tenant|landlord|technician|किरायेदार|मकान मालिक|तकनीशियन",
    "scope": r"only|all|every|each|हर|सभी|केवल|सिर्फ",
}


def text_changes(row):
    import re

    original, simplified = row["original"], row["simplified"]
    changes = []
    for label, pattern in LEGAL_PATTERNS.items():
        if label not in ("number_or_date", "date"):
            pattern = rf"(?<![\w\u0900-\u097f])(?:{pattern})(?![\w\u0900-\u097f])"
        before = re.findall(pattern, original, flags=re.I)
        after = re.findall(pattern, simplified, flags=re.I)
        if Counter(x.casefold() for x in before) != Counter(x.casefold() for x in after):
            changes.append({"possible_change": label, "original_matches": before,
                            "simplified_matches": after})
    return changes


def legal_features(row, mode):
    changes = {item["possible_change"] for item in text_changes(row)}
    context = row.get("context", "") if mode == "with_context" else ""
    import re

    numbers = lambda text: set(re.findall(r"\d+", text))
    introduced = numbers(row["simplified"]) - numbers(row["original"])
    return [float(label in changes) for label in LEGAL_PATTERNS] + [
        overlap(row["original"], row["simplified"]),
        float(bool(context and introduced and introduced <= numbers(context))),
        float(bool(context and introduced - numbers(context))),
    ]


def head_prediction(vector, row, mode, head):
    import numpy as np

    if "hidden_weights" not in head:
        return float(np.dot(vector, head["weights"]) + head["intercept"])
    fused = np.r_[vector, legal_features(row, mode)]
    hidden = np.maximum(0, np.asarray(head["hidden_weights"]) @ fused + head["hidden_bias"])
    return float(np.dot(hidden, head["weights"]) + head["intercept"])


def pair_inputs(rows, model):
    originals = [row["original"] for row in rows]
    simplified = [row["simplified"] for row in rows]
    separator = f" {model.tokenizer.sep_token} "
    pairs = [a + separator + b for a, b in zip(originals, simplified)]
    tokenizer = model.tokenizer
    contextual = [pair + separator + row["context"] if row.get("context") else pair
                  for pair, row in zip(pairs, rows)]
    for row, combined in zip(rows, contextual):
        if len(tokenizer.encode(combined, verbose=False)) > model.max_seq_length:
            raise ValueError(
                f"Pair {row.get('id', '')} exceeds the token budget; shorten passages or context")
    return originals, simplified, pairs, contextual


def features(rows, model):
    import numpy as np

    originals, simplified, pairs, contextual = pair_inputs(rows, model)
    vectors = model.encode(originals + simplified, normalize_embeddings=True)
    left, right = np.split(vectors, 2)
    return {"overlap": np.array([[overlap(a, b)] for a, b in zip(originals, simplified)]),
            "similarity": (left * right).sum(axis=1, keepdims=True),
            "sentence_only": model.encode(pairs),
            "with_context": model.encode(contextual)}


def metrics(target, predicted):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    residual = np.asarray(predicted) - target
    variable = len(target) >= 2 and np.ptp(target) > 0 and np.ptp(predicted) > 0
    return {"n": len(target), "pearson": float(pearsonr(target, predicted)[0]) if variable else None,
            "spearman": float(spearmanr(target, predicted)[0]) if variable else None,
            "mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt((residual**2).mean()))}


def train(args):
    rows = read_rows(args.dataset)
    validate(rows, args.demo)
    training, test = split_rows(rows, args.seed)
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import Ridge
    import numpy as np

    model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    model.max_seq_length = 256
    matrix = features(training + test, model)
    train_y = np.array([mean(row["ratings"].values()) for row in training])
    test_y = np.array([mean(row["ratings"].values()) for row in test])
    artifact = {"encoder": MODEL, "revision": MODEL_REVISION, "max_seq_length": 256,
                "demo": args.demo, "heads": {},
                "source_ids": sorted({row["source"] for row in rows}),
                "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest()}
    report = {"demo": args.demo, "seed": args.seed, "results": {},
              "train_sources": sorted({r["source"] for r in training}),
              "test_sources": sorted({r["source"] for r in test}),
              "agreement": {lang: agreement([r for r in rows if r["language"] == lang])
                            for lang in LANGUAGES}}
    predictions = [{"id": row["id"], "language": row["language"], "source": row["source"],
                    "target": float(target), "ratings": row["ratings"],
                    "original": row["original"], "simplified": row["simplified"],
                    "context": row["context"], "evidence": evidence(row), "scores": {}}
                   for row, target in zip(test, test_y)]
    for name, vectors in matrix.items():
        x_train, x_test = vectors[:len(training)], vectors[len(training):]
        head = Ridge(alpha=1.0 if vectors.shape[1] > 1 else 0.01).fit(x_train, train_y)
        artifact["heads"][name] = {"weights": head.coef_.tolist(),
                                   "intercept": float(head.intercept_)}
        scores = np.clip(head.predict(x_test), 1, 10)
        report["results"][name] = {"all": metrics(test_y, scores)}
        for item, score in zip(predictions, scores):
            item["scores"][name] = prediction(score)
        for lang in LANGUAGES:
            held = np.array([r["language"] == lang for r in test])
            other = np.array([r["language"] != lang for r in training])
            transfer = Ridge(alpha=head.alpha).fit(x_train[other], train_y[other])
            report["results"][name][lang] = metrics(test_y[held], scores[held])
            report["results"][name][f"transfer_to_{lang}"] = metrics(
                test_y[held], np.clip(transfer.predict(x_test[held]), 1, 10))
    write_json(args.out / "model.json", artifact)
    write_json(args.out / "report.json", report)
    write_json(args.out / "predictions.json", predictions)


def validate_model(artifact):
    if not isinstance(artifact, dict) or (
        artifact.get("encoder"), artifact.get("revision"), artifact.get("max_seq_length")
    ) != (MODEL, MODEL_REVISION, 256):
        raise ValueError("Unsupported encoder, model revision, or sequence length")
    if type(artifact.get("demo")) is not bool:
        raise ValueError("Model must specify whether it is a demo")
    for mode in ("sentence_only", "with_context"):
        heads = artifact.get("heads")
        head = heads.get(mode) if isinstance(heads, dict) else None
        weights = head.get("weights") if isinstance(head, dict) else None
        if isinstance(head, dict) and "hidden_weights" in head:
            import numpy as np

            try:
                hidden = np.asarray(head["hidden_weights"], dtype=float)
                bias = np.asarray(head["hidden_bias"], dtype=float)
                output = np.asarray(weights, dtype=float)
                if hidden.shape == (32, 384 + len(LEGAL_PATTERNS) + 3) and (
                    bias.shape == output.shape == (32,)
                ) and all(np.isfinite(value).all() for value in (hidden, bias, output)) and (
                    math.isfinite(head["intercept"])
                ):
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
            raise ValueError(f"Invalid neural regression head for {mode}")
        if not isinstance(weights, list) or len(weights) != 384 or not all(
            type(value) in (int, float) and math.isfinite(value)
            for value in [*weights, head.get("intercept")]
        ):
            raise ValueError(f"Invalid regression head for {mode}")


class Scorer:
    def __init__(self, path, *, allow_demo=False, local_only=False, require_release=False):
        self.artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(self.artifact, dict):
            raise ValueError("Model artifact must be a JSON object")
        if self.artifact.get("demo") is not False and not allow_demo:
            raise ValueError("A research model is required. Use --allow-demo for local demos.")
        if require_release and not (allow_demo and self.artifact.get("demo") is True) and (
            self.artifact.get("demo") is not False
            or not isinstance(self.artifact.get("release"), dict)
            or self.artifact["release"].get("policy_version") != RELEASE_POLICY_VERSION
        ):
            raise ValueError("The service requires a release-approved model")
        validate_model(self.artifact)
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(
            self.artifact["encoder"], revision=self.artifact["revision"],
            device="cpu", local_files_only=local_only)
        self.encoder.max_seq_length = self.artifact["max_seq_length"]

    def predict(self, rows):
        import numpy as np

        if not rows:
            raise ValueError("The input contains no pairs")
        for row in rows:
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("original", "simplified")
            ):
                raise ValueError("Each scoring input needs nonempty original and simplified text")
            if "context" in row and not isinstance(row["context"], str):
                raise ValueError("Context must be text")
        _, _, pairs, contextual = pair_inputs(rows, self.encoder)
        results = []
        pending = []
        for index, row in enumerate(rows):
            identical = unicodedata.normalize("NFC", row["original"].strip()) == (
                unicodedata.normalize("NFC", row["simplified"].strip()))
            mode = "identical" if identical else (
                "with_context" if row.get("context", "").strip() else "sentence_only")
            item = {**row, "mode": mode, "evidence": evidence(row),
                    "evidence_all": text_changes(row), "demo": self.artifact["demo"],
                    "uncertainty": {"calibrated": False}}
            if identical:
                item.update(prediction(10))
            else:
                pending.append((index, mode))
            results.append(item)
        if pending:
            vectors = self.encoder.encode([
                (contextual if mode == "with_context" else pairs)[index]
                for index, mode in pending])
            for (index, mode), vector in zip(pending, vectors):
                head = self.artifact["heads"][mode]
                results[index].update(prediction(head_prediction(vector, rows[index], mode, head)))
        for item in results:
            if "expected" in item:
                item["passes_expected_decision"] = item["decision"] == item["expected"]
        return results


def score(args):
    scorer = Scorer(args.model, allow_demo=args.allow_demo, local_only=args.local_only)
    write_json(args.out, scorer.predict(read_rows(args.pairs)))


def release(args):
    """Evaluate a candidate on a separate bilingual human-rated release set."""
    rows = read_rows(args.dataset)
    # Reuse the research validation rules, with a smaller independent sample.
    validate(rows, minimum_per_language=30, require_balance=False)
    artifact = json.loads(args.model.read_text(encoding="utf-8"))
    if artifact.get("demo") is not False or not artifact.get("source_ids"):
        raise ValueError("Release requires a research model with recorded source IDs")
    if set(artifact["source_ids"]) & {row["source"] for row in rows}:
        raise ValueError("Release data must use sources absent from the development dataset")
    if args.out.resolve() == args.model.resolve():
        raise ValueError("Write the release artifact to a separate path")
    artifact.pop("release", None)
    scorer = Scorer(args.model, local_only=args.local_only)
    results = {}
    failures = []
    for mode, inputs in (
        ("with_context", rows), ("sentence_only", [{**row, "context": ""} for row in rows]),
    ):
        scored = scorer.predict(inputs)
        results[mode] = {}
        for lang in LANGUAGES:
            subset = [item for item in scored if item["language"] == lang]
            target = [mean(item["ratings"].values()) for item in subset]
            result = metrics(target, [item["score"] for item in subset])
            altered = [item for item, label in zip(subset, target) if label <= 6]
            false_accepts = sum(item["decision"] == "Preserved" for item in altered)
            result.update(altered=len(altered), false_accepts=false_accepts)
            results[mode][lang] = result
            scope = f"{mode}/{lang}"
            if len(altered) < 15 or sum(label >= 7 for label in target) < 10:
                failures.append(f"{scope}: require at least 15 altered and 10 preserved pairs")
            if false_accepts > len(altered) * .10:
                failures.append(f"{scope}: false acceptance exceeds 10%")
            if result["rmse"] > 2 or result["pearson"] is None or result["pearson"] < .5:
                failures.append(f"{scope}: require RMSE <= 2 and Pearson >= 0.5")
    if failures:
        raise ValueError("Release rejected: " + "; ".join(failures))
    artifact["release"] = {"policy_version": RELEASE_POLICY_VERSION, "results": results,
                           "dataset_sha256": sha256(args.dataset.read_bytes()).hexdigest()}
    write_json(args.out, artifact)


def collect(args):
    rows = read_rows(args.pairs)
    by_id = {row["id"]: {**row, "ratings": {}, "rating_origin": "human"} for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("The pair file contains duplicate IDs")
    for path in args.ratings:
        for row in read_rows(path):
            key, reviewer = row.get("id"), row.get("annotator")
            if key not in by_id or not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError(f"Unknown pair or missing reviewer in {path}")
            ratings = by_id[key]["ratings"]
            if reviewer in ratings:
                raise ValueError(f"Duplicate rating for {key} by {reviewer}")
            ratings[reviewer] = row.get("score")
    validate(list(by_id.values()), minimum_per_language=30 if args.release_set else None,
             require_balance=not args.release_set)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                               for row in by_id.values()), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collection = commands.add_parser("collect")
    collection.add_argument("pairs", type=Path)
    collection.add_argument("ratings", type=Path, nargs="+")
    collection.add_argument("--out", type=Path, required=True)
    collection.add_argument("--release-set", action="store_true")
    training = commands.add_parser("train")
    training.add_argument("dataset", type=Path)
    training.add_argument("--out", type=Path, default=Path("out"))
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--demo", action="store_true")
    inference = commands.add_parser("score")
    inference.add_argument("model", type=Path)
    inference.add_argument("pairs", type=Path)
    inference.add_argument("--out", type=Path, default=Path("out/scores.json"))
    inference.add_argument("--allow-demo", action="store_true")
    inference.add_argument("--local-only", action="store_true")
    promotion = commands.add_parser("release")
    promotion.add_argument("model", type=Path)
    promotion.add_argument("dataset", type=Path)
    promotion.add_argument("--out", type=Path, required=True)
    promotion.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    try:
        {"collect": collect, "train": train, "score": score,
         "release": release}[args.command](args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
