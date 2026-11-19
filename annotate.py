"""Collect independent human ratings from an unlabeled JSONL pair file."""

import argparse
import json
from pathlib import Path

from lmp import LANGUAGES, read_rows


def queue(pairs, existing, reviewer, language):
    seen = set()
    pending = []
    for pair in pairs:
        if not isinstance(pair, dict) or any(
            not isinstance(pair.get(key), str) or not pair[key].strip()
            for key in ("id", "original", "simplified", "context", "source", "language")
        ):
            raise ValueError("Each pair needs an ID, original, simplification, context, source and language")
        if pair["id"] in seen or pair["language"] not in LANGUAGES:
            raise ValueError(f"Duplicate ID or unsupported language: {pair['id']}")
        seen.add(pair["id"])
        if pair["language"] == language:
            pending.append(pair)
    completed = set()
    for item in existing:
        if (item.get("annotator") != reviewer or item.get("id") not in seen
                or item["id"] in completed or type(item.get("score")) is not int
                or not 1 <= item["score"] <= 10):
            raise ValueError("Existing output contains an invalid, duplicate or foreign rating")
        completed.add(item["id"])
    return [pair for pair in pending if pair["id"] not in completed]


def annotate(args):
    if args.pairs.resolve() == args.out.resolve():
        raise ValueError("Use a separate output file for ratings")
    reviewer = args.reviewer.strip()
    if not reviewer:
        raise ValueError("Provide a reviewer ID")
    pending = queue(read_rows(args.pairs), read_rows(args.out) if args.out.exists() else [],
                    reviewer, args.language)
    if not pending:
        print("No remaining pairs for this language.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Rate legal preservation from 1 to 10. Enter q to quit and resume later.")
    print("1 = unrelated or fundamentally changed; 2–6 = partial; 7–10 = substantially preserved.")
    with args.out.open("a", encoding="utf-8") as output:
        for index, pair in enumerate(pending, 1):
            print(f"\n{index}/{len(pending)} — {pair['id']}\n"
                  f"Context: {pair['context']}\nOriginal: {pair['original']}\n"
                  f"Simplified: {pair['simplified']}")
            while True:
                try:
                    answer = input("Score [1–10, q]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return
                if answer.lower() == "q":
                    return
                if answer in {str(score) for score in range(1, 11)}:
                    break
                print("Enter an integer from 1 to 10 or q.")
            record = {"id": pair["id"], "annotator": reviewer, "score": int(answer)}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    try:
        annotate(parser.parse_args())
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
