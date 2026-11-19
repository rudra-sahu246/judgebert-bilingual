"""Check independence and coverage before distributing pairs for annotation."""

import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

from annotate import queue
from lmp import LANGUAGES, read_rows


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def audit(development, release):
    corpora = {"development": development, "release": release}
    source_sets = {}
    report = {}
    for name, rows in corpora.items():
        counts = {}
        source_sets[name] = {row["source"] for row in rows}
        for language in LANGUAGES:
            selected = queue(rows, [], "", language)
            sources = Counter(row["source"] for row in selected)
            minimum = 150 if name == "development" else 30
            if len(selected) < minimum:
                raise ValueError(f"{name}/{language} requires at least {minimum} pairs")
            if len(sources) < 2:
                raise ValueError(f"{name}/{language} requires at least two source documents")
            counts[language] = {"pairs": len(selected), "sources": len(sources),
                                "largest_source_pairs": max(sources.values())}
        report[name] = counts
    if report["development"]["en"]["pairs"] != report["development"]["hi"]["pairs"]:
        raise ValueError("Development must have the same number of pairs in each language")
    if source_sets["development"] & source_sets["release"]:
        raise ValueError("Development and release share source IDs")
    paired_inputs = set()
    original_locations = {}
    for name, rows in corpora.items():
        for row in rows:
            original = (row["language"], normalize(row["original"]))
            pair = (*original, normalize(row["simplified"]))
            if pair in paired_inputs:
                raise ValueError(f"Duplicate pair at {name}/{row['id']}")
            paired_inputs.add(pair)
            location = (name, row["source"])
            if original_locations.setdefault(original, location) != location:
                raise ValueError(f"Original text crosses source groups at {name}/{row['id']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(read_rows(args.development), read_rows(args.release)), indent=2))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Corpus audit failed: {exc}\n")
