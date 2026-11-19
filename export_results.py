"""Build submission-ready research notes and a presentation outline from results."""

import argparse
import json
from pathlib import Path


def export(root):
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    challenge = json.loads((root / "independent.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "data/audit.json").read_text(encoding="utf-8"))
    errors = json.loads((root / "independent_errors.json").read_text(encoding="utf-8"))
    head_a, head_b = ("sentence_only", "with_context")
    first, second = (report["internal_test"][mode]["all"] for mode in (head_a, head_b))
    independent = challenge["classification"]
    count = independent[head_b]["all"]["n"]
    counts = ", ".join(f"{key} {value['pairs']}" for key, value in audit.items())
    rows = ["| Mode | Language | Accuracy | Precision | Recall | F1 | False acceptance | False rejection |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, groups in independent.items():
        for language in ("all", "en", "hi"):
            row = groups[language]
            cells = [f"{row[key]:.1%}" for key in (
                "accuracy", "precision", "recall", "f1", "false_acceptance", "false_rejection")]
            rows.append(f"| {mode} | {language} | " + " | ".join(cells) + " |")
    result_text = "\n".join(rows)
    dataset = f"Synthetic corpus counts are {counts}. The separate challenge contains {count} bilingual pairs."
    summary = (f"Internal test RMSE was {first['rmse']:.3f} for sentence-only and "
               f"{second['rmse']:.3f} with context. On the separate challenge, false acceptance was "
               f"{independent[head_a]['all']['false_acceptance']:.1%} and "
               f"{independent[head_b]['all']['false_acceptance']:.1%}, respectively.")
    destination = root / "submission"
    destination.mkdir(exist_ok=True)
    (destination / "paper-draft.md").write_text(
        "# Evaluating Legal Meaning Preservation in English and Hindi with Synthetic Supervision\n\n"
        "## Abstract\n\nWe evaluate whether bilingual representations and textual change features "
        "detect meaning changes in fictional lease clauses. " + summary + " All supervision is synthetic.\n\n"
        "## I. Introduction\n\nLegal simplification can preserve surface similarity while "
        "changing parties, deadlines, permissions, or exceptions. We compare two transformer regressors "
        "with lexical and embedding baselines.\n\n"
        "## II. Data\n\n" + dataset + " Labels use explicit transformation penalties. Topic groups "
        "stay intact across splits. The independently phrased challenge is evaluated after checkpoint selection.\n\n"
        "## III. Methods\n\nA frozen multilingual MiniLM encoder is fused with legal text features. "
        "Separate neural heads handle sentence-only and context-aware inputs. The bounded experiment "
        "uses validation loss, early stopping, and training-only augmentation.\n\n"
        "## IV. Results\n\n" + summary + "\n\n" + result_text + "\n\n"
        "See the accompanying evaluation tables for baseline and language-specific regression metrics.\n\n"
        "## V. Limitations\n\nSynthetic template performance does not establish legal validity or "
        "human agreement. The grammar-diverse challenge is small, and uncertainty is uncalibrated.\n\n"
        "## References\n\n[1] Beauchemin et al., “JUDGEBERT: Assessing Legal Meaning "
        "Preservation Between Sentences,” EMNLP, 2025. https://aclanthology.org/2025.emnlp-main.5/\n",
        encoding="utf-8")
    (destination / "presentation-outline.md").write_text(
        "# Legal meaning preservation\n\n## 1. Problem\n\nDetect a changed legal effect in a similar clause.\n\n"
        "## 2. Data\n\n" + dataset + "\n\n## 3. Architecture\n\nFrozen multilingual encoder, "
        "text evidence and two regression heads.\n\n## 4. Results\n\n" + summary +
        "\n\n" + result_text + "\n\n## 5. Demo\n\nCompare a deadline change and a faithful "
        "paraphrase in English and Hindi. Show the score and text evidence.\n\n"
        f"## 6. Error analysis\n\n{len(errors[head_b])} challenge predictions were analysed. "
        "Show the largest errors from independent_errors.json and explain the limits of synthetic supervision.\n",
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/synthetic"))
    export(parser.parse_args().out)
