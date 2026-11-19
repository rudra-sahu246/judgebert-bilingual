# Synthetic experiment methodology

## Dataset and labels

The generator creates fictional lease clauses in English and Hindi. Both versions of a topic belong to the same source group. The initial split uses 14 topics for training, three for validation, three for internal test and four for challenge. The generator checks exact duplicates, source disjointness, language script, accidental identical pairs and contradiction label bounds.

An unchanged legal effect scores 10. Subtract impact penalties for changed legal elements and clip to 1–10. Removing a secondary detail costs 3, shrinking scope costs 4, materially changing a condition, amount, deadline or exception costs 6, and changing a central actor or obligation costs 7. Unrelated clauses cost 9. Explicit contradictions are capped at 2. Semantic similarity is a baseline and does not assign labels.

The templates reuse some grammar across topics. Topic isolation measures transfer to unseen clause topics within the same generated style. It does not establish transfer to independently authored contracts or guarantee the absence of phrasing similarities.

The audit reports near duplicates when both sides of a pair have at least 0.90 word-set Jaccard similarity to an example in an earlier split. Separately, `python evaluate_independent.py` evaluates the frozen checkpoint on 44 differently structured challenge pairs from a distinct fixture. It writes regression, classification and ranked error reports without updating any checkpoint.

## Architecture and selection

The frozen multilingual MiniLM encoder reads each pair jointly. The context mode appends the relevant context. Each 384-dimensional pooled embedding is concatenated with word overlap, text-change flags and features for whether new numbers appear in supplied context. Each mode has a separate 32-unit regression head with ReLU, dropout and smooth L1 training loss. The checkpoint stores numerical weights and the exact encoder revision. The tokenizer is loaded from the same pinned local model snapshot.

The bounded experiment trains with deterministic seeds on CPU. It monitors validation MSE and saves the best epoch of each head. Later trials add numerical variations from training topics in high-error validation categories. A replacement must improve mean validation RMSE and keep each language/mode within 0.05 RMSE of the previous best. Internal test and challenge examples are encoded for evaluation after selection.

The expanded experiment adds independently structured development clauses in separate train, validation and test groups. It recalculates the incumbent's validation score on the expanded validation partition and compares candidates against it. The grammar-diverse challenge fixture remains unchanged and is evaluated separately.

## Outputs and interpretation

- `out/synthetic/data/audit.json` records split coverage and label distributions.
- `out/synthetic/trials` records trial metrics and checkpoints.
- `out/synthetic/report.json` and `evaluation.md` contain baseline, language, context, sanity and challenge results.
- `out/synthetic/error_analysis.json` records challenge scores and rule evidence with the largest errors first.
- `out/synthetic/model.json` contains the two selected regression heads for CLI and API use.
- `out/synthetic-v2/diagnostics.json` records three repetitions on the same data and English-to-Hindi and Hindi-to-English transfer. These runs do not change the selected checkpoint.
- `out/synthetic-v2/submission` contains the editable two-column paper, PDF and PowerPoint derived from the measured results.

False acceptance means a synthetic negative receives a Preserved decision. False rejection means a synthetic positive does not. An empty class has an undefined rate. Text evidence reports observed pattern changes and can flag lawful paraphrases or miss meaning changes. The API reports that uncertainty is uncalibrated. Neither the labels nor model outputs represent human legal judgments.
