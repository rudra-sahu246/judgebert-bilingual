"""Create the editable two-column paper from measured experiment outputs."""

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


def format_paper(root):
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    report, challenge, audit = (read(name) for name in (
        "report.json", "independent.json", "data/audit.json"))
    initial, contextual = ("sentence_only", "with_context")
    results = report["internal_test"]
    a, b = (results[mode]["all"] for mode in (initial, contextual))
    external = challenge["classification"]
    x, y = (external[mode]["all"] for mode in (initial, contextual))
    document = Document()
    document.core_properties.title = "Legal meaning preservation in English and Hindi"
    styles = document.styles
    for name, size in (("Normal", 10), ("Title", 21), ("Heading 1", 11), ("Caption", 9)):
        style = styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.line_spacing = 1.03
    for name in ("Title", "Heading 1"):
        properties = styles[name].element.get_or_add_pPr()
        for border in properties.findall(qn("w:pBdr")):
            properties.remove(border)
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = Inches(.72)
    section.top_margin = section.bottom_margin = Inches(.65)
    headline = document.add_paragraph("Legal Meaning Preservation in English and Hindi",
                                      style="Title")
    headline.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph("A comparison of sentence and context models under synthetic supervision")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def para(content):
        paragraph = document.add_paragraph(content)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return paragraph

    def heading(text):
        document.add_heading(text, level=1)

    def table(title, rows):
        caption = document.add_paragraph(title, style="Caption")
        caption.paragraph_format.keep_with_next = True
        grid = document.add_table(rows=0, cols=len(rows[0]))
        grid.style = "Table Grid"
        for index, row in enumerate(rows):
            tr = grid.add_row()
            tr._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            for col, cell in enumerate(tr.cells):
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.keep_with_next = index < len(rows) - 1
                run = p.add_run(str(row[col]))
                run.font.name = "Times New Roman"
                run.font.size = Pt(8)
                run.bold = index == 0

    abstract = para("")
    abstract.add_run("Abstract. ").bold = True
    abstract.add_run(
        "We investigate whether multilingual representations and textual change features can distinguish "
        "faithful rewrites from changes to fictional lease clauses. We compare lexical and embedding "
        "baselines with separate sentence and context regressors. Internal test RMSE fell from "
        f"{a['rmse']:.3f} to {b['rmse']:.3f} with context. On a separate 44-pair challenge, "
        f"context reduced false acceptance from {x['false_acceptance']:.1%} to "
        f"{y['false_acceptance']:.1%}, while false rejection was {y['false_rejection']:.1%} "
        "in both modes. All scores use synthetic supervision.")
    para("Index Terms—legal text simplification, multilingual NLP, synthetic supervision, evaluation")
    second = document.add_section(WD_SECTION.CONTINUOUS)
    cols = second._sectPr.find(qn("w:cols"))
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "330")
    heading("I Introduction")
    para("Lexical overlap can remain high even when a simplification changes a deadline, a party, "
         "or a permission. JUDGEBERT introduced a task-specific preservation score for French insurance "
         "text [1]. We study a related task for English and Hindi fictional rental clauses and test "
         "whether surrounding context helps a model distinguish legal changes.")
    heading("II Data and Supervision")
    para(f"The expanded base corpus has {audit['train']['pairs']} training pairs, "
         f"{audit['validation']['pairs']} validation pairs and {audit['test']['pairs']} internal test pairs. "
         "Each topic is grouped across languages to prevent source overlap. Known transformations assign "
         "a score by deducting penalties from 10. Penalties are based on the legal element changed. "
         "Contradictions are capped at 2. Unchanged legal effects receive 10. The independent challenge "
         "uses distinct wording and includes both preserved and altered pairs.")
    heading("III Model and Experimental Design")
    para("A frozen multilingual MiniLM encoder represents the original and rewrite jointly. The context "
         "mode includes the supplied surrounding clause. Each 384-dimensional representation is combined "
         "with lexical overlap, text change indicators and numerical context features. Two neural "
         "regression heads use a 32-unit hidden layer, ReLU, dropout and smooth L1 loss.")
    para("The bounded training loop adds examples from training topics in high-error validation categories. "
         "The selected checkpoint minimizes mean validation RMSE across languages and modes, subject to a "
         "maximum allowed per-group regression of 0.05. Neither test set informs checkpoint selection.")
    heading("IV Results")
    rows = [("Method", "Pearson", "RMSE", "MAE", "Spearman")]
    for label, method in (("Word overlap", "word_overlap"), ("Embedding", "embedding_similarity"),
                          ("Sentence", initial), ("Context", contextual)):
        entry = results[method]["all"]
        rows.append((label, *(f"{entry[key]:.3f}" for key in ("pearson", "rmse", "mae", "spearman"))))
    table("TABLE I  Internal test regression results", rows)
    para(f"With context, internal test RMSE was {results[contextual]['en']['rmse']:.3f} for English "
         f"and {results[contextual]['hi']['rmse']:.3f} for Hindi. The independent challenge shows "
         "that performance depends strongly on wording. Prediction errors and evidence are available "
         "for all challenge cases.")
    rows = [("Mode", "Accuracy", "F1", "FAR", "FRR")]
    for label, entry in (("Sentence", x), ("Context", y)):
        rows.append((label, *(f"{entry[key]:.1%}" for key in (
            "accuracy", "f1", "false_acceptance", "false_rejection"))))
    table("TABLE II  Independent synthetic challenge", rows)
    heading("V Limitations")
    para("The generated labels measure agreement with prescribed transformations. They have not been "
         "validated against human legal judgments. Development templates share linguistic structure, "
         "and the independent challenge is small. Text evidence reports observed token matches. "
         "It cannot determine the full legal effect, and model uncertainty is uncalibrated.")
    heading("VI Conclusion")
    para("Context improved internal synthetic regression and reduced false acceptance on the separate "
         "challenge. High false rejection limits this model to research and demonstration.")
    heading("References")
    para("[1] D. Beauchemin et al., “JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences,” "
         "in Proc. EMNLP, 2025, pp. 92–118.")
    output = root / "submission" / "judgebert-paper.docx"
    output.parent.mkdir(exist_ok=True)
    document.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/synthetic-v2"))
    format_paper(parser.parse_args().out)
