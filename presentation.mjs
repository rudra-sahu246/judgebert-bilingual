// Build an editable presentation from a finished experiment.
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const root = path.resolve(process.argv[2] || "out/synthetic");
const read = async name => JSON.parse(await fs.readFile(path.join(root, name), "utf8"));
const report = await read("report.json");
const challenge = await read("independent.json");
const audit = await read("data/audit.json");
const requireFromRuntime = createRequire(path.join(process.env.LMP_NODE_MODULES, "_entry.cjs"));
const { Presentation, PresentationFile } = await import(
  pathToFileURL(requireFromRuntime.resolve("@oai/artifact-tool")).href);
const { finalizePresentation } = await import(pathToFileURL(path.join(
  process.env.LMP_PRESENTATION_SKILL, "container_tools/artifact_tool_utils.mjs")).href);
const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const ink = "#13243A", muted = "#4D6076", blue = "#155EB5", paper = "#F3F7FC", white = "#FFFFFF";
const font = "Arial";

function text(slide, value, left, top, width, height, size, color = ink, bold = false) {
  const box = slide.shapes.add({ geometry: "textbox", name: value.slice(0, 50),
    position: { left, top, width, height }, fill: "none",
    line: { fill: "none", width: 0 } });
  box.text = value;
  box.text.style = { fontFamily: font, fontSize: size, color, bold };
}

function body(title, subtitle) {
  const slide = deck.slides.add();
  slide.background.fill = paper;
  text(slide, title, 72, 62, 1130, 80, 48, ink, true);
  text(slide, subtitle, 72, 141, 1130, 70, 25, muted);
  return slide;
}

function row(slide, top, label, detail) {
  text(slide, label, 72, top, 340, 68, 27, blue, true);
  text(slide, detail, 415, top, 790, 80, 25, ink);
}

const cover = deck.slides.add();
cover.background.fill = ink;
text(cover, "Legal meaning preservation", 70, 145, 1132, 105, 65, white, true);
text(cover, "English and Hindi clause scoring with synthetic supervision",
  72, 274, 1090, 110, 35, "#C9DAF0");
text(cover, "Offline bilingual NLP research prototype", 72, 566, 1100, 56, 25, white);

const problem = body("Small edits can change legal meaning",
  "The model compares an original clause with a proposed simplification.");
row(problem, 250, "Original", "The tenant must give notice within 10 days.");
row(problem, 355, "Rewrite", "The tenant may give notice within 20 days.");
row(problem, 460, "Changes", "The rewrite changes both the obligation and the deadline.");

const datasets = body("Synthetic labels and separated sources",
  "Known transformations provide scores from 1 to 10 in both languages.");
row(datasets, 230, "Development", `${audit.train.pairs} training pairs and ${audit.validation.pairs} validation pairs.`);
row(datasets, 330, "Internal test", `${audit.test.pairs} pairs from held-out source groups.`);
row(datasets, 430, "Independent challenge", `${challenge.classification.with_context.all.n} differently worded pairs, scored after model selection.`);
row(datasets, 530, "Label origin", "Synthetic supervision. No human legal judgments.");

const architecture = body("A shared encoder with two scoring modes",
  "The same pinned multilingual encoder processes both forms of input.");
row(architecture, 230, "01  Encode", "Read the original and rewrite together. Append context when provided.");
row(architecture, 330, "02  Combine features", "Fuse the pooled embedding with overlap and structured text-change features.");
row(architecture, 430, "03  Predict", "Use a separate regression head for each mode and return a score from 1 to 10.");
row(architecture, 530, "04  Explain", "Report matched text changes separately from the score.");

const results = body("Internal test results",
  "Lower RMSE is better. All targets are synthetic labels.");
for (const [index, [label, key]] of [
  ["Word overlap", "word_overlap"], ["Embedding similarity", "embedding_similarity"],
  ["Sentence-only", "sentence_only"], ["With context", "with_context"]
].entries()) {
  const metrics = report.internal_test[key].all;
  row(results, 226 + index * 93, label,
    `RMSE ${metrics.rmse.toFixed(3)}    Pearson ${metrics.pearson?.toFixed(3) ?? "undefined"}`);
}

const pct = value => value === null ? "undefined" : `${(value * 100).toFixed(1)}%`;
const external = body("Independent challenge results",
  "A separate phrasing set tests whether the result transfers beyond topic templates.");
for (const [index, [label, metric]] of [
  ["False acceptance", "false_acceptance"], ["False rejection", "false_rejection"],
  ["Accuracy", "accuracy"]
].entries()) {
  row(external, 228 + index * 100, label,
    `Sentence-only ${pct(challenge.classification.sentence_only.all[metric])}    With context ${pct(challenge.classification.with_context.all[metric])}`);
}
text(external, "These measurements describe the synthetic challenge only.", 72, 575, 1120, 56, 24, muted);

const demo = body("Demonstration and limits",
  "Compare a faithful rewrite and a changed obligation in each language.");
row(demo, 235, "Run offline", "The Windows and Docker entry points use a cached encoder.");
row(demo, 335, "Inspect the result", "Compare the score, decision and rule evidence for every pair.");
row(demo, 435, "Interpret carefully", "The challenge measures synthetic transformations. Human legal judgment and uncertainty calibration remain unvalidated.");

const workspace = process.cwd();
const stage = path.join(root, "presentation-build");
const output = path.join(root, "submission", "judgebert.pptx");
const draft = path.join(stage, "draft.pptx");
await fs.mkdir(stage, { recursive: true });
await fs.mkdir(path.dirname(output), { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(draft);
await finalizePresentation({
  workspaceDir: workspace, candidatePath: draft, finalPath: output,
  pythonExecutable: process.env.LMP_RUNTIME_PYTHON,
  integrityValidatorPath: path.join(process.env.LMP_PRESENTATION_SKILL, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(process.env.LMP_PRESENTATION_SKILL, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
  fontPolicy: { basis: "design", families: [font] },
  explicitTotalSlideCount: 7,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stage, "validation.json"),
});
for (let i = 0; i < deck.slides.items.length; i++) {
  const png = await deck.export({ slide: deck.slides.items[i], format: "png", scale: 1 });
  await fs.writeFile(path.join(stage, `slide-${i+1}.png`), new Uint8Array(await png.arrayBuffer()));
}
