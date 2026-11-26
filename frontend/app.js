const state = { overview: null, language: 'all', caseLanguage: 'all' };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const fmt = (value, digits = 3) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(digits);
const pct = (value, digits = 1) => value == null || Number.isNaN(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(digits)}%`;
const comma = value => Number(value || 0).toLocaleString('en-US');
const safe = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const methodNames = { word_overlap: 'Word overlap', embedding_similarity: 'Embedding similarity', sentence_only: 'Sentence only', with_context: 'With context' };

async function loadOverview() {
  const refresh = $('#refresh-button');
  refresh.classList.add('spinning');
  try {
    const response = await fetch('/api/overview', { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error((await response.json()).detail || `Could not load experiment results (${response.status})`);
    state.overview = await response.json();
    renderOverview();
    toast('Experiment data refreshed');
  } catch (error) {
    $('#run-label').textContent = 'Experiment data unavailable';
    $('#stat-grid').innerHTML = `<div class="load-error panel"><strong>Could not load the experiment report</strong><p>${safe(error.message)}</p><small>Set LMP_RESULTS_DIR to a completed run containing report.json.</small></div>`;
  } finally { refresh.classList.remove('spinning'); }
}

function renderOverview() {
  const data = state.overview;
  if (!data) return;
  const report = data.report || {};
  const test = report.internal_test || {};
  const context = test.with_context?.all || {};
  const sentence = test.sentence_only?.all || {};
  const audit = data.audit || {};
  const trainCount = audit.train?.pairs ?? report.trials?.at(-1)?.training_pairs ?? 0;
  const validationCount = audit.validation?.pairs ?? 0;
  const testCount = audit.test?.pairs ?? sentence.n ?? 0;
  const independent = data.independent?.classification || {};
  const challengeCount = independent.with_context?.all?.n ?? 0;
  $('#run-label').textContent = data.run_name ? `Run · ${data.run_name}` : 'Experiment run';
  $('#crumb-label').textContent = 'Overview';
  $('#test-count').textContent = `${comma(testCount)} held-out pairs`;
  $('#challenge-pairs').textContent = `${comma(challengeCount)} pairs`;

  const cards = [
    ['TRAINING PAIRS', trainCount, 'Generated development set', '◌'],
    ['INTERNAL TEST', testCount, 'Held-out source groups', '⌗'],
    ['CHALLENGE SET', challengeCount, 'Independent wording · synthetic', '⌁'],
    ['CONTEXT RMSE', fmt(context.rmse, 2), `Pearson ${fmt(context.pearson, 2)} · lower error is better`, '↗'],
  ];
  $('#stat-grid').innerHTML = cards.map(([label, value, foot, icon]) => `<article class="stat-card"><div class="stat-label">${label}<span class="stat-icon">${icon}</span></div><div class="stat-value">${typeof value === 'number' ? comma(value) : safe(value)}</div><div class="stat-foot">${safe(foot)}</div></article>`).join('');

  renderPerformance(test);
  const delta = sentence.rmse && context.rmse ? ((sentence.rmse - context.rmse) / sentence.rmse) * 100 : null;
  $('#rmse-change').textContent = delta == null ? '—' : `${delta >= 0 ? '−' : '+'}${Math.abs(delta).toFixed(1)}%`;
  $('#finding-title').textContent = delta == null ? 'Context comparison unavailable' : delta >= 0 ? 'Context improves the score fit' : 'Sentence-only fits this test better';
  $('#finding-copy').textContent = delta == null ? 'The report does not include both scoring modes.' : `On this held-out synthetic set, the context-aware model has RMSE ${fmt(context.rmse, 2)} compared with ${fmt(sentence.rmse, 2)} for sentence-only scoring.`;
  renderMetrics(test);
  renderChallenge(independent);
  renderSplits(audit, data.independent_audit, { trainCount, validationCount, testCount, challengeCount });
  renderEnvironment(data, report);
  renderDiagnostics(data.diagnostics);
  renderTrials(report.trials || []);
  renderRawOutputs(data);
  renderCases();
}

function renderPerformance(test) {
  const names = ['word_overlap', 'embedding_similarity', 'sentence_only', 'with_context'];
  const html = names.filter(key => test[key]?.all).map(key => {
    const row = test[key].all;
    const pearsonWidth = Math.max(0, Math.min(100, (row.pearson || 0) * 100));
    const rmseWidth = Math.max(4, Math.min(100, (row.rmse || 0) / 4 * 100));
    const highlight = key === 'with_context' ? '<small>CONTEXT-AWARE</small>' : key.includes('overlap') || key.includes('similarity') ? '<small>BASELINE</small>' : '<small>MODEL</small>';
    return `<div class="chart-method"><div class="method-name">${methodNames[key]}${highlight}</div><div class="method-bars"><div class="bar-line" title="Pearson ${fmt(row.pearson)}"><span class="bar-fill pearson" style="width:${pearsonWidth}%"></span><span class="bar-line-label">r ${fmt(row.pearson, 2)}</span></div><div class="bar-line" title="RMSE ${fmt(row.rmse)}"><span class="bar-fill rmse" style="width:${rmseWidth}%"></span><span class="bar-line-label">RMSE ${fmt(row.rmse, 2)}</span></div></div></div>`;
  }).join('');
  $('#performance-chart').innerHTML = html || '<div class="no-evidence">No internal test metrics were included in the report.</div>';
}

function renderMetrics(test) {
  const language = state.language;
  const rows = ['word_overlap', 'embedding_similarity', 'sentence_only', 'with_context'];
  $('#metrics-body').innerHTML = rows.filter(key => test[key]?.[language]).map(key => {
    const row = test[key][language];
    const best = key === 'with_context' ? 'best-cell' : '';
    const type = key.includes('overlap') || key.includes('similarity') ? 'BASELINE' : 'MODEL';
    return `<tr><td class="method-cell">${methodNames[key]}<span class="method-type">${type}</span></td><td>${fmt(row.pearson)}</td><td>${fmt(row.spearman)}</td><td>${fmt(row.mae)}</td><td class="${best}">${fmt(row.rmse)}</td><td>${comma(row.n)}</td></tr>`;
  }).join('') || '<tr><td colspan="6">No metrics are available for this selection.</td></tr>';
}

function renderChallenge(classification) {
  const keys = ['sentence_only', 'with_context'];
  const metrics = [['Accuracy', 'accuracy'], ['F1 score', 'f1'], ['False acceptance', 'false_acceptance'], ['False rejection', 'false_rejection']];
  $('#challenge-grid').innerHTML = keys.map(key => {
    const all = classification[key]?.all || {};
    const en = classification[key]?.en || {};
    const hi = classification[key]?.hi || {};
    return `<article class="challenge-card"><div class="challenge-card-head"><h3>${methodNames[key]}</h3><span class="challenge-label">${comma(all.n)} cases</span></div><div class="challenge-values">${metrics.map(([title, prop]) => `<div class="challenge-value"><small>${title}</small><strong class="${prop.includes('false') ? 'warn' : ''}">${pct(all[prop])}</strong></div>`).join('')}</div><div class="challenge-lang"><span>EN accuracy <b>${pct(en.accuracy)}</b></span><span>HI accuracy <b>${pct(hi.accuracy)}</b></span></div></article>`;
  }).join('') || '<article class="challenge-card">Independent challenge results are not available in this run.</article>';
}

function renderSplits(audit, independentAudit, fallback) {
  const values = [
    ['Train', audit.train?.pairs ?? fallback.trainCount, ''],
    ['Validation', audit.validation?.pairs ?? fallback.validationCount, 'validation'],
    ['Internal test', audit.test?.pairs ?? fallback.testCount, 'test'],
    ['Generated challenge', audit.challenge?.pairs ?? 0, 'challenge'],
    ['Independent challenge', independentAudit?.independent?.pairs ?? fallback.challengeCount, 'challenge'],
  ];
  const max = Math.max(...values.map(([, count]) => Number(count) || 0), 1);
  $('#split-list').innerHTML = values.map(([label, count, cls]) => `<div class="split-item"><span>${label}</span><div class="split-track"><div class="split-fill ${cls}" style="width:${Math.max(2, (Number(count) || 0) / max * 100)}%"></div></div><span class="split-count">${comma(count)}</span></div>`).join('');
}

function renderEnvironment(data, report) {
  const env = data.environment || {};
  const model = report.selected_trial == null ? 'Demo / selected run not recorded' : `Selected experiment ${report.selected_trial}`;
  const items = [
    ['Model selection', model],
    ['Training labels', report.synthetic ? 'Synthetic transformations' : 'Human-rated dataset'],
    ['Language coverage', 'English + Hindi'],
    ['Live scorer model', data.model_path || 'Configured by LMP_MODEL_PATH'],
    ['Runtime', env.python ? `Python ${env.python} · ${env.platform || 'platform not recorded'}` : 'Environment manifest not found'],
    ['Result folder', data.results_dir || '—'],
  ];
  $('#environment-list').innerHTML = items.map(([name, value]) => `<div class="detail-row"><span>${safe(name)}</span><strong>${safe(value)}</strong></div>`).join('');
}

function renderDiagnostics(diagnostics) {
  const content = $('#diagnostics-content');
  if (!diagnostics) { content.innerHTML = '<div class="no-evidence">No seed or transfer diagnostics were found in this run.</div>'; return; }
  const stability = diagnostics.stability || {};
  const transfer = diagnostics.language_transfer || {};
  const entries = [];
  for (const [mode, name] of [['sentence_only', 'Sentence only'], ['with_context', 'With context']]) {
    const result = stability[mode]?.all;
    if (result) entries.push([`${name} · seed stability`, `Mean RMSE ${fmt(result.mean_rmse)} · sample SD ${fmt(result.sample_sd)}`]);
  }
  for (const [key, name] of [['en_to_hi', 'English → Hindi transfer'], ['hi_to_en', 'Hindi → English transfer']]) {
    for (const [mode, label] of [['sentence_only', 'sentence'], ['with_context', 'context']]) {
      const result = transfer[key]?.[mode];
      if (result) entries.push([`${name} · ${label}`, `Pearson ${fmt(result.pearson)} · RMSE ${fmt(result.rmse)} · ${comma(result.n)} pairs`]);
    }
  }
  content.innerHTML = entries.length ? `<div class="diagnostic-grid">${entries.map(([label, value]) => `<div class="diagnostic-card"><strong>${safe(label)}</strong><span>${safe(value)}</span></div>`).join('')}</div>` : '<div class="no-evidence">Diagnostics were present but contain no recognized stability or transfer metrics.</div>';
}

function renderTrials(trials) {
  const content = $('#trials-content');
  if (!trials.length) { content.innerHTML = '<div class="no-evidence">This report does not contain experiment trial history.</div>'; return; }
  content.innerHTML = `<div class="table-wrap"><table class="compact-table"><thead><tr><th>TRIAL</th><th>TRAINING PAIRS</th><th>VALIDATION RMSE · SENTENCE</th><th>VALIDATION RMSE · CONTEXT</th><th>SELECTION</th></tr></thead><tbody>${trials.map(trial => {
    const sentence = trial.validation?.sentence_only?.all?.rmse;
    const context = trial.validation?.with_context?.all?.rmse;
    const accepted = trial.accepted === true;
    return `<tr><td class="method-cell">${trial.trial ?? '—'}</td><td>${comma(trial.training_pairs)}</td><td>${fmt(sentence)}</td><td>${fmt(context)}</td><td><span class="tag ${accepted ? 'correct' : ''}">${accepted ? 'Accepted' : trial.accepted === false ? 'Not selected' : 'Recorded'}</span></td></tr>`;
  }).join('')}</tbody></table></div>`;
}

function renderRawOutputs(data) {
  const content = $('#raw-content');
  const artifacts = [
    ['Experiment report', data.report],
    ['Independent challenge metrics', data.independent],
    ['Dataset audit', data.audit],
    ['Independent challenge audit', data.independent_audit],
    ['Seed and transfer diagnostics', data.diagnostics],
    ['Challenge error analysis', data.error_analysis],
    ['Runtime environment', data.environment],
  ].filter(([, value]) => value != null);
  content.innerHTML = `<div class="raw-toolbar"><label for="artifact-select">Backend artifact</label><select id="artifact-select">${artifacts.map(([label], index) => `<option value="${index}">${safe(label)}</option>`).join('')}</select><button class="button button-quiet" id="copy-artifact" type="button">Copy JSON</button></div><pre id="artifact-json" class="json-view"></pre>`;
  const select = $('#artifact-select', content);
  const output = $('#artifact-json', content);
  const render = () => { output.textContent = JSON.stringify(artifacts[Number(select.value)]?.[1] ?? {}, null, 2); };
  select.addEventListener('change', render);
  $('#copy-artifact', content).addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(output.textContent); toast('JSON copied'); }
    catch { toast('Clipboard access is unavailable in this browser'); }
  });
  render();
}

function cases() {
  const list = state.overview?.error_analysis;
  if (Array.isArray(list)) return list;
  if (list && typeof list === 'object') return Object.values(list).flat().filter(item => item && typeof item === 'object');
  return [];
}

function renderCases() {
  const list = cases();
  const lang = state.caseLanguage;
  const filtered = list.filter(item => lang === 'all' || item.language === lang);
  const total = list.length;
  const wrong = list.filter(item => item.correct === false || item.predicted_decision !== item.decision).length;
  const avg = total ? list.reduce((sum, item) => sum + Number(item.absolute_error || 0), 0) / total : 0;
  $('#challenge-summary').innerHTML = `<div class="summary-chip"><span>Cases available</span><strong>${comma(total)}</strong></div><div class="summary-chip"><span>Mismatch cases</span><strong>${comma(wrong)}</strong></div><div class="summary-chip"><span>Mean absolute error</span><strong>${fmt(avg, 2)}</strong></div><div class="summary-chip"><span>Showing</span><strong>${comma(filtered.length)}</strong></div>`;
  $('#case-list').innerHTML = filtered.length ? filtered.slice(0, 60).map(renderCase).join('') : '<article class="panel case-card"><div class="no-evidence">No individual error-analysis cases were found. Generate the independent evaluation output to populate this review.</div></article>';
}

function renderCase(item) {
  const correct = item.correct === true;
  const category = String(item.category || item.error_type || 'case').replaceAll('_', ' ');
  const score = item.predicted_score ?? item.score;
  const expected = item.expected_score ?? item.score;
  const evidence = Array.isArray(item.evidence) ? item.evidence.map(entry => entry.possible_change || entry).join(', ') : item.evidence;
  return `<article class="panel case-card"><div class="case-head"><span class="case-id">${safe(item.id || 'Challenge pair')} · ${safe(item.language || '')}</span><span class="case-tags"><span class="tag">${safe(category)}</span><span class="tag ${correct ? 'correct' : 'incorrect'}">${correct ? 'Matches label' : 'Model mismatch'}</span></span></div><div class="case-grid"><div><div class="case-text-label">ORIGINAL</div><div class="case-text" lang="${item.language === 'hi' ? 'hi' : 'en'}">${safe(item.original || '')}</div></div><div><div class="case-text-label">SIMPLIFIED</div><div class="case-text" lang="${item.language === 'hi' ? 'hi' : 'en'}">${safe(item.simplified || '')}</div></div></div>${item.context ? `<div class="case-context"><b>Context</b> · ${safe(item.context)}</div>` : ''}<div class="case-foot"><span>Expected label <strong>${fmt(expected, 1)}</strong></span><span>Predicted <strong>${fmt(score, 1)}</strong></span><span>Absolute error <strong>${fmt(item.absolute_error, 2)}</strong></span>${evidence ? `<span>Evidence <strong>${safe(evidence)}</strong></span>` : ''}</div></article>`;
}

function showView(name) {
  $$('.view').forEach(view => view.classList.toggle('active', view.id === `view-${name}`));
  $$('.nav-link').forEach(button => button.classList.toggle('active', button.dataset.view === name));
  const label = { overview: 'Overview', evaluate: 'Evaluate clauses', challenge: 'Challenge review' }[name] || 'Overview';
  $('#crumb-label').textContent = label;
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function toast(message) {
  const el = $('#toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), 2300);
}

function renderScore(result, input, origin) {
  const score = Math.max(1, Math.min(10, Number(result.score) || 1));
  const decisionColors = { 'Preserved': '#3b7b61', 'Partially Preserved': '#ac7a43', 'Not Preserved': '#b35449' };
  const color = decisionColors[result.decision] || '#3b7b61';
  const changes = result.evidence_all || (result.evidence ? [result.evidence] : []);
  const evidenceHTML = changes.length ? `<div class="evidence-list">${changes.map(change => {
    const label = typeof change === 'string' ? change : String(change.possible_change || 'Text signal').replaceAll('_', ' ');
    const before = typeof change === 'object' ? change.original_matches : null;
    const after = typeof change === 'object' ? change.simplified_matches : null;
    return `<div class="evidence-item"><strong>${safe(label)}</strong><p>${before?.length ? `Original: ${safe(before.join(', '))}` : 'No matching marker in original'} &nbsp;→&nbsp; ${after?.length ? `Rewrite: ${safe(after.join(', '))}` : 'No matching marker in rewrite'}</p></div>`;
  }).join('')}</div>` : '<div class="no-evidence">No configured text-change markers were detected. This does not prove the meaning is preserved.</div>';
  $('#empty-result').classList.add('hidden');
  const resultEl = $('#score-result');
  resultEl.classList.remove('hidden');
  resultEl.innerHTML = `<div class="result-top"><span class="eyebrow">MODEL OUTPUT</span><span class="origin-pill">${safe(origin || 'local model')} · ${safe(result.mode || (input.context ? 'with context' : 'sentence only'))}</span></div><div class="score-hero"><div class="score-ring" style="--score-fill:${score * 10}"><div class="score-number">${score.toFixed(1)}<small>/10</small></div></div><div class="decision-copy"><small>PREDICTED DECISION</small><strong style="color:${color}">${safe(result.decision || '—')}</strong><p>${input.context ? 'Context-aware input' : 'Sentence-only input'} · ${input.language === 'hi' ? 'Hindi' : 'English'}</p></div></div><div class="result-block"><h4>Meaning preservation score <span>${score.toFixed(1)} / 10</span></h4><div class="score-scale"><span style="width:${score * 10}%;background:${color}"></span></div><div class="score-bands"><span>Not preserved</span><span>Partial</span><span>Preserved</span></div></div><div class="result-block"><h4>Possible text changes <span>${changes.length} signals</span></h4>${evidenceHTML}</div><div class="uncertainty-note">${result.uncertainty?.calibrated === false ? 'Model uncertainty is not calibrated. These rule-based markers can miss meaning changes or flag harmless paraphrases.' : 'Possible text-change indicators are clues for review, not a legal explanation.'}</div>`;
}

async function submitScore(event) {
  event.preventDefault();
  const token = $('#token-input').value.trim();
  const errorEl = $('#score-error');
  errorEl.classList.add('hidden');
  if (token.length < 32) { errorEl.textContent = 'Enter the local API token (at least 32 ASCII characters) in the settings below the form.'; errorEl.classList.remove('hidden'); return; }
  const input = {
    id: `web-${Date.now()}`,
    language: $('#language-input').value,
    original: $('#original-input').value.trim(),
    simplified: $('#rewrite-input').value.trim(),
    context: $('#context-input').value.trim(),
  };
  if (!input.original || !input.simplified) return;
  const button = $('#score-button');
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Scoring…';
  try {
    const response = await fetch('/v1/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ pairs: [input] }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `Scoring request failed (${response.status})`);
    renderScore(body.predictions?.[0] || {}, input, body.model_origin);
  } catch (error) {
    errorEl.textContent = error.message.includes('fetch') ? 'Cannot reach the local scoring service. Start the backend and reload this page.' : error.message;
    errorEl.classList.remove('hidden');
  } finally {
    button.disabled = false;
    button.innerHTML = 'Score this pair <span>→</span>';
  }
}

function wireUI() {
  $$('.nav-link').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));
  $$('[data-open-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.openView)));
  $$('.segment').forEach(button => button.addEventListener('click', () => {
    state.language = button.dataset.language;
    $$('.segment').forEach(item => item.classList.toggle('active', item === button));
    if (state.overview) renderMetrics(state.overview.report.internal_test || {});
  }));
  $('#case-language').addEventListener('change', event => { state.caseLanguage = event.target.value; renderCases(); });
  for (const [buttonId, contentId] of [['diagnostics-toggle', 'diagnostics-content'], ['trials-toggle', 'trials-content'], ['raw-toggle', 'raw-content']]) {
    $(`#${buttonId}`).addEventListener('click', event => {
      const button = event.currentTarget;
      button.classList.toggle('open');
      $(`#${contentId}`).classList.toggle('hidden');
    });
  }
  $('#refresh-button').addEventListener('click', loadOverview);
  $('#score-form').addEventListener('submit', submitScore);
  $('#clear-form').addEventListener('click', () => {
    $('#score-form').reset();
    $('#score-result').classList.add('hidden');
    $('#empty-result').classList.remove('hidden');
    $('#score-error').classList.add('hidden');
  });
  const initial = location.hash.slice(1);
  if (['overview', 'evaluate', 'challenge'].includes(initial)) showView(initial);
  window.addEventListener('hashchange', () => {
    const next = location.hash.slice(1);
    if (['overview', 'evaluate', 'challenge'].includes(next)) showView(next);
  });
}

wireUI();
loadOverview();
