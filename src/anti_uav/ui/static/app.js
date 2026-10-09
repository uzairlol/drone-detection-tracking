/* anti-uav operator console
 *
 * No framework, no bundler. Every request goes to the local API and every value
 * comes from disk via the API, so the console cannot drift from the pipeline.
 *
 * Two presentation rules this file follows on purpose:
 *
 *   1. A metric is never shown without its caveat. "precision 0.98" on a dataset
 *      with no birds is a meaningless number, and presenting it bare is worse
 *      than not showing it.
 *   2. Missing data says "missing", not "0". A run that has not trained has no
 *      mAP, and rendering that as 0.0000 invites a comparison that is not real.
 */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  health: null,
  runs: [],
  datasets: [],
  matrix: null,
  runDetail: null,
  charts: {},
};

// --------------------------------------------------------------------- utils

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || JSON.stringify(payload);
    } catch (_) { /* not json */ }
    throw new Error(detail);
  }
  return response.json();
}

const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
};

/** '-' for absent, never '0.0000' - a missing metric is not a zero metric. */
const num = (value, digits = 4) =>
  value === null || value === undefined || Number.isNaN(value) ? '–' : Number(value).toFixed(digits);

const int = (value) =>
  value === null || value === undefined ? '–' : Number(value).toLocaleString();

const pill = (text, kind) => el('span', { class: `pill ${kind}` }, text);

function table(headers, rows) {
  const thead = el('thead', {}, el('tr', {}, headers.map((h) =>
    el('th', typeof h === 'string' ? {} : h, typeof h === 'string' ? h : h.label))));
  const tbody = el('tbody', {}, rows);
  return el('table', {}, thead, tbody);
}

function alertBox(lines, kind = 'warn') {
  return el('div', { class: `alert-box ${kind}` },
    lines.map((line) => el('div', {}, line)));
}

// ------------------------------------------------------------------ overview

async function loadHealth() {
  try {
    const health = await api('/api/health');
    state.health = health;
    $('#version').textContent = 'v' + health.version;

    const cards = [
      card('status', health.status, health.status === 'ok' ? 'ok' : 'warn'),
      card('profile', health.profile || '–', ''),
      card('gpu', health.gpu || 'none', health.gpu ? 'ok' : 'warn'),
      card('cuda', health.cuda_available ? 'available' : 'unavailable',
        health.cuda_available ? 'ok' : 'warn'),
      card('datasets', int(health.datasets_present.length) + ' / ' + int(state.datasets.length || 4), ''),
      card('runs', int(health.runs), ''),
    ];
    $('#health-cards').replaceChildren(...cards);

    const lines = [`interpreter : ${health.status === 'ok' ? '' : ''}anti-uav v${health.version}`];
    lines.push(`profile     : ${health.profile}`);
    lines.push(`gpu         : ${health.gpu || 'none detected'}`);
    lines.push(`converted   : ${health.datasets_present.join(', ') || 'none yet'}`);
    if (health.warnings.length) {
      lines.push('');
      lines.push('not converted yet:');
      health.warnings.forEach((w) => lines.push('  - ' + w));
    }
    $('#health-detail').textContent = lines.join('\n');
  } catch (err) {
    $('#health-detail').textContent = 'could not reach the API: ' + err.message;
  }
}

const card = (k, v, kind) => el('div', { class: `card ${kind || ''}` },
  el('div', { class: 'k' }, k), el('div', { class: 'v' }, v));

// ---------------------------------------------------------------------- runs

async function loadRuns() {
  try {
    state.runs = await api('/api/runs');
  } catch (err) {
    $('#runs-table').replaceChildren(alertBox(['could not load runs: ' + err.message]));
    return;
  }

  if (!state.runs.length) {
    $('#runs-table').replaceChildren(
      alertBox(['no runs yet. Train one with: anti-uav train --model yolo11n --combo dvb'], 'warn'));
    return;
  }

  const rows = state.runs.map((run) => {
    const m = run.best_metrics || {};
    return el('tr', {},
      el('td', {}, el('a', {
        href: '#', onclick: (e) => { e.preventDefault(); showRun(run.run_name); },
      }, run.run_name)),
      el('td', {}, run.model || '–'),
      el('td', {}, run.combo || '–'),
      el('td', { class: 'num' }, run.profile || '–'),
      el('td', { class: 'num' }, int(run.epochs_completed) + ' / ' + int(run.epochs_requested)),
      el('td', { class: 'num' }, num(m['metrics/mAP50(B)'])),
      el('td', { class: 'num' }, num(m['metrics/mAP50-95(B)'])),
      el('td', { class: 'num' }, num(m['metrics/precision(B)'])),
      el('td', { class: 'num' }, num(m['metrics/recall(B)'])),
      el('td', {}, run.has_weights ? pill('yes', 'ok') : pill('none', 'none')),
      el('td', { class: 'num' }, run.duration_s ? (run.duration_s / 60).toFixed(1) + 'm' : '–'),
    );
  });

  $('#runs-table').replaceChildren(table(
    ['run', 'model', 'combo', 'profile', 'epochs', 'mAP50', 'mAP50-95', 'prec', 'recall', 'best.pt', 'time'],
    rows));

  fillRunSelectors();
}

async function showRun(runName) {
  try {
    const detail = await api('/api/runs/' + encodeURIComponent(runName));
    state.runDetail = detail;
    $('#run-select').value = runName;

    const parts = [];
    parts.push(`combo        : ${detail.combo}`);
    parts.push(`model        : ${detail.model}`);
    parts.push(`profile      : ${detail.profile}  (imgsz ${detail.imgsz}, batch ${detail.batch}, amp ${detail.amp})`);
    if (detail.best_epoch !== null && detail.best_epoch !== undefined) {
      parts.push(`best epoch   : ${detail.best_epoch}`);
    }
    const env = detail.environment || {};
    if (env.torch) parts.push(`torch        : ${env.torch} (CUDA ${env.torch_cuda || 'cpu'})`);
    if (env.devices && env.devices.length) {
      parts.push(`gpu          : ${env.devices.map((d) => `${d.name} sm=${d.capability}`).join(', ')}`);
    }
    if (detail.best_weights) parts.push(`best weights : ${detail.best_weights}`);
    parts.push('');
    parts.push(`download     : /api/runs/${encodeURIComponent(runName)}/weights`);

    const children = [el('h2', {}, runName), el('pre', {}, parts.join('\n'))];

    if (detail.curve && detail.curve.length) {
      children.push(el('h3', {}, 'Training curve'));
      children.push(el('div', { class: 'chart-box' }, el('canvas', { id: 'curve-chart' })));
    }
    if (detail.warnings && detail.warnings.length) {
      children.push(alertBox(detail.warnings, 'warn'));
    }
    $('#run-detail').replaceChildren(...children);
    if (detail.curve && detail.curve.length) drawCurve(detail.curve);
  } catch (err) {
    $('#run-detail').replaceChildren(alertBox(['could not load run: ' + err.message]));
  }
}

function drawCurve(curve) {
  const canvas = document.getElementById('curve-chart');
  if (!canvas) return;
  if (state.charts.curve) state.charts.curve.destroy();

  const epochs = curve.map((p) => p.epoch);
  const series = [
    { key: 'metrics/mAP50-95(B)', label: 'mAP50-95', color: '#4c9aff' },
    { key: 'metrics/mAP50(B)', label: 'mAP50', color: '#3fb950' },
    { key: 'metrics/precision(B)', label: 'precision', color: '#d29922' },
    { key: 'metrics/recall(B)', label: 'recall', color: '#a371f7' },
    { key: 'train/box_loss', label: 'box loss', color: '#f85149' },
  ].filter((s) => curve.some((p) => Number.isFinite(p.metrics[s.key])));

  state.charts.curve = new Chart(canvas, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: series.map((s) => ({
        label: s.label,
        data: curve.map((p) => (Number.isFinite(p.metrics[s.key]) ? p.metrics[s.key] : null)),
        borderColor: s.color,
        backgroundColor: s.color + '22',
        borderWidth: 1.6,
        pointRadius: 0,
        tension: 0.2,
        spanGaps: true,
        yAxisID: s.label.includes('loss') ? 'y1' : 'y',
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { title: { display: true, text: 'epoch' }, grid: { color: '#222' } },
        y: { position: 'left', min: 0, max: 1, grid: { color: '#222' } },
        y1: { position: 'right', grid: { drawOnChartArea: false } },
      },
      plugins: {
        legend: { labels: { color: '#d6deeb', boxWidth: 10 } },
        tooltip: { callbacks: { title: (items) => 'epoch ' + items[0].label } },
      },
    },
  });
}

// -------------------------------------------------------------------- matrix

async function loadMatrix() {
  try {
    const matrix = await api('/api/matrix');
    state.matrix = matrix;

    const birdless = new Set(state.datasets.filter((d) => !d.has_bird_negatives).map((d) => d.alias));

    const rows = matrix.cells.map((cell) => {
      const sources = cell.combo.split('+');
      const noBirds = sources.every((s) => birdless.has(s));
      return el('tr', {},
        el('td', {}, cell.model),
        el('td', {}, cell.combo),
        el('td', { class: 'num' }, int(cell.epochs_completed) || '–'),
        el('td', { class: 'num' }, num(cell.map50)),
        el('td', { class: 'num' }, num(cell.map50_95)),
        el('td', { class: 'num' }, num(cell.precision)),
        el('td', { class: 'num' }, num(cell.recall)),
        el('td', {},
          noBirds
            ? pill('no bird negatives', 'warn')
            : pill('bird negatives', 'ok')),
        el('td', {}, cell.has_weights
          ? pill('ok', 'ok')
          : (cell.epochs_completed ? pill('no weights', 'warn') : pill('not run', 'none'))),
      );
    });

    const children = [table(
      ['model', 'combo', 'epochs', 'mAP50', 'mAP50-95', 'prec', 'recall', 'class coverage', 'status'],
      rows)];

    const unfilled = matrix.cells.filter((c) => !c.has_weights);
    if (unfilled.length) {
      children.push(el('h3', {}, 'Not yet run'));
      children.push(el('pre', {}, unfilled
        .map((c) => `anti-uav train --model ${c.model} --combo ${c.combo}`)
        .join('\n')));
    }
    $('#matrix-table').replaceChildren(...children);
  } catch (err) {
    $('#matrix-table').replaceChildren(alertBox(['could not load matrix: ' + err.message]));
  }
}

async function loadCross() {
  const run = $('#cross-run-select').value;
  if (!run) return;
  $('#cross-table').replaceChildren(el('p', { class: 'muted' }, 'evaluating...'));
  try {
    const results = await api('/api/runs/' + encodeURIComponent(run) + '/cross');
    const rows = Object.entries(results).map(([alias, result]) => {
      const meta = state.datasets.find((d) => d.alias === alias);
      const drone = (result.per_class && result.per_class.drone) || {};
      const small = ((result.per_class_and_scale || {}).drone || {}).small || {};
      return el('tr', {},
        el('td', {}, alias),
        el('td', { class: 'num' }, num(result.metrics.map50)),
        el('td', { class: 'num' }, num(result.metrics['map50-95'])),
        el('td', { class: 'num' }, num(result.metrics.precision)),
        el('td', { class: 'num' }, num(result.metrics.recall)),
        el('td', { class: 'num' }, num(small['map50-95'])),
        el('td', {}, meta && meta.has_bird_negatives
          ? pill('falsifiable', 'ok')
          : pill('drone-only', 'warn')),
      );
    });
    const notes = Object.values(results).flatMap((r) => r.notes || []);
    const children = [table(
      ['dataset', 'mAP50', 'mAP50-95', 'prec', 'recall', 'drone AP_S', 'falsifiable?'], rows)];
    if (notes.length) {
      children.push(alertBox(notes.map((n) => '• ' + n), 'warn'));
    }
    $('#cross-table').replaceChildren(...children);
  } catch (err) {
    $('#cross-table').replaceChildren(alertBox(['cross-eval failed: ' + err.message]));
  }
}

// ------------------------------------------------------------------ datasets

async function loadDatasets() {
  try {
    state.datasets = await api('/api/datasets');
  } catch (err) {
    $('#datasets-table').replaceChildren(alertBox(['could not load datasets: ' + err.message]));
    return;
  }

  const rows = state.datasets.map((d) => el('tr', {},
    el('td', {}, d.alias),
    el('td', {}, d.display_name),
    el('td', {}, d.modalities.join(' + ')),
    el('td', { class: 'num' }, d.median_target_px ? d.median_target_px.toFixed(0) : '–'),
    el('td', { class: 'num' }, int(d.approx_frames)),
    el('td', {}, d.has_bird_negatives ? pill('yes', 'ok') : pill('NO', 'bad')),
    el('td', { class: 'num' }, d.converted ? int(d.frames) : '–'),
    el('td', {}, d.converted ? pill('converted', 'ok') : pill('not converted', 'none')),
    el('td', {}, d.has_track_ids ? pill('MOT', 'ok') : '–'),
    el('td', {}, d.official_metric || '–'),
  ));

  const caveats = state.datasets
    .filter((d) => !d.has_bird_negatives)
    .map((d) => `${d.alias}: no bird annotations - precision on this dataset cannot falsify a false-positive claim`);

  const children = [table(
    ['alias', 'name', 'modality', 'median px', 'approx frames', 'bird neg.', 'converted', 'status', 'track ids', 'official metric'],
    rows)];
  if (caveats.length) children.push(alertBox(caveats, 'warn'));

  const sources = state.datasets.flatMap((d) =>
    d.sources.map((s) => `  ${d.alias} :: ${s.kind.padEnd(11)} ${s.url}\n` +
      `      ${s.approx_size_gb} GB${s.needs_credentials ? '  [needs credentials]' : ''}`));
  children.push(el('h3', {}, 'Acquisition sources'));
  children.push(el('pre', {}, sources.join('\n')));

  $('#datasets-table').replaceChildren(...children);

  const select = $('#stats-select');
  select.replaceChildren(el('option', { value: '' }, '-'),
    ...state.datasets.filter((d) => d.converted)
      .map((d) => el('option', { value: d.alias }, d.alias)));
  select.onchange = () => loadDatasetStats(select.value);
}

async function loadDatasetStats(alias) {
  if (!alias) { $('#stats-detail').textContent = 'select a dataset'; return; }
  try {
    const s = await api(`/api/datasets/${alias}/stats`);
    const lines = [
      `frames        : ${s.frames.toLocaleString()}  (empty ${s.empty_frames.toLocaleString()})`,
      `sequences     : ${s.sequences.toLocaleString()}`,
      `boxes         : ${s.boxes.toLocaleString()}`,
      `classes       : ${JSON.stringify(s.class_counts)}`,
      `bird seqs     : ${s.bird_sequences}  -> bird negatives: ${s.provides_bird_negatives ? 'yes' : 'NO'}`,
      `target size px: ${JSON.stringify(s.size_percentiles)}`,
      '',
      'box size distribution:',
      ...Object.entries(s.size_histogram || {}).map(([k, v]) => `  ${k.padEnd(12)} ${v}`),
    ];
    if (s.warnings && s.warnings.length) {
      lines.push('', 'WARNINGS', ...s.warnings.map((w) => '  ! ' + w));
    }
    $('#stats-detail').textContent = lines.join('\n');
  } catch (err) {
    $('#stats-detail').textContent = err.message;
  }
}

// ---------------------------------------------------------------- playground

function fillRunSelectors() {
  for (const id of ['#run-select', '#cross-run-select', '#pg-run']) {
    const select = $(id);
    if (!select) continue;
    const previous = select.value;
    select.replaceChildren(el('option', { value: '' }, '-'),
      ...state.runs.map((r) => el('option', { value: r.run_name }, r.run_name)));
    if (previous) select.value = previous;
  }
}

async function loadSamples() {
  const combo = $('#pg-combo').value;
  if (!combo) return;
  try {
    const data = await api(`/api/samples?limit=48&split=val&combo=${encodeURIComponent(combo)}`);
    const select = $('#pg-image');
    if (data.note) {
      select.replaceChildren(el('option', { value: '' }, data.note.slice(0, 60)));
      $('#pg-result').replaceChildren(el('p', { class: 'muted' }, data.note));
      return;
    }
    select.replaceChildren(...data.images.map((img) =>
      el('option', { value: img.path }, img.name)));
  } catch (err) {
    $('#pg-image').replaceChildren(el('option', { value: '' }, 'failed: ' + err.message));
  }
}

async function runPrediction() {
  const run = $('#pg-run').value;
  const image = $('#pg-image').value;
  if (!run || !image) { alert('pick a run and a sample'); return; }

  $('#pg-result').replaceChildren(el('p', { class: 'muted' }, 'running...'));
  try {
    const result = await api('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        run,
        image_path: image,
        tracker: $('#pg-tracker').value,
        conf: parseFloat($('#pg-conf').value),
        return_overlay: true,
      }),
    });

    const children = [];
    if (result.overlay_png_b64) {
      children.push(el('img', { src: 'data:image/png;base64,' + result.overlay_png_b64 }));
    }
    const side = [];
    side.push(el('div', { class: 'muted' },
      `${result.width}×${result.height}  ·  ${result.detections.length} detections  ·  ` +
      `${result.tracks.length} tracks  ·  ${result.elapsed_ms.toFixed(1)} ms`));
    side.push(el('h3', {}, 'Detections'));
    side.push(table(['class', 'conf', 'track', 'box'], result.detections.map((d) =>
      el('tr', {},
        el('td', {}, d.class_name || String(d.class_id)),
        el('td', { class: 'num' }, d.confidence.toFixed(3)),
        el('td', { class: 'num' }, d.track_id ?? '–'),
        el('td', { class: 'num' }, d.box.map((v) => Math.round(v)).join(', '))))));
    (result.notes || []).forEach((n) => side.push(alertBox([n], 'warn')));
    children.push(el('div', {}, ...side));
    $('#pg-result').replaceChildren(...children);
  } catch (err) {
    $('#pg-result').replaceChildren(alertBox(['inference failed: ' + err.message]));
  }
}

// --------------------------------------------------------------------- rules

async function loadRules() {
  try {
    const data = await api('/api/rules');
    $('#rules-json').textContent = JSON.stringify(data.rules, null, 2);
    if (data.problems && data.problems.length) {
      $('#rules-problems').replaceChildren(
        alertBox(['cross-rule problems:'].concat(data.problems.map((p) => '  • ' + p))));
    } else {
      $('#rules-problems').replaceChildren(
        alertBox(['no cross-rule problems found'], 'ok'));
    }
  } catch (err) {
    $('#rules-json').textContent = err.message;
  }
}

async function explainRule() {
  const body = {
    box: [400, 200, 424, 224],
    confidence: parseFloat($('#re-conf').value),
    hits: parseInt($('#re-hits').value, 10),
    duration_s: parseFloat($('#re-dur').value),
    camera_id: $('#re-cam').value,
    image_height_px: 1080,
    global_id: parseInt($('#re-gid').value, 10) || null,
    velocity_m_s: [parseFloat($('#re-speed').value), 0],
    ground_xy: [300, 300],
    ground_z: 25,
    sightings: {
      [$('#re-gid').value || 7]: [
        { camera_id: 'fixed-001', node: 'node-00', t: 1.0, ground_xy: [300, 300] },
        { camera_id: 'fixed-040', node: 'node-00', t: 1.0, ground_xy: [301, 300] },
      ],
    },
  };
  try {
    const result = await api('/api/rules/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const header = alertBox(
      [result.alerted
        ? `ALERTED at severity "${result.severity}"`
        : `not alerted - first failing gate: ${result.first_failure || 'none'}`],
      result.alerted ? 'ok' : '');
    const gates = result.gates.map((g) => el('div', { class: `gate ${g.outcome}` },
      el('div', { class: 'name' }, `[${g.outcome}] ${g.name}`),
      g.detail ? el('div', { class: 'detail' }, g.detail) : null,
      Object.keys(g.measured).length
        ? el('div', { class: 'measured' },
          Object.entries(g.measured).map(([k, v]) => `${k}=${v}`).join('  '))
        : null));
    $('#re-result').replaceChildren(header, ...gates);
  } catch (err) {
    $('#re-result').replaceChildren(alertBox([err.message]));
  }
}

// ------------------------------------------------------------------ coverage

async function loadCoverage() {
  try {
    const data = await api('/api/coverage');
    const s = data.summary;
    const cards = [
      card('cameras', int(s.cameras), ''),
      card('fixed / PTZ', `${s.fixed} / ${s.ptz}`, ''),
      card('worker nodes', int(s.nodes), ''),
      card('overlap zones', int(s.overlap_zones), ''),
      card('protected zones', int(s.protected_zones), ''),
      card('uncovered', int(s.uncovered_zones.length),
        s.all_zones_covered ? 'ok' : 'bad'),
    ];
    const children = [el('div', { class: 'grid' }, ...cards)];

    if (!s.all_zones_covered) {
      children.push(alertBox(
        [`uncovered zones: ${s.uncovered_zones.join(', ')}`,
         'the coverage scheduler will refuse to move the last camera covering a zone'],
        'warn'));
    }

    children.push(el('h3', {}, 'Cameras per node'));
    children.push(table(['node', 'cameras'], Object.entries(s.cameras_per_node).map(([node, count]) =>
      el('tr', {}, el('td', {}, node), el('td', { class: 'num' }, count)))));

    children.push(el('h3', {}, 'Protected zones'));
    children.push(table(['id', 'priority', 'height m', 'description'],
      data.protected_zones.map((z) => el('tr', {},
        el('td', {}, z.id),
        el('td', { class: 'num' }, z.priority),
        el('td', { class: 'num' }, z.height_m),
        el('td', {}, z.description || '')))));

    children.push(el('h3', {}, 'Geofences'));
    children.push(table(['id', 'action', 'height m'], data.geofences.map((g) => el('tr', {},
      el('td', {}, g.id),
      el('td', {}, pill(g.action, g.action === 'suppress' ? 'bad' : 'warn')),
      el('td', { class: 'num' }, g.height_m)))));

    $('#coverage-summary').replaceChildren(...children);
  } catch (err) {
    $('#coverage-summary').replaceChildren(alertBox(['could not load coverage: ' + err.message]));
  }
}

// ---------------------------------------------------------------- bootstrap

function initTabs() {
  $$('nav button').forEach((button) => {
    button.addEventListener('click', () => {
      $$('nav button').forEach((b) => b.classList.remove('active'));
      $$('.tab').forEach((t) => t.classList.remove('active'));
      button.classList.add('active');
      $('#' + button.dataset.tab).classList.add('active');
      if (button.dataset.tab === 'coverage') loadCoverage();
    });
  });
}

async function init() {
  initTabs();
  await loadHealth();
  await loadDatasets();
  await loadHealth();          // now knows the dataset count
  await loadRuns();
  await loadMatrix();
  await loadRules();

  const comboSelect = $('#pg-combo');
  if (state.matrix) {
    comboSelect.replaceChildren(...state.matrix.combos.map((c) => el('option', { value: c }, c)));
  }
  comboSelect.onchange = loadSamples;

  $('#run-select').onchange = (e) => e.target.value && showRun(e.target.value);
  $('#cross-load').onclick = loadCross;
  $('#re-eval').onclick = explainRule;
  $('#pg-run-btn').onclick = runPrediction;

  loadSamples();
}

document.addEventListener('DOMContentLoaded', init);
