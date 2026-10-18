/* Smoke Sentinel demo UI.
 *
 * This replays a pack built by scripts/build_demo_pack.py. Nothing is computed
 * here and nothing is invented here: every probability, every agent decision and
 * every trace line comes from the committed evaluation artefacts, so what the
 * screen shows is what the reported numbers were computed from.
 */
'use strict';

const PACK_ROOT = 'pack';
const FPS = 4;

const el = (id) => document.getElementById(id);
const state = { index: null, pack: null, i: 0, timer: null };

async function boot() {
  state.index = await (await fetch(`${PACK_ROOT}/index.json`)).json();
  el('thrval').textContent = state.index.threshold.toFixed(3);
  el('thrmark').style.left = `${pct(state.index.threshold)}%`;

  const bar = el('seqbar');
  state.index.sequences.forEach((s, n) => {
    const b = document.createElement('button');
    b.innerHTML = `${shortName(s.sequence)}<span>${s.detector_alerts} alerts &middot; ` +
      `${s.agent_escalations} escalated &middot; ${s.agent_suppressions} suppressed</span>`;
    b.onclick = () => load(n);
    bar.appendChild(b);
  });
  await load(0);
}

/* "20190924_FIRE_sm-n-mobo-c" -> "sm-n  2019-09-24" : the camera and the date,
 * which is what an operator would actually read. */
function shortName(seq) {
  const m = seq.match(/^(\d{4})(\d{2})(\d{2})_(?:.*?_)?([a-z0-9-]+?)-mobo-c$/i);
  if (!m) return seq;
  return `${m[4]} &nbsp; ${m[1]}-${m[2]}-${m[3]}`;
}

const pct = (p) => Math.max(0, Math.min(100, p * 100));

async function load(n) {
  stop();
  const seq = state.index.sequences[n].sequence;
  state.pack = await (await fetch(`${PACK_ROOT}/${seq}/pack.json`)).json();
  [...el('seqbar').children].forEach((b, k) => b.classList.toggle('on', k === n));

  const frames = state.pack.frames;
  el('scrub').max = frames.length - 1;
  buildTrack(frames);

  // Preload so headless recording never catches a blank frame.
  frames.forEach((f) => { const im = new Image(); im.src = `${PACK_ROOT}/${seq}/${f.image}`; });

  // Start a little before ignition: the interesting part is the transition.
  const ign = frames.findIndex((f) => f.label === 1);
  show(Math.max(0, (ign < 0 ? 0 : ign) - 10));
}

function cellColour(f, thr) {
  if (f.prob === null) return '#1b232d';            // no reference / nothing to score
  if (f.prob < thr) return '#243244';               // below the alert threshold
  if (f.agent && f.agent.action === 'suppress') return '#8a6bd8';
  return '#3fb6ff';
}

function buildTrack(frames) {
  const t = el('track');
  t.innerHTML = '';
  const w = 100 / frames.length;
  frames.forEach((f, k) => {
    const d = document.createElement('div');
    d.className = 'cell';
    d.style.left = `${k * w}%`;
    d.style.width = `${w + 0.12}%`;
    d.style.background = cellColour(f, state.pack.threshold);
    d.title = `${f.offset_s} s`;
    t.appendChild(d);
  });
  const ign = frames.findIndex((f) => f.label === 1);
  if (ign >= 0) {
    const m = document.createElement('div');
    m.className = 'ign';
    m.style.left = `${ign * w}%`;
    t.appendChild(m);
  }
  const cur = document.createElement('div');
  cur.className = 'cur';
  cur.id = 'curmark';
  t.appendChild(cur);
}

function signed(s) { return (s >= 0 ? '+' : '-') + Math.abs(s); }

function show(i) {
  const frames = state.pack.frames;
  state.i = Math.max(0, Math.min(frames.length - 1, i));
  const f = frames[state.i];
  const seq = state.pack.sequence;
  const thr = state.pack.threshold;

  el('shot').src = `${PACK_ROOT}/${seq}/${f.image}`;
  el('scrub').value = state.i;
  el('curmark').style.left = `${(state.i / frames.length) * 100}%`;

  const post = f.label === 1;
  el('tag').className = `tag ${post ? 'post' : 'pre'}`;
  el('tag').textContent = post ? 'plume visible' : 'before plume';
  el('clock').textContent = `t ${signed(f.offset_s)} s   frame ${state.i + 1}/${frames.length}`;

  // --- detector panel ---
  const p = f.prob;
  el('bar').style.width = p === null ? '0%' : `${pct(p)}%`;
  el('bar').style.background = p !== null && p >= thr ? 'var(--hot)' : 'var(--cool)';
  el('pval').textContent = p === null ? 'n/a' : p.toFixed(3);
  el('truth').textContent = `ground truth ${post ? 'smoke' : 'clear'}`;

  const v = el('verdict');
  const alert = p !== null && p >= thr;
  v.className = `verdict ${alert ? 'is-alert' : p === null ? '' : 'is-clear'}`;
  if (p === null) {
    el('vlbl').textContent = f.has_reference ? 'nothing to score' : 'warming up';
    el('vwhy').textContent = f.has_reference
      ? 'no region deviates enough from the reference to measure'
      : 'not enough history yet to build a clean reference';
  } else if (alert) {
    el('vlbl').textContent = 'ALERT';
    el('vwhy').textContent = `${post ? 'true positive' : 'false alarm'} · ` +
      `classical CV score ${f.cv_score.toFixed(3)}`;
  } else {
    el('vlbl').textContent = 'below threshold';
    el('vwhy').textContent = post
      ? 'missed: plume is visible but scores under the operating point'
      : 'correctly quiet';
  }

  renderAgent(f, post);
  renderTrace(f);
}

function renderAgent(f, post) {
  const card = el('esc');
  const a = f.agent;
  if (!a) {
    card.className = 'card esc none';
    el('ehead').textContent = 'No alert';
    el('emeta').textContent = '';
    el('eev').innerHTML = '';
    return;
  }
  const escalated = a.action === 'escalate';
  card.className = `card esc${escalated ? '' : ' sup'}`;
  el('ehead').textContent = escalated
    ? `SMOKE SUSPECTED — ${shortName(state.pack.sequence).replace(/&nbsp;/g, ' ')}`
    : 'Suppressed before reaching the operator';
  el('emeta').textContent = `confidence ${a.confidence.toFixed(3)} · ` +
    `${a.reruns} re-examination${a.reruns === 1 ? '' : 's'}` +
    (a.corroborated ? ' · corroborated' : '');

  const ev = [];
  ev.push(`${escalated ? 'Escalating' : 'Suppressing'}: ${a.reason}.`);
  ev.push(`Candidate region in the band above the estimated horizon, ` +
    `classical score ${f.cv_score.toFixed(3)} at t ${signed(f.offset_s)} s.`);
  if (a.corroborated) {
    const nb = (state.pack.stats.neighbours || [])[0];
    ev.push(`A second camera${nb ? ` (${shortName(nb).replace(/&nbsp;/g, ' ')})` : ''} ` +
      `observed the same event independently.`);
  } else if (escalated) {
    ev.push('No neighbouring camera confirmed it; confidence alone carried the escalation.');
  }
  if (a.guardrail) ev.push(`Guardrail overrode the policy: ${a.guardrail}.`);
  if (!escalated && post) {
    ev.push('Ground truth says this was real smoke. This suppression is a miss, ' +
      'and is counted in the 9.3% recall cost.');
  }
  el('eev').innerHTML = ev.map((t) => `<li>${t}</li>`).join('');
}

function fmtDetail(d) {
  return Object.entries(d).map(([k, v]) =>
    `${k}=${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}`).join('  ');
}

function renderTrace(f) {
  const ol = el('trace');
  if (!f.agent) {
    ol.innerHTML = '<li class="empty">Below the alert threshold, so no verification ran. ' +
      'The agent only sees candidates the detector already flagged.</li>';
    return;
  }
  ol.innerHTML = f.agent.trace.map((t) =>
    `<li><span class="k ${t.kind}">${t.kind.replace('_', ' ')}</span>` +
    `<span class="n">${t.name}</span>` +
    `<span class="d">${fmtDetail(t.detail)}</span></li>`).join('');
  ol.scrollTop = 0;
}

function stop() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  el('play').textContent = 'Play';
}

function play() {
  if (state.timer) return stop();
  if (state.i >= state.pack.frames.length - 1) show(0);
  el('play').textContent = 'Pause';
  state.timer = setInterval(() => {
    if (state.i >= state.pack.frames.length - 1) return stop();
    show(state.i + 1);
  }, 1000 / FPS);
}

el('play').onclick = play;
el('scrub').oninput = (e) => { stop(); show(+e.target.value); };
el('toign').onclick = () => {
  stop();
  const k = state.pack.frames.findIndex((f) => f.label === 1);
  show(k < 0 ? 0 : k);
};
document.addEventListener('keydown', (e) => {
  if (e.key === ' ') { e.preventDefault(); play(); }
  if (e.key === 'ArrowRight') { stop(); show(state.i + 1); }
  if (e.key === 'ArrowLeft') { stop(); show(state.i - 1); }
});

/* Hooks the Playwright recorder drives, so the recording does not depend on
 * clicking pixel coordinates that could shift. */
window.sentinel = {
  select: (n) => load(n),
  goto: (i) => { stop(); show(i); },
  gotoIgnition: () => el('toign').click(),
  play, stop,
  firstAlert: (opts) => {
    const want = (opts || {}).action;
    const k = state.pack.frames.findIndex((f) => f.agent && (!want || f.agent.action === want));
    if (k >= 0) { stop(); show(k); }
    return k;
  },
  frames: () => state.pack.frames.length,
  at: () => state.i,
};

boot();
