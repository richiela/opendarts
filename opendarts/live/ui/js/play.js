// ---------------------------------------------------------------------------
// play -- the scoreboard. Read from the oche, 2.4 m away.
// ---------------------------------------------------------------------------

// The rig's one status, derived in one place: the status chip, the board
// veil and the session sheet all read this, so they can never disagree.
// Every branch is assigned from the facts, so any state can move both ways.
const TRIGGER_DETAIL = {
  IDLE: (n) => (n === null ? 'Waiting for a dart' : 'Waiting for dart ' + (n + 1)),
  MOTION_DETECTED: () => 'Dart in flight',
  SETTLING: () => 'Settling',
  READY_TO_CAPTURE: () => 'Scoring',
  TAKEOUT_WAITING: (n) => (n === null ? 'Take the darts out' : plural(n, 'dart') + ' on the board'),
};

function rigPhase() {
  if (!S.connected) {
    return S.everConnected
      ? {key: 'offline', tone: 'bad', label: 'Offline', detail: 'Can’t reach the rig — retrying'}
      : {key: 'connecting', tone: 'idle', label: 'Connecting', detail: 'Reaching the rig…'};
  }
  const trig = S.trigger || {};
  const cap = S.capture;
  if (trig.available === false) {
    return {key: 'nocapture', tone: 'idle', label: 'No capture', detail: 'This process has no capture loop — it only serves recorded throws.'};
  }
  if (cap && !cap.running) {
    let detail = 'Cameras are off.';
    if (S.startError) detail = 'The last start failed: ' + S.startError;
    else if (S.idleStopped) detail = 'Stopped itself after ' + duration(S.idleStopped.sec) + ' without a dart.';
    return {key: 'stopped', tone: 'idle', label: 'Stopped', detail};
  }
  // A calibration anyone started (this page, another, or Start itself) is
  // the phase, while it runs.
  if (S.calibrating || (S.calibProgress && S.calibProgress.active)) return {key: 'calibrating', tone: 'busy', label: 'Calibrating', detail: 'Keep the board clear — this takes about half a minute.'};
  if (cap && cap.starting) return {key: 'starting', tone: 'busy', label: 'Starting', detail: 'Opening the cameras and checking calibration…'};
  if (!trig.state) return {key: 'warming', tone: 'busy', label: 'Warming up', detail: 'Waiting for the first frames…'};
  const n = trig.dart_count === undefined || trig.dart_count === null ? null : trig.dart_count;
  const fn = TRIGGER_DETAIL[trig.state];
  if (trig.state === 'TAKEOUT_WAITING') return {key: 'takeout', tone: 'warn', label: 'Remove darts', detail: fn(n)};
  return {key: 'ready', tone: 'live', label: 'Ready', detail: fn ? fn(n) : trig.state};
}

// A throw as the scoreboard reads it: the corrected call when a person
// gave one, otherwise the engine's.
function effectiveCall(t) {
  if (!t) return null;
  if (t.corrected_ring) return {sector: t.corrected_sector, ring: t.corrected_ring, corrected: true};
  return {sector: t.sector, ring: t.ring, corrected: false};
}

const Play = (() => {
  const rows = $$('#turn .dart');
  let recent = [];

  function renderTurn() {
    const throws = S.visit.throws || [];
    let total = 0, scored = 0;
    rows.forEach((row, i) => {
      const t = throws[i];
      const call = effectiveCall(t);
      const unscored = !!t && (t.ok === false || !t.ring) && !(call && call.corrected);
      row.classList.toggle('filled', !!t);
      row.classList.toggle('unscored', unscored);
      row.classList.toggle('corrected', !!(call && call.corrected));
      row.disabled = !t;
      const callEl = $('.call', row), ptsEl = $('.pts', row), subEl = $('.sub', row);
      row.classList.toggle('fresh', !!t && i === throws.length - 1 && throws.length < 3);
      if (!t) {
        callEl.textContent = 'No dart yet'; ptsEl.textContent = ''; subEl.textContent = '';
        row.removeAttribute('aria-label');
        return;
      }
      if (unscored) {
        callEl.textContent = 'NO READ';
        ptsEl.textContent = '';
        subEl.textContent = 'Tap — what landed?';
        row.setAttribute('aria-label', 'Dart ' + (i + 1) + ': not scored. Tap to correct.');
        return;
      }
      const pts = points(call.sector, call.ring);
      callEl.textContent = callLabel(call.sector, call.ring);
      ptsEl.textContent = pts === null ? '' : String(pts);
      subEl.textContent = call.corrected ? 'Corrected' : '';
      if (pts !== null) { total += pts; scored++; }
      row.setAttribute('aria-label', 'Dart ' + (i + 1) + ': ' + callLabel(call.sector, call.ring)
        + ', ' + pts + ' points. Tap to correct.');
    });
    const tot = $('#turn-total');
    tot.textContent = String(total);
    $('#total-slab').classList.toggle('filled', throws.length > 0);
    $('#turn-count').textContent = throws.length ? plural(Math.min(throws.length, 3), 'DART', 'DARTS') : 'NO DARTS YET';
    $('#turn-note').textContent = throws.length > 3 ? plural(throws.length, 'dart') + ' recorded this turn — showing the first 3' : '';
  }

  // Earlier visits run along the bottom like a broadcast ticker, newest first.
  function renderRecent() {
    const list = $('#recent-turns');
    const turns = recent.slice(-8).reverse();
    if (!turns.length) { list.replaceChildren(h('li', {class: 'empty'}, 'NO EARLIER VISITS YET')); return; }
    list.replaceChildren(...turns.map((v) => h('li', {class: 'recent'},
      h('b', {}, String(v.total)),
      (v.darts || []).map((d) => (d.ring ? callLabel(d.sector === 0 ? null : d.sector, d.ring) : '?')).join(' · '),
      h('small', {}, clock(v.completed_at_utc, false)),
    )));
  }

  async function loadRecent() {
    try {
      const body = await api.get('/api/live/recent?limit=8');
      if (body && Array.isArray(body.visits)) { recent = body.visits; renderRecent(); }
    } catch (err) { /* the history is a courtesy; the turn never depends on it */ }
  }

  function renderVeil() {
    const p = rigPhase();
    const veil = $('#board-veil');
    const show = p.key !== 'ready' && p.key !== 'takeout' && p.key !== 'warming';
    veil.hidden = !show;
    $('#play').dataset.phase = p.key;
    $('#play').classList.toggle('veiled', show);
    if (!show) return;
    veil.dataset.tone = p.tone;
    veil.dataset.key = p.key;
    $('#veil-kicker').textContent = {offline: 'NO SIGNAL', nocapture: 'OFF AIR', stopped: 'OFF AIR'}[p.key] || 'STAND BY';
    $('#veil-title').textContent = {
      offline: 'Can’t reach the rig', connecting: 'Connecting', nocapture: 'No capture here',
      stopped: 'Scoring is off', starting: 'Starting', calibrating: 'Calibrating',
    }[p.key] || p.label;
    $('#veil-detail').textContent = p.detail;
    const btn = $('#veil-start');
    btn.hidden = p.key !== 'stopped';
    btn.disabled = !!Session.busy;
    $('#veil-spin').hidden = !(p.tone === 'busy' || p.key === 'connecting' || p.key === 'offline');
  }

  function renderMissed() {
    // Only offered where it can work: a buffer that exists and holds frames.
    const fr = S.runtime && S.runtime.frame_ring;
    const ok = !!(fr && fr.attached && fr.configured_seconds > 0) && rigPhase().key !== 'stopped';
    $('#missed-open').hidden = !ok;
  }

  // Tap a dart -> say what actually landed.
  rows.forEach((row, i) => row.addEventListener('click', () => {
    const t = (S.visit.throws || [])[i];
    if (!t) return;
    const pkg = Throws.find(t.session, t.throw_id);
    const cands = pkg ? Throws.candidatesFor(pkg) : [{sector: t.sector, ring: t.ring, who: 'Scored', source: 'manual', xy: t.board_xy_mm, primary: true}];
    // The visit's total as the scoreboard shows it, so the sheet can say
    // what a correction would make it.
    let visitTotal = 0;
    (S.visit.throws || []).slice(0, 3).forEach((tt) => { const c = effectiveCall(tt); visitTotal += (c && points(c.sector, c.ring)) || 0; });
    const cur = effectiveCall(t);
    Truth.open({
      visitId: t.visit_id || S.visit.id, index: i, session: t.session, throwId: t.throw_id,
      title: 'Dart ' + (i + 1), fromPlay: true, when: t.captured_at_utc ? clock(t.captured_at_utc) : '',
      visitTotal, scoredPts: (cur && points(cur.sector, cur.ring)) || 0,
      scored: {sector: t.sector, ring: t.ring},
      candidates: cands,
      confirmed: t.corrected_ring ? {sector: t.corrected_sector, ring: t.corrected_ring} : null,
    });
  }));

  // ---- missed a dart --------------------------------------------------------
  function openMissed() {
    const fr = (S.runtime && S.runtime.frame_ring) || {};
    const ring = fr.ring || {};
    $('#missed-reach').textContent = ring.span_s
      ? 'The buffer holds the last ' + Math.round(ring.span_s) + ' seconds — save it before the moment scrolls out.'
      : 'The buffer keeps the last ' + (fr.configured_seconds || '?') + ' seconds of every camera.';
    $('#missed-reason').value = '';
    $('#missed-error').textContent = '';
    sheet.open('sheet-missed');
    setTimeout(() => $('#missed-reason').focus(), 50);
  }
  async function saveMissed() {
    const why = ($('#missed-reason').value || '').trim() || 'a dart landed and nothing was scored';
    const btn = $('#missed-save');
    btn.disabled = true;
    const act = activity('Missed dart', 'pending', 'saving the buffer…');
    const body = await api.post('/api/frame-ring/capture-missed', {reason: why}).catch(() => null);
    btn.disabled = false;
    if (!body || !body.ok) {
      $('#missed-error').textContent = (body && body.reason) || 'Could not reach the rig.';
      act.done('bad', (body && body.reason) || 'request failed');
      return;
    }
    act.done('ok', 'writing ' + (body.job ? body.job.mb_total + ' MB' : 'the buffer'));
    toast('Saving the last few seconds of every camera', 'ok');
    sheet.close('sheet-missed');
    Rig.refreshConfig();
  }
  $('#missed-open').addEventListener('click', openMissed);
  $('#missed-save').addEventListener('click', saveMissed);
  $('#veil-start').addEventListener('click', () => Session.start());

  // Under the board, who scored it: the primary engine, its voters, the
  // cameras. The rig's credits, the way a broadcast names its team.
  function renderCredits() {
    const ec = S.engineConfig || {};
    const n = (ec.also_run || []).length;
    $('#board-credits').textContent = ec.primary
      ? 'Scored by ' + ec.primary + (n ? ' · ' + plural(n, 'engine') + ' voting' : '') + ' · ' + plural(CAM_IDS.length, 'camera')
      : '';
  }

  function render() { renderTurn(); renderVeil(); renderMissed(); LiveBoard.draw(); }

  on('visit', render);
  on('engines', renderCredits);
  on('phase', () => { renderVeil(); renderMissed(); });
  on('config', renderMissed);
  on('landed', (i) => {
    const row = rows[i];
    if (row) { row.classList.remove('arrive'); void row.offsetWidth; row.classList.add('arrive'); }
    LiveBoard.landed(i);
  });
  on('turnended', loadRecent);
  on('connected', loadRecent);

  // Side by side, or darts above the board: decided by Play's own box, so a
  // phone, a tablet on a stand and the half of a split display agree.
  const stage = $('#play');
  function shape() {
    const r = stage.getBoundingClientRect();
    if (!r.width || !r.height) return;
    stage.classList.toggle('stacked', r.width < 760 || r.width / r.height < 1);
  }
  if (window.ResizeObserver) new ResizeObserver(shape).observe(stage);
  window.addEventListener('resize', shape);

  return {render, loadRecent, openMissed, shape};
})();
