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
  const hold = heldBy();
  if (hold) return hold;
  return {key: 'ready', tone: 'live', label: 'Ready', detail: fn ? fn(n) : trig.state};
}

// The oracle's one status -- Autodarts' board, as every indicator shows it
// (bar readout, Play, a display in any layout) and as its sheet explains it.
// Grey: not in play (off / no listener / not connected). Red: its board is
// stopped. Yellow: takeout; blinking once it has been stuck there
// ORACLE_STUCK_S. Green: ready and scoring alongside.
const ORACLE_STUCK_S = 30;
function adBoardAge() {
  const b = S.adBoard || {};
  return typeof b.since === 'number' ? Math.max(0, (Date.now() - b.since) / 1000) : null;
}
function oracleState() {
  const ad = S.ad || (S.runtime && S.runtime.ad);
  const cfg = S.config;
  const b = S.adBoard || {};
  const off = (key, word, label, detail) => ({key, tone: 'off', word, label, detail, blink: false, live: false});
  if (!S.connected) return off('unknown', '—', 'Autodarts: unknown', 'OpenDarts itself cannot be reached, so neither can its oracle.');
  if (cfg && cfg.ad_enabled === false) return off('off', 'OFF', 'Autodarts is off', 'Switched off in Rig › Autodarts — never contacted.');
  if (ad && ad.available === false) return off('none', 'NONE', 'No Autodarts here', (ad && ad.reason) || 'This rig has no Autodarts listener.');
  if (!ad || !ad.connected) {
    return off('offline', 'OFFLINE', 'Autodarts is offline',
      'OpenDarts cannot reach it' + (cfg && cfg.ad_base_url ? ' at ' + cfg.ad_base_url : '') + ' — no comparisons are being recorded.');
  }
  // Autodarts reads OD's virtual cameras; it can lose them (e.g. ~a minute
  // after OD's process dies) while its own status still says ready. The
  // rig sees its camera fps fall to 0 and reports cameras_down: not ready,
  // like stopped -- red, no grace, no blink.
  if (b.status === 'cameras_down') {
    return {key: 'cameras', tone: 'stopped', word: 'CAMERAS DOWN', label: 'Autodarts’ cameras are down',
      detail: 'It is getting no frames, so it cannot see the board and nothing is compared.', blink: false, live: true};
  }
  if (b.status === 'stopped') {
    return {key: 'stopped', tone: 'stopped', word: 'STOPPED', label: 'Autodarts has stopped', detail: 'Its board is not scoring, so nothing is compared. Start it here.', blink: false, live: true};
  }
  if (b.status === 'takeout') {
    const age = adBoardAge();
    if (age !== null && age >= ORACLE_STUCK_S) {
      return {key: 'stuck', tone: 'takeout', word: 'STUCK', label: 'Autodarts is stuck in takeout',
        detail: 'For ' + duration(age) + ' — it is not scoring. Clear the board, or reset it.', blink: true, live: true};
    }
    return {key: 'takeout', tone: 'takeout', word: 'TAKEOUT', label: 'Autodarts: takeout', detail: 'Waiting for the darts to come out.', blink: false, live: true};
  }
  if (b.status === 'ready') {
    return {key: 'ready', tone: 'ready', word: 'READY', label: 'Autodarts is ready', detail: 'Scoring alongside — every throw is compared.', blink: false, live: true};
  }
  return off('connected', 'CONNECTED', 'Autodarts is connected', 'Its board has not said what it is doing yet.');
}

// The rig's own LED, in the oracle's colours so the two lights read alike:
// grey = not connected, red = stopped (not scoring), yellow = on its way or
// waiting on you (starting, calibrating, warming up, darts to remove),
// green = ready for a dart.
const RIG_LED = {
  offline: 'off', connecting: 'off',
  stopped: 'stopped', nocapture: 'stopped',
  starting: 'takeout', calibrating: 'takeout', warming: 'takeout', takeout: 'takeout',
  ready: 'ready',
};
const rigLed = (p) => p.led || RIG_LED[p.key] || 'off';

// Holding for the oracle (opendarts/live/oracle_hold.py): with the oracle
// on, OD does not score while Autodarts is stopped, offline or stuck in
// takeout -- and says so, by name, wherever the rig's status is shown. The
// server's answer is used as given, except that a takeout gets the same
// grace the scoring gate gives it: Autodarts enters takeout a moment before
// OD at the end of a visit, and that is not a hold.
const HOLD_GRACE_S = 3;
function heldBy() {
  const h = S.oracleHold;
  if (!h || !h.code) return null;
  if (h.code === 'takeout') {
    const age = adBoardAge();
    if (age !== null && age < HOLD_GRACE_S) return null;
  }
  const reason = h.reason || 'oracle not ready';
  const what = {
    takeout: 'Autodarts is in takeout. Pull the darts, then Reset.',
    stopped: 'Autodarts has stopped. Start it, or Reset.',
    offline: 'OpenDarts cannot reach Autodarts.',
    cameras: 'Autodarts is getting no frames from its cameras.',
    unknown: 'Autodarts has not said it is ready.',
  }[h.code] || '';
  // A takeout hold is yellow, like any takeout. Every other hold is PURPLE
  // -- the cabinet strip's "OD ready, held by the oracle" -- never red: red
  // is OD itself not running, and a red light while Autodarts was the one
  // stopped read as "OD isn't started".
  if (h.code === 'takeout') {
    return {key: 'held', tone: 'warn', led: 'takeout', label: reason.charAt(0).toUpperCase() + reason.slice(1),
      title: 'Oracle in takeout', detail: 'Not scoring — ' + what, hold: h};
  }
  const why = {stopped: 'stopped', offline: 'offline', cameras: 'cameras down', unknown: 'not ready'}[h.code]
    || reason.replace(/^oracle\s+/, '');
  return {key: 'held', tone: 'held', led: 'waiting', label: 'Waiting on oracle: ' + why,
    title: 'Waiting on oracle', detail: 'OpenDarts is ready; not scoring — ' + what, hold: h};
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
    $('#veil-kicker').textContent = {offline: 'NO SIGNAL', nocapture: 'OFF AIR', stopped: 'OFF AIR', held: 'ON HOLD'}[p.key] || 'STAND BY';
    $('#veil-title').textContent = {
      offline: 'Can’t reach the rig', connecting: 'Connecting', nocapture: 'No capture here',
      stopped: 'Scoring is off', starting: 'Starting', calibrating: 'Calibrating',
    }[p.key] || p.title || p.label;
    $('#veil-detail').textContent = p.detail;
    const btn = $('#veil-start');
    // Stopped: start scoring. Held: one Reset clears OD and the oracle both.
    btn.hidden = p.key !== 'stopped' && p.key !== 'held';
    btn.firstChild.textContent = p.key === 'held' ? 'RESET BOTH ▸' : 'START SCORING ▸';
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
  $('#veil-start').addEventListener('click', () => (rigPhase().key === 'held' ? Session.reset() : Session.start()));

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
