// ---------------------------------------------------------------------------
// calib -- a calibration, shown the way it happens: the whole screen, a real
// bar, the stage it is in, each camera as it finishes.
//
// Calibration was the most opaque thing the rig does: half a minute with
// nothing moving. The rig reports its stages (CALIBRATION_PROGRESS,
// opendarts/live/calibration_progress.py); this draws them, on every screen
// -- the TV included, since the person who must keep the board clear is
// standing in front of it. Also the rig's connection strip (conn), below.
// ---------------------------------------------------------------------------

const Calib = (() => {
  const box = $('#calib');
  let hiddenByHand = false, closeTimer = null, clockTimer = null, t0 = null, shownAt = 0;

  const cameraName = (c) => 'Cam ' + (Number(c) + 1);
  // Progress that belongs to the calibration on screen now, not a
  // previous one's leftovers.
  const current = () => { const p = S.calibProgress; return p && p._at >= shownAt ? p : null; };

  function show(p) {
    clearTimeout(closeTimer);
    shownAt = p ? p._at : performance.now();
    t0 = performance.now() - (p && typeof p.elapsed_s === 'number' ? p.elapsed_s * 1000 : 0);
    box.hidden = false;
    box.dataset.tone = 'busy';
    $('#calib-kicker').textContent = 'CALIBRATING';
    $('#calib-title').textContent = 'Keep the board clear';
    $('#calib-sub').textContent = 'No darts in the board, no hands in front of it.';
    clearInterval(clockTimer);
    clockTimer = setInterval(tick, 500);
    tick();
  }

  function render() {
    const p = S.calibProgress;
    const running = !!S.calibrating || !!(p && p.active);
    if (running && box.hidden && !hiddenByHand) show(p && p.active ? p : null);
    if (box.hidden) return;
    const cp = current();
    const frac = cp ? cp.fraction || 0 : 0;
    $('#calib-fill').style.width = Math.round(frac * 100) + '%';
    $('#calib-pct').textContent = Math.round(frac * 100) + '%';
    $('#calib-stage').textContent = cp && cp.label ? cp.label + (cp.detail ? ' — ' + cp.detail : '') : 'Starting…';
    $('#calib-steps').replaceChildren(...((cp && cp.stages) || []).map((st) => h('li', {class: 'st-' + st.state},
      h('i', {'aria-hidden': 'true'}, st.state === 'done' ? '✓' : st.state === 'now' ? '▸' : '·'), st.label)));
    $('#calib-cams').replaceChildren(...Object.entries((cp && cp.cameras) || {}).map(([c, st]) => h('span', {class: 'cc cc-' + st},
      cameraName(c), h('small', {}, {waiting: 'waiting', working: 'working', done: 'done', failed: 'gave up'}[st] || st))));
    if (!running) finished(cp);
  }

  function tick() { $('#calib-time').textContent = Math.round((performance.now() - t0) / 1000) + ' s'; }

  // Say how it ended, then get out of the way -- unless it failed and a
  // person can act on it. The rig's own report when there is one; else
  // what this page's Calibrate request was answered with.
  function finished(p) {
    clearInterval(clockTimer);
    const mine = S.calibResult || {};
    const ok = p && p.succeeded !== null && p.succeeded !== undefined ? !!p.succeeded : !!mine.ok;
    const cams = p && p.cameras ? Object.values(p.cameras) : null;
    const good = cams ? cams.filter((st) => st === 'done').length : mine.good;
    const total = cams ? cams.length : mine.total;
    box.dataset.tone = ok ? 'ok' : 'bad';
    $('#calib-fill').style.width = '100%';
    if (ok) $('#calib-pct').textContent = '100%';
    $('#calib-kicker').textContent = ok ? 'CALIBRATED' : 'CALIBRATION FAILED';
    $('#calib-title').textContent = ok ? (total ? good + ' of ' + total + ' cameras' : 'Done') : 'It did not take';
    $('#calib-sub').textContent = ok
      ? (total && good < total ? 'A camera gave up — scoring uses the others. Rig › Cameras has the details.' : 'Throw when ready.')
      : ((p && p.error) || mine.error || 'No camera could be calibrated.') + ' The previous calibration is still in use.';
    clearTimeout(closeTimer);
    closeTimer = setTimeout(close, ok ? 2600 : (ROLE === 'display' ? 9000 : 30000));
  }

  function close() { box.hidden = true; clearInterval(clockTimer); }

  function onProgress(m) {
    const was = S.calibProgress && S.calibProgress.active;
    S.calibProgress = Object.assign({}, m, {_at: performance.now()});
    if (m.active && !was) { hiddenByHand = false; if (!box.hidden) shownAt = S.calibProgress._at; }
    emit('phase');
    render();
  }

  $('#calib-hide').addEventListener('click', () => { hiddenByHand = true; close(); });
  on('calibration', render);                       // this page's own Calibrate
  // A page that opens part-way through one catches up.
  api.get('/api/calibration/progress').then((p) => { if (p && p.active) onProgress(p); }).catch(() => {});

  return {onProgress, render};
})();

// The connection, on the pages with no scoreboard veil to say it: when the
// rig cannot be reached, or is not scoring, the Throws list is not going to
// move -- and a list that simply stops looks exactly like a quiet board.
const Conn = (() => {
  const el = $('#conn-strip');
  function render() {
    const p = rigPhase();
    const hide = p.key === 'ready' || p.key === 'takeout' || p.key === 'warming' || p.key === 'calibrating' || !Shell.showing('throws') && !Shell.showing('rig');
    el.hidden = hide;
    document.body.classList.toggle('conn-on', !hide);
    if (hide) return;
    el.dataset.tone = p.tone;
    $('#conn-title').textContent = {offline: 'RIG OFFLINE', connecting: 'CONNECTING', stopped: 'SCORING IS OFF', nocapture: 'NO CAPTURE', starting: 'STARTING'}[p.key] || p.label.toUpperCase();
    $('#conn-detail').textContent = p.key === 'stopped' ? 'New darts will not appear until scoring starts.' : p.detail;
    measure();
  }
  // The page below moves down by the strip's real height -- one line on a
  // wide screen, two on a phone -- so nothing is ever hidden under it.
  function measure() {
    if (el.hidden) return;
    document.body.style.setProperty('--conn-h', Math.ceil(el.getBoundingClientRect().height) + 'px');
  }
  window.addEventListener('resize', measure);
  on('phase', render);
  on('connected', render);
  on('view', render);
  return {render};
})();
