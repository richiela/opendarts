// ---------------------------------------------------------------------------
// shell -- the frame around the three spaces: routing, the one status chip,
// the Session and This-screen sheets, the events socket, start-up.
// ---------------------------------------------------------------------------

const VIEWS = ['play', 'throws', 'rig'];

const Shell = (() => {
  let view = 'play';
  const kiosk = new URLSearchParams(window.location.search).has('kiosk');

  // 'split' exists only on a display: Play and Throws side by side.
  function showing(name) {
    if (document.hidden) return false;
    return view === name || (view === 'split' && (name === 'play' || name === 'throws'));
  }

  function go(name, opts) {
    const allowed = ROLE === 'display' ? ['play', 'throws', 'split'] : VIEWS;
    if (allowed.indexOf(name) < 0) name = 'play';
    if (kiosk && ROLE !== 'display') name = 'play';
    view = name;
    document.body.dataset.view = name;
    const shown = name === 'split' ? ['play', 'throws'] : [name];
    $$('.views a').forEach((a) => a.setAttribute('aria-current', a.dataset.view === name ? 'page' : 'false'));
    $$('main > section').forEach((s) => { s.hidden = shown.indexOf(s.id) < 0; });
    if (ROLE !== 'display' && (!opts || opts.hash !== false)) setHash(name, true);
    emit('view', name);
    if (showing('play')) { Play.shape(); LiveBoard.draw(); }
  }

  function setHash(hsh, replace) {
    if (location.hash === '#' + hsh) return;
    try {
      if (replace) history.replaceState(null, '', '#' + hsh); else location.hash = hsh;
    } catch (err) { location.hash = hsh; }
  }

  function fromHash() {
    const raw = decodeURIComponent((location.hash || '').replace(/^#/, ''));
    const [name, arg] = raw.split('/');
    go(VIEWS.indexOf(name) >= 0 ? name : 'play', {hash: false});
    if (name === 'throws' && arg) {
      // A deep link can arrive before the throws do: try again as they land,
      // until it is found once.
      let found = Throws.selectById(arg);
      if (!found) on('packages', () => { if (!found) found = Throws.selectById(arg); });
    }
    if (name === 'rig' && arg) {
      const el = document.getElementById('rig-' + arg);
      if (el) requestAnimationFrame(() => el.scrollIntoView({block: 'start'}));
    }
  }

  // ---- the status chip ----
  // The show says it like a broadcast (LIVE, OFF AIR, STAND BY, NO SIGNAL);
  // the lab says it like an instrument. Same phase, same place.
  const ON_AIR = {ready: 'LIVE', takeout: 'LIVE', stopped: 'OFF AIR', nocapture: 'OFF AIR', offline: 'NO SIGNAL'};
  function renderStatus() {
    const p = rigPhase();
    const chip = $('#status');
    chip.dataset.tone = p.tone;
    $('#status-tag').textContent = ON_AIR[p.key] || 'STAND BY';
    $('#status-label').textContent = p.label;
    $('#status-detail').textContent = p.key === 'ready' || p.key === 'takeout' || p.key === 'stopped' ? p.detail.replace(/\.$/, '') : '';
    chip.title = p.label + (p.detail ? ' — ' + p.detail : '') + '. Session controls.';
    renderSession();
    renderReadout();
  }

  // The lab's readout: the oracle and the cameras at a glance, and a clock.
  function renderReadout() {
    const ad = S.ad || (S.runtime && S.runtime.ad);
    const b = S.adBoard || {};
    const o = $('#ro-oracle');
    let tone = 'idle', word = '—';
    if (S.config && S.config.ad_enabled === false) word = 'OFF';
    else if (ad && ad.available === false) word = 'NONE';
    else if (ad && !ad.connected) { tone = 'bad'; word = 'DOWN'; }
    else if (b.status === 'takeout' && typeof b.age === 'number' && b.age >= 30) { tone = 'bad'; word = 'STUCK'; }
    else if (ad && ad.connected) { tone = 'live'; word = 'READY'; }
    o.dataset.tone = tone;
    o.replaceChildren('ORACLE ', h('b', {}, word));
    const cams = $('#ro-cams');
    const rows = S.camRows ? Object.values(S.camRows) : null;
    const running = !!(S.capture && S.capture.running);
    const open = rows ? rows.filter((r) => r.opened && r.last_read_ok !== false).length : null;
    cams.dataset.tone = !running ? 'idle' : open === null || open === CAM_IDS.length ? 'live' : 'warn';
    cams.replaceChildren('CAMS ', h('b', {}, !running ? 'OFF' : open === null ? 'ON' : open + '/' + CAM_IDS.length));
  }

  // ---- the session sheet ----
  function renderSession() {
    const p = rigPhase();
    $('#sess-title').textContent = p.label;
    $('#sess-detail').textContent = p.detail || '';
    $('#sess-dot').dataset.tone = p.tone;
    const running = !!(S.capture && S.capture.running);
    const noCapture = p.key === 'nocapture' || p.key === 'offline' || p.key === 'connecting';
    $('#sess-start').hidden = running || noCapture;
    $('#sess-stop').hidden = !running || noCapture;
    const busy = p.key === 'starting' || p.key === 'calibrating' || Session.busy;
    ['#sess-start', '#sess-stop', '#sess-calibrate', '#sess-reset'].forEach((sel) => { $(sel).disabled = busy || noCapture; });
    $('#sess-reset').disabled = busy || !running;
  }

  function renderActivity() {
    const list = $('#sess-activity');
    if (!S.activity.length) { list.replaceChildren(h('li', {class: 'muted'}, 'Nothing yet this visit to the page.')); return; }
    list.replaceChildren(...S.activity.map((a) => h('li', {class: 'act act-' + a.status},
      h('span', {class: 'act-time'}, clock(a.started) + (a.finished ? '–' + clock(a.finished) : '')),
      h('span', {class: 'act-label'}, a.label),
      h('span', {class: 'act-detail'}, a.detail))));
  }

  // ---- this screen ----
  function renderScreen() {
    const snd = Sound.status();
    const btn = $('#screen');
    btn.dataset.state = snd.state;
    btn.title = snd.label + ' — this screen’s sound and display';
    $('#screen-label').textContent = snd.state === 'blocked' ? 'Tap for sound' : '';
    $('#scr-device').textContent = 'Settings for this screen only — every screen keeps its own.';
    $('#scr-sound').checked = Sound.settings.enabled;
    $('#scr-note').textContent = snd.detail;
    $('#scr-note').className = 'note ' + ({blocked: 'bad', error: 'bad', unsupported: 'bad', on: 'ok'}[snd.state] || '');
    const vol = Math.round(Sound.settings.volume * 100);
    if (document.activeElement !== $('#scr-volume')) $('#scr-volume').value = String(vol);
    $('#scr-volume-val').textContent = vol + '%';
    $$('#scr-boardview button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === LiveBoard.view)));
    $('#scr-overlay').checked = Cameras.overlay;
    const theme = prefs.get('theme', 'light');
    $$('#scr-theme button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === theme)));
    const text = prefs.get('textSize', 'normal');
    $$('#scr-text button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === text)));
  }

  async function fillVoices() {
    const sel = $('#scr-voice');
    try {
      const cat = await Sound.fetchCatalog(false);
      const sets = (cat && cat.voices) || [];
      if (!sets.length) { sel.replaceChildren(h('option', {value: ''}, 'None installed')); sel.disabled = true; return; }
      sel.disabled = false;
      sel.replaceChildren(...sets.map((s) => h('option', {value: s.voice},
        s.voice + (s.voice === cat.default ? ' (default)' : '') + (s.complete ? '' : ' — ' + s.present + '/' + s.total + ' calls'))));
      sel.value = Sound.chosenVoice();
    } catch (err) { sel.replaceChildren(h('option', {value: ''}, 'Unavailable')); }
  }

  // The theme is the lab's paper -- paper or blueprint. The show is always
  // dark: it is a scoreboard in a dim room.
  // A display passes its own (from the rig); every other screen reads its
  // saved preference.
  function applyTheme(theme, textSize) {
    const t = theme || prefs.get('theme', 'light');
    if (t === 'auto') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t === 'dark' ? 'dark' : 'light';
    const size = textSize || prefs.get('textSize', 'normal');
    if (size === 'large' || size === 'huge') document.documentElement.dataset.text = size;
    else delete document.documentElement.dataset.text;
    emit('theme');
  }

  function openScreen() { fillVoices(); renderScreen(); sheet.open('sheet-screen'); }

  // A blocking full-page notice, for the one moment the page cannot work:
  // the rig restarting underneath it.
  function overlay(title, detail) {
    const o = $('#blocker');
    o.hidden = !title;
    if (title) { $('#blocker-title').textContent = title; $('#blocker-detail').textContent = detail || ''; }
  }

  // ---- wiring ----
  $$('.views a').forEach((a) => a.addEventListener('click', (ev) => { ev.preventDefault(); go(a.dataset.view); }));
  window.addEventListener('hashchange', () => { if (ROLE !== 'display') fromHash(); });
  $('#status').addEventListener('click', () => { renderSession(); renderActivity(); sheet.open('sheet-session'); });
  $('#screen').addEventListener('click', openScreen);
  $('#sess-start').addEventListener('click', () => Session.start());
  $('#sess-stop').addEventListener('click', () => Session.stop());
  $('#sess-calibrate').addEventListener('click', () => { sheet.close('sheet-session'); Session.calibrate(); });
  $('#sess-reset').addEventListener('click', () => Session.reset());
  $('#scr-sound').addEventListener('change', (ev) => Sound.setEnabled(ev.target.checked));
  $('#scr-voice').addEventListener('change', (ev) => Sound.setVoice(ev.target.value));
  $('#scr-volume').addEventListener('input', (ev) => Sound.setVolume(Number(ev.target.value) / 100, false));
  $('#scr-volume').addEventListener('change', (ev) => Sound.setVolume(Number(ev.target.value) / 100, true));
  $('#scr-test').addEventListener('click', async () => {
    const r = await Sound.test();
    if (r === 'blocked') toast('The browser is holding sound back — tap the page once, and check an iPad’s silent switch', 'warn');
    else if (r !== 'played') toast('Could not play the test call', 'bad');
  });
  $$('#scr-boardview button').forEach((b) => b.addEventListener('click', () => { LiveBoard.setView(b.dataset.v); renderScreen(); }));
  $('#scr-overlay').addEventListener('change', (ev) => Cameras.setOverlay(ev.target.checked));
  $$('#scr-theme button').forEach((b) => b.addEventListener('click', () => { prefs.set('theme', b.dataset.v); applyTheme(); renderScreen(); }));
  $$('#scr-text button').forEach((b) => b.addEventListener('click', () => { prefs.set('textSize', b.dataset.v); applyTheme(); renderScreen(); }));
  $('#scr-fullscreen').addEventListener('click', () => {
    const el = document.documentElement;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (el.requestFullscreen) el.requestFullscreen().catch(() => toast('This browser would not go full screen', 'warn'));
  });

  document.addEventListener('keydown', (ev) => {
    if (ROLE === 'display') return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey || typingIn(ev.target)) return;
    const i = ['1', '2', '3'].indexOf(ev.key);
    if (i >= 0) { ev.preventDefault(); go(VIEWS[i]); }
  });

  on('phase', renderStatus);
  on('adboard', renderReadout);
  on('camstatus', renderReadout);
  on('config', renderReadout);
  setInterval(() => { const el = $('#ro-clock'); if (el) el.textContent = clock(new Date()); }, 1000);
  on('connected', renderStatus);
  on('activity', renderActivity);
  on('sound', renderScreen);
  on('boardview', renderScreen);
  on('screenprefs', renderScreen);

  function start() {
    applyTheme(ROLE === 'display' ? 'dark' : undefined);
    $('#rig-name').textContent = OD.host_label || 'opendarts';
    if (ROLE === 'display') {
      Display.start();
    } else {
      if (kiosk) document.body.classList.add('kiosk');
      fromHash();
    }
    renderStatus();
    renderScreen();
  }

  return {start, go, showing, setHash, overlay, openScreen, applyTheme, get view() { return view; }};
})();

// ---- session actions ------------------------------------------------------------
// Start / Stop / Reset / Calibrate. Each is one activity line, updated in
// place; the status everywhere follows from the stamped server replies.

const Session = (() => {
  let busy = false;

  function phaseChanged() { emit('phase'); }

  async function resync() {
    try {
      const st = await api.get('/api/state');
      if (st && st.capture_loop) applyState(st);
    } catch (err) { /* the socket will bring it */ }
  }

  async function start() {
    if (busy) return;
    busy = true;
    S.idleStopped = null;
    S.capture = Object.assign({}, S.capture || {}, {running: true, starting: true});
    phaseChanged();
    const act = activity('Start', 'pending', 'opening the cameras…');
    try {
      const body = await api.post('/api/start');
      if (!body || body.ok === false) {
        S.startError = (body && body.reason) || 'unknown';
        act.done('bad', S.startError);
      } else {
        S.startError = null;
        if (acceptCapture(body)) S.capture = body;
        act.done('ok', body.already_running ? 'already running' : 'cameras open');
      }
    } catch (err) {
      act.done('bad', 'could not reach the rig');
    } finally {
      busy = false;
      phaseChanged();
      resync();
    }
  }

  async function stop() {
    if (busy) return;
    busy = true;
    const act = activity('Stop', 'pending');
    try {
      const body = await api.post('/api/stop');
      if (!body || body.ok === false) act.done('bad', (body && body.reason) || 'failed');
      else { if (acceptCapture(body)) S.capture = body; act.done('ok', body.already_stopped ? 'already stopped' : 'cameras closed'); }
    } catch (err) { act.done('bad', 'could not reach the rig'); }
    finally { busy = false; phaseChanged(); resync(); }
  }

  async function reset() {
    const act = activity('Reset visit', 'pending');
    try {
      const body = await api.post('/api/reset');
      if (body && body.loop_listening) { act.done('ok', 'the board as it is now is the new empty board'); toast('Visit reset', 'ok'); }
      else act.done('bad', 'no capture loop in this process');
    } catch (err) { act.done('bad', 'could not reach the rig'); }
  }

  async function calibrate() {
    if (busy || S.calibrating) return;
    S.calibrating = true;
    S.calibResult = null;
    Cameras.suspend();
    emit('calibration'); phaseChanged();
    const act = activity('Calibrate', 'pending', 'keep the board clear…');
    const t0 = performance.now();
    try {
      const body = await api.post('/api/calibration/refresh');
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      if (body && body.cameras) S.calibration = Object.assign({}, S.calibration || {}, body);
      if (!body || body.calibration_error || body.ok === false) {
        S.calibResult = {ok: false, error: (body && (body.calibration_error || body.reason)) || 'The rig did not answer.'};
        act.done('bad', ((body && (body.calibration_error || body.reason)) || 'failed') + ' (' + secs + ' s)');
      } else {
        const cams = Object.values(body.cameras || {});
        const ok = cams.filter((c) => c.ok).length;
        S.calibResult = {ok: ok > 0, good: ok, total: cams.length, error: ok ? null : 'No camera could be calibrated.'};
        const moved = body.ring_geometry_relearned;
        act.done(moved ? 'warn' : 'ok', ok + ' of ' + cams.length + ' cameras calibrated in ' + secs + ' s' + (moved ? ' — cameras had moved, layout relearned' : ''));
        // the calibration screen says how it ended (calib.js)
      }
    } catch (err) {
      S.calibResult = {ok: false, error: 'Could not reach the rig.'};
      act.done('bad', 'could not reach the rig');
    } finally {
      S.calibrating = false;
      Cameras.resume();
      await resync();
      emit('calibration'); phaseChanged();
    }
  }

  return {start, stop, reset, calibrate, get busy() { return busy; }};
})();

// ---- state in ---------------------------------------------------------------------

function applyState(st) {
  if (!st) return;
  S.state = st;
  if (st.capture_loop && acceptCapture(st.capture_loop)) S.capture = st.capture_loop;
  S.trigger = st.trigger || {};
  if (st.capture_loop && st.capture_loop.last_start_error) S.startError = st.capture_loop.last_start_error;
  if (st.visit) {
    S.visit = {id: st.visit.visit_id || null, throws: (st.visit.throws || []).slice(),
      photoVersion: st.visit.board_photo_version, available: st.visit.available !== false};
    LiveBoard.setPhotoVersion(st.visit.board_photo_version);
  }
  if (st.calibration) S.calibration = st.calibration;
  if (st.engine_config) S.engineConfig = st.engine_config;
  S.adBoard = {status: st.ad_board_status, age: st.ad_board_status_age_sec};
  emit('state'); emit('phase'); emit('visit'); emit('calibration'); emit('engines'); emit('adboard');
}

function onThrow(ev) {
  if (ev.visit_id && ev.visit_id !== S.visit.id) { S.visit.id = ev.visit_id; S.visit.throws = []; }
  const i = ev.visit_index;
  if (i === null || i === undefined) S.visit.throws.push(ev); else S.visit.throws[i] = ev;
  emit('visit');
  emit('landed', i === null || i === undefined ? S.visit.throws.length - 1 : i);
}

// A page that reconnects to a newer rig reloads itself, once.
function reloadIfStale(serverVersion) {
  const mine = OD.page_version;
  if (!serverVersion || !mine || serverVersion === mine) return false;
  let already = null;
  try { already = window.sessionStorage.getItem('opendarts.reloadedForPage'); } catch (err) { already = null; }
  if (already === serverVersion) { console.warn('page still differs after a reload — not reloading again'); return false; }
  try { window.sessionStorage.setItem('opendarts.reloadedForPage', serverVersion); } catch (err) { /* private mode */ }
  window.location.reload();
  return true;
}

function connect() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/api/events');
  ws.onopen = () => { S.connected = true; S.everConnected = true; emit('connected'); emit('phase'); };
  ws.onclose = () => {
    const was = S.connected;
    S.connected = false;
    Cameras.stopAll('Reconnecting…');
    if (was) emit('phase');
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    let m;
    try { m = JSON.parse(event.data); } catch (err) { return; }
    switch (m.type) {
      case 'HELLO':
        if (reloadIfStale(m.page_version)) return;
        applyState(m.state);
        Throws.ingest(m.packages);
        Throws.reconcile(m.count);
        break;
      case 'PACKAGES_UPDATED':
        Throws.ingest(m.packages);
        Throws.reconcile(m.count);
        break;
      case 'DART_CALL':
        Sound.play(m.phrase);
        break;
      case 'THROW_DETECTED':
        onThrow(m);
        Throws.provisional(m);
        break;
      case 'BOARD_PHOTO':
        LiveBoard.setPhotoVersion(m.version);
        break;
      case 'VISIT_CLEARED':
        S.visit.id = m.visit_id || S.visit.id;
        S.visit.throws = [];
        emit('visit'); emit('turnended');
        break;
      case 'THROW_CORRECTED': {
        if (m.visit_id !== S.visit.id) break;
        const t = S.visit.throws[m.visit_index];
        if (t) { t.corrected_sector = m.corrected_sector; t.corrected_ring = m.corrected_ring; emit('visit'); }
        break;
      }
      case 'TRIGGER_STATE':
        S.trigger = {available: true, state: m.state, dart_count: m.dart_count, last_event_utc: m.ts};
        if (S.capture && acceptCapture(m)) S.capture.starting = false;
        emit('phase');
        break;
      case 'CAPTURE_LOOP_STATUS':
        if (!acceptCapture(m)) break;
        S.capture = m;
        if (m.ok === false && m.reason) S.startError = m.reason;
        else if (m.ok) S.startError = null;
        emit('phase');
        break;
      case 'CALIBRATION_STATUS':
        S.calibration = Object.assign({}, S.calibration || {}, {cameras: m.cameras, ring_geometry_relearned: m.ring_geometry_relearned});
        if (m.source === 'startup' || m.source === 'startup_reused') {
          const cams = Object.values(m.cameras || {});
          activity('Calibration at start', cams.some((c) => c.ok) || m.source === 'startup_reused' ? 'ok' : 'bad',
            m.source === 'startup' ? cams.filter((c) => c.ok).length + ' of ' + cams.length + ' cameras' : 'reused the saved calibration');
        }
        if (m.ring_geometry_relearned && m.ring_geometry_relearned.note) activity('Cameras moved', 'warn', 'layout relearned');
        emit('calibration');
        // The package id that keys the overlay only comes with the full state.
        api.get('/api/state').then((st) => { if (st && st.calibration) { S.calibration = st.calibration; emit('calibration'); } }).catch(() => {});
        break;
      case 'DASHBOARD_SWITCHED':
        // The rig's switch flipped: / serves the other dashboard now. A
        // display, and a screen opened with ?ui=, chose their page and stay.
        if (ROLE !== 'display' && !new URLSearchParams(location.search).get('ui')) location.reload();
        break;
      case 'CALIBRATION_PROGRESS':
        Calib.onProgress(m);
        break;
      case 'AD_BOARD_STATUS':
        S.adBoard = {status: m.status, age: 0};
        emit('adboard');
        break;
      case 'AD_CONNECTION':
        if (acceptAd(m)) { S.ad = m; emit('adboard'); }
        break;
      case 'DISPLAY_UPDATED':
      case 'DISPLAY_IDENTIFY':
      case 'DISPLAY_RELOAD':
      case 'DISPLAY_FORGOTTEN':
        if (ROLE === 'display') Display.onMessage(m);
        else emit('displays', m);
        break;
      case 'IDLE_TIMEOUT':
        S.idleStopped = {sec: m.idle_timeout_sec};
        activity('Auto-stop', 'ok', 'no darts for ' + duration(m.idle_timeout_sec));
        emit('phase');
        break;
      default:
        break;
    }
  };
}

async function loadInitial() {
  try {
    const [st, pk] = await Promise.all([api.get('/api/state'), api.get('/api/packages')]);
    applyState(st);
    if (Array.isArray(pk)) Throws.ingest(pk);
  } catch (err) { console.error('initial load failed', err); }
  Rig.refreshConfig();
  Rig.refreshHealth();
  Play.loadRecent();
}

Shell.start();
Sound.start();
loadInitial();
connect();
