// ---------------------------------------------------------------------------
// rig -- set-up: the cameras first (can the rig see the board?), then every
// setting, then the machine itself.
// ---------------------------------------------------------------------------

// ---- cameras ----------------------------------------------------------------
// Live MJPEG previews with the calibration overlay as a transparent layer.
//
// The connection rules are the current dashboard's, and each one is a scar:
//  * A stream exists only while the Rig view is showing AND this window has
//    focus AND capture is not positively stopped. The browser allows ~6
//    sockets per origin across every tab; three streams per window, twice,
//    froze the whole dashboard once.
//  * Clearing img.src is what closes the connection; hiding does not.
//  * A multipart stream that ends fires no error in Chrome, so the camera
//    status poll is what notices a camera that stopped producing.
//  * The overlay is fetched once per recalibration, and a starved fetch is
//    retried a bounded number of times -- never polled.
//  * Calibrate closes the streams for its duration (it needs the sockets
//    and the CPU), and restores whatever the view state says afterwards.

const Cameras = (() => {
  const TICK_MS = 3000, CONNECT_DEADLINE_MS = 10000;
  const OVERLAY_RETRY_MS = 1200, OVERLAY_MAX_RETRIES = 6;
  const st = {}, startedAt = {}, overlayOk = {}, retries = {}, retryTimer = {};
  let token = '';
  let suspended = false;
  let overlayPref = prefs.get('overlay', true) !== false;

  function build() {
    const box = $('#cams');
    // Fig. 1, the rig plan: the board from above, each camera drawn where
    // its calibrated pose puts it, looking in. A knocked camera shows here
    // before it shows anywhere else.
    // Beside it, the schedule -- the three cameras side by side, the way a
    // drawing tabulates its parts -- so one that has drifted or fits worse
    // than the others stands out without opening each camera's details.
    const planBox = h('div', {class: 'cropbox'}, h('canvas', {id: 'plan-canvas', 'aria-label': 'Where the cameras sit around the board'}));
    const plan = h('div', {class: 'plan'},
      h('figure', {class: 'plan-fig'}, planBox,
        h('figcaption', {class: 'figcap'}, h('span', {}, 'Fig. 1 — rig plan'), h('span', {}, 'from the calibrated poses'))),
      h('div', {class: 'schedule'},
        h('div', {class: 'h'}, h('b', {}, 'Schedule — cameras'), h('span', {id: 'sched-when'}, '')),
        h('div', {class: 'table-wrap'}, h('table', {class: 'sched', id: 'cam-schedule'})),
        h('p', {class: 'sched-note', id: 'sched-note'}, '')));
    box.replaceChildren(plan, ...CAM_IDS.map((c) => h('article', {class: 'cam', id: 'cam-' + c},
      h('div', {class: 'cropbox media'},
        h('img', {class: 'cam-img', id: 'cam-img-' + c, alt: 'Camera ' + (c + 1) + ' live', hidden: true}),
        h('img', {class: 'cam-ov', id: 'cam-ov-' + c, alt: '', hidden: true}),
        h('div', {class: 'cam-ph', id: 'cam-ph-' + c},
          h('span', {class: 'ph-title'}, 'No picture'),
          h('span', {class: 'ph-sub', id: 'cam-ph-sub-' + c}, ''))),
      h('div', {class: 'cam-cap'},
        h('div', {class: 't'}, h('b', {class: 'cam-name'}, 'Cam ' + (c + 1)), h('span', {id: 'cam-where-' + c}, '')),
        h('div', {class: 's'}, h('span', {class: 'cam-live', id: 'cam-live-' + c}, ''), h('span', {class: 'cam-cal', id: 'cam-cal-' + c}, '—'))),
      h('select', {class: 'cam-src', id: 'cam-src-' + c, dataset: {slot: String(c)}, 'aria-label': 'Camera ' + (c + 1) + ' source'}),
      h('details', {class: 'cam-more'}, h('summary', {}, 'Details'), h('dl', {class: 'kv small', id: 'cam-detail-' + c})),
    )));
    if (window.ResizeObserver) new ResizeObserver(drawPlan).observe(planBox);
    for (const c of CAM_IDS) { st[c] = 'idle'; startedAt[c] = 0; overlayOk[c] = false; retries[c] = 0; retryTimer[c] = null; }
  }

  const img = (c) => document.getElementById('cam-img-' + c);
  const ph = (c) => document.getElementById('cam-ph-' + c);

  function placeholder(c, sub) {
    img(c).hidden = true;
    ph(c).hidden = false;
    $('#cam-ph-sub-' + c).textContent = sub;
  }

  function stop(c) {
    if (st[c] === 'idle') return;
    const el = img(c);
    el.onload = null; el.onerror = null;   // this stop is not a failure
    el.src = '';
    el.removeAttribute('src');
    st[c] = 'idle';
  }

  function stopAll(sub) {
    for (const c of CAM_IDS) { stop(c); placeholder(c, sub || 'Paused'); hideOverlay(c); }
  }

  function wanted() {
    return Shell.showing('rig') && document.hasFocus() && !suspended;
  }

  function update() {
    if (!$('#cams').children.length) return;
    const showing = Shell.showing('rig');
    const focused = document.hasFocus();
    const cap = S.capture;
    const stopped = !!(cap && !cap.running);
    for (const c of CAM_IDS) {
      if (!wanted() || stopped) {
        stop(c);
        hideOverlay(c);
        if (stopped) placeholder(c, cap.starting ? 'Starting…' : 'Scoring is off — press Start');
        else if (suspended) placeholder(c, 'Paused while calibrating');
        else if (showing && !focused) placeholder(c, 'Paused — click this window to resume');
        else placeholder(c, 'Paused');
        $('#cam-live-' + c).textContent = '';
        $('#cam-live-' + c).dataset.tone = 'idle';
        continue;
      }
      if (st[c] === 'live') {
        const row = S.camRows && S.camRows[String(c)];
        if (row && (row.opened === false || row.last_read_ok === false)) {
          stop(c); hideOverlay(c); placeholder(c, 'No frames from this camera');
          $('#cam-live-' + c).textContent = 'no frames';
          $('#cam-live-' + c).dataset.tone = 'bad';
        }
        continue;
      }
      if (st[c] === 'connecting') {
        if (Date.now() - startedAt[c] < CONNECT_DEADLINE_MS) continue;
        stop(c);                              // a silent stall: retry now
      }
      st[c] = 'connecting';
      startedAt[c] = Date.now();
      $('#cam-live-' + c).textContent = 'connecting…';
      const el = img(c);
      el.onload = () => {
        st[c] = 'live';
        el.hidden = false;
        ph(c).hidden = true;
        renderLive(c);
        refreshOverlays();
      };
      el.onerror = () => {
        st[c] = 'idle';
        placeholder(c, 'Camera unreachable');
        $('#cam-live-' + c).textContent = 'unreachable';
        $('#cam-live-' + c).dataset.tone = 'bad';
        hideOverlay(c);
      };
      el.src = '/api/cameras/' + c + '/stream.mjpg?t=' + Date.now();
    }
  }

  function renderLive(c) {
    const row = S.camRows && S.camRows[String(c)];
    const el = $('#cam-live-' + c);
    if (st[c] !== 'live') return;
    // Falling behind is said where you are looking: a 30 fps camera read
    // at 21 degrades scoring silently (docs/LIVE_API.md, frame-health).
    const fh = S.frameHealth && Array.isArray(S.frameHealth.capture)
      ? S.frameHealth.capture.find((x) => x.camera === c) : null;
    const behind = !!(fh && fh.opened && fh.keeping_up === false);
    el.dataset.tone = behind ? 'warn' : 'live';
    if (!row) { el.textContent = 'live'; return; }
    const fps = row.effective_fps || row.actual_fps;
    el.textContent = (behind ? 'falling behind' : 'live') + (fps ? ' · ' + Math.round(fps) + ' fps' : '');
    el.title = behind ? 'Frames are being read slower than this camera sends them.' : '';
  }

  // ---- overlay ----
  function hideOverlay(c) {
    const ov = document.getElementById('cam-ov-' + c);
    if (!ov) return;
    ov.hidden = true;
    ov.removeAttribute('src');
    if (retryTimer[c]) { clearTimeout(retryTimer[c]); retryTimer[c] = null; }
  }

  function refreshOverlays() {
    for (const c of CAM_IDS) {
      const ov = document.getElementById('cam-ov-' + c);
      if (!ov) continue;
      if (!overlayPref || !overlayOk[c] || st[c] !== 'live') { hideOverlay(c); continue; }
      const want = '/api/cameras/' + c + '/overlay-rgba.png?v=' + encodeURIComponent(token || 'none');
      if (ov.getAttribute('src') === want && !ov.hidden) continue;
      ov.onload = () => { ov.hidden = false; retries[c] = 0; };
      ov.onerror = () => {
        ov.hidden = true;
        ov.removeAttribute('src');
        if (retries[c] < OVERLAY_MAX_RETRIES) {
          retries[c] += 1;
          if (retryTimer[c]) clearTimeout(retryTimer[c]);
          retryTimer[c] = setTimeout(() => { retryTimer[c] = null; refreshOverlays(); }, OVERLAY_RETRY_MS);
        }
      };
      ov.setAttribute('src', want);
    }
  }

  // ---- calibration ----
  function renderCalibration() {
    const cal = S.calibration || {};
    const src = cal.live_source || {};
    const next = src.calibration_package_id || src.checked_at_utc || '';
    if (next !== token) for (const c of CAM_IDS) retries[c] = 0;
    token = next;
    const cams = cal.cameras || {};
    let ok = 0;
    for (const c of CAM_IDS) {
      const row = cams[String(c)];
      const badge = document.getElementById('cam-cal-' + c);
      if (!badge) continue;
      if (S.calibrating) {
        badge.className = 'cam-cal busy'; badge.textContent = 'Calibrating';
        overlayOk[c] = false;
        continue;
      }
      const good = !!row && row.ok === true;
      if (good) ok++;
      overlayOk[c] = good;
      badge.className = 'cam-cal ' + (!row ? 'idle' : good ? 'ok' : 'bad');
      const err = row && row.reprojection_error_px;
      badge.textContent = !row ? 'Not calibrated' : good ? (err !== null && err !== undefined ? 'Cal ' + err.toFixed(2) + ' px' : 'Calibrated') : 'Not calibrated';
      badge.title = row
        ? (row.reprojection_error_px !== null && row.reprojection_error_px !== undefined
          ? 'Reprojection error ' + row.reprojection_error_px.toFixed(2) + ' px'
          : good ? 'Loaded from disk — press Calibrate to measure its error afresh' : (row.reason || ''))
        : 'No calibration data yet';
    }
    const summary = $('#calib-summary');
    if (S.calibrating) {
      summary.textContent = 'Calibrating — keep the board clear.';
    } else if (!src.checked_at_utc) {
      summary.textContent = 'Not calibrated yet. Press Calibrate — or Start calibrates automatically.';
    } else {
      const from = {startup: 'at start-up', manual: 'by hand', persisted: 'loaded from disk'}[src.source] || src.source || '';
      summary.textContent = (ok === CAM_IDS.length ? 'All ' + CAM_IDS.length + ' cameras calibrated' : ok + ' of ' + CAM_IDS.length + ' cameras calibrated')
        + ' · ' + ago(src.checked_at_utc) + (from ? ' · ' + from : '');
    }
    const moved = cal.ring_geometry_relearned;
    const mv = $('#calib-moved');
    mv.hidden = !(moved && moved.note);
    if (moved && moved.note) mv.textContent = 'The cameras had moved — the layout was relearned. ' + moved.note;
    refreshOverlays();
  }

  // Where each camera sits, in the words of someone standing at the board:
  // a clock position, how far off the board's face it looks, how far away.
  // From /api/calibration, which derives it from the solved pose.
  let geometry = null;
  function placement(c) {
    const g = geometry && geometry[String(c)];
    if (!g || !g.position_mm) return null;
    const [x, y] = g.position_mm;
    const deg = (Math.atan2(x, y) * 180 / Math.PI + 360) % 360;       // clockwise from 12
    const hour = Math.round(deg / 30) % 12 || 12;
    return {hour, elevation: g.elevation_deg, distance: g.distance_mm, focal: g.focal_length_px};
  }
  async function refreshGeometry() {
    const body = await api.get('/api/calibration').catch(() => null);
    geometry = body && body.ok !== false && body.cameras ? body.cameras : null;
    renderStatus();
    drawPlan();
  }

  // The rig plan: board from above at true scale, each camera at its
  // calibrated position with a sight line and a field-of-view wedge.
  function drawPlan() {
    const cv = document.getElementById('plan-canvas');
    if (!cv) return;
    const fit = fitCanvas(cv, true);
    if (!fit) return;
    const {c, w} = fit;
    c.clearRect(0, 0, w, w);
    const INK = cssVar('--ink', '#15140f'), INK2 = cssVar('--ink2', '#56534a'), INK3 = cssVar('--ink3', '#8b8676'), CARD = cssVar('--card', '#f6f2ea');
    const cams = CAM_IDS.map((k) => ({k, g: geometry && geometry[String(k)]})).filter((x) => x.g && x.g.position_mm);
    const reach = Math.max(260, ...cams.map((x) => Math.hypot(x.g.position_mm[0], x.g.position_mm[1])));
    const ppm = (w / 2 - 34) / reach, cx = w / 2, cy = w / 2;
    drawDiagram(c, cx, cy, ppm, {ink: labInk(), numbers: false, wire: .6});
    c.font = '600 12px "IBM Plex Mono", monospace'; c.textAlign = 'center'; c.textBaseline = 'middle';
    if (!cams.length) {
      c.fillStyle = INK2; c.fillText('CALIBRATE TO PLACE THE CAMERAS', cx, w - 16);
      return;
    }
    for (const {k, g} of cams) {
      const x = cx + g.position_mm[0] * ppm, y = cy - g.position_mm[1] * ppm;
      const aim = Math.atan2(cy - y, cx - x);
      c.fillStyle = 'rgba(255,79,0,.09)';
      c.beginPath(); c.moveTo(x, y); c.arc(x, y, Math.hypot(cx - x, cy - y) * 1.25, aim - .42, aim + .42); c.closePath(); c.fill();
      c.setLineDash([4, 4]); c.strokeStyle = INK3; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x, y); c.lineTo(cx, cy); c.stroke(); c.setLineDash([]);
      c.fillStyle = INK; c.beginPath(); c.arc(x, y, 14, 0, Math.PI * 2); c.fill();
      c.fillStyle = CARD; c.fillText(String(k + 1), x, y + 1);
      const d = Math.hypot(g.position_mm[0], g.position_mm[1]) || 1;
      c.fillStyle = INK2;
      c.fillText(Math.round(g.distance_mm || d) + ' MM', x + g.position_mm[0] / d * 32, y - g.position_mm[1] / d * 32);
    }
    c.fillStyle = cssVar('--signal', '#ff4f00'); c.beginPath(); c.arc(cx, cy, 3, 0, Math.PI * 2); c.fill();
  }

  function renderStatus() {
    for (const c of CAM_IDS) {
      const dl = document.getElementById('cam-detail-' + c);
      if (!dl) continue;
      const row = S.camRows && S.camRows[String(c)];
      const at = placement(c);
      const where = at ? [['Sits at', at.hour + ' o’clock · ' + Math.round(at.elevation) + '° off the board · ' + Math.round(at.distance) + ' mm away']] : [];
      const whereEl = document.getElementById('cam-where-' + c);
      if (whereEl) whereEl.textContent = at ? at.hour + ' o’clock · ' + Math.round(at.elevation) + '° · ' + Math.round(at.distance) + ' mm' : '';
      if (!row) { dl.replaceChildren(...where.concat([['Status', S.camStatusReason || 'No status yet']]).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)])); continue; }
      const kv = where.concat([
        ['Opened', row.opened ? 'yes' : 'no' + (row.last_error ? ' — ' + row.last_error : '')],
        ['Backend', row.backend_used || '—'],
        ['Resolution', (row.actual_width || '?') + '×' + (row.actual_height || '?') + ' (asked ' + row.requested_width + '×' + row.requested_height + ')'],
        ['Frame rate', (row.actual_fps ? row.actual_fps.toFixed(1) : '—') + ' fps nominal' + (row.effective_fps ? ', ' + row.effective_fps.toFixed(1) + ' measured' : '')],
        ['JPEG', row.jpeg_passthrough ? 'the camera’s own' : row.jpeg_synthetic ? 'encoded here, q' + row.jpeg_synthetic_quality : '—'],
        ['Frames read', String(row.frame_count)],
        ['Open took', row.open_latency_s ? row.open_latency_s.toFixed(2) + ' s' : '—'],
      ]);
      dl.replaceChildren(...kv.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
      renderLive(c);
    }
    renderSchedule();
  }

  // The schedule: one row per camera, the numbers that say whether the rig
  // is set up evenly and calibrated well. A fit worse than FIT_WARN_PX, or
  // a camera much further out than the others, is marked.
  const FIT_WARN_PX = 1.0, DISTANCE_WARN = 0.10;
  function renderSchedule() {
    const table = document.getElementById('cam-schedule');
    if (!table) return;
    const cal = S.calibration || {};
    const rows = CAM_IDS.map((c) => ({c, g: geometry && geometry[String(c)], at: placement(c),
      cal: (cal.cameras || {})[String(c)], live: S.camRows && S.camRows[String(c)]}));
    const dists = rows.map((r) => r.g && r.g.distance_mm).filter((d) => typeof d === 'number').sort((a, b) => a - b);
    const median = dists.length ? dists[Math.floor(dists.length / 2)] : null;
    const cell = (text, cls, title) => h('td', {class: cls || '', title: title || null}, text);
    const head = h('thead', {}, h('tr', {}, ['Cam', 'Sits at', 'Away', 'Angle', 'Focal length', 'Reproj. error', 'Live'].map((t, i) => h('th', {class: i > 1 && i < 6 ? 'r' : ''}, t))));
    let flagged = [];
    const body = h('tbody', {}, rows.map(({c, g, at, cal: cc, live}) => {
      const ok = cc && cc.ok === true;
      const err = cc && typeof cc.reprojection_error_px === 'number' ? cc.reprojection_error_px
        : g && typeof g.reprojection_error_px === 'number' ? g.reprojection_error_px : null;
      const far = median && g && typeof g.distance_mm === 'number' && Math.abs(g.distance_mm - median) / median > DISTANCE_WARN;
      const loose = err !== null && err > FIT_WARN_PX;
      if (far) flagged.push('Cam ' + (c + 1) + ' sits ' + Math.round(Math.abs(g.distance_mm - median)) + ' mm ' + (g.distance_mm > median ? 'further out' : 'closer in') + ' than the others');
      if (loose) flagged.push('Cam ' + (c + 1) + ' fits its calibration loosely (' + err.toFixed(2) + ' px) — worth a recalibrate');
      const fps = live && (live.effective_fps || live.actual_fps);
      return h('tr', {class: ok ? '' : 'uncal'},
        h('td', {}, h('span', {class: 'cn'}, h('span', {class: 'dot', dataset: {tone: !cc ? 'idle' : ok ? (loose ? 'warn' : 'live') : 'bad'}}), 'Cam ' + (c + 1))),
        cell(at ? at.hour + ' o’clock' : '—'),
        cell(g && g.distance_mm ? Math.round(g.distance_mm) + ' mm' : '—', 'r' + (far ? ' flag' : ''), far ? 'Noticeably further from the board than the others' : 'Distance from the bull'),
        cell(at && typeof at.elevation === 'number' ? Math.round(at.elevation) + '°' : '—', 'r', 'How far off the board’s face it looks'),
        cell(g && g.focal_length_px ? Math.round(g.focal_length_px) + ' px' : '—', 'r', 'The lens focal length the calibration solved for -- how zoomed-in this camera is. Identical cameras should agree'),
        cell(err !== null ? err.toFixed(2) + ' px' : ok ? 'saved' : '—', 'r' + (loose ? ' flag' : ''),
          err !== null ? 'Reprojection error: how far the calibrated board lands from the real one, on average. Under 1 px is good.'
            : ok ? 'Loaded from disk — Calibrate measures its fit afresh' : 'Not calibrated'),
        cell(live && live.opened ? (fps ? fps.toFixed(0) + ' fps' : '—') + ' · ' + (live.actual_width || '?') + '×' + (live.actual_height || '?') : 'off', 'dim'));
    }));
    table.replaceChildren(head, body);
    const from = {startup: 'at start-up', manual: 'by hand', persisted: 'loaded from disk'}[cal.source] || cal.source || '';
    $('#sched-when').textContent = cal.checked_at_utc ? 'calibrated ' + ago(cal.checked_at_utc) + (from ? ' · ' + from : '') : 'not calibrated yet';
    const note = $('#sched-note');
    note.textContent = flagged.length ? flagged.join('. ') + '.' : (dists.length === CAM_IDS.length ? 'All cameras even and well fitted.' : '');
    note.classList.toggle('warn', flagged.length > 0);
  }

  async function refreshStatus() {
    try {
      const data = await api.get('/api/cameras/status');
      S.camRows = data && data.available ? data.cameras || {} : null;
      S.camStatusReason = (data && data.reason) || '';
      emit('camstatus');
      renderStatus();
      update();
    } catch (err) { /* the next tick tries again */ }
  }

  // ---- sources (which device or stream feeds each slot) ----
  const URL_PREFIX = 'url:', URL_NEW = '__url__';
  function prettyStream(url) {
    try {
      const u = new URL(url);
      const cam = (u.pathname.split('/cameras/')[1] || '').split('/')[0];
      return 'Stream · ' + u.hostname + (u.port && u.port !== '80' ? ':' + u.port : '') + (cam ? ' cam ' + (Number(cam) + 1) : '');
    } catch (err) { return 'Stream · ' + (url.length > 34 ? url.slice(0, 31) + '…' : url); }
  }
  function renderSources() {
    const cams = S.runtime && S.runtime.cameras;
    const selects = CAM_IDS.map((c) => document.getElementById('cam-src-' + c)).filter(Boolean);
    if (!cams || cams.available === false) {
      selects.forEach((s) => { s.disabled = true; s.title = (cams && cams.reason) || 'Camera assignment unavailable'; });
      return;
    }
    const names = cams.device_names || [];
    const live = cams.devices || [];
    let total = Number.isInteger(cams.device_count) && cams.device_count > 0 ? cams.device_count : 10;
    live.forEach((d) => { if (Number.isInteger(d) && d >= total) total = d + 1; });
    const own = new Set(cams.own_virtual_devices || []);
    const urls = Array.from(new Set((cams.urls || []).concat(cams.saved_urls || []).filter(Boolean)));
    selects.forEach((sel) => {
      const slot = Number(sel.dataset.slot);
      const opts = [];
      for (let d = 0; d < total; d++) {
        if (own.has(d) && !live.includes(d)) continue;   // our own virtual camera: feeding it back in is a loop
        opts.push(h('option', {value: String(d)}, 'Device ' + d + (names[d] ? ' — ' + names[d] : '')));
      }
      urls.forEach((u) => opts.push(h('option', {value: URL_PREFIX + u}, prettyStream(u))));
      opts.push(h('option', {value: URL_NEW}, 'Read from a stream…'));
      if (document.activeElement !== sel) {
        sel.replaceChildren(...opts);
        const savedUrl = cams.saved_urls ? cams.saved_urls[slot] : null;
        const url = (cams.urls || [])[slot] || null;
        const shown = savedUrl || url;
        sel.value = shown ? URL_PREFIX + shown : String(live[slot]);
        sel.dataset.last = sel.value;
        sel.dataset.device = String(live[slot]);
        sel.title = url ? 'Reading frames from ' + url
          : shown ? 'Will read from ' + shown + ' after a restart'
            : 'Device ' + live[slot] + (names[live[slot]] ? ' (' + names[live[slot]] + ')' : '');
      }
      sel.disabled = false;
    });
  }

  function assignment() {
    const devices = [], urls = [];
    CAM_IDS.forEach((c) => {
      const sel = document.getElementById('cam-src-' + c);
      const v = String(sel.value || '');
      if (v.startsWith(URL_PREFIX)) { urls.push(v.slice(URL_PREFIX.length)); devices.push(Number(sel.dataset.device || c)); }
      else { urls.push(null); devices.push(Number(v)); }
    });
    return {devices, urls};
  }

  async function applySources() {
    const {devices, urls} = assignment();
    const local = devices.filter((d, i) => !urls[i]);
    if (new Set(local).size !== local.length) {
      toast('That device already feeds another camera slot', 'bad');
      renderSources();
      return;
    }
    await Rig.save({camera_devices: devices, camera_urls: urls}, null, 'Camera sources',
      (body, o) => o.applied.length ? 'Applied' + (body.restarted_capture ? ' — capture restarted' : '') : (o.note || 'Saved — applies after a restart'));
  }

  function normaliseUrl(raw, slot) {
    let v = String(raw || '').trim();
    if (!v) return null;
    if (!/^https?:\/\//i.test(v)) v = 'http://' + v;
    if (v.indexOf('/api/cameras/') === -1) v = v.replace(/\/+$/, '') + '/api/cameras/' + slot + '/stream.mjpg?full=1';
    return v;
  }
  let urlSlot = null;
  function openStream(slot) {
    urlSlot = slot;
    $('#stream-sub').textContent = 'Camera ' + (slot + 1) + ' will read frames from another rig instead of local hardware.';
    const cur = String(document.getElementById('cam-src-' + slot).dataset.last || '');
    $('#stream-url').value = cur.startsWith(URL_PREFIX) ? cur.slice(URL_PREFIX.length) : '';
    $('#stream-error').textContent = '';
    $('#stream-all-row').hidden = true;
    sheet.open('sheet-stream');
    setTimeout(() => $('#stream-url').focus(), 50);
  }
  $('#stream-url').addEventListener('input', (ev) => {
    const v = ev.target.value.trim();
    $('#stream-all-row').hidden = !(v && v.indexOf('/api/cameras/') === -1);
  });
  $('#stream-save').addEventListener('click', () => {
    const raw = $('#stream-url').value;
    if (!normaliseUrl(raw, urlSlot)) { $('#stream-error').textContent = 'Enter an address.'; return; }
    const allSlots = !$('#stream-all-row').hidden && $('#stream-all').checked;
    for (const c of allSlots ? CAM_IDS : [urlSlot]) {
      const sel = document.getElementById('cam-src-' + c);
      const url = normaliseUrl(raw, c);
      if (!Array.from(sel.options).some((o) => o.value === URL_PREFIX + url)) {
        sel.insertBefore(h('option', {value: URL_PREFIX + url}, prettyStream(url)), sel.lastElementChild);
      }
      sel.value = URL_PREFIX + url;
      sel.dataset.last = sel.value;
    }
    sheet.close('sheet-stream');
    applySources();
  });
  $('#sheet-stream').addEventListener('close', () => {
    if (urlSlot === null) return;
    const sel = document.getElementById('cam-src-' + urlSlot);
    if (sel && sel.value === URL_NEW) sel.value = sel.dataset.last || '';
    urlSlot = null;
  });
  document.addEventListener('change', (ev) => {
    const sel = ev.target;
    if (!sel.classList || !sel.classList.contains('cam-src')) return;
    if (sel.value === URL_NEW) { openStream(Number(sel.dataset.slot)); return; }
    sel.dataset.last = sel.value;
    applySources();
  });

  function suspend() { suspended = true; for (const c of CAM_IDS) { stop(c); hideOverlay(c); } update(); }
  function resume() { suspended = false; update(); refreshOverlays(); }
  function setOverlay(v) { overlayPref = !!v; prefs.set('overlay', overlayPref); refreshOverlays(); renderOverlayToggle(); emit('screenprefs'); }
  function renderOverlayToggle() {
    $$('#cam-overlay-toggle button').forEach((b) => b.setAttribute('aria-pressed', String((b.dataset.v === 'on') === overlayPref)));
  }
  $$('#cam-overlay-toggle button').forEach((b) => b.addEventListener('click', () => setOverlay(b.dataset.v === 'on')));

  build();
  renderOverlayToggle();
  on('theme', drawPlan);
  setInterval(update, TICK_MS);
  setInterval(() => { if (Shell.showing('rig')) refreshStatus(); }, TICK_MS);
  window.addEventListener('focus', () => { update(); refreshOverlays(); });
  window.addEventListener('blur', () => { update(); refreshOverlays(); });
  document.addEventListener('visibilitychange', () => { update(); refreshOverlays(); });
  // A reload or close must release the sockets explicitly: a multipart
  // connection left to the browser lingers half-open against the budget.
  window.addEventListener('pagehide', () => stopAll());
  on('view', () => { update(); refreshOverlays(); if (Shell.showing('rig')) { refreshStatus(); refreshGeometry(); } });
  on('phase', update);
  on('calibration', () => { renderCalibration(); if (!S.calibrating && Shell.showing('rig')) refreshGeometry(); });
  on('config', renderSources);
  on('framehealth', () => CAM_IDS.forEach(renderLive));

  return {update, stopAll, suspend, resume, refreshStatus, renderCalibration, setOverlay, get overlay() { return overlayPref; }};
})();

// ---- settings -----------------------------------------------------------------

const Rig = (() => {
  function applyConfig(body) {
    if (!body || !body.ok || !body.config) return false;
    S.config = body.config;
    if (body.runtime) {
      const heldAd = S.ad;
      S.runtime = body.runtime;
      if (body.runtime.ad && acceptAd(body.runtime.ad)) S.ad = body.runtime.ad;
      else if (heldAd) S.runtime.ad = heldAd;
    }
    S.restartRequired = body.restart_required || [];
    emit('config');
    return true;
  }

  async function refreshConfig(names) {
    try { applyConfig(await api.get('/api/config' + (names ? '?refresh_camera_names=true' : ''))); } catch (err) { /* retried on the next event */ }
  }

  // Save a partial config document and answer inline, beside the control:
  // "Saved", "Saved — applies after restart", or the server's own reason.
  async function save(patch, noteEl, label, describe) {
    const keys = Object.keys(patch);
    if (noteEl) { noteEl.className = 'note saving'; noteEl.textContent = 'Saving…'; }
    const act = activity(label, 'pending', 'saving…');
    let body;
    try { body = await api.patch('/api/config', patch); } catch (err) { body = null; }
    if (!body || !body.ok) {
      const errors = (body && body.errors) || {};
      const reason = keys.map((k) => errors[k]).find(Boolean) || Object.values(errors)[0] || (body && body.reason) || 'Could not reach the rig';
      if (noteEl) { noteEl.className = 'note bad'; noteEl.textContent = reason; }
      act.done('bad', reason);
      refreshConfig();
      return null;
    }
    applyConfig(body);
    const pending = keys.filter((k) => (body.restart_required || []).indexOf(k) >= 0);
    const note = keys.map((k) => (body.notes || {})[k]).find(Boolean) || null;
    const applied = keys.filter((k) => (body.applied_live || []).indexOf(k) >= 0);
    const text = describe ? describe(body, {pending, note, applied}) : (note || (pending.length ? 'Saved — applies after a restart' : 'Saved'));
    if (noteEl) {
      noteEl.className = 'note ' + (pending.length || note ? 'warn' : 'ok');
      noteEl.textContent = text;
      // A plain "Saved" fades; the row then goes back to its standing note.
      if (!pending.length && !note) {
        setTimeout(() => {
          if (noteEl.textContent !== text) return;
          noteEl.className = 'note';
          noteEl.textContent = '';
          emit('config');
        }, 2500);
      }
    }
    act.done(pending.length || note ? 'warn' : 'ok', text);
    return body;
  }

  // A note is showing the answer to a save (Saving, Saved, a refusal):
  // a re-render must not paint the row's standing note over it.
  const answering = (el) => /\b(saving|ok|warn|bad)\b/.test(el.className);
  const standing = (sel, text, cls) => {
    const el = $(sel);
    if (answering(el)) return;
    el.className = 'note' + (cls ? ' ' + cls : '');
    el.textContent = text;
  };

  const val = (key, fallback) => (S.config && S.config[key] !== undefined && S.config[key] !== null ? S.config[key] : fallback);
  const idle = (el) => document.activeElement !== el;
  const pressSeg = (sel, v) => $$(sel + ' button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === String(v))));

  function frameRingPrice(fr, seconds) {
    const perS = fr && fr.estimated_bytes_per_s ? fr.estimated_bytes_per_s : 0;
    return bytes(perS * seconds);
  }
  function ringNote(fr, seconds) {
    if (!fr) return '';
    if (!seconds) return 'Off — a missed or misscored dart cannot be captured afterwards.';
    const parts = ['About ' + frameRingPrice(fr, seconds) + ' of memory once full'
      + (fr.estimate_source === 'measured' ? ', from what this rig holds now.' : ' (uncompressed estimate).')];
    const ring = fr.ring || {};
    if (!fr.attached) parts.push(fr.reason || 'Not running in this process.');
    else if (ring.sets) {
      parts.push('Holding ' + ring.span_s + ' s now.');
      if (ring.capped) parts.push('Capped — shorter than set.');
      if (ring.paused) parts.push('Paused for a capture in progress.');
    }
    return parts.join(' ');
  }

  function renderAll() {
    if (!S.config) return;
    // Scoring
    const frames = S.config.lifecycle_settings ? S.config.lifecycle_settings.dart_stable_frames : null;
    if (frames !== null && frames !== undefined) pressSeg('#set-speed', frames);
    const idleSel = $('#set-idle');
    const idleSec = val('idle_timeout_sec', null);
    if (idleSec !== null && idle(idleSel)) {
      const want = String(idleSec);
      if (!Array.from(idleSel.options).some((o) => o.value === want)) {
        idleSel.append(h('option', {value: want}, String(Math.round(idleSec / 60))));
      }
      idleSel.value = want;
      idleUnit();
    }
    const ec = S.config.engine_config || {};
    $('#engine-summary').textContent = (ec.primary || '—')
      + ((ec.also_run || []).length ? ' — voting ' + ec.also_run.join(' · ') : '')
      + (ec.timeout_s ? ' · ' + ec.timeout_s + ' s limit' : '');
    // Recording
    $('#set-store').checked = !!val('store_packages', true);
    const running = S.runtime && S.runtime.store_packages;
    const differs = running && running.running !== undefined && running.running !== null && running.running !== !!val('store_packages', true);
    standing('#note-store', differs ? 'This session is still ' + (running.running ? 'saving' : 'not saving') + ' — the change takes effect at the next Start.' : '', differs ? 'info' : '');
    const clips = $('#set-clips');
    if (idle(clips)) clips.value = val('video_record_mode', VIDEO_RECORD_MODE);
    const fr = S.runtime && S.runtime.frame_ring;
    const ringIn = $('#set-ring');
    if (idle(ringIn)) ringIn.value = String(val('frame_ring_seconds', fr ? fr.configured_seconds : 0));
    standing('#note-ring', ringNote(fr, parseFloat(ringIn.value)));
    const floor = $('#set-floor');
    if (idle(floor)) floor.value = String(val('min_free_disk_gb', 5));
    // Autodarts
    renderAd();
    // System
    const port = $('#set-port');
    if (idle(port)) port.value = String(val('port', ''));
    $('#host-label').textContent = (S.runtime && S.runtime.host && S.runtime.host.active) || val('host', '—');
    $('#set-always').checked = !!val('always_update', false);
    renderMaintenance();
    renderBanner();
  }

  function setNote(sel, cls, text) { const el = $(sel); el.className = 'note ' + (cls || ''); el.textContent = text; }

  function renderAd() {
    const ad = S.ad || (S.runtime && S.runtime.ad);
    const on_ = $('#set-ad'), url = $('#set-ad-url');
    if (!S.config || !ad || ad.available === false) {
      on_.disabled = true; url.disabled = true;
      setNote('#note-ad', '', (ad && ad.reason) || 'Autodarts is not wired in this process.');
      $('#ad-dot').dataset.tone = 'idle';
      return;
    }
    on_.disabled = false; url.disabled = false;
    on_.checked = !!S.config.ad_enabled;
    if (idle(url)) url.value = S.config.ad_base_url || '';
    const tone = !S.config.ad_enabled ? 'idle' : ad.connected ? 'live' : 'bad';
    $('#ad-dot').dataset.tone = tone;
    setNote('#note-ad', tone === 'bad' ? 'bad' : '', !S.config.ad_enabled
      ? 'Off — Autodarts is never contacted. On Windows and Linux this also removes the virtual cameras.'
      : ad.connected ? 'Connected to ' + S.config.ad_base_url + '.'
        : 'Not connected to ' + S.config.ad_base_url + ' — no comparisons are being recorded.');
    const b = S.adBoard || {};
    const stuck = b.status === 'takeout' && typeof b.age === 'number' && b.age >= 30;
    $('#ad-board').textContent = !S.config.ad_enabled ? '' : b.status
      ? 'Autodarts’ board: ' + ({ready: 'ready', takeout: 'takeout', stopped: 'stopped'}[b.status] || b.status)
        + (stuck ? ' — stuck for ' + Math.round(b.age) + ' s, so it is not scoring. Clear the board.' : '')
      : '';
    $('#ad-board').className = 'note' + (stuck ? ' bad' : '');
  }

  function renderMaintenance() {
    const always = val('always_update', null);
    const queued = val('update_on_next_restart', null);
    $('#btn-update-restart').hidden = !!always;
    $('#note-restart').textContent = always === null ? ''
      : always ? 'This rig follows its branch: every restart pulls the latest code first.'
        : queued ? 'An update is queued — the next restart pulls first.'
          : 'Restart relaunches the code already on this rig.';
  }

  function renderBanner() {
    const keys = S.restartRequired || [];
    const b = $('#restart-banner');
    b.hidden = keys.length === 0;
    if (keys.length) $('#restart-keys').textContent = keys.join(', ');
  }

  // ---- system ----
  function meter(el, pct, label, tone) {
    el.querySelector('.meter-fill').style.width = Math.max(0, Math.min(100, pct || 0)).toFixed(0) + '%';
    el.querySelector('.meter-fill').dataset.tone = tone || '';
    el.querySelector('.meter-val').textContent = label;
  }
  function renderLoad(cfg) {
    if (!cfg) return;
    const cpu = cfg.cpu, mem = cfg.memory, disk = cfg.disk;
    if (cpu && cpu.busy_pct !== undefined) meter($('#meter-cpu'), cpu.busy_pct, Math.round(cpu.busy_pct) + '%' + (cpu.cores ? ' of ' + cpu.cores + ' cores' : ''), cpu.busy_pct > 85 ? 'bad' : cpu.busy_pct > 60 ? 'warn' : '');
    if (mem && mem.total_bytes) meter($('#meter-mem'), mem.used_pct, bytes(mem.available_bytes) + ' free of ' + bytes(mem.total_bytes), mem.used_pct > 90 ? 'bad' : mem.used_pct > 80 ? 'warn' : '');
    if (disk && disk.total_bytes) {
      const used = 100 - (disk.free_pct || 0);
      const g = disk.guard || {};
      meter($('#meter-disk'), used, bytes(disk.free_bytes) + ' free' + (g.below_floor ? ' — below the floor, saving stopped' : ''), g.below_floor ? 'bad' : used > 90 ? 'warn' : '');
    }
  }
  function renderBuild() {
    const b = S.build;
    if (!b) return;
    const kv = [
      ['Machine', b.hostname || '—'],
      ['Version', (b.app_version || '?') + (b.code_version ? ' · ' + b.code_version.slice(0, 7) : '')],
      ['Running for', duration(b.uptime_s)],
      ['Platform', (b.platform || '') + ' ' + (b.platform_release || '')],
      ['Python', b.python_version || '—'],
      ['OpenCV', (b.opencv_version || '—') + (b.opencv_threads ? ' · ' + b.opencv_threads + ' threads' : '')],
    ];
    $('#sys-rig').replaceChildren(...kv.flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, v)]));
    $('#sys-host').textContent = b.hostname || '';
  }
  async function refreshHealth() {
    try {
      const body = await api.get('/api/health');
      if (body && body.build) { S.build = Object.assign({}, body.build, {pid: body.pid}); renderBuild(); }
    } catch (err) { /* shown as it was */ }
  }
  async function refreshLoad() {
    try {
      const st = await api.get('/api/state');
      if (st && st.config) renderLoad(st.config);
    } catch (err) { /* next tick */ }
  }
  // Frames keeping up on both sides (docs/LIVE_API.md, /api/frame-health):
  // is each camera read at its own rate, and is whatever reads our virtual
  // cameras collecting every frame we hand it.
  const fourccMatches = (want, got) => {
    const w = String(want).toUpperCase(), g = String(got).toUpperCase();
    return w === g || (w === 'BGR24' && g === 'BGR3') || (w === 'MJPEG' && g === 'MJPG');
  };
  function renderPublishing(fh) {
    const list = $('#vcam-list');
    const pub = (fh && fh.publish) || {};
    const line = (tone, name, state, what) => h('li', {class: 'vcam'}, h('span', {class: 'vn'}, name), h('span', {class: 'vs', dataset: {tone}}, state), h('span', {class: 'vw'}, what));
    if (!fh) { list.replaceChildren(h('li', {class: 'muted'}, 'Unavailable.')); return; }
    if (!pub.enabled) { list.replaceChildren(line('idle', 'ALL', 'Off', 'No virtual cameras are being fed.')); return; }
    if (fh.frame_sink_attached === false) { list.replaceChildren(line('bad', 'ALL', 'Not wired', 'Publishers exist but nothing feeds them — frames are captured and never offered.')); return; }
    const slots = pub.slots || [];
    if (!slots.length) { list.replaceChildren(line('warn', 'ALL', 'No slots', 'Publishing is on, but no virtual camera exists.')); return; }
    list.replaceChildren(...slots.map((sl) => {
      const name = 'CAM ' + ((sl.slot === undefined ? 0 : sl.slot) + 1);
      if (sl.open === false || (sl.open === undefined && sl.published === undefined)) {
        return line('warn', name, 'Not open', (sl.device ? sl.device + ' · ' : '') + (sl.last_error || 'nothing published yet'));
      }
      const neg = sl.negotiated || {};
      const res = neg.width && neg.height ? neg.width + '×' + neg.height : (sl.width && sl.height ? sl.width + '×' + sl.height : null);
      const fmtBad = neg.fourcc && sl.format && !fourccMatches(sl.format, neg.fourcc);
      const bits = [];
      if (sl.device) bits.push(sl.device);
      bits.push(fmtBad ? sl.format + ' asked, ' + neg.fourcc + ' in force' : (sl.format || '?'));
      if (res) bits.push(res);
      bits.push((sl.published || 0).toLocaleString() + ' frames');
      let tone = 'live';
      if (sl.write_errors) { bits.push(sl.write_errors + ' write errors'); tone = 'bad'; }
      if (sl.consumer_attached === false) { bits.push('nobody reading'); tone = tone === 'bad' ? 'bad' : 'warn'; }
      else if (sl.consumer_attached === true) {
        bits.push((sl.read || 0).toLocaleString() + ' read');
        if (sl.missed) { bits.push(sl.missed + ' missed'); tone = 'warn'; }
        if (sl.torn) { bits.push(sl.torn + ' torn'); tone = 'bad'; }
      }
      if (fmtBad) tone = 'bad';
      const state = tone === 'live' ? 'Publishing' : tone === 'warn' ? (sl.missed ? 'Falling behind' : 'Nobody reading') : 'Trouble';
      return line(tone, name, state, bits.join(' · '));
    }));
  }
  async function refreshFrameHealth() {
    const fh = await api.get('/api/frame-health').catch(() => null);
    S.frameHealth = fh && fh.ok !== false ? fh : null;
    renderPublishing(S.frameHealth);
    emit('framehealth');
  }

  // ---- restart ----
  async function restart(update) {
    const yes = await confirmSheet({
      title: update ? 'Update and restart?' : 'Restart the rig?',
      body: '<p>' + (update ? 'The rig pulls the latest code for its branch, then starts again.' : 'The same code starts again.')
        + ' Scoring stops now, and <strong>does not start again on its own</strong> — press Start when the page comes back.</p>',
      confirm: update ? 'Update and restart' : 'Restart',
    });
    if (!yes) return;
    const act = activity(update ? 'Update and restart' : 'Restart', 'pending', 'restarting…');
    $('#btn-restart').disabled = true; $('#btn-update-restart').disabled = true;
    let body = null;
    try { body = await api.post('/api/restart', update ? {update: true} : undefined); } catch (err) { body = null; }
    if (!body || !body.ok) {
      act.done('bad', (body && body.reason) || 'request failed');
      $('#btn-restart').disabled = false; $('#btn-update-restart').disabled = false;
      return;
    }
    Shell.overlay('Restarting…', 'The page reloads when the rig answers.');
    const deadline = Date.now() + 240000;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 1000));
      try {
        const h2 = await (await fetch('/api/health', {cache: 'no-store'})).json();
        // A different pid is the proof: the old process can still answer
        // for a moment after it was asked to go.
        if (h2 && (h2.pid === undefined || h2.pid !== body.pid)) { act.done('ok', 'back — reloading'); location.reload(); return; }
      } catch (err) { /* down, as expected */ }
    }
    act.done('bad', 'no answer after 4 minutes — it may still be installing; reload to check');
    Shell.overlay(null);
    $('#btn-restart').disabled = false; $('#btn-update-restart').disabled = false;
  }

  // ---- diagnostics ----
  function diagnosticsText() {
    const L = [];
    const b = S.build || {};
    L.push('opendarts ' + (b.app_version || '?') + ' (' + (b.code_version || '?') + ') on ' + (b.hostname || '?'));
    L.push('platform: ' + (b.platform || '') + ' ' + (b.platform_release || '') + ', python ' + (b.python_version || '?') + ', opencv ' + (b.opencv_version || '?'));
    L.push('uptime: ' + duration(b.uptime_s) + ', pid ' + (b.pid || '?'));
    const p = rigPhase();
    L.push('status: ' + p.label + (p.detail ? ' — ' + p.detail : ''));
    const cal = (S.calibration && S.calibration.live_source) || {};
    L.push('calibration: ' + (cal.calibration_package_id || 'none') + ' (' + (cal.source || '?') + ', ' + (cal.checked_at_utc || '?') + ')');
    for (const c of CAM_IDS) {
      const r = S.camRows && S.camRows[String(c)];
      L.push('cam' + c + ': ' + (r ? [r.backend_used, r.actual_width + 'x' + r.actual_height, (r.actual_fps || 0).toFixed(1) + 'fps',
        'opened=' + r.opened, 'frames=' + r.frame_count, r.jpeg_passthrough ? 'jpeg=camera' : r.jpeg_synthetic ? 'jpeg=synthetic' : 'jpeg=none',
        r.last_error ? 'error=' + r.last_error : ''].filter(Boolean).join(' ') : 'no status'));
    }
    const ad = S.ad || {};
    L.push('autodarts: ' + (S.config && S.config.ad_enabled ? 'on' : 'off') + ', ' + (ad.connected ? 'connected' : 'not connected') + ' (' + ((S.config && S.config.ad_base_url) || '') + ')');
    const snd = Sound.status();
    L.push('sound here: ' + snd.label + ' — ' + snd.detail);
    L.push('restart required: ' + ((S.restartRequired || []).join(', ') || 'none'));
    L.push('config: ' + JSON.stringify(Object.assign({}, S.config || {}, {capabilities: undefined})));
    return L.join('\n');
  }
  async function copyDiagnostics() {
    const text = diagnosticsText();
    try {
      await navigator.clipboard.writeText(text);
      toast('Diagnostics copied — paste them into the bug report', 'ok');
    } catch (err) {
      const ta = h('textarea', {class: 'offscreen'}, text);
      document.body.append(ta);
      ta.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (err2) { ok = false; }
      ta.remove();
      toast(ok ? 'Diagnostics copied' : 'This browser would not copy — select the text from the console instead', ok ? 'ok' : 'warn');
      if (!ok) console.log(text);
    }
  }

  // ---- wiring ----
  $$('#set-speed button').forEach((b) => b.addEventListener('click', () => {
    pressSeg('#set-speed', b.dataset.v);
    save({lifecycle_settings: {dart_stable_frames: Number(b.dataset.v)}}, $('#note-speed'), 'Detection speed',
      (body, o) => o.note || 'Saved — ' + plural(Number(b.dataset.v), 'still frame') + ' before a dart is scored');
  }));
  // "Never" has no unit; the slot keeps its width so the row still lines up.
  function idleUnit() { $('#idle-unit').classList.toggle('blank', $('#set-idle').value === '0'); }
  $('#set-idle').addEventListener('change', (ev) => {
    idleUnit();
    const sec = Number(ev.target.value);
    save({idle_timeout_sec: sec}, $('#note-idle'), 'Auto-stop', () => sec ? 'Saved — stops after ' + Math.round(sec / 60) + ' min without a dart' : 'Saved — never stops on its own');
  });
  $('#set-store').addEventListener('change', (ev) => save({store_packages: ev.target.checked}, $('#note-store'), 'Save throws',
    () => (ev.target.checked ? 'Saving throws' : 'Not saving throws') + ' from the next Start'));
  $('#set-clips').addEventListener('change', (ev) => save({video_record_mode: ev.target.value}, $('#note-clips'), 'Video clips'));
  $('#set-ring').addEventListener('input', (ev) => { $('#note-ring').className = 'note'; $('#note-ring').textContent = ringNote(S.runtime && S.runtime.frame_ring, parseFloat(ev.target.value)); });
  $('#set-ring').addEventListener('change', (ev) => {
    const s = parseFloat(ev.target.value);
    save({frame_ring_seconds: s}, $('#note-ring'), 'Throw buffer', (body) => {
      const fr = body.runtime && body.runtime.frame_ring;
      return s ? 'Saved — ' + s + ' s' + (fr && fr.estimated_label ? ', about ' + fr.estimated_label : '') : 'Off';
    });
  });
  $('#set-floor').addEventListener('change', (ev) => save({min_free_disk_gb: parseFloat(ev.target.value)}, $('#note-floor'), 'Disk floor'));
  $('#set-ad').addEventListener('change', (ev) => {
    save({ad_enabled: ev.target.checked}, $('#note-ad'), 'Autodarts', (body) => body.config.ad_enabled ? 'On — connecting…' : 'Off');
    setTimeout(refreshConfig, 2000);   // the socket comes up a moment later
  });
  $('#set-ad-url').addEventListener('change', (ev) => {
    const v = ev.target.value.trim();
    if (!v) { renderAd(); return; }
    save({ad_base_url: v}, $('#note-ad-url'), 'Autodarts address');
    setTimeout(refreshConfig, 2000);
  });
  $('#set-port').addEventListener('change', (ev) => save({port: Number(ev.target.value)}, $('#note-port'), 'Port'));
  $('#set-always').addEventListener('change', (ev) => save({always_update: ev.target.checked}, $('#note-always'), 'Always update',
    () => ev.target.checked ? 'On — every restart pulls the latest code' : 'Off — restarts keep the code already here'));
  $('#btn-restart').addEventListener('click', () => restart(false));
  $('#btn-update-restart').addEventListener('click', () => restart(true));
  $('#restart-now').addEventListener('click', () => restart(false));
  $('#btn-copy-diag').addEventListener('click', copyDiagnostics);
  $('#btn-delete-recorded2').addEventListener('click', () => $('#delete-recorded').click());
  $('#btn-calibrate').addEventListener('click', () => Session.calibrate());
  $('#btn-relearn').addEventListener('click', async () => {
    const yes = await confirmSheet({title: 'Relearn the ring layout?', body: '<p>Forgets this rig’s learned camera layout; the next calibration learns it again from scratch. Rarely needed — a calibration that finds the cameras moved does this by itself.</p>', confirm: 'Relearn'});
    if (!yes) return;
    const act = activity('Relearn ring layout', 'pending');
    const body = await api.post('/api/calibration/relearn-ring-geometry').catch(() => null);
    act.done(body && body.ok ? 'ok' : 'bad', body && body.ok ? (body.cleared ? 'cleared — calibrate to relearn' : 'nothing was stored') : (body && body.reason) || 'request failed');
  });
  $('#screens-open').addEventListener('click', () => Shell.openScreen());

  async function renderRecorded() {
    const el = $('#disk-summary');
    const snap = await api.get('/api/recorded-data').catch(() => null);
    if (!snap || snap.ok === false) { el.textContent = 'Could not read what is on disk.'; return; }
    const pk = snap.packages || {}, cp = snap.captures || {};
    el.textContent = plural(pk.count || 0, 'throw') + ' (' + (pk.label || '0 B') + ') and ' + plural(cp.count || 0, 'capture') + ' (' + (cp.label || '0 B') + ') on this rig'
      + (snap.disk ? ' · ' + bytes(snap.disk.free_bytes) + ' free' : '');
  }

  async function renderScreens() {
    const body = await api.get('/api/audio/clients').catch(() => null);
    const list = $('#screens-list');
    const rows = (body && body.clients) || [];
    if (!rows.length) { list.replaceChildren(h('li', {class: 'muted'}, 'No screen has reported in yet.')); return; }
    list.replaceChildren(...rows.map((c) => {
      let tone = 'idle', what = 'Sound off';
      if (c.blocked) { tone = 'bad'; what = 'Blocked — needs a tap'; }
      else if (!c.enabled) { tone = 'idle'; what = 'Sound off'; }
      else if (c.load_state === 'error') { tone = 'bad'; what = 'Voice failed to load'; }
      else if (c.context_state !== 'running') { tone = 'warn'; what = c.context_state; }
      else if (!c.ready) { tone = 'warn'; what = 'Loading…'; }
      else { tone = 'live'; what = 'Speaking · ' + (c.voice || '?'); }
      // The audio clock falling behind is what a clipped call looks like
      // from inside the page.
      if (c.stalls) { what += ' · ' + plural(c.stalls, 'audio stall') + (c.worst_stall_ms ? ' (worst ' + c.worst_stall_ms + ' ms)' : ''); if (tone === 'live') tone = 'warn'; }
      if (typeof c.output_latency_ms === 'number') what += ' · ' + (c.base_latency_ms || 0) + '+' + c.output_latency_ms + ' ms out';
      return h('li', {class: 'screen-row'},
        h('span', {class: 'dot', dataset: {tone}}),
        h('span', {class: 'screen-name'}, (c.label || 'Unknown screen') + (c.client_id === Sound.clientId ? ' — this one' : '')),
        h('span', {class: 'screen-state'}, what),
        h('span', {class: 'age'}, Math.round(c.age_s) + ' s ago'));
    }));
  }

  // Which dashboard the rig serves (opendarts/live/dashboard_choice.py).
  // Flipping it reloads every open dashboard onto the other page -- this
  // one included, via DASHBOARD_SWITCHED.
  async function renderDashboard() {
    const body = await api.get('/api/dashboard').catch(() => null);
    const ui = body && body.ui;
    $$('#set-dashboard button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === ui)));
  }
  $$('#set-dashboard button').forEach((b) => b.addEventListener('click', async () => {
    if (b.getAttribute('aria-pressed') === 'true') return;
    const note = $('#note-dashboard');
    note.className = 'note saving';
    note.textContent = 'Switching every screen to the ' + b.dataset.v + ' dashboard…';
    const r = await api.put('/api/dashboard', {ui: b.dataset.v}).catch(() => null);
    if (!r || r.ok === false) { note.className = 'note bad'; note.textContent = (r && r.reason) || 'Could not reach the rig'; return; }
    renderDashboard();
  }));
  on('view', (name) => { if (name === 'rig') renderDashboard(); });
  on('config', renderAll);
  on('adboard', renderAd);
  on('state', () => { if (S.state && S.state.config) renderLoad(S.state.config); });
  // The index follows the reading: whichever section is at the top of the
  // scroll is the one marked.
  function spy() {
    const body = $('#rig-body');
    const top = body.getBoundingClientRect().top + 40;
    let current = 'cameras';
    for (const a of $$('.rig-index a')) {
      const sec = document.getElementById('rig-' + a.dataset.sec);
      if (sec && sec.getBoundingClientRect().top <= top) current = a.dataset.sec;
    }
    $$('.rig-index a').forEach((a) => a.classList.toggle('on', a.dataset.sec === current));
  }
  $('#rig-body').addEventListener('scroll', spy, {passive: true});

  on('view', (name) => {
    if (name !== 'rig') return;
    requestAnimationFrame(spy);
    refreshConfig(); refreshHealth(); renderRecorded(); renderScreens(); refreshFrameHealth();
  });
  on('packages', () => { if (Shell.showing('rig')) renderRecorded(); });
  setInterval(() => { if (Shell.showing('rig')) refreshLoad(); }, 2500);
  setInterval(() => { if (Shell.showing('rig')) refreshHealth(); }, 30000);
  setInterval(() => { if (Shell.showing('rig')) { renderScreens(); refreshFrameHealth(); } }, 15000);

  return {save, refreshConfig, applyConfig, refreshHealth, renderScreens};
})();
