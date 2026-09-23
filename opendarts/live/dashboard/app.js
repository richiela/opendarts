// Everything the server computes for this page -- the camera count and the
// board vocabulary below -- arrives in ONE JSON block, the
// <script id="bootstrap" type="application/json"> in index.html, read once
// here. Nothing else in this file is substituted: it is a plain static
// asset, which is the point of it no longer living inside a Python
// f-string, where every brace had to be written doubled and three separate
// substitution slips shipped a page that died at the first JS error.
const OD_BOOTSTRAP = JSON.parse(
  document.getElementById('bootstrap').textContent
);

const CAM_IDS = OD_BOOTSTRAP.cam_ids;

// SCREENS FOLLOW THE SERVER'S BUILD. A page runs the JS it loaded, so after an
// update every open screen keeps the old page until someone reloads it -- and
// a kiosk has nobody to do that. Every WebSocket HELLO carries the server's
// page fingerprint; a screen that reconnects to a different one reloads
// itself. The URL survives a reload, so `?sound=on#engines` comes back as it
// was.
//
// AT MOST ONCE PER NEW VERSION, remembered for this tab: if a reload somehow
// serves the old page again (a caching proxy, a server mid-restart), the
// next HELLO would otherwise reload it again, forever. Missing on either side
// (an older server, a garbled bootstrap) means "cannot tell", never "reload".
// Returns true when it has asked for a reload, so the caller stops there.
function reloadIfPageIsStale(serverVersion, myVersion, store, reload) {
  if (!serverVersion || !myVersion || serverVersion === myVersion) return false;
  const KEY = 'opendarts.reloadedForPage';
  let already = null;
  try { already = store && store.getItem(KEY); } catch (err) { already = null; }
  if (already === serverVersion) {
    console.warn('server page ' + serverVersion + ' still differs from this one ('
                 + myVersion + ') after a reload -- not reloading again');
    return false;
  }
  try { if (store) store.setItem(KEY, serverVersion); } catch (err) { /* private mode */ }
  reload();
  return true;
}
// Housekeeping tick for the camera tiles, NOT a frame rate: the live
// preview is an MJPEG stream the browser pulls continuously (see the
// cameras section). This cadence only drives overlay-still refreshes
// and retry/teardown of stream connections.
const CAMERA_FEED_TICK_MS = 3000;

// Board vocabulary for the "Which was actually right?" modal's manual
// sector/ring picker -- injected from opendarts.geometry.board itself
// (SECTOR_NUMBERS_CLOCKWISE, and board_ring_names() which derives the
// ring names by probing sector_ring_for_point), NOT a hand-typed list in
// this template. A human's manually-confirmed segment has to be spelled
// exactly the way opendarts's own scorer spells it or it could never
// compare equal to any engine's answer.
const BOARD_SECTORS = OD_BOOTSTRAP.board_sectors;
const BOARD_RINGS = OD_BOOTSTRAP.board_rings;
// "never" | "mismatch" | "all" -- see the Save-frames gate in
// fmtAdWrongCell(). Restart-scoped, so reading it once here is safe.
const VIDEO_RECORD_MODE = OD_BOOTSTRAP.video_record_mode;
const CAMERA_STATUS_REFRESH_MS = 3000;

// -- action log + shared control-busy state ----------------------------
// Added 2026-08-12, the project's real, live-confirmed incident: clicking
// Calibrate with no visible confirmation caused 4 rapid duplicate clicks
// in ~4s (real log evidence). Two real fixes,
// both here: (1) EVERY control button click gets an immediate, persistent,
// timestamped log line (logAction) that stays on screen and updates in
// place once the request resolves -- not just a transient button-label
// change that's easy to miss; (2) ALL FOUR control buttons (not just the
// one clicked) are disabled together for the duration of any single
// in-flight action (setControlsBusy) -- a second click on the SAME button,
// or any other control button, simply cannot register while one request
// is still outstanding, closing the actual race that let 4 Calibrate
// clicks land as 4 separate real recalibration solves.
const ACTION_LOG_MAX_LINES = 10;

function fmtClockTime(d) {
  d = d || new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
}

// Returns the created line element so a caller (see the button handlers
// below) can update it in place once its own request resolves, instead of
// appending a SECOND line for the same click's outcome.
function logAction(label, status, detail) {
  const log = document.getElementById('action-log');
  if (!log) return null;
  const line = document.createElement('div');
  line.className = 'action-line ' + status;
  line.dataset.startedAt = fmtClockTime(); // preserved across updateActionLine() below
  line.innerHTML = '<span class="t">' + line.dataset.startedAt + '</span>' +
    '<span class="l">' + label + '</span>' +
    (detail ? ' \u2014 ' + detail : '');
  log.prepend(line);
  while (log.children.length > ACTION_LOG_MAX_LINES) log.lastChild.remove();
  return line;
}

// 2026-08-16: an action line shows its completion time as well as its
// start time -- the ORIGINAL start
// timestamp (line.dataset.startedAt, set once by logAction() above) is
// now kept as-is rather than overwritten with "now" on every update, and
// a real completion time is appended alongside it once the action
// actually resolves, so a single line shows both "when this started" and
// "when this finished" instead of losing the start time on resolution.
function updateActionLine(line, status, label, detail) {
  if (!line) { logAction(label, status, detail); return; }
  line.className = 'action-line ' + status;
  const started = line.dataset.startedAt || fmtClockTime();
  const finished = fmtClockTime();
  const timeText = status === 'pending' ? started : started + ' \u2192 ' + finished;
  line.innerHTML = '<span class="t">' + timeText + '</span>' +
    '<span class="l">' + label + '</span>' +
    (detail ? ' \u2014 ' + detail : '');
}

const CONTROL_BUTTON_IDS = ['btn-start', 'btn-stop', 'btn-reset', 'btn-refresh-calib'];
function setControlsBusy(busy) {
  for (const id of CONTROL_BUTTON_IDS) {
    const b = document.getElementById(id);
    if (b) b.disabled = busy;
  }
}

function fmtBool(v) {
  if (v === true) return '<span class="badge ok">ok</span>';
  if (v === false) return '<span class="badge bad">no</span>';
  return '<span class="badge unknown">unknown</span>';
}

function fmtVal(v, suffix) {
  if (v === null || v === undefined) return '&mdash;';
  return v + (suffix || '');
}

// Engine-row-vs-AD-row sector/ring comparison badge -- same 3-state color
// convention as fmtBool (green/red/grey). "PASS"/"FAIL".
function fmtMatch(v) {
  if (v === true) return '<span class="badge ok">PASS</span>';
  if (v === false) return '<span class="badge bad">FAIL</span>';
  return '<span class="badge unknown">&mdash;</span>';
}

// Attribute-safe text -- session/throw_id/note end up inside HTML
// attributes (data-session/data-throw/title) built via string
// concatenation, not a templating engine that escapes automatically.
function escapeAttr(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/"/g, '&quot;')
    .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// The "AD wrong" human-judgment toggle. A visible "AD wrong (human)"
// badge -- deliberately worded "(human)" so it never reads as something
// opendarts itself decided -- plus a toggle button whose label/data-wrong
// flips between marking and unmarking. data-session/data-throw carry
// what POST /api/packages/{session}/{throw_id}/mark-ad-wrong needs;
// reading them off the clicked button (event delegation, see below)
// means this never closes over stale package data from an earlier render.
// AD row's own "match" cell -- 2026-08-13 restructure: the AD row is the
// reference, so it doesn't get a PASS/FAIL badge (that's what its own
// engine rows below it get, against IT) -- it gets the "Mark AD wrong"
// human-judgment control instead. ALWAYS shown now.
// AD answer latency for the throw, signed ms (`ad_latency_ms` off
// /api/packages, itself ad_ground_truth.json's `staleness_sec`).
// NEGATIVE means AD answered before we did, so a negative number here
// is not an error state. null/undefined renders as an em dash, never 0: "no
// matched AD event" and "AD answered at the same instant we did" are
// different facts and must not share a rendering.
function fmtLatency(ms) {
  if (ms === null || ms === undefined) return '<span class="badge unknown">&mdash;</span>';
  const v = Number(ms);
  if (!isFinite(v)) return '<span class="badge unknown">&mdash;</span>';
  // Coloured by sign: green for a positive delta, red for a negative one.
  // Colour on the figure itself, not a filled badge -- this column is
  // scanned down a list, and a stack of filled chips would carry more
  // weight than the numbers. Exactly 0.0 stays neutral: rounding can
  // land there from either side.
  const cls = v > 0 ? 'lat-ahead' : (v < 0 ? 'lat-behind' : '');
  const txt = (v > 0 ? '+' : '') + v.toFixed(1);
  return cls ? '<span class="' + cls + '">' + txt + '</span>' : txt;
}

function fmtAdWrongCell(p, sections) {
  // 2026-08-13, only show the mark-wrong control when there's
  // actually a miss to flag -- if every engine agreed with AD on this
  // throw, there's nothing to second-guess. Already-marked throws
  // always keep showing (badge + "Unmark") regardless of the current
  // miss state, since a human judgment call already on record must
  // stay visible/reversible, not disappear because engines happen to
  // agree with AD now.
  // A REAL miss is an engine disagreeing with an AD answer that actually
  // arrived: sector_match is false. When AD is offline / never matched it
  // is null (not false), so there is nothing to second-guess and the AD
  // controls stay hidden -- otherwise every throw looked like a miss and
  // the row filled with dead buttons.
  const realMiss = (sections || []).some((s) => s.sector_match === false);
  const parts = [];

  if (p.ad_operator_marked_wrong || realMiss) {
    let html = '';
    if (p.ad_operator_marked_wrong) {
      const title = p.ad_operator_note ? ' title="' + escapeAttr(p.ad_operator_note) + '"' : '';
      html += '<span class="badge warn"' + title + '>AD wrong (human)</span> ';
    }
  // AD's OWN pass/fail, once a human has confirmed what was actually
  // right (2026-08-13). DESIGN CALL, stated plainly rather than assumed:
  // the AD row deliberately has no PASS/FAIL badge in the normal case --
  // it's the reference the engine rows below are graded against, not a
  // graded row itself (see this function's own header comment). But once
  // a human confirms a DIFFERENT segment, AD demonstrably got this throw
  // wrong, and the row would be lying by omission to show only the
  // control. So: a FAIL badge appears if and only if the confirmed
  // segment differs from AD's own raw sector/ring. The converse case (a
  // human marks AD wrong but then confirms the very segment AD already
  // had -- e.g. they disagreed about the tip position, not the segment)
  // deliberately shows NO badge at all rather than a PASS, because
  // "PASS" would contradict the "AD wrong (human)" badge sitting right
  // next to it; the human's own flag is the honest signal there.
  if (p.ad_operator_confirmed_ring) {
    const adDiffers = (p.ad_operator_confirmed_sector !== p.ad_sector)
      || (p.ad_operator_confirmed_ring !== p.ad_ring);
    if (adDiffers) html += fmtMatch(false) + ' ';
    const src = p.ad_operator_confirmed_source || 'manual';
    html += '<span class="badge ok" title="Operator-confirmed truth \u2014 every engine row on this throw is graded against THIS, not AD.">truth: ' +
      fmtSectorRing(p.ad_operator_confirmed_sector, p.ad_operator_confirmed_ring) +
      ' (' + escapeAttr(src) + ')</span> ';
  }
    const label = p.ad_operator_marked_wrong ? 'Unmark' : 'AD Wrong';
    const nextWrong = p.ad_operator_marked_wrong ? '0' : '1';
    html += '<button class="action ad-wrong-btn" type="button"' +
      ' data-session="' + escapeAttr(p.session) + '"' +
      ' data-throw="' + escapeAttr(p.throw_id) + '"' +
      ' data-wrong="' + nextWrong + '">' + label + '</button>';
    parts.push(html);
  }
  // Save frames (manual misscore capture). Hidden outright when the rig
  // records every throw: the button could then only ever duplicate a clip
  // the rig already took.
  //
  // Gated on the MODE, not on p.has_video, deliberately. The clip is
  // finalised a moment after the package is saved, so a freshly-saved
  // throw still reads has_video=false -- gating on that alone made the
  // useless button flash up on every single dart before the clip landed.
  // The mode is restart-scoped, so the bootstrap value cannot go stale.
  if (VIDEO_RECORD_MODE !== 'all' && !p.has_video) {
    parts.push('<button class="action misscore-capture-btn" type="button"' +
      ' title="Save the raw camera frames from around this throw (about a second, centred on when it was actually captured) so the detection and settling can be replayed. Refuses, with the numbers, if the throw is older than the buffer."' +
      ' data-session="' + escapeAttr(p.session) + '"' +
      ' data-throw="' + escapeAttr(p.throw_id) + '">Save frames</button>');
  }
  return parts.join(' ') || '&mdash;';
}

// VIEW lives in its OWN column, not in the match cell (2026-09-21). On a
// corrected throw that cell already carries up to four things -- "AD wrong
// (human)", AD's own FAIL, the "truth: D5 (Athena)" badge and Unmark -- and
// a fifth control pushed the row wide enough to clip the truth badge, so
// the one button you always want was the one squeezed out. Its own narrow
// column keeps it in the same place on every throw regardless of what the
// match cell happens to be carrying.
//
// Once per THROW, not per row: it opens the throw's viewer, and a copy on
// every engine row would be four buttons doing one thing.
function fmtViewCell(p) {
  return '<button class="action view-throw-btn" type="button"' +
    ' title="Open a viewer for this throw: the bg/after grid, plus the video clip if one was recorded."' +
    ' data-session="' + escapeAttr(p.session) + '"' +
    ' data-throw="' + escapeAttr(p.throw_id) + '">View</button>';
}

// -- tabs -------------------------------------------------------------

// -- Scoring tab: live board + current visit ----------------------------
// PACKAGE-FREE BY DESIGN. Everything below is fed by THROW_DETECTED /
// VISIT_CLEARED / THROW_CORRECTED over the websocket, plus /api/state's
// own `visit` section on load (the same THROW_DETECTED payloads the
// server accumulated for the turn in progress -- see AppState's
// visit_throws). Nothing here reads allKnownPackages or /api/packages,
// deliberately (2026-09-08): package writing may be switched off, and
// scoring must then run purely on the JSON events. This panel therefore keeps working unchanged the day package writing
// is switched off, while the Engines tab (which is ABOUT the saved
// packages) honestly stops having anything to show.

const DART_COLORS = ['#4fc4b0', '#d68a4c', '#e0665a'];

// Regulation radii in mm, matching opendarts.geometry.board's own bands.
// board_xy_mm arrives in the same mm space, so markers need no scaling
// beyond board-radius -> pixels.
const BOARD_MM = {
  doubleOuter: 170.0, doubleInner: 162.0,
  trebleOuter: 107.0, trebleInner: 99.0,
  outerBull: 15.9, bull: 6.35,
};

// Points per ring, from the real ring vocabulary fmtSectorRing() already
// speaks. 'outside' is a real, scoreable answer worth 0 -- a miss, not
// missing data; a null/absent ring is the genuinely-unknown case and is
// handled separately by the callers below.
const RING_POINTS = {
  bull: () => 50,
  outer_bull: () => 25,
  outside: () => 0,
  single_inner: (s) => s,
  single_outer: (s) => s,
  treble: (s) => s * 3,
  double: (s) => s * 2,
};

function throwPoints(t) {
  if (!t || !t.ring) return 0;
  const fn = RING_POINTS[t.ring];
  if (!fn) return 0;
  const sector = t.sector === null || t.sector === undefined ? 0 : Number(t.sector);
  return fn(sector) || 0;
}

// A throw's displayed call, preferring an operator correction when one
// exists (THROW_CORRECTED) over what the engine originally said -- the
// same precedence the Engines tab applies to its own rows.
function throwCall(t) {
  if (!t) return '&ndash;';
  const sector = t.corrected_ring ? t.corrected_sector : t.sector;
  const ring = t.corrected_ring || t.ring;
  if (!ring) return '?';
  return fmtSectorRing(sector, ring);
}

let visitThrows = [];   // THROW_DETECTED payloads, in visit_index order
let visitId = null;
// Mirrors /api/state's visit.available -- false when this process has no
// live event queue wired, i.e. no THROW_DETECTED will ever arrive.
let liveEventsAvailable = true;

const boardCanvas = document.getElementById('board-canvas');
const boardCtx = boardCanvas ? boardCanvas.getContext('2d') : null;
let boardPx = 0;

function sectorAngles(i) {
  // Canvas angles: 0 = +x, -PI/2 = straight up. BOARD_SECTORS is clockwise
  // from 20 at the top (opendarts.geometry.board's SECTOR_NUMBERS_CLOCKWISE),
  // so sector i is centred at -PI/2 + i*step, spanning half a step either side.
  const step = (Math.PI * 2) / 20;
  const a0 = -Math.PI / 2 + (i - 0.5) * step;
  return [a0, a0 + step];
}

function wedge(c, cx, cy, r0, r1, a0, a1) {
  c.beginPath();
  c.arc(cx, cy, r1, a0, a1, false);
  c.arc(cx, cy, r0, a1, a0, true);
  c.closePath();
}

function resizeBoard() {
  if (!boardCanvas) return;
  const box = boardCanvas.parentElement.getBoundingClientRect();
  // A hidden panel measures 0x0 -- skip rather than committing a 0-size
  // canvas we'd then have to detect and undo. activateTab() re-calls this
  // on every switch into Scoring, so the real size lands then.
  if (box.width < 2 || box.height < 2) return;
  const dpr = window.devicePixelRatio || 1;
  // Square, and as large as the stage allows in BOTH axes -- the dart
  // strip is a separate flex row above, so whatever height is left here
  // is genuinely the board's to use.
  const size = Math.max(160, Math.min(box.width, box.height));
  boardCanvas.width = size * dpr;
  boardCanvas.height = size * dpr;
  boardCanvas.style.width = size + 'px';
  boardCanvas.style.height = size + 'px';
  boardCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  boardPx = size;
  drawBoard();
}

function drawBoard() {
  if (!boardCtx || !boardPx) return;
  const c = boardCtx, s = boardPx, cx = s / 2, cy = s / 2;
  const R = s * 0.41;                       // pixels per doubleOuter
  const mm = (v) => (v / BOARD_MM.doubleOuter) * R;
  c.clearRect(0, 0, s, s);

  c.beginPath();
  c.arc(cx, cy, R * 1.22, 0, Math.PI * 2);
  c.fillStyle = '#0a0c0f';
  c.fill();
  c.strokeStyle = 'rgba(255,255,255,0.06)';
  c.lineWidth = 1;
  c.stroke();

  for (let i = 0; i < 20; i++) {
    const a = sectorAngles(i);
    const dark = i % 2 === 0;
    c.fillStyle = dark ? '#14171b' : '#ddd3bb';
    wedge(c, cx, cy, mm(BOARD_MM.outerBull), mm(BOARD_MM.trebleInner), a[0], a[1]); c.fill();
    wedge(c, cx, cy, mm(BOARD_MM.trebleOuter), mm(BOARD_MM.doubleInner), a[0], a[1]); c.fill();
    c.fillStyle = dark ? '#a2312a' : '#17663f';
    wedge(c, cx, cy, mm(BOARD_MM.trebleInner), mm(BOARD_MM.trebleOuter), a[0], a[1]); c.fill();
    wedge(c, cx, cy, mm(BOARD_MM.doubleInner), mm(BOARD_MM.doubleOuter), a[0], a[1]); c.fill();
  }

  c.strokeStyle = 'rgba(6,7,9,0.85)';
  c.lineWidth = 1;
  for (let i = 0; i < 20; i++) {
    const a0 = sectorAngles(i)[0];
    c.beginPath();
    c.moveTo(cx + Math.cos(a0) * mm(BOARD_MM.outerBull), cy + Math.sin(a0) * mm(BOARD_MM.outerBull));
    c.lineTo(cx + Math.cos(a0) * mm(BOARD_MM.doubleOuter), cy + Math.sin(a0) * mm(BOARD_MM.doubleOuter));
    c.stroke();
  }
  [BOARD_MM.trebleInner, BOARD_MM.trebleOuter, BOARD_MM.doubleInner, BOARD_MM.doubleOuter].forEach((r) => {
    c.beginPath(); c.arc(cx, cy, mm(r), 0, Math.PI * 2); c.stroke();
  });

  c.beginPath(); c.arc(cx, cy, mm(BOARD_MM.outerBull), 0, Math.PI * 2);
  c.fillStyle = '#17663f'; c.fill(); c.stroke();
  c.beginPath(); c.arc(cx, cy, mm(BOARD_MM.bull), 0, Math.PI * 2);
  c.fillStyle = '#a2312a'; c.fill(); c.stroke();

  c.font = '600 ' + Math.round(s * 0.030) + 'px ui-monospace, Menlo, monospace';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillStyle = 'rgba(215,222,229,0.62)';
  BOARD_SECTORS.forEach((n, i) => {
    const a = sectorAngles(i);
    const mid = (a[0] + a[1]) / 2;
    c.fillText(String(n), cx + Math.cos(mid) * R * 1.11, cy + Math.sin(mid) * R * 1.11);
  });

  // Oldest first, so a later dart draws over an earlier one. +y is up in
  // board space, hence the y negation into canvas space.
  visitThrows.forEach((t, idx) => {
    if (!t || !t.board_xy_mm) return;
    const px = cx + mm(t.board_xy_mm[0]);
    const py = cy - mm(t.board_xy_mm[1]);
    const col = DART_COLORS[idx % DART_COLORS.length];
    c.beginPath(); c.arc(px, py, s * 0.026, 0, Math.PI * 2);
    c.strokeStyle = col; c.lineWidth = 1.5; c.globalAlpha = 0.45; c.stroke();
    c.globalAlpha = 1;
    c.beginPath(); c.arc(px, py, s * 0.0125, 0, Math.PI * 2);
    c.fillStyle = col; c.fill();
    c.strokeStyle = 'rgba(0,0,0,0.72)'; c.lineWidth = 1.5; c.stroke();
  });
}

if (boardCanvas && window.ResizeObserver) {
  new ResizeObserver(() => resizeBoard()).observe(boardCanvas.parentElement);
}

function renderVisit() {
  for (let i = 0; i < 3; i++) {
    const el = document.getElementById('visit-slot-' + i);
    if (!el) continue;
    const t = visitThrows[i];
    const vEl = el.querySelector('.vs-v');
    const subEl = el.querySelector('.vs-sub');
    // A dart the engine could not place (ok=false, or no ring at all) still
    // occupies its slot, flagged -- never silently omitted, so the slot
    // count always matches the darts actually thrown this turn.
    const unscored = !!t && (t.ok === false || !t.ring);
    el.classList.toggle('filled', !!t);
    el.classList.toggle('unscored', unscored);
    if (!t) {
      vEl.innerHTML = '&ndash;';
      subEl.textContent = '';
      continue;
    }
    if (unscored) {
      vEl.textContent = 'n/a';
      subEl.textContent = t.reason || 'no board position';
      continue;
    }
    vEl.innerHTML = throwCall(t);
    const pts = throwPoints(t.corrected_ring ? {sector: t.corrected_sector, ring: t.corrected_ring} : t);
    const spread = t.max_ray_disagreement_mm;
    subEl.textContent = pts + ' pts' +
      (spread !== null && spread !== undefined ? ' · ' + spread.toFixed(1) + 'mm spread' : '');
  }
  // Turn total, summed from the SAME throwPoints() the per-dart
  // subtitles use, so the parts can never disagree with the whole.
  // Unscored darts contribute nothing and are not silently counted as 0
  // -- the slot already shows them as n/a.
  const totEl = document.getElementById('visit-total');
  if (totEl) {
    let total = 0, scored = 0;
    for (const t of visitThrows) {
      if (!t || t.ok === false || !t.ring) continue;
      const pts = throwPoints(t.corrected_ring ? {sector: t.corrected_sector, ring: t.corrected_ring} : t);
      if (typeof pts === 'number' && isFinite(pts)) { total += pts; scored++; }
    }
    totEl.classList.toggle('filled', scored > 0);
    totEl.querySelector('.vt-v').innerHTML = scored > 0 ? String(total) : '&ndash;';
  }
  const idEl = document.getElementById('visit-id-label');
  // Empty, not a dash, when there's no open visit -- a lone em-dash
  // right-aligned under the dart boxes reads as leftover debris rather
  // than as "no visit yet", which the empty dart boxes already say.
  if (idEl) idEl.textContent = visitId || '';
  const note = document.getElementById('visit-note');
  if (note) {
    if (!liveEventsAvailable) {
      note.textContent = 'No live capture loop in this process -- this board stays empty until one is running.';
    } else if (visitThrows.length > 3) {
      note.textContent = visitThrows.length + ' darts recorded for this visit -- showing the first 3';
    } else {
      note.textContent = '';
    }
  }
  drawBoard();
}

// visit_index is the authoritative slot (0-2) -- ordering by arrival would
// put a late-delivered dart in the wrong slot. An absent index (an emitter
// predating the field) appends rather than being dropped.
function onThrowDetected(ev) {
  if (ev.visit_id && ev.visit_id !== visitId) {
    visitId = ev.visit_id;
    visitThrows = [];
  }
  const idx = ev.visit_index;
  if (idx === null || idx === undefined) visitThrows.push(ev);
  else visitThrows[idx] = ev;
  renderVisit();
}

function onVisitCleared(ev) {
  visitId = ev.visit_id || visitId;
  visitThrows = [];
  renderVisit();
}

// A correction lands on the throw already in the visit, if it's still the
// open one -- the same in-place patch AppState does server-side.
function onThrowCorrectedLive(ev) {
  if (!ev || ev.visit_id !== visitId) return;
  const t = visitThrows[ev.visit_index];
  if (!t) return;
  t.corrected_sector = ev.corrected_sector;
  t.corrected_ring = ev.corrected_ring;
  renderVisit();
}

// -- tabs ---------------------------------------------------------------
// Tab selection lives in location.hash so a refresh (or a bookmark, or a
// link pasted to someone else) lands back on the tab you were actually on
// (2026-09-08). Hash, not localStorage: the tab is part
// of WHERE you are, so it belongs in the URL, and two windows can then sit
// on two different tabs instead of fighting over one shared key.
const TABS = ['scoring', 'engines', 'config', 'info'];

function activateTab(name, opts) {
  const push = !opts || opts.push !== false;
  if (TABS.indexOf(name) === -1) name = TABS[0];
  document.querySelectorAll('.tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === name);
  });
  document.querySelectorAll('.tab-panel').forEach((p) => {
    p.classList.toggle('active', p.id === 'tab-' + name);
  });
  // The board canvas is sized from its container's real box, which is 0x0
  // while the panel is display:none -- so a switch INTO Scoring must
  // re-measure. Cheap and idempotent; harmless on every other tab.
  if (name === 'scoring') resizeBoard();
  // Camera previews stream only while the Config tab is showing (see
  // the cameras section below) -- both edges matter: entering starts
  // the streams, leaving closes them. Deferred to a task because this
  // function also runs once synchronously during script setup, before
  // the camera-feed consts below it are initialized.
  setTimeout(() => {
    updateCameraFeeds(); refreshCalibrationOverlays(); renderPackagesTableIfStale();
    if (name === 'config' || name === 'info') refreshCameraStatus();
    if (name === 'config') refreshAudioClients();
  }, 0);
  if (push && location.hash !== '#' + name) location.hash = name;
}

function tabShowing(name) {
  const panel = document.getElementById('tab-' + name);
  return !document.hidden && !!panel && panel.classList.contains('active');
}

function tabFromHash() {
  return (location.hash || '').replace('#', '') || TABS[0];
}

document.querySelectorAll('.tab').forEach((btn) => {
  btn.onclick = () => activateTab(btn.dataset.tab);
});
// Back/forward and manual hash edits route through the same path, without
// re-pushing the hash they just came from.
window.addEventListener('hashchange', () => activateTab(tabFromHash(), {push: false}));
activateTab(tabFromHash(), {push: false});

// -- cameras ------------------------------------------------------------

// LIVE MJPEG PREVIEWS, 2026-09-11 -- the raw preview is no longer a
// snapshot.png re-fetched on a timer (a hand entering frame took up to
// 3 seconds to appear); each tile's <img> now points at
// /api/cameras/{cam}/stream.mjpg, a multipart/x-mixed-replace stream
// the browser renders as live video with no JS in the frame path. The
// JS here only manages the CONNECTION: streams exist
// exclusively while the Config tab is showing AND the page is visible
// AND capture is running -- three continuously-encoding streams on the
// same 4-core box as the scoring pipeline is CPU nobody gets back, so a
// preview nobody is looking at must cost exactly nothing. Clearing
// img.src is what actually closes the connection; merely hiding the
// panel does not, which is why activateTab/visibilitychange both route
// here.
//
// BROKEN-IMAGE FALLBACK (mechanism kept from the snapshot era, incident
// history preserved): img.onerror used to only update
// cam-fetch-status-*'s text ("unreachable") -- it never touched img.src
// or hid the <img> itself, so a refused fetch (cameras not
// started/stopped -- a normal, expected state now that cameras don't
// auto-open at process startup, see AppState.start_capture()) left the
// browser's own native broken-image icon showing in the tile, on top of
// the separate text label most people never look at. Fixed by keeping a
// real sibling placeholder <div id="cam-placeholder-{cam}"> (see this
// function's HTML, and the .cam-placeholder CSS) permanently in the DOM
// next to each <img>, and toggling which one is visible on every
// load/error outcome -- BOTH directions, not just once: error hides the
// <img> (style.display='none', so the browser has nothing left to
// render a broken-icon for) and un-hides the placeholder; a SUBSEQUENT
// success re-shows the <img> and re-hides the placeholder.
// Per-camera overlay state -- fully automatic now, no manual per-camera button anymore. Driven by
// renderCalibration() below: true only once THAT camera's own
// calibration is confirmed ok, forced false the instant a manual
// Calibrate click starts (see btn-refresh-calib's onclick), and comes
// back on its own the moment fresh calibration data lands (manual
// result, WS CALIBRATION_STATUS broadcast, or the initial state poll --
// renderCalibration()'s 3 call sites all feed this same state, so all 3
// paths behave identically without extra wiring here).
const camOverlayOn = {};
for (const c of CAM_IDS) { camOverlayOn[c] = false; }

// Per-camera stream connection state: 'idle' (no stream on the img --
// placeholder, overlay still, or nothing), 'connecting' (src set, no
// frame yet), 'live' (frames arriving). This is what makes the periodic
// tick idempotent -- a healthy stream is left completely alone rather
// than reconnected every 3 seconds.
const camStreamState = {};
// When each 'connecting' attempt began -- a stream that never delivers
// a first frame fires no event at all in some engines, and without a
// deadline it would sit in 'connecting' forever, blocking every retry.
const camStreamStartedAt = {};
for (const c of CAM_IDS) { camStreamState[c] = 'idle'; camStreamStartedAt[c] = 0; }
const CAM_STREAM_CONNECT_DEADLINE_MS = 10000;

// Latest /api/cameras/status rows (renderCameraStatus stores them).
// Needed here because a multipart stream that ENDS does not reliably
// fire img.onerror -- Chrome just leaves the last frame painted. When
// the server closes a stalled stream (camera stopped producing), this
// is how the tile finds out and drops to the placeholder instead of
// freezing while labelled 'live'.
let lastCamRows = null;

function stopCamStream(c) {
  if (camStreamState[c] === 'idle') return;
  const img = document.getElementById('cam-img-' + c);
  // Handlers first: setting src to '' fires an error event in some
  // engines, and this stop must not be reported as a fetch failure.
  img.onload = null;
  img.onerror = null;
  // Clearing src is the part that actually aborts the MJPEG connection
  // and frees the server-side encoder slot.
  img.src = '';
  img.removeAttribute('src');
  camStreamState[c] = 'idle';
}

// Close EVERY preview stream and drop each tile to its placeholder. Used
// when the whole page or the server is going away (page unload, or the
// events WebSocket dropping on an OD restart) -- see stranded-socket note
// on the pagehide/ws.onclose handlers below.
function stopAllCamStreams() {
  for (const c of CAM_IDS) {
    stopCamStream(c);
    const img = document.getElementById('cam-img-' + c);
    const placeholder = document.getElementById('cam-placeholder-' + c);
    if (img) img.style.display = 'none';
    if (placeholder) placeholder.hidden = false;
    hideCalibrationOverlay(c);
  }
}

function updateCameraFeeds() {
  // FOCUS, not just visibility (2026-09-21). document.hidden is false for
  // EVERY visible window, so two Config windows side by side both streamed
  // -- three never-ending MJPEG connections each, against a browser budget
  // of ~6 sockets PER ORIGIN shared across every window and tab of the
  // profile. Six streams used the whole budget, so every subsequent fetch
  // (state/status/clients) queued forever with no error at all: the page
  // just froze and read as "the rig died". It is origin-keyed on
  // scheme+host+port, so the hostname and the IP are separate budgets --
  // which is exactly why "it works on the IP" looked like a DNS problem for
  // hours and was not.
  //
  // Gating on hasFocus() means only the window actually being worked in
  // streams: two windows cost three sockets, not six. It is also the same
  // principle this section already states -- a preview nobody is looking at
  // must cost exactly nothing -- extended from "hidden tab" to "unfocused
  // window", which is the case that was missed.
  const onConfig = !document.hidden &&
    document.getElementById('tab-config').classList.contains('active');
  const focused = document.hasFocus();
  const showing = onConfig && focused;
  // Only a POSITIVE "not running" stops the connection attempts --
  // pillCaptureLoop is null before the first state load (and forever in
  // standalone mode), and in that case attempting costs one immediate,
  // honest 503 per tick, while NOT attempting could leave a genuinely
  // running rig with no previews if the WS never delivered a status.
  const stopped = !!(pillCaptureLoop && !pillCaptureLoop.running);
  for (const c of CAM_IDS) {
    const img = document.getElementById('cam-img-' + c);
    const placeholder = document.getElementById('cam-placeholder-' + c);
    const status = document.getElementById('cam-fetch-status-' + c);
    if (!showing || stopped) {
      // Overlay stills hold no connection, so when merely hidden they
      // are left painted for an instant tab return; live streams always
      // close. On a positive stop the placeholder is shown either way
      // -- a stale frame pretending to be a feed is the one thing this
      // tile must never do.
      if (camStreamState[c] !== 'idle') {
        stopCamStream(c);
        img.style.display = 'none';
        placeholder.hidden = false;
      }
      if (stopped) {
        img.style.display = 'none';
        placeholder.hidden = false;
        // Start clicked -> {running:false, starting:true} until the
        // CAPTURE_LOOP_STATUS broadcast lands; say so instead of
        // labelling a rig that is coming up as 'stopped'.
        status.textContent = pillCaptureLoop.starting ? 'starting\u2026' : 'stopped';
      } else if (onConfig && !focused) {
        // Say WHY it is not live. Silence here would leave the operator
        // staring at a dark tile on a perfectly healthy rig, and a frozen
        // last frame would be the stale-feed lie this tile must never tell.
        img.style.display = 'none';
        placeholder.hidden = false;
        status.textContent = 'paused \u2014 click this window to resume';
      }
      continue;
    }
    // NO overlay branch here any more (2026-09-12). This used to hand the
    // tile to the overlay poller the moment a camera calibrated, which
    // took it off the stream and onto a 3-second still -- a working rig
    // showed a slideshow. The overlay is now a transparent layer ABOVE
    // this img (#cam-overlay-*), so the stream runs regardless.
    if (camStreamState[c] === 'live') {
      const row = lastCamRows && lastCamRows[String(c)];
      if (row && (row.opened === false || row.last_read_ok === false)) {
        // See lastCamRows' comment: the stream's end is invisible to
        // the img element, but the status poll knows the camera stopped
        // producing. Drop to the placeholder; the tick retries later.
        stopCamStream(c);
        img.style.display = 'none';
        placeholder.hidden = false;
        status.textContent = 'no frames';
        hideCalibrationOverlay(c); // never leave it floating over "No signal"
      }
      continue;
    }
    if (camStreamState[c] === 'connecting') {
      if (Date.now() - camStreamStartedAt[c] < CAM_STREAM_CONNECT_DEADLINE_MS) continue;
      stopCamStream(c); // silent connect stall -- reset so this pass retries
    }
    camStreamState[c] = 'connecting';
    camStreamStartedAt[c] = Date.now();
    img.onload = () => {
      camStreamState[c] = 'live';
      status.textContent = 'live';
      img.style.display = '';
      placeholder.hidden = true;
      refreshCalibrationOverlays(); // the layer waits for a picture to sit on
    };
    img.onerror = () => {
      camStreamState[c] = 'idle';
      status.textContent = 'unreachable';
      img.style.display = 'none';
      placeholder.hidden = false;
      hideCalibrationOverlay(c);
    };
    // Cache-buster for proxies only -- each attempt is a fresh
    // long-lived connection, not a cacheable resource.
    img.src = '/api/cameras/' + c + '/stream.mjpg?t=' + Date.now();
  }
}

// -- calibration overlay, as a LAYER ------------------------------------
//
// The overlay confirms calibration against a static board: it changes
// only when the calibration itself is re-derived, and never between
// frames. It used to be fetched on the 3-second camera tick anyway, as a
// baked-in still that REPLACED the tile's stream -- so a camera reporting
// ok calibration lost its live picture, which is the one state where you
// least want to lose it.
//
// Now: /api/cameras/{c}/overlay-rgba.png is a transparent PNG drawn over
// the running stream, and it is fetched once per RECALIBRATION. The
// cache-busting token is the calibration package id (falling back to its
// timestamp), so the src only changes when the calibration does -- and
// setting src is guarded on the URL actually differing, which is what
// makes this one request per recalibration rather than a slower poll.
let lastCalibrationToken = '';

function hideCalibrationOverlay(c) {
  const ov = document.getElementById('cam-overlay-' + c);
  if (!ov) return;
  ov.hidden = true;
  // Drop the src too: a hidden <img> still holds its decoded bitmap, and
  // the next show must re-check the calibration rather than flash the
  // previous one over a freshly reopened camera.
  ov.removeAttribute('src');
}

function refreshCalibrationOverlays() {
  for (const c of CAM_IDS) {
    const ov = document.getElementById('cam-overlay-' + c);
    if (!ov) continue;
    // Three conditions, all necessary: a confirmed calibration to draw,
    // a live stream to draw it over, and the Config tab actually showing.
    if (!camOverlayOn[c] || camStreamState[c] !== 'live') {
      hideCalibrationOverlay(c);
      continue;
    }
    const want = '/api/cameras/' + c + '/overlay-rgba.png?v='
      + encodeURIComponent(lastCalibrationToken || 'none');
    if (ov.getAttribute('src') === want) continue; // already the current one
    ov.onload = () => { ov.hidden = false; };
    // A camera with no calibration answers 404 -- show the stream alone
    // rather than an empty layer or a broken-image icon.
    ov.onerror = () => { ov.hidden = true; };
    ov.setAttribute('src', want);
  }
}

// Page-visibility edge: a hidden page must not keep three MJPEG
// encoders busy on the rig. updateCameraFeeds() closes on hidden and
// reconnects on return; the overlay layer follows whatever the streams
// end up doing (it is hidden while they are not live).
document.addEventListener('visibilitychange', () => {
  updateCameraFeeds();
  refreshCalibrationOverlays();
});

// Window focus is the OTHER edge, and the one that was missing: two
// side-by-side windows are both visible, so visibilitychange never fires
// for either and both kept streaming. See updateCameraFeeds()'s note on the
// per-origin socket budget -- these two listeners are what keep the stream
// count at one window's worth no matter how many windows are open.
window.addEventListener('focus', () => {
  updateCameraFeeds();
  refreshCalibrationOverlays();
});
window.addEventListener('blur', () => {
  updateCameraFeeds();
  refreshCalibrationOverlays();
});

// Page teardown: a reload, navigation, or close must release the MJPEG
// stream sockets explicitly. A multipart/x-mixed-replace connection does
// not close cleanly on its own -- left to the browser it lingers as a
// half-open socket against Chrome's ~6-per-origin connection pool. Enough
// of them stranded (three per page life, and this rig gets reloaded and
// restarted constantly) and every later fetch to this origin has no free
// socket and sits "pending" forever -- the whole dashboard wedges on the
// hostname while the IP (a separate pool) looks fine. pagehide is the
// reliable teardown edge: it covers reload, navigation, close, and the
// bfcache freeze, where beforeunload does not.
window.addEventListener('pagehide', () => {
  stopAllCamStreams();
  try { if (liveWs) liveWs.close(); } catch (e) { /* already gone */ }
});

function renderCameraStatus(data) {
  const cams = (data && data.cameras) || {};
  // Feed the stream-liveness safety net (lastCamRows' comment above) --
  // only real rows; an unavailable status report proves nothing about
  // whether frames are flowing.
  lastCamRows = data && data.available ? cams : null;
  for (const c of CAM_IDS) {
    const el = document.getElementById('cam-detail-' + c);
    if (!el) continue;
    const row = cams[String(c)];
    if (!data.available) {
      el.textContent = data.reason || 'camera status unavailable';
      continue;
    }
    if (!row) {
      el.textContent = 'no status for this camera index';
      continue;
    }
    el.textContent =
      'opened: ' + row.opened + (row.opened ? '' : ' (' + (row.last_error || 'unknown reason') + ')') + '\n' +
      'backend: ' + (row.backend_used || '\u2014') + '\n' +
      'requested: ' + row.requested_width + 'x' + row.requested_height + '@' + row.requested_fps + 'fps' + '\n' +
      'actual: ' + row.actual_width + 'x' + row.actual_height + '@' + (row.actual_fps || 0).toFixed(1) + 'fps' + '\n' +
      'open latency: ' + (row.open_latency_s !== null && row.open_latency_s !== undefined ? row.open_latency_s.toFixed(3) + 's' : '\u2014') + '\n' +
      'first frame latency: ' + (row.first_frame_latency_s !== null && row.first_frame_latency_s !== undefined ? row.first_frame_latency_s.toFixed(3) + 's' : '\u2014') + '\n' +
      'frames read: ' + row.frame_count + '\n' +
      'last read ok: ' + row.last_read_ok +
      (row.last_error ? '\nlast error: ' + row.last_error : '');
  }
}

async function refreshCameraStatus() {
  try {
    const resp = await fetch('/api/cameras/status');
    renderCameraStatus(await resp.json());
  } catch (err) {
    console.error('camera status fetch failed', err);
  }
}

// -- calibration ----------------------------------------------------

// renderLiveCalibrationAge -- added 2026-08-26 alongside calibration
// persistence across restarts (CalibrationStore's snapshot_path). Renders
// calibration.live_source (CalibrationStore.meta(): source/checked_at_utc)
// into #live-calib-age -- the one place an operator can actually see
// "when was the calibration the capture loop is scoring against right
// now actually taken", which matters now that a persisted calibration can
// silently outlive a hardware re-seat across a restart (see that class's
// own docstring for the accepted trade-off this exists to make visible).
// Only present on state.calibration (initial /api/state load) and the
// manual-refresh POST response -- the WS CALIBRATION_STATUS push (both
// the dashboard's own manual-refresh broadcast and the capture loop's
// async auto-calibrate confirmation) doesn't carry live_source, so this
// function leaves the element's last-known text untouched when it's
// absent, rather than blanking a real value on an unrelated update.
function renderLiveCalibrationAge(liveSource) {
  const el = document.getElementById('live-calib-age');
  if (!el || !liveSource) return;
  const n = liveSource.n_cameras || 0;
  if (!n || !liveSource.checked_at_utc) {
    el.textContent = 'Live calibration: none yet -- click Calibrate, or Start will auto-calibrate.';
    return;
  }
  const sourceLabel = {
    startup: 'this process\u2019s own startup bootstrap',
    manual: 'a manual Calibrate click',
    persisted: 'a PRIOR process (persisted to disk)',
  }[liveSource.source] || liveSource.source || 'unknown';
  let when;
  try {
    when = new Date(liveSource.checked_at_utc).toLocaleString();
  } catch (e) {
    when = liveSource.checked_at_utc;
  }
  // No hardcoded "stale after N hours" alarm -- there is no principled
  // threshold (a rig can sit calibrated for days with zero hardware
  // change and be perfectly fine); this is deliberately a plain fact,
  // not a warning, so the operator's own judgment ("did I just move a
  // camera?") is what decides whether to hit Refresh.
  el.textContent = 'Live calibration (' + n + ' camera(s)): captured ' + when + ', from ' + sourceLabel + '.';
}

// Every camera's badge -> "calibrating", for the duration of a manual
// Calibrate. 2026-09-08: the click already pulled each camera's overlay
// (so stale geometry stopped being drawn) but left the badge alone, so a
// camera kept showing a green "calibrated" while its calibration was
// being recomputed and the answer might be about to change -- the one
// moment the badge is guaranteed NOT to describe current state.
//
// Uses the same outerHTML replacement renderCalibration() does, so
// whichever runs last wins cleanly instead of one leaving behind a class
// the other set. renderCalibration() overwrites these the instant real
// results land.
function setCalibratingBadges() {
  for (const c of CAM_IDS) {
    const badge = document.getElementById('calib-badge-' + c);
    if (!badge) continue;
    badge.outerHTML = '<span id="calib-badge-' + c + '" class="badge warn">calibrating</span>';
  }
}

// The action log is a narrow box, so a moved-camera line says the fact and
// the number and nothing else; the full explanation lives on the
// calibration panel, and on this line's own tooltip.
function movedShort(relearned) {
  const drift = relearned && typeof relearned.drift_deg === 'number'
    ? relearned.drift_deg.toFixed(1) + '°' : 'a lot';
  return 'gaps shifted ' + drift + ' — layout relearned';
}

// "A camera moved" -- shown until the next calibration says otherwise.
function renderCameraMovedNote(relearned) {
  const el = document.getElementById('calib-moved-note');
  if (!el) return;
  if (!relearned || !relearned.note) {
    el.hidden = true;
    el.textContent = '';
    return;
  }
  el.hidden = false;
  const when = relearned.at_utc ? new Date(relearned.at_utc) : null;
  el.textContent = 'Cameras moved: ' + relearned.note
    + (when && !isNaN(when) ? ' (' + fmtClockTime(when) + ')' : '');
}

function renderCalibration(calibration) {
  renderLiveCalibrationAge(calibration && calibration.live_source);
  renderCameraMovedNote(calibration && calibration.ring_geometry_relearned);
  // The overlay's change token. calibration_package_id is written fresh
  // by every real re-derivation, so it is the exact "the overlay is now
  // stale" signal -- and, just as importantly, it does NOT change on the
  // state polls that call this function several times a minute, which is
  // what keeps the overlay off a polling cadence. checked_at_utc is the
  // fallback for a rig whose store has not recorded a package id; the
  // top-level calibration.checked_at_utc is deliberately NOT used, since
  // that is "when this status was computed" and moves constantly.
  const src = (calibration && calibration.live_source) || {};
  lastCalibrationToken = src.calibration_package_id || src.checked_at_utc || '';
  const cams = (calibration && calibration.cameras) || {};
  for (const c of CAM_IDS) {
    const key = String(c);
    const badge = document.getElementById('calib-badge-' + c);
    const errEl = document.getElementById('calib-err-' + c);
    const noteEl = document.getElementById('calib-note-' + c);
    if (!badge) continue;
    const row = cams[key];
    if (!row) {
      badge.outerHTML = '<span id="calib-badge-' + c + '" class="badge unknown">unknown</span>';
      errEl.textContent = '\u2014';
      noteEl.textContent = 'no calibration data yet';
      // No calibration data at all -> no overlay to draw (see
      // camOverlayOn's own comment above for the full automatic-overlay
      // design).
      camOverlayOn[c] = false;
      continue;
    }
    const cls = row.ok === true ? 'ok' : row.ok === false ? 'bad' : 'unknown';
    badge.outerHTML = '<span id="calib-badge-' + c + '" class="badge ' + cls + '">' + (row.ok === true ? 'calibrated' : row.ok === false ? 'not calibrated' : 'unknown') + '</span>';
    // A NUMBER, or WHY THERE ISN'T ONE. `ok: true` with a null
    // reprojection error is a real and common state -- a calibration
    // loaded from disk is valid, but its error was measured in the
    // session that solved it, not this one. Rendering that as a bare
    // em-dash beside a green "calibrated" badge reads as missing data or
    // a broken row, when the honest answer is "not measured here".
    const err = row.reprojection_error_px;
    const metricEl = document.getElementById('calib-metric-' + c);
    if (err === null || err === undefined) {
      errEl.textContent = row.ok === true ? 'not measured' : '\u2014';
      if (metricEl) {
        metricEl.classList.toggle('calib-metric-absent', row.ok === true);
        if (row.ok === true) {
          metricEl.title = 'This calibration was loaded from disk rather than solved in this '
            + 'session, so its reprojection error was measured elsewhere. Press Calibrate to '
            + 're-solve and get a fresh number.';
        }
      }
    } else {
      errEl.textContent = err.toFixed(2);
      if (metricEl) metricEl.classList.remove('calib-metric-absent');
    }
    noteEl.textContent = row.reason || (row.landmark_spread_ok === false ? 'poor landmark spread' : '');
    // Automatic overlay: on exactly when THIS camera's own calibration
    // is confirmed ok, off otherwise (not calibrated, unknown, or a
    // manual Calibrate currently in flight -- see btn-refresh-calib's
    // onclick, which forces this false before this function ever runs
    // with fresh data).
    camOverlayOn[c] = row.ok === true;
  }
  if (typeof updateCameraFeeds === 'function') { updateCameraFeeds(); refreshCalibrationOverlays(); }
  // 2026-08-13, the left-hand controls panel was stripped to the
  // essentials -- the dashboard-last-refreshed/live-source summary
  // line that used to render into #calib-checked here was removed along
  // with that element; per-camera calibration state is still fully
  // visible on the Cameras tab's own cards (calib-badge-/calib-err-/
  // calib-note- above), which this function keeps updating unchanged.
}

// Every handler below follows the SAME real shape (added to, not
// replacing, the pre-existing per-button disable+relabel): (1)
// setControlsBusy(true) disables ALL FOUR control buttons, not just the
// one clicked -- no other action can race this one; (2) logAction()
// appends an immediate, persistent "requested" line to the sidebar's
// action log and keeps a handle to it; (3) the SAME line is updated in
// place (updateActionLine) to ok/bad once the request resolves, so the
// log always shows exactly what happened to exactly this click, not a
// vanishing button-label flash. See ACTION_LOG_MAX_LINES's own comment
// above for the real incident this fixes.
document.getElementById('btn-refresh-calib').onclick = async () => {
  const btn = document.getElementById('btn-refresh-calib');
  setControlsBusy(true);
  btn.textContent = 'Calibrating\u2026';
  // Pill state pill -- see manualCalibrating's own comment above for the
  // real gap this closes (2026-08-14). Set/cleared around the SAME
  // request, rendered immediately rather than waiting on a poll, exactly
  // the pattern the Start/Stop handlers already use for their own pill
  // fields.
  manualCalibrating = true;
  renderPill();
  // 2026-08-16: the overlay is hidden from the Calibrate click until
  // the new calibration arrives -- force every camera's
  // automatic overlay off the instant the click happens, not just once
  // the (possibly slow, retry-looped) new calibration actually lands.
  // renderCalibration() below turns it back on per-camera once the real
  // result arrives, same as every other path that calls it.
  for (const c of CAM_IDS) { camOverlayOn[c] = false; }
  setCalibratingBadges();
  // RELEASE THE CAMERA STREAMS FOR THE DURATION. Overlays have just gone
  // off; rather than swap each tile back to its live stream (what this
  // used to do), close the streams outright until the result lands. Two
  // independent reasons, one fix.
  //
  // The browser one, measured on an iPad on 2026-09-15: the Config tab
  // holds one never-ending MJPEG connection per camera. With the
  // WebSocket that is four of WebKit's ~6-connections-per-host budget,
  // and the calibrate POST makes five. That leaves ONE socket for the
  // status poll, three overlay images and the audio heartbeat -- so all
  // of them fail together, for exactly as long as calibration runs.
  // "TypeError: Load failed" on every request at once, which reads as the
  // rig having died. It had not: measured during a calibration it was
  // serving status polls in 0.6ms. The PAGE had run out of sockets, and
  // nothing on the rig could ever have shown that.
  //
  // The rig one: three MJPEG encoders competing with the most CPU-hungry
  // operation this box performs is waste, and what they encode meanwhile
  // is a board mid-measurement that nobody needs to watch.
  //
  // Restored in the `finally` via updateCameraFeeds(), which reads the
  // live tab/visibility/capture state rather than a remembered list -- so
  // switching tabs mid-calibration is honoured instead of being
  // overridden by whatever happened to be true at click time.
  for (const c of CAM_IDS) stopCamStream(c);
  const line = logAction('Calibrate', 'pending', 'requested\u2026');
  // Wall clock for the action line. performance.now(), not Date.now():
  // this is an elapsed-time measurement and must not be skewed by a
  // system clock adjustment landing mid-calibration. Measured around the
  // fetch, so it is what the OPERATOR waited -- a hair more than the
  // server's own bootstrap_calibrations TOTAL, which excludes HTTP. On
  // this LAN that gap is tens of ms against a ~25s operation; the exact
  // server figure stays in the PHASE WALL CLOCK log and each package.
  const calibStartedMs = performance.now();
  try {
    const resp = await fetch('/api/calibration/refresh', { method: 'POST' });
    const body = await resp.json();
    renderCalibration(body);
    const okCams = Object.values(body.cameras || {}).filter((c) => c.ok).length;
    const totalCams = Object.keys(body.cameras || {}).length;
    const calibSecs = ((performance.now() - calibStartedMs) / 1000).toFixed(1);
    if (body.calibration_error) {
      // 2026-08-16: refresh returns an error when the cameras are off
      // -- the server can be up
      // while the cameras themselves aren't
      // producing frames; the backend now fails this fast instead of
      // grinding through a long retry loop. Same visible treatment as
      // the od_reachable===false case above.
      updateActionLine(line, 'bad', 'Calibrate',
        body.calibration_error + ' (after ' + calibSecs + 's)');
    } else {
      const moved = body.ring_geometry_relearned;
      updateActionLine(line, moved ? 'warn' : 'ok', 'Calibrate',
        'done \u2014 ' + okCams + '/' + totalCams + ' camera(s) ok in ' + calibSecs + 's'
        + (moved ? ' \u2014 cameras moved, ' + movedShort(moved) : ''));
      if (moved && moved.note) line.title = moved.note;
    }
  } catch (err) {
    console.error('calibration refresh failed', err);
    // Nothing is calibrating any more, so the badges must stop saying so
    // -- but they cannot simply go back to "calibrated" either, because
    // this client no longer knows whether the backend applied anything
    // before it failed. Re-read real state; if even that is unreachable,
    // fall back to "unknown". A badge stranded on "calibrating" forever
    // is the one outcome worse than either.
    try {
      const again = await fetch('/api/state');
      renderCalibration((await again.json()).calibration);
    } catch (reReadErr) {
      console.error('calibration re-read after failure also failed', reReadErr);
      for (const c of CAM_IDS) {
        const b = document.getElementById('calib-badge-' + c);
        if (b) b.outerHTML = '<span id="calib-badge-' + c + '" class="badge unknown">unknown</span>';
      }
    }
    updateActionLine(line, 'bad', 'Calibrate', 'request failed \u2014 see browser console');
  } finally {
    setControlsBusy(false);
    btn.textContent = 'Calibrate';
    manualCalibrating = false;
    renderPill();
    // Restore the previews closed above. In `finally` because they were
    // closed unconditionally: a failed, refused or aborted calibration
    // must not be able to leave every tile dark until the next tab
    // switch.
    updateCameraFeeds();
    refreshCalibrationOverlays();
    // The pill was reported stuck on Starting "after Calibrate" too. The
    // calibrate itself never touches `starting`, but it is the moment an
    // operator looks at the pill and expects the truth -- re-read it
    // rather than keep whatever this tab last pieced together.
    await resyncCaptureLoop();
  }
};

// Start/Stop -- POST /api/start, /api/stop (see those routes' own
// docstrings, opendarts/live/server.py). Same fetch-call/user-feedback shape
// as Calibrate above. pillCaptureLoop/lastStartError (defined below,
// alongside renderPill()) are updated here too so the status pill reacts
// to THIS tab's own click immediately, without waiting on the
// CAPTURE_LOOP_STATUS broadcast round-trip (which still fires, for every
// OTHER connected tab).
document.getElementById('btn-start').onclick = async () => {
  const btn = document.getElementById('btn-start');
  setControlsBusy(true);
  btn.textContent = 'Starting\u2026';
  // 2026-08-15: after pressing Start, the pill took 2-3 seconds to
  // show "starting". Root cause: renderPill() was
  // only called in this handler's `finally`, i.e. AFTER the whole
  // POST /api/start round trip (which opens 3 real cameras, ~2.3s each
  // per this project's own measured open_latency) had already
  // completed -- the pill sat frozen on its previous state for the
  // entire camera-open duration with zero feedback. Fixed the same way
  // manualCalibrating already fixes this for Calibrate: set an
  // optimistic starting state and render it BEFORE the fetch, not
  // after. The real response (below) overwrites this with the true
  // server state once it actually arrives.
  //
  // `running: true`, not false (2026-09-22): renderPill() checks
  // `!cl.running` (Stopped) BEFORE `cl.starting`, so the old
  // `{running: false, starting: true}` rendered as Stopped and never
  // showed the Starting this block exists to show. Deliberately unstamped
  // (see acceptCaptureStatus()): it is a guess, and any real snapshot
  // must be free to replace it -- resyncCaptureLoop() in the `finally`
  // guarantees one does even if the fetch below throws.
  pillCaptureLoop = { running: true, starting: true };
  renderPill();
  const line = logAction('Start', 'pending', 'requested\u2026');
  try {
    const resp = await fetch('/api/start', { method: 'POST' });
    const body = await resp.json();
    if (body.ok === false) {
      lastStartError = body.reason || 'unknown';
      updateActionLine(line, 'bad', 'Start', 'failed \u2014 ' + (body.reason || 'unknown'));
    } else {
      lastStartError = null;
      startingCalibDetail = '';
      // Through the ordering guard: this response is routinely OLDER than
      // the TRIGGER_STATE push that already ended Starting (see
      // acceptCaptureStatus()), and applying it anyway is the bug.
      if (acceptCaptureStatus(body)) pillCaptureLoop = body;
      updateActionLine(
        line, 'ok', 'Start',
        body.already_running ? 'already running' : 'cameras opened \u2014 starting session (auto-calibrate check pending)'
      );
    }
  } catch (err) {
    console.error('start failed', err);
    updateActionLine(line, 'bad', 'Start', 'request failed \u2014 see browser console');
  } finally {
    setControlsBusy(false);
    btn.textContent = 'Start';
    renderPill();
    await resyncCaptureLoop();
  }
};

document.getElementById('btn-stop').onclick = async () => {
  const btn = document.getElementById('btn-stop');
  setControlsBusy(true);
  btn.textContent = 'Stopping\u2026';
  const line = logAction('Stop', 'pending', 'requested\u2026');
  try {
    const resp = await fetch('/api/stop', { method: 'POST' });
    const body = await resp.json();
    if (body.ok === false) {
      updateActionLine(line, 'bad', 'Stop', 'failed \u2014 ' + (body.reason || 'unknown'));
    } else {
      // Same guard as Start: a Stop response can equally be overtaken
      // by a newer push (e.g. another tab's Start right after).
      if (acceptCaptureStatus(body)) pillCaptureLoop = body;
      updateActionLine(line, 'ok', 'Stop', body.already_stopped ? 'already stopped' : 'session stopped, camera hub closed');
    }
  } catch (err) {
    console.error('stop failed', err);
    updateActionLine(line, 'bad', 'Stop', 'request failed \u2014 see browser console');
  } finally {
    setControlsBusy(false);
    btn.textContent = 'Stop';
    renderPill();
    await resyncCaptureLoop();
  }
};

// Reset -- POST /api/reset (see that route's own docstring,
// opendarts/live/server.py). Same fetch-call/user-feedback shape as
// Calibrate above (disable + relabel while in flight, always restore in
// finally), following that real pattern directly rather than inventing a
// new one.
document.getElementById('btn-reset').onclick = async () => {
  const btn = document.getElementById('btn-reset');
  setControlsBusy(true);
  btn.textContent = 'Resetting\u2026';
  const line = logAction('Reset', 'pending', 'requested\u2026');
  try {
    const resp = await fetch('/api/reset', { method: 'POST' });
    const body = await resp.json();
    if (body.loop_listening) {
      updateActionLine(line, 'ok', 'Reset', 'requested \u2014 capture loop will re-baseline next iteration');
    } else {
      updateActionLine(line, 'bad', 'Reset', 'no capture loop in this process (standalone dashboard mode)');
    }
  } catch (err) {
    console.error('reset failed', err);
    updateActionLine(line, 'bad', 'Reset', 'request failed \u2014 see browser console');
  } finally {
    setControlsBusy(false);
    btn.textContent = 'Reset';
  }
};

// -- scoring (recent throw packages) ---------------------------------

function fmtXy(xy) {
  if (!xy || xy.length !== 2) return '&mdash;';
  return '(' + xy[0].toFixed(1) + ', ' + xy[1].toFixed(1) + ')';
}

// Signed component-wise delta between an engine's own board_xy_mm and
// AD's own tip_xy_mm -- 2026-08-13, engine rows show DELTA-from-AD here,
// not their own raw position (that's what the AD row's own "board xy"
// cell shows instead, via plain fmtXy() above). Explicit +/- sign on
// both components (toFixed doesn't add one for positive numbers) so a
// glance tells you the DIRECTION of the miss, not just its size --
// that's the whole point of showing a vector instead of just reusing
// the already-existing scalar tip_distance_mm column.
function fmtDeltaXy(engineXy, adXy) {
  if (!engineXy || engineXy.length !== 2 || !adXy || adXy.length !== 2) return '&mdash;';
  const dx = engineXy[0] - adXy[0];
  const dy = engineXy[1] - adXy[1];
  const sign = (v) => (v >= 0 ? '+' : '') + v.toFixed(1);
  return '(' + sign(dx) + ', ' + sign(dy) + ')';
}

// "2026-08-12T19:22:23.902480+00:00" (stored UTC) -> "2026-08-12 12:22:23"
// in the browser's LOCAL time zone.
// Storage stays UTC on disk (opendarts.capture.throw_package's own real
// timestamps are unaffected) -- this is a display-only conversion.
function fmtCaptured(iso) {
  if (!iso) return '&mdash;';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso);
  const pad = (n) => String(n).padStart(2, '0');
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
    ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
}

// SUPERSEDED same session -- first asked to shrink "captured" to
// Combine sector+ring into standard dart shorthand instead of two columns
//: S5/s5 = single outer/inner, T5 = treble, D5 =
// double, 25/50 = outer/inner bull, MISS = outside the board. Ring names
// match opendarts.geometry.board.sector_ring_for_point()'s real vocabulary.
function fmtSectorRing(sector, ring) {
  if (ring === 'bull') return '50';
  if (ring === 'outer_bull') return '25';
  if (ring === 'outside') return 'MISS';
  if (sector === null || sector === undefined || !ring) return '&mdash;';
  const prefix = {single_outer: 'S', single_inner: 's', treble: 'T', double: 'D'}[ring];
  return prefix ? (prefix + sector) : '&mdash;';
}

// Row numbers ("dart N") are CHRONOLOGICAL (oldest = 1) within whatever
// set is CURRENTLY PASSED IN (the full known history, or the "since
// session start" filtered set -- see viewFilterSinceMs below), even
// though `packages` itself keeps arriving newest-first (see
// renderScoringTable's own sort), 2026-08-12, so a specific throw can
// be referred to by number ("dart 5") -- oldest=1 is the only scheme where that phrase keeps meaning the SAME
// throw as the session goes on: newest-first numbering would renumber
// every earlier row the instant a new dart lands, which is exactly the
// opposite of a stable conversational reference. `packages` has length
// n and is sorted newest-first, so the row at index i (0 = newest) is
// chronologically the (n - i)th throw in the set -- no separate sort
// needed.
// -- Scoring table row groups
// One AD row (the reference) + one row per engine that actually ran on
// THIS throw (primary + whatever's in p.engines -- read per-package, not
// from current Config-tab state, so a throw scored under an earlier
// engine-config still shows exactly what ran for IT, not today's
// settings). No more dynamic per-engine COLUMNS (the old
// updateAlsoRunColumns/alsoRunEngineColumns mechanism) -- engines are
// rows now, so the column set is fixed regardless of how many engines
// exist across the visible throws.

function fmtEngineSectorCell(section) {
  if (!section) return '&mdash;';
  if (section.timed_out) return '<span class="badge bad">timeout</span>';
  if (section.ok === false) {
    // A genuine non-result (engine errored/couldn't score), not a wrong
    // answer -- engines are expected to always produce one, so a
    // non-result is its own problem. Shown plainly (the real reason IS still
    // there, on hover) rather than folded into PASS/FAIL, which is
    // reserved for "the engine DID produce a sector and it disagrees
    // with AD." Short fixed label + title= (not the full reason inline)
    // -- 2026-08-13, the sector column was narrowed: a full-sentence
    // failure reason inline was exactly the kind of long field that
    // column can no longer afford to render in place.
    const title = section.reason ? ' title="' + escapeAttr(section.reason) + '"' : '';
    return '<span class="badge bad"' + title + '>no result</span>';
  }
  return fmtSectorRing(section.sector, section.ring);
}

// One throw's engine list, normalized to a uniform shape -- the primary
// engine's fields live at the TOP LEVEL of `p` (discover_packages()'s
// original shape, unchanged), also-run engines live in `p.engines[name]`
// (docs/ENGINES.md) -- this merges both into the same
// {name, ok, sector, ring, board_xy_mm, reason, timed_out, sector_match,
// tip_distance_mm} shape so fmtEngineSectorCell/the row renderer below
// never need to know which kind of engine they're looking at.
function engineSectionsFor(p) {
  const sections = [];
  if (p.primary_engine) {
    sections.push({
      name: p.primary_engine,
      // The answer, not a peer. Everything below this row is a voter
      // feeding it -- see the .primary rule in CSS for the treatment.
      primary: true,
      ok: p.ok, sector: p.sector, ring: p.ring, board_xy_mm: p.board_xy_mm,
      reason: p.reason, timed_out: false,
      sector_match: p.sector_match, tip_distance_mm: p.tip_distance_mm,
    });
  }
  for (const name of Object.keys(p.engines || {})) {
    const s = p.engines[name];
    sections.push({
      name: name,
      ok: s.ok, sector: s.sector, ring: s.ring, board_xy_mm: s.board_xy_mm,
      reason: s.reason, timed_out: s.timed_out,
      sector_match: s.sector_match, tip_distance_mm: s.tip_distance_mm,
    });
  }
  return sections;
}

// Rank map {engine name -> position} from the CURRENT sidebar Engines
// config (primary first, then also_run in its configured order, then
// any remaining available_engines) -- 2026-08-14, the row order of
// engines in the scoring tab follows the configured engine order. A
// throw's own
// `p.engines` object-key order reflects whatever order happened to be
// configured back when THAT throw was captured, which can silently
// drift from today's sidebar order as the config changes over a
// session -- this reads the LIVE config instead, so every row in the
// table (old throws included) reflects the order visible in the
// sidebar right now. Falls back to `null` (no reordering) before the
// first real config has loaded.
function engineOrderRank() {
  const cfg = lastEngineConfig;
  if (!cfg) return null;
  // cfg.* and a throw package's own sections both carry the SAME real
  // engine names, so these compare directly.
  const order = [];
  if (cfg.primary) order.push(cfg.primary);
  for (const name of (cfg.also_run || [])) {
    if (!order.includes(name)) order.push(name);
  }
  for (const name of (cfg.available_engines || [])) {
    if (!order.includes(name)) order.push(name);
  }
  const rank = new Map();
  order.forEach((name, idx) => rank.set(name, idx));
  return rank;
}

// Reorders `sections` (engineSectionsFor()'s own output, in whatever
// order that throw's package happens to store them) to match
// engineOrderRank() above -- stable sort, so an engine this config
// doesn't know about at all (removed since, or a future name) keeps its
// original relative position rather than jumping unpredictably.
function orderedEngineSections(sections) {
  const rank = engineOrderRank();
  if (!rank) return sections;
  return sections.slice().sort((a, b) => {
    const ra = rank.has(a.name) ? rank.get(a.name) : Infinity;
    const rb = rank.has(b.name) ? rank.get(b.name) : Infinity;
    return ra - rb;
  });
}

function renderPackages(packages, emptyMessage) {
  const tbody = document.getElementById('packages-tbody');
  const totalCols = 9;  // + the View column
  if (!packages || packages.length === 0) {
    tbody.innerHTML = '<tr><td colspan="' + totalCols + '">' + (emptyMessage || 'no packages saved yet') + '</td></tr>';
    return;
  }
  const n = packages.length;
  tbody.innerHTML = packages.map((p, i) => {
    const sections = orderedEngineSections(engineSectionsFor(p));

    // AD row -- the reference: its own raw sector/ring and tip position.
    // "match" cell shows the AD-Wrong control only when there's an
    // actual miss to flag (2026-08-13, see fmtAdWrongCell's own comment).
    // AD off for this throw (ad_matched null, no ad_ground_truth.json):
    // render no AD row at all rather than an empty one. A blank row reads
    // as "AD failed here"; the honest picture is that AD was not part of
    // this throw.
    // An AD row only when AD actually ANSWERED (2026-09-21). It used to
    // render on ad_matched === false too, as "Autodarts / no data yet" with
    // a dash for a sector -- a row whose every cell says "nothing here".
    // That is not merely noise: marking a throw wrong when AD had no answer
    // WRITES an ad_ground_truth.json (match_reason
    // operator_marked_no_ad_data) to record the human's judgement, so the
    // act of correcting a throw conjured an empty AD row underneath it, and
    // the correction badges then had to live in that row's match cell.
    // Correcting a throw should not invent a reference row for a reference
    // that never spoke.
    const adAsked = (p.ad_matched === true);
    // View sits beside the throw number: both identify the THROW, and the
    // number is the thing an operator scans for before opening it.
    let html = !adAsked ? '' : ('<tr class="ad-row"><td>' + (n - i) +
      '</td><td class="col-view">' + fmtViewCell(p) +
      '</td><td>' + fmtCaptured(p.captured_at_utc) +
      '</td><td>Autodarts' +
      '</td><td>' + fmtSectorRing(p.ad_sector, p.ad_ring) +
      '</td><td>' + fmtAdWrongCell(p, sections) +
      '</td><td>' + fmtLatency(p.ad_latency_ms) +
      '</td><td>&mdash;' +
      '</td><td>' + fmtXy(p.ad_tip_xy_mm) + '</td></tr>');

    // One row per engine that actually ran, in the configured order
    // (see orderedEngineSections() above) -- blank #/captured (same
    // throw group, not a new one), delta-from-AD in the last two
    // columns (fmtDeltaXy/tip_distance_mm), not raw values.
    // With AD off there is no AD row, so the first engine row -- the
    // primary, i.e. Zeus -- becomes the group header and carries the
    // throw number and capture time the AD row would have shown.
    // Without this the whole group renders with a blank number and no
    // timestamp, which reads as a continuation of the throw above it.
    return html + sections.map((s, si) => (
      '<tr class="engine-row' + (s.primary ? ' primary' : '') + '">' +
      '<td>' + (!adAsked && si === 0 ? (n - i) : '') + '</td>' +
      // View once per throw, beside the number: on the first engine row
      // when there is no AD row above to hold it, blank on the rest.
      '<td class="col-view">' + (!adAsked && si === 0 ? fmtViewCell(p) : '') + '</td>' +
      '<td>' + (!adAsked && si === 0 ? fmtCaptured(p.captured_at_utc) : '') + '</td><td>' +
      escapeAttr(s.name) +
      '</td><td>' + fmtEngineSectorCell(s) +
      // With no AD row, the first engine row carries the throw-level
      // correction controls -- they are THROW-level, not AD-level, and a
      // human verdict already on record must stay visible and reversible
      // even though the reference it overrides never answered. Same row
      // that already carries the throw number and capture time in that
      // case. Nothing is displaced: `match` grades against AD, and with no
      // AD there is no comparison to show here.
      '</td><td>' + (!adAsked && si === 0
        ? fmtAdWrongCell(p, sections)
        : (s.ok === false || s.timed_out ? '&mdash;' : fmtMatch(s.sector_match))) +
      // Latency is a THROW-level fact -- the engines run concurrently
      // off one capture and share a single answer instant, so there is
      // no honest per-engine number. It shows once, on the AD row.
      '</td><td></td>' +
      '<td>' + (s.ok === false || s.timed_out ? '&mdash;' : fmtVal(s.tip_distance_mm !== null && s.tip_distance_mm !== undefined ? s.tip_distance_mm.toFixed(1) : null, ' mm')) +
      '</td><td>' + (s.ok === false || s.timed_out ? '&mdash;' : fmtDeltaXy(s.board_xy_mm, p.ad_tip_xy_mm)) +
      '</td></tr>'
    )).join('');
  }).join('');
}

// Event delegation on the (regenerated-every-render) tbody, bound ONCE at
// script load -- individual <button> onclick handlers would need
// re-binding every renderPackages() call since innerHTML replaces the
// whole tbody each time; delegation avoids that entirely.
document.getElementById('packages-tbody').addEventListener('click', (ev) => {
  // Checked BEFORE the AD-wrong branch below, and returning either way:
  // both buttons live in the same cell, and `closest('.ad-wrong-btn')`
  // on a misscore click would walk past it to nothing -- but only
  // because the two classes differ, which is exactly the kind of thing
  // that stops being true when a third button arrives. Explicit.
  const captureBtn = ev.target.closest('.misscore-capture-btn');
  if (captureBtn) {
    captureMisscore(captureBtn.dataset.session, captureBtn.dataset.throw);
    return;
  }
  // The third button (see the note above): open the throw viewer window.
  const viewBtn = ev.target.closest('.view-throw-btn');
  if (viewBtn) {
    const s = encodeURIComponent(viewBtn.dataset.session);
    const t = encodeURIComponent(viewBtn.dataset.throw);
    window.open('/packages/' + s + '/' + t + '/viewer', '_blank', 'noopener');
    return;
  }
  const btn = ev.target.closest('.ad-wrong-btn');
  if (!btn) return;
  const session = btn.dataset.session;
  const throwId = btn.dataset.throw;
  const wrong = btn.dataset.wrong === '1';
  if (!wrong) {
    // UNMARK -- no modal. There's nothing to ask: clearing the flag
    // clears its whole explanation (note + confirmed truth) server-side
    // (see mark_operator_ad_wrong's own docstring), and a confirm-dialog
    // on an already-undoable toggle is friction for nothing.
    btn.disabled = true;
    postMarkAdWrong(session, throwId, {wrong: false}).catch(() => { btn.disabled = false; });
    return;
  }
  // MARK -- ask WHICH answer was actually right first.
  // Nothing is POSTed until the modal's own Confirm button is clicked.
  openAdWrongModal(session, throwId);
});

// The single fetch() every mark/unmark goes through (modal Confirm, the
// Unmark button, both). `extra` is merged into the request body, so the
// pre-modal shapes ({wrong:true}/{wrong:false}) are still exactly
// what goes over the wire when there's no confirmation to send.
async function postMarkAdWrong(session, throwId, extra) {
  const resp = await fetch(
    '/api/packages/' + encodeURIComponent(session) + '/' + encodeURIComponent(throwId) + '/mark-ad-wrong',
    {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(extra || {}),
    }
  );
  const body = await resp.json();
  if (body && body.ok && body.package) {
    // Apply the server's authoritative updated package through the
    // SAME ingestPackages()/renderScoringTable() path every other live
    // update uses -- re-enables the row's button as part of the normal
    // re-render, rather than hand-patching just this row. Every OTHER
    // connected tab gets the same update via the PACKAGES_UPDATED
    // broadcast this endpoint also sends.
    ingestPackages([body.package]);
    return body;
  }
  console.error('mark-ad-wrong failed', body);
  throw new Error((body && body.reason) || 'mark-ad-wrong failed');
}

// -- "Which was actually right?" modal ---------------------------------
// 2026-08-13: clicking AD Wrong asks which engine (or a manually
// entered segment) was actually right -- the answer picked here is persisted onto the
// throw's own ad_ground_truth.json (operator_confirmed_source/_sector/
// _ring) and becomes what discover_packages() grades every engine row on
// this throw against, instead of AD's own (just-rejected) answer.
//
// Hand-rolled, no modal library -- same self-contained constraint as the
// rest of this dashboard. State is two plain variables, not a framework:
// the throw being confirmed, and nothing else (the chosen option is read
// straight off the DOM at submit time, so it can never go stale).
let adWrongModalSession = null;
let adWrongModalThrowId = null;

function findPackage(session, throwId) {
  for (const p of allKnownPackages.values()) {
    if (p.session === session && p.throw_id === throwId) return p;
  }
  return null;
}

function openAdWrongModal(session, throwId) {
  const p = findPackage(session, throwId);
  if (!p) { console.error('no such package in this tab', session, throwId); return; }
  adWrongModalSession = session;
  adWrongModalThrowId = throwId;

  document.getElementById('ad-wrong-modal-sub').innerHTML =
    'Throw ' + escapeAttr(throwId) + ' \u2014 AD said <b>' +
    fmtSectorRing(p.ad_sector, p.ad_ring) + '</b>. Pick the answer that was actually right; ' +
    'every engine on this throw is then scored against it instead of AD.';
  document.getElementById('ad-wrong-modal-error').textContent = '';

  // One radio per engine that actually ran on THIS throw (same
  // engineSectionsFor() the table rows themselves are built from, so the
  // options can never list an engine the row grid doesn't show), plus a
  // manual entry. Engines that produced no result (failed/timed out)
  // still appear but are disabled -- hiding them entirely would leave a
  // human wondering whether the engine ran at all.
  const sections = engineSectionsFor(p);
  const opts = sections.map((s, i) => {
    const usable = s.ok !== false && !s.timed_out && !!s.ring;
    const seg = usable ? fmtSectorRing(s.sector, s.ring) : (s.timed_out ? 'timed out' : 'no result');
    return '<label class="modal-opt">' +
      '<input type="radio" name="ad-wrong-choice" value="engine:' + escapeAttr(s.name) + '"' +
      ' data-sector="' + escapeAttr(s.sector === null || s.sector === undefined ? '' : s.sector) + '"' +
      ' data-ring="' + escapeAttr(s.ring || '') + '"' +
      (usable ? '' : ' disabled') + '>' +
      '<span class="opt-name">' + escapeAttr(s.name) + '</span>' +
      '<span class="opt-seg">' + seg + '</span></label>';
  }).join('');
  document.getElementById('ad-wrong-options').innerHTML = opts +
    '<label class="modal-opt">' +
    '<input type="radio" name="ad-wrong-choice" value="manual">' +
    '<span class="opt-name">None of these</span>' +
    '<span class="opt-seg">enter the segment manually</span></label>';

  // Manual pickers, populated from the REAL board vocabulary injected
  // server-side (BOARD_SECTORS/BOARD_RINGS -- see their own comment).
  // The empty sector option is not a placeholder: bull/outer_bull/
  // outside genuinely have no sector in
  // opendarts.geometry.board.sector_ring_for_point's return shape.
  const selSector = document.getElementById('ad-wrong-manual-sector');
  const selRing = document.getElementById('ad-wrong-manual-ring');
  selSector.innerHTML = '<option value="">(none \u2014 bull/miss)</option>' +
    BOARD_SECTORS.map((s) => '<option value="' + s + '">' + s + '</option>').join('');
  selRing.innerHTML = BOARD_RINGS.map((r) => '<option value="' + r + '">' + r + '</option>').join('');

  // Re-opening an already-confirmed throw pre-selects what's on record,
  // so a correction is a one-click change rather than a blank re-entry.
  if (p.ad_operator_confirmed_ring) {
    const src = p.ad_operator_confirmed_source;
    const engineRadio = src && src !== 'manual'
      ? document.querySelector('#ad-wrong-options input[value="engine:' + CSS.escape(src) + '"]')
      : null;
    if (engineRadio && !engineRadio.disabled) {
      engineRadio.checked = true;
    } else {
      const manualRadio = document.querySelector('#ad-wrong-options input[value="manual"]');
      if (manualRadio) manualRadio.checked = true;
      selSector.value = p.ad_operator_confirmed_sector || '';
      selRing.value = p.ad_operator_confirmed_ring;
    }
  }
  syncAdWrongManualVisibility();

  document.getElementById('ad-wrong-modal').hidden = false;
}

// The manual sector/ring row is only meaningful under the "None of
// these" option -- shown/hidden on every choice change rather than
// permanently visible, so the modal reads as one decision, not two.
function syncAdWrongManualVisibility() {
  const checked = document.querySelector('#ad-wrong-options input[name="ad-wrong-choice"]:checked');
  document.getElementById('ad-wrong-manual').hidden = !(checked && checked.value === 'manual');
}

document.getElementById('ad-wrong-options').addEventListener('change', syncAdWrongManualVisibility);

function closeAdWrongModal() {
  document.getElementById('ad-wrong-modal').hidden = true;
  adWrongModalSession = null;
  adWrongModalThrowId = null;
  // The row's own button was never disabled on this path (nothing was
  // sent), so there's nothing to re-enable here -- Cancel really is a
  // no-op beyond closing.
}

document.getElementById('ad-wrong-cancel').onclick = closeAdWrongModal;

// Backdrop click and Escape both cancel -- standard modal affordances,
// and both go through the same closeAdWrongModal() (no request is ever
// issued on any cancel path).
document.getElementById('ad-wrong-modal').addEventListener('click', (ev) => {
  if (ev.target.id === 'ad-wrong-modal') closeAdWrongModal();
});
document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && !document.getElementById('ad-wrong-modal').hidden) closeAdWrongModal();
});

document.getElementById('ad-wrong-submit').onclick = async () => {
  const errEl = document.getElementById('ad-wrong-modal-error');
  const checked = document.querySelector('#ad-wrong-options input[name="ad-wrong-choice"]:checked');
  if (!checked) { errEl.textContent = 'Pick which answer was actually right.'; return; }

  let confirmedSource, confirmedSector, confirmedRing;
  if (checked.value === 'manual') {
    confirmedSource = 'manual';
    confirmedSector = document.getElementById('ad-wrong-manual-sector').value || null;
    confirmedRing = document.getElementById('ad-wrong-manual-ring').value || null;
    if (!confirmedRing) { errEl.textContent = 'Pick a ring.'; return; }
  } else {
    confirmedSource = checked.value.slice('engine:'.length);
    confirmedSector = checked.dataset.sector || null;
    confirmedRing = checked.dataset.ring || null;
  }

  const btn = document.getElementById('ad-wrong-submit');
  btn.disabled = true;
  try {
    await postMarkAdWrong(adWrongModalSession, adWrongModalThrowId, {
      wrong: true,
      confirmed_source: confirmedSource,
      confirmed_sector: confirmedSector,
      confirmed_ring: confirmedRing,
    });
    closeAdWrongModal();
  } catch (err) {
    console.error('mark-ad-wrong request failed', err);
    errEl.textContent = 'Request failed \u2014 see browser console.';
  } finally {
    btn.disabled = false;
  }
};

// -- "Start new session view" -- a PURELY VISUAL, client-side filter on
// the Scoring table (2026-08-12): lets the operator visually start a
// clean session whatever is on disk -- explicitly NOT a
// data-deletion/archival feature (docs/DESIGN.md's "Replay is the
// source of truth": every
// throw package must stay on disk, forever, replayable). Nothing here
// ever issues a fetch()/DELETE/file operation of any kind -- it only
// changes what this ONE browser tab currently RENDERS.
//
// DESIGN DECISION: a client-side-only
// timestamp marker, not a server-side per-connection marker. Rationale:
// (1) simplicity -- no new server state, no new endpoint, nothing that
// could ever be confused with a real data operation; (2) safety -- by
// construction there is no code path here that could touch
// opendarts/live/capture_daemon.py's saved packages on disk, since this
// code never talks to the server about it at all; (3) "resets on reload"
// is the CORRECT behavior for a "view", not a bug -- a fresh page load is
// a fresh look at the board; there is no persisted client-side view
// filter anywhere. The only real cost is that
// a WebSocket reconnect note: reconnects don't clear the marker (it's a
// plain JS variable, untouched by connectWebSocket()) -- only an actual
// page reload does, so a brief network hiccup won't silently un-clear the
// view underneath the user.
let viewFilterSinceMs = null; // null = show everything; else epoch ms.

// One random id per PAGE LOAD (not per browser/user -- a reload gets a
// fresh id, so an old tab's last report just stops updating rather than
// being overwritten by the new one). The audio-clients panel reports
// under it (POST /api/audio/clients), which is how several
// simultaneously-open screens stay distinguishable from each other.
const debugTabId = 'tab_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);

// Every payload this page has ever received (initial /api/packages load,
// every HELLO/PACKAGES_UPDATED WebSocket push, capped to the most recent
// 20 -- see AppState._package_poll_loop/_handle_live_event) is merged in
// here, keyed by each package's unique on-disk path -- NOT replaced
// wholesale -- so the "N earlier throws hidden" count stays honest even
// after a live push only carries the most recent 20 (without this, the
// hidden-count could silently shrink/lie after the marker's own moment
// scrolls out of the last-20 window).
let allKnownPackages = new Map();

// The single place packages ever reach the DOM from -- initial load,
// every WebSocket push, and (historically) the AD-refresh button, all
// call this instead of renderPackages() directly, so the view filter is
// applied uniformly no matter which path a package arrived through. This
// is also what makes item 3 (a live-pushed new throw still appearing
// while a filter is active) true: PACKAGES_UPDATED's handler calls this
// same function, which re-applies the SAME viewFilterSinceMs check to the
// merged set including the brand-new package -- there is no separate
// "live" code path that could accidentally bypass the filter.
function ingestPackages(packages) {
  for (const p of (packages || [])) {
    if (p && p.path) allKnownPackages.set(p.path, p);
  }
  renderScoringTable();
}

// Self-heal, added 2026-08-24 -- real incident (flagged by the pull/QA
// process, caught because the pulled corpus disagreed with the
// dashboard): session `20260822-002507` captured 120 real throws, the
// dashboard showed 102 and computed every Scoring-tab percentage over
// that undercounted subset, with nothing on the page indicating it was
// incomplete. Root cause: HELLO/PACKAGES_UPDATED both cap `packages` to
// the newest 20 (harmless at normal pace, the list is sorted newest-
// first) -- but a WebSocket that drops for longer than it takes >20
// throws to land loses everything older than the newest 20 on
// reconnect, silently (confirmed: a real 3m16s outage, 38 darts landed
// inside it). PACKAGES_UPDATED already sent a true, uncapped `count`
// that nothing read; HELLO now sends one too (see events_ws()'s own
// dated docstring). This compares that real total against what this
// tab actually has and, on a mismatch, refetches the uncapped
// /api/packages and re-ingests -- self-healing regardless of how large
// the gap is, not special-cased to the WebSocket-outage shape
// specifically (a brand-new tab landing between loadInitial() and the
// first HELLO would hit the exact same check, for the same reason).
// Visibly honest about it via the ws-indicator badge rather than
// silently reverting to a "connected" look mid-resync or if the resync
// itself fails -- a wrong number that looks authoritative is worse
// than a visible gap.
let resyncInFlight = false;
async function checkPackageCountAndResync(count) {
  if (typeof count !== 'number' || count === allKnownPackages.size || resyncInFlight) return;
  resyncInFlight = true;
  const indicator = document.getElementById('ws-indicator');
  const priorText = indicator ? indicator.textContent : null;
  const priorClass = indicator ? indicator.className : null;
  if (indicator) {
    indicator.textContent = 'resyncing (' + allKnownPackages.size + ' of ' + count + ')';
    indicator.className = 'badge unknown';
  }
  try {
    const resp = await fetch('/api/packages');
    const fresh = await resp.json();
    // REPLACE, don't merge (2026-09-17): /api/packages is the whole list,
    // and merging never dropped a package that had left the disk -- the
    // count then never matched, and every later update refetched the
    // whole archive again.
    if (Array.isArray(fresh)) allKnownPackages.clear();
    ingestPackages(fresh);
  } catch (err) {
    console.error(
      'package count mismatch (have ' + allKnownPackages.size + ', server has ' +
      count + ') but the resync fetch itself failed', err,
    );
  } finally {
    resyncInFlight = false;
    if (indicator) {
      if (allKnownPackages.size === count) {
        indicator.textContent = priorText;
        indicator.className = priorClass;
      } else {
        // Resync fetch either failed, or genuinely raced a real change
        // on the server in between -- stay visibly honest rather than
        // reverting to a "connected" look that implies fully caught up.
        indicator.textContent = 'count mismatch (' + allKnownPackages.size + ' of ' + count + ')';
        indicator.className = 'badge bad';
      }
    }
  }
}

function renderScoringTable() {
  const all = Array.from(allKnownPackages.values())
    .sort((a, b) => (b.captured_at_utc || '').localeCompare(a.captured_at_utc || ''));
  let visible = all;
  if (viewFilterSinceMs !== null) {
    // Date.parse() (not a raw string compare) -- captured_at_utc is
    // written by Python's datetime.isoformat() (6-digit microseconds,
    // "+00:00" offset) while viewFilterSinceMs comes from JS Date.now()
    // -- comparing epoch milliseconds numerically is correct regardless
    // of either side's string formatting, where a naive string compare
    // between those two differently-shaped ISO-8601 forms would not be
    // reliable in every edge case. A package with no captured_at_utc at
    // all is hidden while a filter is active (can't be sure it's recent)
    // -- still fully visible again via "Show all throws".
    visible = all.filter((p) => p.captured_at_utc && Date.parse(p.captured_at_utc) >= viewFilterSinceMs);
  }
  const hiddenCount = all.length - visible.length;
  renderFilterIndicator(hiddenCount);
  renderEngineTally(visible);
  const emptyMessage = viewFilterSinceMs !== null
    ? 'no throws captured since the view was cleared yet' + (hiddenCount > 0 ? ' (' + hiddenCount + ' earlier, still saved on disk)' : '')
    : 'no packages saved yet';
  // The table is one row per engine per throw, rebuilt whole -- thousands
  // of rows on a large archive, twice per throw. Only while it can be
  // seen; otherwise it is rebuilt when the Engines tab is next opened or
  // the page becomes visible again (2026-09-17).
  if (packagesTableShowing()) {
    packagesTableStale = false;
    renderPackages(visible, emptyMessage);
  } else {
    packagesTableStale = true;
  }
}

let packagesTableStale = false;

function packagesTableShowing() {
  return tabShowing('engines');
}

function renderPackagesTableIfStale() {
  if (packagesTableStale && packagesTableShowing()) renderScoringTable();
}

document.addEventListener('visibilitychange', renderPackagesTableIfStale);

// Running per-engine accuracy vs AD, PLUS AD's own accuracy vs the
// human "Mark AD wrong" flag. Deliberately
// scoped to `visible` (respects the same view filter as the row
// numbers/table itself -- "how is each engine doing THIS session/view"),
// same reasoning the summary line this replaces already used. Engine
// names are discovered from the data itself (primary_engine + whatever
// keys show up in p.engines across the visible throws), never
// hardcoded, so this needs no code change when the configured engine
// set changes.
function computeEngineTally(visible) {
  const order = [];
  const counts = new Map(); // name -> {correct, total}
  const bump = (name, isCorrect) => {
    if (!counts.has(name)) { order.push(name); counts.set(name, {correct: 0, total: 0}); }
    const c = counts.get(name);
    c.total += 1;
    if (isCorrect) c.correct += 1;
  };
  for (const p of visible) {
    // AD's own row: "correct" = not flagged wrong by a human. Only
    // counted for throws that actually have real AD ground truth --
    // a "no data yet" throw isn't AD being right OR wrong, it's just
    // absent, and folding it into either bucket would misstate the rate.
    // -- exception: once a human has CONFIRMED what was actually right
    // on this throw (the "Which was actually right?" modal), AD's
    // absence stops being "absent" and becomes a real miss -- we now
    // know the true answer and know AD didn't have it. So a confirmed
    // throw always counts for AD, matched or not.
    // null means AD was never ASKED -- switched off for this throw, so no
    // ad_ground_truth.json was written at all. That is distinct from
    // false, which means it WAS asked and had nothing. Counting null
    // would credit AD with throws it never saw and quietly inflate its
    // accuracy, which is the one number this comparison exists to
    // produce.
    const adWasAsked = (p.ad_matched === true || p.ad_matched === false);
    if (adWasAsked && (p.ad_matched !== false || p.ad_operator_confirmed_ring)) {
      bump('AD', !p.ad_operator_marked_wrong);
    }
    // A second external oracle's answer, when wanted, comes from
    // running a non-voting registry engine -- it shows
    // up in engineSectionsFor(p) below like any other engine, no special
    // casing here anymore.
    for (const s of engineSectionsFor(p)) {
      // An engine that timed out or failed to produce a score didn't
      // get this throw right -- counts toward the denominator (it had
      // its shot), not toward correct.
      //
      // OPERATOR TRUTH IS ALREADY BAKED IN HERE. `s.sector_match` comes straight from the server's own
      // discover_packages(), which grades every engine section against
      // the human-confirmed segment whenever one exists (see
      // _operator_truth_for()) and against AD otherwise -- so an engine
      // that was actually right stops being penalised in this percentage
      // the moment a human confirms it, with no second copy of that
      // preference logic living here to drift from the server's.
      //
      // A null sector_match means NO COMPARISON WAS POSSIBLE -- no matched
      // AD ground truth and no operator-confirmed answer, so there is
      // nothing to be right or wrong against. Those throws are skipped
      // entirely rather than counted as misses: with the Autodarts
      // comparison switched off every throw is ungradeable, and counting them dragged every
      // engine's percentage down against a reference that was never
      // consulted.
      //
      // This is NOT the same as an engine that ran and failed. That one
      // has a section, is compared, and comes back false -- it had its
      // shot and missed, so it still counts toward the denominator.
      if (s.sector_match === null || s.sector_match === undefined) continue;
      bump(s.name, s.sector_match === true);
    }
  }
  return order.map((name) => {
    const c = counts.get(name);
    return {name: name, correct: c.correct, total: c.total};
  });
}

function renderEngineTally(visible) {
  const el = document.getElementById('engine-tally-bar');
  if (!visible || visible.length === 0) {
    el.textContent = '';
    return;
  }
  const tally = computeEngineTally(visible);
  el.innerHTML = tally.map((t) => {
    // 2026-08-26: an operator correction did not update the % in the
    // top bar. Root cause, confirmed by direct
    // reproduction (both a 1-throw and a 6-throw synthetic session, real
    // HTTP route + real client re-render, not just code review): the
    // recompute was ALWAYS correct -- computeEngineTally() reruns fresh
    // off allKnownPackages on every render, same data source the row's
    // own PASS/FAIL badges use. The bug is display-only: whole-number
    // Math.round() hides a single-throw correction's effect on the
    // percentage 44% of the time at realistic session sizes (measured --
    // e.g. 51/101 -> 50/101 both round to "50%"), so the number on
    // screen looked frozen even though the underlying math (and the
    // row) had already updated. One decimal place makes every single
    // correction visibly move the number, regardless of session size.
    const pct = t.total > 0 ? ((100 * t.correct) / t.total).toFixed(1) : '0.0';
    const perfect = t.total > 0 && t.correct === t.total;
    return '<span class="tally' + (perfect ? ' perfect' : '') + '">' +
      '<span class="tally-name">' + escapeAttr(t.name) + '</span>' +
      '<span class="tally-val">' + t.correct + '/' + t.total + '</span>' +
      '<span class="tally-pct">' + pct + '%</span></span>';
  }).join('');
}

function renderFilterIndicator(hiddenCount) {
  const el = document.getElementById('view-filter-indicator');
  const showAllBtn = document.getElementById('btn-show-all');
  if (viewFilterSinceMs === null) {
    el.textContent = '';
    showAllBtn.style.display = 'none';
    return;
  }
  const t = new Date(viewFilterSinceMs).toLocaleTimeString();
  el.textContent = 'Showing throws since ' + t + ' \u2014 ' + hiddenCount +
    (hiddenCount === 1 ? ' earlier throw hidden' : ' earlier throws hidden') +
    ' (still saved on disk, not deleted).';
  showAllBtn.style.display = '';
}

document.getElementById('btn-new-session-view').onclick = () => {
  // No fetch(), no POST, nothing sent to the server at all -- this button
  // only ever mutates a JS variable in THIS tab. See the design-decision
  // comment above.
  viewFilterSinceMs = Date.now();
  renderScoringTable();
};

document.getElementById('btn-show-all').onclick = () => {
  viewFilterSinceMs = null;
  renderScoringTable();
};

// Delete recorded data -- 2026-08-14, widened to both kinds 2026-09-17.
// The one genuinely destructive control on this page: it removes real,
// permanent, server-side files (POST /api/packages/delete-all -- see that
// route's own docstring for exactly what it does and does not touch).
//
// IT TAKES BOTH KINDS NOW. It used to delete throw packages only, while
// "Save a missed dart" sat a few inches to the left writing frame-ring
// captures that nothing in the product ever removed -- and the captures
// are the bigger of the two by an order of magnitude. A rig you cannot
// clear from the screen is a rig somebody clears by ssh, or does not
// clear at all.
//
// THE QUESTION NAMES WHAT IS GOING. A GET of /api/recorded-data first, so
// the confirm() carries the real counts and the real sizes of each kind
// ("Delete 312 throw packages (2.0 GB) and 3 captures (420 MB)?") rather
// than a generic "are you sure?" -- that fetch reads and deletes nothing,
// and the destructive POST is reached only after a human has said yes to
// those numbers.
function countLabel(n, one, many) {
  return n + ' ' + (n === 1 ? one : many);
}

function recordedDataPhrase(snap) {
  const p = snap.packages || {}, c = snap.captures || {};
  return countLabel(p.count || 0, 'throw package', 'throw packages')
    + ' (' + (p.label || '0 B') + ') and '
    + countLabel(c.count || 0, 'capture', 'captures')
    + ' (' + (c.label || '0 B') + ')';
}

document.getElementById('btn-delete-recorded').onclick = async () => {
  const btn = document.getElementById('btn-delete-recorded');
  btn.disabled = true;
  try {
    let snap;
    try {
      snap = await (await fetch('/api/recorded-data')).json();
    } catch (err) {
      console.error('recorded-data count failed', err);
      logAction('Delete recorded data', 'bad',
        'could not read what is on disk \u2014 nothing was deleted');
      return;
    }
    // Reported rather than silently allowed to fail on the server: a dump
    // in flight is a real, temporary "not now", and saying so beats a
    // refusal the operator has to decode.
    if (snap.busy) {
      logAction('Delete recorded data', 'warn',
        'a capture is being written right now \u2014 nothing was deleted. Try again when it finishes.');
      return;
    }
    const total = snap.total || {};
    if (!total.count && !total.bytes) {
      logAction('Delete recorded data', 'ok',
        'nothing to delete \u2014 no throw packages and no captures on this rig');
      return;
    }
    const ok = window.confirm(
      'Delete ' + recordedDataPhrase(snap) + '? This cannot be undone.\n\n'
      + 'It removes the real files under\n  ' + (snap.packages || {}).root
      + '\n  ' + (snap.captures || {}).root
      + '\n\nAnything you still need must be copied off this rig first.'
    );
    if (!ok) return;
    const line = logAction('Delete recorded data', 'pending',
      'deleting ' + recordedDataPhrase(snap) + '\u2026');
    try {
      const resp = await fetch('/api/packages/delete-all', {method: 'POST'});
      const body = await resp.json();
      if (body.ok) {
        // The SERVER's counts, not the ones the question was asked with:
        // a package written between the two calls, or a session that
        // failed to delete, makes those two numbers differ, and the line
        // in the log has to be what really happened.
        updateActionLine(line, 'ok', 'Delete recorded data',
          recordedDataPhrase({
            packages: {count: (body.packages || {}).deleted, label: (body.packages || {}).label},
            captures: {count: (body.captures || {}).deleted, label: (body.captures || {}).label},
          }) + ' deleted');
        allKnownPackages.clear();
        viewFilterSinceMs = null;
        renderScoringTable();
        refreshConfig();
      } else {
        updateActionLine(line, 'bad', 'Delete recorded data', body.reason || 'unknown error');
      }
    } catch (err) {
      console.error('delete recorded data failed', err);
      updateActionLine(line, 'bad', 'Delete recorded data',
        'request failed \u2014 see browser console');
    }
  } finally {
    btn.disabled = false;
  }
};

// -- status pill (header) ----------------------------------------------
// Maps opendarts.capture.trigger_state.ThrowState -> {color, short primary
// label, secondary detail text}. **CHANGED 2026-08-12**:
// the original labels/detail sentences were too narrative/wordy.
// The status vocabulary here is short, direct words --
// "Stopped" / "Throw" / "Calibrating" / "Error" / "Takeout" or "Wait" --
// not full sentences; opendarts's own ThrowState enum has its own
// names, so the labels below are a terse remap in that same spirit:
// IDLE -> "Ready", MOTION_DETECTED -> "Motion", SETTLING -> "Settling",
// READY_TO_CAPTURE -> "Capturing", TAKEOUT_WAITING -> "Takeout".
// Colors unchanged from the original judgment call, still on the usual
// dashboard convention
// (#57d17a green / #ff6b6b red / #6aa8ff blue / #888 grey / #f0b429
// amber): IDLE=grey (neutral, not "bad"), MOTION_DETECTED/SETTLING=blue
// (same color for both -- cosmetically different phases of the same
// "something's happening" wait, per throw_trigger.py's own module
// docstring), READY_TO_CAPTURE=green, TAKEOUT_WAITING=amber.
//
// dart_count (opendarts.capture.trigger_state.ThrowTriggerState.dart_count,
// 0-3, added to the TRIGGER_STATE event payload alongside this pill --
// see opendarts/live/capture_daemon.py's on_event calls) is kept in the
// secondary text -- still directly useful ("Settling -- dart 2 of 3" says
// more than "Settling" alone) -- just shortened to match the terser
// primary label instead of a full sentence.
// **RESTRUCTURED 2026-08-12** (bug #1: the top-bar ready pill needed
// separate ready / wait (or starting) / takeout / stopped states plus a
// throw/takeout detail). Verified against a real two-axis
// pill before building this, not guessed from memory. Two axes: a
// PRIMARY status (Stopped/Error/Calibrating/Wait/Takeout/Throw, computed
// from board status/phase/detection substate + cams_ok/score_armed/
// capture_frozen) and a SECONDARY detail (the last
// discrete event, e.g. "Throw detected"), both rendered by
// setStatus() below into exactly the
// #statusText/#phaseText id shape opendarts's own pill already used
// (status-text/status-detail) -- confirmed real, not assumed.
//
// opendarts does NOT have cams_ok/score_armed/capture_frozen/
// orientation_confirmed signals -- a different architecture, adapted
// here rather than literally copied (stated plainly, not glossed over).
// PRIMARY_INFO below is the real primary axis this project actually has:
// CaptureLoopController.running/starting (see AppState.capture_starting's
// own docstring, opendarts/live/server.py) for Stopped/Starting, and
// ThrowTriggerState.state for Throw-vs-Takeout once a session is
// genuinely live. PHASE_DETAIL is the secondary axis -- the same real
// ThrowState sub-states the old (pre-2026-08-12) single-axis
// TRIGGER_STATE_INFO this replaces already covered, just no longer also
// deciding the dot color.
// WAITING covers two real reasons the system can't accept a throw right
// now -- opening cameras + auto-calibrating on Start, or a manual
// mid-session recalibrate -- unified under one main-state label
//, with the reason itself carried as the detail text below,
// same pattern PHASE_DETAIL already uses for Throw's own sub-phases.
// Was two separate ideas before this date: STARTING had its own primary
// label/color, and a manual recalibrate had NO primary-axis
// representation at all -- the real bug ("I press Calibrate and it
// still says Throw") this section fixes.
const PRIMARY_INFO = {
  NONE: { color: '#555', label: 'No live capture' },
  CONNECTING: { color: '#555', label: 'Connecting\u2026' },
  STOPPED: { color: '#888', label: 'Stopped' },
  WAITING: { color: '#6aa8ff', label: 'Waiting' },
  THROW: { color: '#57d17a', label: 'Throw' },
  TAKEOUT: { color: '#f0b429', label: 'Takeout' },
};

// dart_count (opendarts.capture.trigger_state.ThrowTriggerState.dart_count,
// 0-3) kept in the secondary text, same terser form the old pill used
// ("Settling -- dart 2 of 3"). IDLE -> '' -- a bare "Throw" primary with
// no secondary detail already reads as "ready and waiting," matching
// the project's own word for this ("ready") without saying it twice.
const PHASE_DETAIL = {
  IDLE: (n) => '',
  MOTION_DETECTED: (n) => 'Motion' + (n !== null ? ' \u2014 dart ' + (n + 1) + ' of 3' : ''),
  SETTLING: (n) => 'Settling' + (n !== null ? ' \u2014 dart ' + (n + 1) + ' of 3' : ''),
  READY_TO_CAPTURE: (n) => 'Capturing' + (n !== null ? ' \u2014 dart ' + n + ' of 3' : ''),
  TAKEOUT_WAITING: (n) => (n !== null ? n + ' thrown' : ''),
};

// Live, mutable client-side view of the two real backend signals the pill
// needs -- kept in sync by renderState() (full /api/state loads, WS
// HELLO), the WS handlers below (TRIGGER_STATE/CAPTURE_LOOP_STATUS/
// CALIBRATION_STATUS), and the Start/Stop button handlers' own immediate
// (pre-broadcast) updates above -- so THIS tab's own click updates the
// pill without waiting on its own broadcast round-trip, while every
// OTHER connected tab still gets it live via the broadcast.
let pillCaptureLoop = null; // state.capture_loop shape, or null (standalone -- no controller)
let pillTrigger = {}; // state.trigger shape
let lastStartError = null; // mirrors AppState.capture_last_start_error
let startingCalibDetail = ''; // set from the async CALIBRATION_STATUS(source=startup*) confirmation, bug #3
// True for the duration of THIS tab's own manual Calibrate click only --
// 2026-08-14, the WAITING/Calibrating sub-state. No backend event backs
// this (a manual recalibrate is one plain blocking POST, see
// AppState.refresh_calibration's own docstring) -- deliberately v1-scoped
// to "the tab that clicked it sees it," not broadcast to every open tab.
// Set true immediately before the fetch below, false in its `finally`,
// with an explicit renderPill() on both edges so it never waits on a
// poll to appear or clear.
let manualCalibrating = false;

// ORDERING GUARD for pillCaptureLoop -- 2026-09-22, "the pill stays on
// Waiting . Starting after Start (or Calibrate); a refresh clears it."
//
// The server's copy was right and this tab's was stale. This tab learns
// the capture-loop status from TWO channels that nothing orders against
// each other: WebSocket pushes (CAPTURE_LOOP_STATUS, TRIGGER_STATE) and
// the HTTP response bodies of its own Start/Stop clicks. With a valid
// calibration already in place, the capture thread skips auto-calibrate
// and sends its first TRIGGER_STATE -- the one message that ends
// Starting -- within milliseconds of the start being accepted. That
// routinely landed BEFORE the POST /api/start response, whose
// `starting: true` then put Starting straight back. Nothing clears it a
// second time: the capture thread only sends TRIGGER_STATE when the
// trigger state CHANGES, so an idle board kept the stale pill until the
// next dart or a reload (renderState() -- the one path that was always
// right).
//
// Rather than trusting any particular arrival order (which is exactly
// what failed), every snapshot the server sends carries a
// {status_epoch, status_seq} stamp taken in the same step as the state
// it describes -- see AppState._capture_status_stamp() in server.py.
// acceptCaptureStatus() lets a snapshot through only if it is not older
// than the newest one already applied, so whichever order the channels
// deliver in, the pill ends on the newest server state. Same guard for
// every writer (WS pushes, POST responses, /api/state loads), so other
// tabs -- which only ever see the pushes -- get the same protection.
//
// A different epoch means a different server process (a restart resets
// its counter to zero), so it is accepted and becomes the new baseline
// rather than being compared numerically. A snapshot with no stamp at all
// (the standalone server's null capture_loop, a failure body, this tab's
// own optimistic pre-fetch state) is accepted and leaves the baseline
// alone -- it has no ordering claim to check, and dropping it would
// reintroduce a stale-state bug of its own.
let captureStatusEpoch = null;
let captureStatusSeq = -1;
function acceptCaptureStatus(snap) {
  if (!snap || typeof snap.status_seq !== 'number') return true;
  if (snap.status_epoch === captureStatusEpoch && snap.status_seq < captureStatusSeq) {
    return false;
  }
  captureStatusEpoch = snap.status_epoch;
  captureStatusSeq = snap.status_seq;
  return true;
}

// Re-read the authoritative capture-loop status after this tab's own
// Start/Stop/Calibrate request settles. Belt to acceptCaptureStatus()'s
// braces: the guard stops an old snapshot overwriting a new one, and this
// covers the case where the NEWEST thing this tab holds is its own
// unstamped optimistic state -- e.g. the Start fetch threw (network
// blip, server restarting) after the pill was already set to Starting,
// and no push is coming to correct it. Goes through the same guard, so
// a slow response here can never undo a push that overtook it.
async function resyncCaptureLoop() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    const state = await r.json();
    if (!acceptCaptureStatus(state.capture_loop)) return;
    pillCaptureLoop = state.capture_loop;
    pillTrigger = state.trigger || pillTrigger;
    if (state.capture_loop && state.capture_loop.last_start_error) {
      lastStartError = state.capture_loop.last_start_error;
    }
    renderPill();
  } catch (e) {
    // Best-effort: the pushes remain the primary channel, and a failed
    // re-read must not surface as a failed Start/Stop/Calibrate.
    console.error('capture-loop status re-read failed', e);
  }
}

function renderPill() {
  const dot = document.getElementById('status-dot');
  const textEl = document.getElementById('status-text');
  const sepEl = document.getElementById('status-sep');
  const detailEl = document.getElementById('status-detail');
  const pill = document.getElementById('status-pill');

  function apply(info, detail, title) {
    dot.style.background = info.color;
    dot.style.boxShadow = info.color === '#555' ? 'none' : '0 0 0 3px ' + info.color + '33';
    textEl.textContent = info.label;
    detailEl.textContent = detail || '';
    sepEl.style.display = detail ? '' : 'none';
    pill.title = title || '';
  }

  const trig = pillTrigger || {};
  const cl = pillCaptureLoop;

  // Honest "no live data" case -- unchanged in spirit from before this
  // task (standalone `-m opendarts.live.server`, no opendarts.live.run_product
  // capture loop sharing this process at all). Never invent a default
  // state (e.g. quietly defaulting to Ready) -- same discipline as
  // /api/state's trigger.available flag and the AD ground-truth
  // "\u2014" convention for data that genuinely isn't there.
  if (!trig.available) {
    apply(PRIMARY_INFO.NONE, '', trig.reason || 'no live capture loop in this process');
    return;
  }

  // Real capture-loop lifecycle (Stopped/Starting) takes priority over
  // the trigger's own last-known state once it's available -- a session
  // that's Stopped or still Starting must never keep showing a stale
  // Throw/Takeout left over from a PREVIOUS session. This is the actual
  // fix for bug #2 ("start/stop buttons do something in the background
  // but the pill doesn't reflect it").
  if (cl && !cl.running) {
    apply(PRIMARY_INFO.STOPPED, lastStartError ? ('last start failed: ' + lastStartError) : '', 'click Start to begin');
    return;
  }
  if (cl && cl.starting) {
    // 2026-08-15: the substate is just 'starting' then 'calibrating',
    // with no long verbose substates. Full detail
    // (which cameras, ok/reused, counts) still goes to the action log
    // via logAction() below -- it's not lost, just no longer crammed
    // into the fixed-width pill itself.
    apply(
      PRIMARY_INFO.WAITING,
      'Starting',
      'auto-calibrates only if no valid calibration exists yet -- see the action log for confirmation once it lands'
    );
    return;
  }

  // Manual mid-session recalibrate -- the other real reason for WAITING,
  // and the specific gap that used to leave the pill stuck on a stale
  // Throw/Takeout while a recalibrate (real, ~30s) was in flight. Checked
  // after Stopped/Starting (a manual calibrate can't happen during
  // either -- the controls are disabled) and before the phase-based
  // fallback below, so it always wins over a stale trig.state.
  if (manualCalibrating) {
    apply(PRIMARY_INFO.WAITING, 'Calibrating', 'manual recalibrate in progress (POST /api/calibration/refresh) -- this tab only, see the action log');
    return;
  }

  if (!trig.state) {
    // live_events_enabled is true but the first real TRIGGER_STATE hasn't
    // arrived yet -- brief in practice, still a real, distinct,
    // honestly-labeled moment rather than a blank/default flash.
    apply(PRIMARY_INFO.CONNECTING, '', 'live event push is wired but no TRIGGER_STATE has arrived yet');
    return;
  }

  const detailFn = PHASE_DETAIL[trig.state];
  const dartCount = (trig.dart_count === undefined || trig.dart_count === null) ? null : trig.dart_count;
  if (!detailFn) {
    // Unknown state name (e.g. a future ThrowState this dashboard hasn't
    // been updated for) -- show it honestly rather than silently mapping
    // to some other color.
    dot.style.background = '#f0b429';
    dot.style.boxShadow = '0 0 0 3px #f0b42933';
    textEl.textContent = trig.state;
    sepEl.style.display = 'none';
    detailEl.textContent = '';
    pill.title = 'unrecognized trigger state';
    return;
  }
  const primary = trig.state === 'TAKEOUT_WAITING' ? PRIMARY_INFO.TAKEOUT : PRIMARY_INFO.THROW;
  apply(
    primary,
    detailFn(dartCount),
    (trig.reason || '') + (trig.last_event_utc ? ' \u2014 last update ' + trig.last_event_utc : '')
  );
}

// -- diagnostics -------------------------------------------------------
// Was renderConfig() into the Info tab's #config-tbody until 2026-09-13.
// Same rows minus host/port, which are real controls now (see the config
// document and renderPortPanel below) -- renamed because "config" was
// never what these are: they report how this PROCESS was launched, and
// nothing on this page can change any of them.

// The rows are ordered by who needs them: what this IS, then where its
// output goes, then the runtime a bug report has to quote. Build first
// because it is the field every report starts with.
//
// `lastBuild` is fetched once per load rather than per state tick --
// nothing in it changes within a process except uptime, which is derived
// from started_at_epoch client-side so the table does not need polling to
// stay honest.
let lastBuild = null;
let lastDiagCfg = null;
let lastCapabilities = null;

function fmtUptime(startedEpoch) {
  if (!startedEpoch) return null;
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - startedEpoch));
  const d = Math.floor(secs / 86400), h = Math.floor(secs % 86400 / 3600);
  const m = Math.floor(secs % 3600 / 60);
  const started = new Date(startedEpoch * 1000).toLocaleString();
  const ago = d ? (d + 'd ' + h + 'h') : (h ? (h + 'h ' + m + 'm') : (m + 'm'));
  return started + ' (' + ago + ')';
}

// A missing value is rendered as an explicit "unknown", never blank: a
// blank cell reads as a broken row. code_version in particular is null
// whenever this runs from a tarball rather than a git checkout, which
// stops being an edge case the moment this repo is copied to a public git.
// GB, not bytes: nobody reads 468331323392. The percentage rides along
// because "436 GB free" means nothing without knowing whether that is most
// of the disk or the last of it.
//
// Warns below 10% and calls it bad below 5%, rather than reporting a number
// and leaving the reader to judge -- the whole point of putting it on this
// page is that someone notices before the scoring path does.
//
// THE FLOOR RIDES ALONG (2026-09-17). A percentage says how full the disk
// is; it does not say what this rig has already stopped doing. There is a
// real threshold -- min_free_disk_gb, 5 GB by default -- below which the
// capture daemon skips writing throw packages and a ring dump is refused,
// and a row that reported "4.10 GB free (3%)" while both of those were
// true was reporting the symptom and hiding the consequence. The verdict
// comes from the guard itself (opendarts.disk_space, via /api/state's
// disk.guard), never from a comparison redone here -- two implementations
// of one rule is two implementations to disagree.
function fmtDisk(d) {
  if (!d || !d.total_bytes) return null;
  const gb = (n) => (n / (1024 * 1024 * 1024)).toFixed(1);
  const pct = d.free_pct;
  const guard = d.guard || {};
  let text = gb(d.free_bytes) + ' GB free of ' + gb(d.total_bytes) + ' GB'
           + (pct === null || pct === undefined ? '' : ' (' + pct + '%)');
  if (guard.enabled === false) {
    text += ' \u2014 free-space guard off (min_free_disk_gb is negative)';
  } else if (guard.below_floor) {
    text += ' \u2014 BELOW the ' + guard.floor_label + ' floor: throw packages are '
          + 'not being written, and a capture would be refused';
  } else if (guard.floor_label) {
    text += ' \u2014 floor ' + guard.floor_label;
  }
  if (guard.below_floor) {
    return '<span class="badge bad">' + escapeAttr(text) + '</span>';
  }
  if (pct !== null && pct !== undefined && pct < 5) {
    return '<span class="badge bad">' + escapeAttr(text) + '</span>';
  }
  if (pct !== null && pct !== undefined && pct < 10) {
    return '<span class="badge warn">' + escapeAttr(text) + '</span>';
  }
  return escapeAttr(text);
}

// RAM, rendered like disk: a plain string normally, a badge when it is
// tight. Phrased in the same "X free of Y" shape as the disk row so the
// two capacity lines read as a pair. `available` is authoritative on
// Linux/Windows and an estimate on macOS -- the caveat is on the row's
// tooltip rather than crammed inline.
function fmtMem(m) {
  if (!m || !m.total_bytes) return null;
  const gb = (n) => (n / (1024 * 1024 * 1024)).toFixed(1);
  const used = m.used_pct;
  const text = gb(m.available_bytes) + ' GB free of ' + gb(m.total_bytes) + ' GB'
             + (used === null || used === undefined ? '' : ' (' + used + '% used)');
  if (used !== null && used !== undefined && used >= 95) {
    return '<span class="badge bad">' + escapeAttr(text) + '</span>';
  }
  if (used !== null && used !== undefined && used >= 85) {
    return '<span class="badge warn">' + escapeAttr(text) + '</span>';
  }
  return escapeAttr(text);
}

// CPU load, rendered like memory/disk: a badge when it is high. busy_pct is
// null for the first sample after start (no baseline yet) -- shown as
// "measuring…" rather than a fake 0%, so the row is never silently wrong.
function fmtCpu(c) {
  if (!c) return null;
  const cores = c.cores ? ' · ' + c.cores + ' cores' : '';
  if (c.busy_pct === null || c.busy_pct === undefined) {
    return '<span class="diag-unknown">measuring…</span>' + escapeAttr(cores);
  }
  const text = c.busy_pct + '%' + cores;
  if (c.busy_pct >= 90) return '<span class="badge bad">' + escapeAttr(text) + '</span>';
  if (c.busy_pct >= 75) return '<span class="badge warn">' + escapeAttr(text) + '</span>';
  return escapeAttr(text);
}

// The same guard, said again where the recording controls are (Engines
// tab, beside Delete recorded data and "Save a missed dart"). Not a second
// calculation and not a second reading -- the same disk.guard object the
// Info row uses, because the operator standing at the board is not going
// to go and look at another tab to find out why nothing is being saved.
//
// SILENT WHILE THERE IS ROOM. A line that is always on is a line nobody
// reads; this one appears when the answer is something other than "fine",
// which is exactly when it is worth the space.
function renderRecordedDataNote(cfg) {
  const el = document.getElementById('recorded-data-note');
  if (!el) return;
  const guard = ((cfg || {}).disk || {}).guard;
  if (!guard) { el.hidden = true; el.textContent = ''; return; }
  if (guard.below_floor) {
    el.textContent = 'Only ' + guard.free_label + ' free \u2014 below the '
      + guard.floor_label + ' floor. Throw packages are not being written, and a '
      + 'capture would be refused. Copy what you need off this rig, then delete '
      + 'the recorded data here.';
    el.className = 'panel-sub disk-floor-bad';
    el.hidden = false;
    return;
  }
  if (guard.enabled === false) {
    el.textContent = 'Free-space guard off (min_free_disk_gb is negative): '
      + guard.free_label + ' free, and nothing will stop packages or captures '
      + 'being written as the disk fills.';
    el.className = 'panel-sub disk-floor-warn';
    el.hidden = false;
    return;
  }
  if (guard.free_bytes === null || guard.free_bytes === undefined) {
    el.textContent = 'Free space could not be read, so the ' + guard.floor_label
      + ' floor is not being enforced on this rig.';
    el.className = 'panel-sub disk-floor-warn';
    el.hidden = false;
    return;
  }
  el.hidden = true;
  el.textContent = '';
}

function diagVal(v, suffix) {
  if (v === null || v === undefined || v === '') {
    return '<span class="diag-unknown">unknown</span>';
  }
  if (typeof v === 'boolean') return fmtBool(v);
  return escapeAttr(String(v)) + (suffix || '');
}

let lastPublish = null;

// WHY THE NEGOTIATED FOURCC IS SHOWN NEXT TO THE REQUESTED FORMAT, and
// not folded into one value: they disagree exactly when something has
// gone wrong, and that disagreement is otherwise invisible. v4l2loopback
// keeps its existing format when a consumer already holds the device and
// returns SUCCESS anyway, so "we asked for BGR24" and "the kernel gave us
// BGR24" are genuinely different facts. The consuming app then decodes
// the wrong thing and renders black while looking perfectly alive.
function renderPublishing(fh) {
  if (fh) lastPublish = fh;
  const d = lastPublish || {};
  const pub = d.publish || {};
  const body = document.getElementById('publish-tbody');
  if (!body) return;

  if (!pub.enabled) {
    body.innerHTML = '<tr><td>virtual cameras</td><td>'
      + '<span class="diag-unknown">not publishing</span>'
      + ' &mdash; no virtual cameras are being fed</td></tr>';
    return;
  }
  // A set exists but no sink reaches the pump: the publishers will never
  // be called, so every row would read "not open" with no hint that the
  // fault is upstream of them. Say it once, at the top, rather than three
  // times in a way that points at the wrong thing.
  if (d.frame_sink_attached === false) {
    body.innerHTML = '<tr><td>virtual cameras</td><td>'
      + '<span class="badge bad">not wired</span> &mdash; publishers exist but '
      + 'nothing is feeding them; frames are captured and never offered</td></tr>';
    return;
  }
  const slots = pub.slots || [];
  if (!slots.length) {
    body.innerHTML = '<tr><td>virtual cameras</td><td>'
      + '<span class="diag-unknown">no slots</span></td></tr>';
    return;
  }
  const rows = slots.map((s) => {
    const label = 'cam ' + (s.slot === undefined ? '?' : s.slot);
    if (s.open === false || (s.open === undefined && s.published === undefined)) {
      // NO "unknown" HERE. The device is a Linux concept; on Windows
      // there is none, and printing "unknown" for a field that does not
      // exist reads as a broken row rather than an absent concept. That
      // is what a Windows rig actually showed: "unknown not open", which
      // told an operator nothing at all.
      const where = s.device ? (escapeAttr(String(s.device)) + ' &middot; ') : '';
      const why = s.last_error
        ? (' &mdash; ' + escapeAttr(String(s.last_error)))
        : ' &mdash; nothing has been published yet';
      return '<tr><td>' + label + '</td><td>' + where
        + '<span class="diag-unknown">not open</span>' + why + '</td></tr>';
    }
    const neg = (s.negotiated || {});
    // Linux reports what the kernel AGREED to; Windows reports what it
    // configured. Prefer the negotiated one where it exists -- that is
    // the number that can differ from what was asked for.
    const res = (neg.width && neg.height) ? (neg.width + '×' + neg.height)
              : ((s.width && s.height) ? (s.width + '×' + s.height) : null);
    const want = s.format || '?';
    const got = neg.fourcc || null;
    // A mismatch is the whole reason this row exists -- call it out rather
    // than printing two values side by side and hoping someone compares.
    const fmt = (got && want && !fourccMatches(want, got))
      ? ('<span class="badge bad">' + escapeAttr(want) + ' requested, '
         + escapeAttr(got) + ' in force</span>')
      : (escapeAttr(want) + (got ? ' (' + escapeAttr(got) + ')' : ''));
    const bits = [];
    // BOTH BACKENDS RENDER HERE, and they can honestly say different
    // things. Linux names a /dev node and the fourcc the kernel agreed
    // to; Windows has neither -- it writes into shared memory -- but its
    // reader writes counters BACK, so it alone can report whether the
    // consumer is actually collecting frames. Each field is shown when
    // present rather than blanked to "unknown" on the platform that has
    // no such concept.
    if (s.device) bits.push(escapeAttr(String(s.device)));
    bits.push(fmt);
    if (res) bits.push(res);
    if (s.quality) bits.push('q' + s.quality);
    bits.push((s.published || 0).toLocaleString() + ' frames');
    if (s.write_errors) bits.push('<span class="badge bad">'
      + s.write_errors + ' write errors</span>');
    // Windows only: the DirectShow filter reports back through the shared
    // header. "Nothing has ever read this" is a different failure from
    // "the reader is behind", so they are separate badges.
    if (s.consumer_attached === false) {
      bits.push('<span class="badge bad">no consumer</span>');
    } else if (s.consumer_attached === true) {
      bits.push((s.read || 0).toLocaleString() + ' read');
      if (s.missed) bits.push('<span class="badge bad">' + s.missed + ' missed</span>');
      if (s.torn) bits.push('<span class="badge bad">' + s.torn + ' torn</span>');
    }
    return '<tr><td>' + label + '</td><td>' + bits.join(' &middot; ') + '</td></tr>';
  });
  body.innerHTML = rows.join('');
}

// BGR24/BGR3 and MJPEG/MJPG are the same thing under two spellings -- the
// name we ask for and the fourcc the kernel answers with. Comparing them
// literally would flag every healthy rig as mismatched.
function fourccMatches(want, got) {
  const w = String(want).toUpperCase(), g = String(got).toUpperCase();
  if (w === g) return true;
  if (w === 'BGR24' && g === 'BGR3') return true;
  if (w === 'MJPEG' && g === 'MJPG') return true;
  return false;
}

async function refreshPublishing() {
  try {
    const r = await fetch('/api/frame-health');
    renderPublishing(await r.json());
  } catch (err) {
    console.error('frame-health fetch failed', err);
  }
}

function renderDiagnostics(cfg) {
  if (cfg) lastDiagCfg = cfg;
  cfg = lastDiagCfg || {};
  const b = lastBuild || {};
  const cv = b.opencv_version
    ? b.opencv_version + (b.opencv_parallel_framework ? ' (' + b.opencv_parallel_framework + ')' : '')
    : null;
  // Grouped by the question each answers, NOT by which endpoint the value
  // came from. The three "is this rig stressed right now" numbers -- cpu,
  // memory, disk -- are the live ones, so they sit together under Load
  // (and re-poll on a timer, see startLoadPolling); everything else is a
  // static fact about what/where this rig is.
  const groups = [
    ['This rig', [
      ['version', b.app_version ? 'v' + b.app_version : null, 'The released version. A published copy has no git checkout, so the build line below is empty there and this is the one to quote in a report.'],
      ['build', b.code_version ? b.code_version.slice(0, 12) : null, b.code_version || 'no git checkout -- running from a copy?'],
      ['running since', fmtUptime(b.started_at_epoch), null],
      ['machine', b.hostname, null],
      ['system', b.platform ? (b.platform + (b.platform_release ? ' ' + b.platform_release : '')) : null, null],
      ['reach this dashboard at', diagReachAt(cfg), 'Open this address from a phone or another screen on the same network.'],
    ]],
    ['Load', [
      ['cpu', fmtCpu(cfg.cpu), 'CPU in use across all cores, measured between polls (100% = every core saturated). Blank for a moment right after start until there are two samples to compare.'],
      ['memory', fmtMem(cfg.memory), 'Physical RAM in use. Like disk, a rig near its ceiling does not fail where the memory went — it shows up as latency on the scoring path. On macOS "available" is an estimate (free + inactive + speculative pages); Linux and Windows report it directly.'],
      ['disk', fmtDisk(cfg.disk), 'Throw packages and frame-ring captures are the two things here that grow without bound, and nothing deletes either of them. A rig that fills its disk does not fail where the space went — it fails at the next write on the scoring path, which is why there is a floor below which both simply stop being written. Clear them from the Engines tab.'],
    ]],
    ['Environment', [
      ['frame source', cfg.frame_source, null],
      ['python', b.python_version, null],
      ['opencv', cv, 'The parallel framework decides whether the thread count below can be trusted -- it does not round-trip on every backend.'],
      ['opencv threads', b.opencv_threads, null],
      ['numpy', b.numpy_version, null],
    ]],
    ['Storage', [
      ['throws are saved to', cfg.package_root, null],
    ]],
    ['Wiring', [
      ['live event push wired', cfg.live_events_enabled, null],
      ['manual recalibrate affects live scoring', cfg.calibration_store_wired, null],
    ]],
  ];
  // fmtCpu/fmtMem/fmtDisk return markup (a badge when load is high), so
  // those rows are already rendered and must not be escaped again --
  // diagVal would print the tags as text.
  const preRendered = new Set(['cpu', 'memory', 'disk']);
  const html = [];
  for (const [heading, rows] of groups) {
    html.push('<tr class="cfg-group"><td colspan="2">' + heading + '</td></tr>');
    for (const [k, v, title] of rows) {
      html.push('<tr><td' + (title ? ' title="' + escapeAttr(title) + '"' : '') + '>' + k + '</td>'
        + '<td>' + (preRendered.has(k) && v ? v : diagVal(v)) + '</td></tr>');
    }
  }
  document.getElementById('diagnostics-tbody').innerHTML = html.join('');
}

// host 0.0.0.0 means "every interface", which is not an address anyone can
// type. The browser already knows which one it reached, so use that.
function diagReachAt(cfg) {
  const port = cfg.port || location.port;
  if (!port) return null;
  return location.protocol + '//' + location.hostname + ':' + port;
}

async function refreshBuildInfo() {
  try {
    const r = await fetch('/api/health');
    if (!r.ok) return;
    const j = await r.json();
    lastBuild = j.build || null;
    // The header shows the RELEASE number, not the git SHA. A published copy
    // ships without .git, so code_version is null for every user who did not
    // clone -- the version is the only thing they can quote back to us.
    const sub = document.getElementById('brand-sub');
    if (sub && lastBuild && lastBuild.app_version) {
      sub.textContent = 'live scoring \u00b7 v' + lastBuild.app_version;
    }
    lastCapabilities = j.capabilities || null;
    renderDiagnostics(null);
  } catch (e) { /* a status table must never break the page */ }
}

// ONE PASTE, not fourteen rows a human retypes. This is the actual point
// of the table for anyone reporting a problem: everything a maintainer
// asks for, already formatted, with no interpretation required from the
// person reporting.
// Plain-text (no markup) capacity line for the copy block, shared by disk
// and memory. `pctIsUsed` says whether the percentage is used or free, so
// the one wording serves both without lying about which number it is.
function plainCapacity(d, pct, pctIsUsed) {
  if (!d || !d.total_bytes) return 'unknown';
  const gb = (n) => (n / (1024 * 1024 * 1024)).toFixed(1);
  const free = d.free_bytes !== undefined ? d.free_bytes : d.available_bytes;
  return gb(free) + ' GB free of ' + gb(d.total_bytes) + ' GB'
    + (pct === null || pct === undefined ? '' : ' (' + pct + '% ' + (pctIsUsed ? 'used' : 'free') + ')');
}

function diagnosticsText() {
  const b = lastBuild || {}, cfg = lastDiagCfg || {};
  const tools = (lastCapabilities && lastCapabilities.tools) || {};
  const absent = Object.keys(tools).filter(k => tools[k] && tools[k].present === false);
  const cams = (lastCamRows && Object.keys(lastCamRows).length)
    ? Object.keys(lastCamRows).map(k => lastCamRows[k]).map(r =>
        (r.actual_width || '?') + 'x' + (r.actual_height || '?')
        + '@' + (r.actual_fps ? Math.round(r.actual_fps) : '?') + 'fps'
      ).join(', ')
    : 'none open';
  const lines = [
    'opendarts diagnostics',
    'version     : ' + (b.app_version ? 'v' + b.app_version : 'unknown'),
    'build       : ' + (b.code_version || 'unknown (not a git checkout)'),
    'started     : ' + (fmtUptime(b.started_at_epoch) || 'unknown'),
    'machine     : ' + (b.hostname || 'unknown') + '  ' + (b.platform || '?') + ' ' + (b.platform_release || ''),
    'cpu         : ' + (cfg.cpu && cfg.cpu.busy_pct !== null && cfg.cpu.busy_pct !== undefined ? cfg.cpu.busy_pct + '% of ' + (cfg.cpu.cores || '?') + ' cores' : 'unknown'),
    'disk        : ' + plainCapacity(cfg.disk, cfg.disk && cfg.disk.free_pct, false),
    'memory      : ' + plainCapacity(cfg.memory, cfg.memory && cfg.memory.used_pct, true),
    'python      : ' + (b.python_version || '?') + '   numpy ' + (b.numpy_version || '?'),
    'opencv      : ' + (b.opencv_version || '?') + ' ' + (b.opencv_parallel_framework || '?')
                     + ', ' + (b.opencv_threads === null || b.opencv_threads === undefined ? '?' : b.opencv_threads)
                     + ' thread(s), ' + (b.cpu_count || '?') + ' cores',
    'cameras     : ' + cams,
    'autodarts   : ' + diagAdSummary(),
    'audio       : ' + diagAudioSummary(),
    'tools absent: ' + (absent.length ? absent.join(', ') : 'none'),
    'capture loop: ' + (pillCaptureLoop ? (pillCaptureLoop.running ? 'running' : 'stopped') : 'unknown')
                     + (pillCaptureLoop && pillCaptureLoop.last_start_error
                        ? '  LAST START ERROR: ' + pillCaptureLoop.last_start_error : ''),
    'packages    : ' + (cfg.package_root || 'unknown'),
  ];
  return lines.join('\n');
}

function diagAdSummary() {
  const n = document.getElementById('ad-config-note');
  return n && n.textContent ? n.textContent.trim() : 'unknown';
}

// THIS screen's audio state first, then every other screen's. Both
// halves belong in a bug report: "the dashboard does not talk" is a
// question about one browser's autoplay permission far more often than
// it is a question about the rig, and the paste-able block is where a
// remote reader finds that out without asking.
function diagAudioSummary() {
  const n = document.getElementById('audio-coverage-note');
  const mine = n && n.textContent ? n.textContent.trim() : 'unknown';
  const others = (audioClientRows || [])
    .filter((c) => c.client_id !== debugTabId)
    .map((c) => (c.label || '?') + '='
      + (c.blocked ? 'BLOCKED' : (c.enabled ? (c.context_state || '?') : 'muted')));
  return mine + (others.length ? '   others: ' + others.join(', ') : '');
}

async function copyDiagnostics() {
  const btn = document.getElementById('btn-copy-diagnostics');
  const text = diagnosticsText();
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch (e) {
    // Clipboard API needs a secure context; a rig reached over plain http
    // on a LAN is not one. Fall back rather than failing silently.
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      ok = document.execCommand('copy');
      document.body.removeChild(ta);
    } catch (e2) { ok = false; }
  }
  if (btn) {
    btn.textContent = ok ? 'Copied' : 'Press Ctrl+C';
    if (!ok) window.prompt('Copy the diagnostics below:', text);
    setTimeout(() => { btn.textContent = 'Copy diagnostics'; }, 2000);
  }
}


// -- THE CONFIG DOCUMENT -------------------------------------------------
//
// One GET populates this whole tab; one PATCH saves one change. Until
// 2026-09-17 every control here owned a route pair of its own -- eight of
// them -- and four were re-fetched on every /api/state tick, so a tab
// sitting open made five requests where it now makes one.
//
// NOTHING ON THIS SIDE DECIDES WHETHER A CHANGE NEEDS A RESTART. The
// reply says so: `restart_required` names the keys that cannot reach the
// running process, `applied_live` the ones that already have, and `notes`
// explains any key that was saved but not applied. This page reports that
// answer instead of holding its own idea of which settings are live --
// which it could only ever get wrong, because "is a session running right
// now" is not a fact a browser has.
let lastConfig = null;        // the effective document from the last answer
let lastConfigRuntime = null; // the facts beside it that are not settings
let configRestartRequired = [];

function configValue(key, fallback) {
  if (!lastConfig || lastConfig[key] === undefined || lastConfig[key] === null) {
    return fallback;
  }
  return lastConfig[key];
}

function configRuntime(section) {
  return (lastConfigRuntime && lastConfigRuntime[section]) || null;
}

// Is the Autodarts listener's socket connected -- the one runtime fact on
// this tab that changes on its own, between requests, and that two
// channels report: the GET/PATCH /api/config bodies (runtime.ad) and the
// AD_CONNECTION push. Until 2026-09-22 only the first existed. Turning
// "Compare against Autodarts" on starts the listener, and the PATCH reply
// is built before its socket has connected, so it truthfully says
// `connected: false`; the socket came up a moment later and nothing told
// the page, which showed "On but NOT connected" in red until a reload.
// Nothing re-reads the document on a timer, so a push is the only way a
// change the operator did not cause (Autodarts going away, or coming
// back) can reach an open tab -- and the only way OTHER tabs hear about
// the toggle at all.
//
// With two channels comes the ordering bug the status pill had
// (acceptCaptureStatus()): the push can overtake the PATCH reply, and the
// older `false` applied last would put the red note back for good. So
// every snapshot carries {connection_epoch, connection_seq}, taken in the
// same step as the server's is_connected() read (see
// AppState.ad_connection_snapshot() in server.py), and one that is older
// than what this tab already applied is dropped. Same rules as the pill:
// a new epoch is a restarted server and becomes the new baseline; an
// unstamped snapshot has no ordering claim and is let through.
let adConnectionEpoch = null;
let adConnectionSeq = -1;
function acceptAdConnection(snap) {
  if (!snap || typeof snap.connection_seq !== 'number') return true;
  if (snap.connection_epoch === adConnectionEpoch && snap.connection_seq < adConnectionSeq) {
    return false;
  }
  adConnectionEpoch = snap.connection_epoch;
  adConnectionSeq = snap.connection_seq;
  return true;
}

// The AD_CONNECTION push. Updates only runtime.ad -- the push carries
// nothing else, and the rest of the document is still whatever the last
// GET/PATCH said. Kept even before the first document has arrived: that
// GET may have been READ on the server before this push and so be
// dropped as older for runtime.ad, and then this is the only copy of the
// newer fact (applyConfigDocument() carries it across).
function applyAdConnection(msg) {
  if (!acceptAdConnection(msg)) return false;
  lastConfigRuntime = lastConfigRuntime || {};
  lastConfigRuntime.ad = {
    available: msg.available, connected: msg.connected, reason: msg.reason,
    connection_epoch: msg.connection_epoch, connection_seq: msg.connection_seq,
  };
  renderAdConfig();
  return true;
}

// Every renderer on this tab reads the two globals above, so one answer
// redraws the lot -- the GET's and the PATCH's, which carry the same
// document, so an optimistic redraw and the next poll cannot disagree.
function applyConfigDocument(body) {
  if (!body || !body.ok || !body.config) return false;
  lastConfig = body.config;
  if (body.runtime) {
    // Everything in the new runtime replaces the old EXCEPT a runtime.ad
    // older than one already applied (see acceptAdConnection()) -- the
    // PATCH reply that says "not connected" while an AD_CONNECTION saying
    // "connected" already landed. The rest of the reply is still applied:
    // only the connection fact has a newer copy to protect.
    const heldAd = lastConfigRuntime && lastConfigRuntime.ad;
    lastConfigRuntime = body.runtime;
    if (!acceptAdConnection(body.runtime.ad) && heldAd) lastConfigRuntime.ad = heldAd;
  }
  configRestartRequired = body.restart_required || [];
  renderPortPanel();
  renderAdConfig();
  renderStorePackagesPanel();
  renderFrameRingPanel();
  renderCameraDevices(cameraDevicePayload());
  renderIdleTimeoutSelect();
  renderDetectionTimeSelect();
  renderAlwaysUpdate();
  renderMaintenanceNote();
  return true;
}

async function refreshConfig(refreshCameraNames) {
  try {
    const url = refreshCameraNames
      ? '/api/config?refresh_camera_names=true' : '/api/config';
    applyConfigDocument(await (await fetch(url)).json());
  } catch (err) { console.error('config fetch failed', err); }
}

// One PATCH, one action-log line. The LEVEL comes from what the server
// said, never from an assumption here: green only when the change is
// really in force now, amber when it was saved and waits for a restart or
// the next Start.
async function patchConfig(patch, label, describe) {
  const keys = Object.keys(patch);
  const line = logAction(label, 'pending', 'saving\u2026');
  let body = null;
  try {
    const resp = await fetch('/api/config', {
      method: 'PATCH', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(patch),
    });
    body = await resp.json();
  } catch (err) {
    console.error(label + ' save failed', err);
    updateActionLine(line, 'bad', label, 'request failed \u2014 see browser console');
    refreshConfig();
    return null;
  }
  if (!body.ok) {
    // Errors are keyed BY KEY, so the one this control sent is the one
    // reported -- not whichever reason happened to come first.
    const errors = body.errors || {};
    const reason = keys.map((k) => errors[k]).find((r) => r)
      || Object.values(errors)[0] || 'not saved';
    updateActionLine(line, 'bad', label, reason);
    // Unconditionally after a rejection: the field must go back to what is
    // actually saved rather than leave the rejected text looking accepted.
    refreshConfig();
    return body;
  }
  applyConfigDocument(body);
  const pending = keys.filter((k) => (body.restart_required || []).indexOf(k) >= 0);
  const note = keys.map((k) => (body.notes || {})[k]).find((x) => x) || null;
  const applied = keys.filter((k) => (body.applied_live || []).indexOf(k) >= 0);
  const outcome = {pending: pending, note: note, applied: applied};
  updateActionLine(line, (pending.length || note) ? 'warn' : 'ok', label,
    describe ? describe(body, outcome) : (note || 'saved'));
  return body;
}


// -- Server: the port this dashboard answers on --------------------------
// The second control on this tab that cannot apply immediately (Recording
// is the other), and for a harder reason: the listening socket is bound
// by uvicorn before this app object is built, so "apply now" would mean
// dropping the socket carrying the response that says it worked. The note
// therefore always names the restart, and says which port is live NOW
// whenever that differs from what is saved.
//
// The bind address is rendered here too and is READ-ONLY -- see the
// comment beside the field in index.html for why. `host` IS a writable
// key of the config document; the decision not to offer it on a web page
// is this page's, not the API's.
function renderPortPanel() {
  const input = document.getElementById('server-port');
  const port = configValue('port', null);
  // Never clobber a half-typed port: the /api/state poll refreshes this
  // every few seconds, and re-seeding the field under the cursor is
  // exactly how "I typed 8500 and it went back to 8420" happens.
  if (input && port !== null && document.activeElement !== input) {
    input.value = String(port);
  }
  const rt = configRuntime('port') || {};
  const hostEl = document.getElementById('server-host');
  const hostRt = configRuntime('host') || {};
  if (hostEl) hostEl.textContent = hostRt.active || configValue('host', null) || '\u2014';
  const note = document.getElementById('server-port-note');
  if (!note || port === null) return;
  note.textContent = (rt.active != null && rt.active !== port)
    ? 'Saved as ' + port + '. Serving on ' + rt.active
      + ' until this process restarts.'
    : 'Serving on ' + port + '. A change applies at the next restart, not now.';
}

document.getElementById('server-port').addEventListener('change', (ev) => {
  // Sent as the raw string: the server takes a number or a numeric string
  // and is the only side that should be deciding what 1-65535 means.
  patchConfig({port: ev.target.value}, 'Port', (body, outcome) => {
    // 'warn' with the restart named, straight from the reply: the write
    // DID succeed, but this dashboard is still answering somewhere else,
    // and a green tick would imply it had moved.
    const port = body.config.port;
    return outcome.pending.length
      ? port + ' \u2014 applies at the next restart'
      : String(port);
  });
});

// -- engine config (docs/ENGINES.md) -----------------------------------
// Live engine config, data only -- no control renders it any more (see
// renderState below). engineOrderRank() reads it to order each throw's
// engine rows in the Engines tab; null until the first /api/state lands,
// which that function already treats as "no reordering".
let lastEngineConfig = null;

// Engine configuration UI removed 2026-09-08 -- Zeus always runs as the
// primary. The
// runtime mutation endpoint went with it (2026-09-09): the engine set is
// configured in data/config.json and nowhere else. /api/state still
// REPORTS the live set, which is what lastEngineConfig above reads.

// capture_loop: null when no capture loop is wired in this process (this
// module's own standalone CLI) -- distinct from "wired but stopped",
// same honest-null convention as calibration.live_source/trigger above.
// Running/starting/stopped state itself still shows via the header
// status pill (renderPill(), unchanged) -- this function's only
// remaining job (2026-08-13, replacing the removed sidebar status text)
// is keeping #idle-timeout-select's selected option in sync with the
// server's real current value.
// `secondsOverride` is for the CAPTURE_LOOP_STATUS push, which carries
// the live idle timeout and arrives before the next config poll -- the
// one caller that already holds a newer value than the document does.
function renderIdleTimeoutSelect(secondsOverride) {
  const sel = document.getElementById('idle-timeout-select');
  const seconds = (secondsOverride === undefined || secondsOverride === null)
    ? configValue('idle_timeout_sec', null) : secondsOverride;
  if (!sel || seconds === null || seconds === undefined) return;
  // Don't clobber a selection the user is actively mid-interaction with
  // -- same reasoning the engine-timeout input this replaced used to
  // have for its own live-poll updates.
  if (document.activeElement === sel) return;
  const minutes = String(Math.round(seconds / 60));
  if (Array.from(sel.options).some((o) => o.value === minutes)) {
    sel.value = minutes;
  }
}

document.getElementById('idle-timeout-select').addEventListener('change', (ev) => {
  const minutes = Number(ev.target.value);
  patchConfig({idle_timeout_sec: minutes * 60}, 'Camera timeout',
    (body, outcome) => outcome.note || (minutes + ' min'));
});

function renderDetectionTimeSelect() {
  const sel = document.getElementById('detection-time-select');
  const settings = configValue('lifecycle_settings', null);
  if (!sel || !settings) return;
  if (document.activeElement === sel) return;
  sel.value = String(settings.dart_stable_frames);
}

// -- Recording: whether throws are written to disk -----------------------
// The odd one out on this tab, and the note says so rather than hiding it.
// Every other control here applies to the running session; this one cannot
// -- run_product reads store_packages once per Start, so a session keeps
// the setting it began with and cannot end up with half a package set.
// Both the panel and the change handler therefore report WHEN the value
// lands, using the `running` flag the endpoint returns alongside it.
function renderStorePackagesPanel() {
  const enabled = configValue('store_packages', null);
  if (enabled === null) return;
  const sel = document.getElementById('store-packages-select');
  if (sel && document.activeElement !== sel) sel.value = enabled ? 'yes' : 'no';
  const note = document.getElementById('store-packages-note');
  if (!note) return;
  const running = !!((configRuntime('store_packages') || {}).running);
  const what = enabled
    ? 'Saving a package per throw.'
    : 'Not saving — no replay corpus, and the Engines tab stays empty.';
  note.textContent = running
    ? what + ' A change applies at the next Start, not to this session.'
    : what;
}

document.getElementById('store-packages-select').addEventListener('change', (ev) => {
  const on = ev.target.value === 'yes';
  patchConfig({store_packages: on}, 'Save packages', (body, outcome) =>
    // 'warn' with the server's own note while a session is running: the
    // write DID succeed, but the rig is still doing the opposite of what
    // the control now shows, and a green tick would say otherwise.
    (on ? 'on' : 'off') + (outcome.note ? ' — ' + outcome.note : ''));
});

// -- Throw capture buffer: seconds in, megabytes shown ------------------
//
// The note under the control is the point of this block. Three 720p
// cameras produce about 270 MB every second, so the difference between 5
// and 22 in that box is the difference between 1.4GB and 6GB of this
// machine's memory -- and a seconds box with no price on it invites
// exactly the choice that takes the rig down mid-session. The figure is
// recomputed from the LIVE slot count the server reports, so a two-camera
// rig is never quoted a three-camera price.
//
// It also reports what the ring is actually holding right now, which is a
// different fact from what it is configured to hold: a ring 3 seconds
// into a 22-second window is either a session that just started or a pump
// that is stalled, and that distinction is what someone is reading this
// number to find out.
let lastFrameRing = null;

function frameRingPrice(state, seconds) {
  const perSecond = state && state.estimated_bytes_per_s ? state.estimated_bytes_per_s : 0;
  const bytes = perSecond * seconds;
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
  if (bytes >= 1e6) return Math.round(bytes / 1e6) + ' MB';
  return Math.round(bytes / 1e3) + ' kB';
}

function renderFrameRingNote(state, secondsOverride) {
  const note = document.getElementById('frame-ring-note');
  if (!note || !state) return;
  const seconds = (secondsOverride === undefined || secondsOverride === null || isNaN(secondsOverride))
    ? state.configured_seconds : secondsOverride;
  if (!seconds) {
    note.textContent = 'Off — no frames are kept, so a missed or misscored dart cannot be captured.';
    return;
  }
  const parts = [];
  parts.push('About ' + frameRingPrice(state, seconds) + ' of memory across '
    + state.n_slots + ' camera' + (state.n_slots === 1 ? '' : 's') + ' once full'
    + (state.estimate_source === 'measured'
        ? ', going by what this rig is holding now.'
        : ' (uncompressed; cameras that send their own JPEG use far less).'));
  const ring = state.ring || {};
  if (!state.attached) {
    // "No ring in this process" and "the ring is empty" are different
    // facts. Saying which is what stops an operator concluding the
    // feature is broken when it was simply never switched on here.
    parts.push(state.reason || 'Not attached in this process.');
  } else if (ring.sets) {
    parts.push('Holding ' + ring.span_s + 's (' + ring.mb + ' MB) right now.');
    if (ring.capped) parts.push('CAPPED — the window is shorter than configured.');
    if (ring.paused) parts.push('PAUSED for a capture in progress.');
  } else {
    parts.push('Nothing buffered yet — the capture loop has not produced a frame.');
  }
  // The ceiling and its price, straight from the server so the screen
  // and the code cannot drift apart.
  if (state.max_seconds) {
    parts.push('The most we record is ' + state.max_seconds + ' seconds, which would use about '
      + state.max_label + '.');
  }
  note.textContent = parts.join(' ');
}

function renderFrameRingPanel() {
  const state = configRuntime('frame_ring');
  if (!state) return;
  lastFrameRing = state;
  const input = document.getElementById('frame-ring-seconds');
  const seconds = configValue('frame_ring_seconds', state.configured_seconds);
  // The standard anti-clobber guard used by every other control on this
  // tab: never re-seed a field someone is typing into.
  if (input && document.activeElement !== input) {
    input.value = String(seconds);
  }
  renderFrameRingNote(state, input ? parseFloat(input.value) : seconds);
  renderCaptureJob(state);
}

// Live pricing WHILE TYPING, not only on save: the number is being chosen
// in the box, so the cost has to be visible in the box, not one round trip
// later when it has already been committed.
const frameRingMissedBtn = document.getElementById('frame-ring-missed-btn');
if (frameRingMissedBtn) frameRingMissedBtn.addEventListener('click', captureMissedDart);

const frameRingInput = document.getElementById('frame-ring-seconds');
if (frameRingInput) {
  frameRingInput.addEventListener('input', (ev) => {
    renderFrameRingNote(lastFrameRing, parseFloat(ev.target.value));
  });
  frameRingInput.addEventListener('change', (ev) => {
    const seconds = parseFloat(ev.target.value);
    patchConfig({frame_ring_seconds: seconds}, 'Capture buffer', (body, outcome) => {
      // 'warn' when the value was saved but there was no live ring to
      // apply it to: the write really did succeed, and a plain green tick
      // would claim the memory changed when it did not. Which of the two
      // it is comes from `applied_live`, not from anything guessed here.
      if (!seconds) return 'off';
      const ring = (body.runtime && body.runtime.frame_ring) || {};
      return seconds + 's — ' + ring.estimated_label
        + (outcome.applied.length ? '' : ', saved for the next Start');
    });
  });
}

// -- The two capture triggers -------------------------------------------
//
// A dump is megabytes to gigabytes of disk, so both of these report
// PROGRESS (bytes written against bytes total) rather than leaving a
// button that looks hung. a measured NVMe rig does 3.3 GB/s and finishes a
// 6GB dump in about two seconds; a Mac on SATA or a VM on shared storage
// will not, and that is the case this readout exists for.
function renderCaptureJob(state) {
  const el = document.getElementById('frame-ring-job');
  if (!el || !state) return;
  const writer = state.writer || {};
  const job = writer.current || writer.last;
  if (!job) { el.textContent = ''; el.hidden = true; return; }
  el.hidden = false;
  if (job.state === 'writing' || job.state === 'queued') {
    el.textContent = 'Saving ' + job.kind.replace('_', ' ') + ' capture — '
      + job.mb_written + ' of ' + job.mb_total + ' MB';
  } else if (job.state === 'failed') {
    el.textContent = 'Last capture FAILED: ' + (job.error || 'no reason recorded');
  } else {
    el.textContent = 'Last capture: ' + job.kind.replace('_', ' ') + ', '
      + job.mb_total + ' MB in ' + (job.duration_s || 0).toFixed(1) + 's'
      + (job.reason ? ' — ' + job.reason : '');
  }
}

async function captureMissedDart() {
  // The reason is asked for, not optional-by-default: six months later
  // "why is there a 6GB dump from Tuesday" has to be answerable from the
  // dump itself, and nobody will remember.
  const why = window.prompt(
    // The escape is doubled because this whole page is a Python
    // f-string; the browser receives a single one.
    'Missed dart — what happened? (saved with the capture)\n\n'
    + (lastFrameRing && lastFrameRing.flight_note ? lastFrameRing.flight_note : ''),
    'a dart landed and nothing was scored');
  if (why === null) return;
  const line = logAction('Missed dart', 'pending', 'saving the whole buffer…');
  try {
    const resp = await fetch('/api/frame-ring/capture-missed', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reason: why}),
    });
    const body = await resp.json();
    updateActionLine(line, body.ok ? 'ok' : 'bad', 'Missed dart',
      body.ok ? 'writing ' + (body.job ? body.job.mb_total : '?') + ' MB'
              : (body.reason || 'refused'));
  } catch (err) {
    console.error('missed-dart capture failed', err);
    updateActionLine(line, 'bad', 'Missed dart', 'request failed');
  }
  refreshConfig();
}

async function captureMisscore(session, throwId) {
  const line = logAction('Misscore capture', 'pending', throwId + '…');
  try {
    const resp = await fetch('/api/packages/' + encodeURIComponent(session)
      + '/' + encodeURIComponent(throwId) + '/capture-misscore', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({reason: 'operator marked this throw misscored'}),
    });
    const body = await resp.json();
    // An AGED-OUT refusal is reported in full, with the numbers the
    // server worked out. This is the case the operator most needs to
    // understand -- they walked to the screen and the frames expired --
    // and a bare "failed" would send them looking for a bug.
    updateActionLine(line, body.ok ? 'ok' : 'warn', 'Misscore capture',
      body.ok ? 'writing ' + (body.job ? body.job.mb_total : '?') + ' MB'
              : (body.reason || 'refused'));
  } catch (err) {
    console.error('misscore capture failed', err);
    updateActionLine(line, 'bad', 'Misscore capture', 'request failed');
  }
  refreshConfig();
}

// -- Audio: spoken dart calls, played HERE ------------------------------
//
// THE WHOLE FEATURE MOVED INTO THIS PAGE on 2026-09-15. The server used
// to run `afplay`/`aplay`/`winsound` on itself; it now broadcasts a
// DART_CALL message carrying a finished phrase ("treble 20") and every
// browser decides for itself whether to make a noise. The reason is
// simply where the people are: the rig is a headless box on a shelf and
// the player is looking at an iPad beside the board or a TV above it.
//
// EVERY SETTING HERE IS PER DEVICE, in localStorage, never on the
// server. On/off, volume and voice belong to the screen, not the rig:
// the TV above the board wants to be loud, the iPad in someone's hand
// wants to be off, and one server-side switch cannot express that. Two
// screens both unmuted will both speak, a few tens of milliseconds
// apart. That is expected and is precisely why each one has its own
// mute.
//
// WEB AUDIO, NOT AN <audio> ELEMENT, and this is the decision that
// matters most for how it FEELS. Pointing an element at a URL per call
// pays fetch + decode on the first play of every one of the 64 phrases
// -- on the throw path, at the exact moment the sound is supposed to be
// instant. That would be worse than the server playback it replaces.
// Instead the whole vocabulary is fetched and decodeAudioData'd into
// buffers up front, the moment sound is armed, and a call is then a
// createBufferSource().start() against memory. ~600 KB of MP3 once, in a
// settings panel, versus a stutter on every new phrase for the first
// hour of use.
const AUDIO_SETTINGS_KEY = 'opendarts.audio.v1';
// Three missed beats is what the server calls stale (AUDIO_CLIENT_STALE_S
// = 45s), so report at a third of it. Pinned to that ratio on purpose: a
// healthy screen must never flicker off the other dashboards' list.
const AUDIO_REPORT_INTERVAL_MS = 15000;
const AUDIO_CLIENTS_REFRESH_MS = 20000;

// The defaults a device that has never been configured starts with.
// OFF, deliberately. A screen that starts talking the first time it is
// opened is a worse default than a silent one -- and the on/off toggle
// being the thing you press is what earns the autoplay gesture (below),
// so "off until someone chooses" costs nothing and buys everything.
const AUDIO_DEFAULTS = {enabled: false, volume: 0.8, voice: ''};

let audioSettings = readAudioSettings();
let audioCtx = null;
let audioGain = null;
// phrase -> AudioBuffer, and which voice they were decoded from. Null
// until a set has been loaded; a voice change throws the whole map away
// rather than patching it, because a half-swapped set would speak two
// voices in one visit.
let audioBuffers = null;
let audioBufferVoice = null;
let audioCatalog = null;
let audioLoadState = 'idle';   // idle | loading | ready | error
let audioLoadDetail = '';
// TRUE means this browser refused to make a sound because no one has
// interacted with the document yet. The single most important piece of
// state on this page that nobody can see by looking at the room.
let audioBlocked = false;
// Re-entry guard for recovery: rebuilding a context changes its state,
// which fires onstatechange, which would call recovery again.
let audioRecovering = false;
let audioSpokenCount = 0;
let audioLastPhrase = '';
let audioReportTimer = null;
let audioClientRows = [];

function readAudioSettings() {
  // localStorage throws outright in some privacy modes rather than
  // returning null, so this is a try/catch and not a null check. A
  // device that cannot persist its settings still gets to use them for
  // the session, which is a far better failure than a page that does not
  // load.
  try {
    const raw = window.localStorage.getItem(AUDIO_SETTINGS_KEY);
    if (!raw) return Object.assign({}, AUDIO_DEFAULTS);
    const got = JSON.parse(raw);
    return {
      enabled: got.enabled === true,
      volume: typeof got.volume === 'number' && got.volume >= 0 && got.volume <= 1
        ? got.volume : AUDIO_DEFAULTS.volume,
      voice: typeof got.voice === 'string' ? got.voice : '',
    };
  } catch (err) {
    console.warn('audio settings unreadable, using defaults', err);
    return Object.assign({}, AUDIO_DEFAULTS);
  }
}

function writeAudioSettings() {
  try {
    window.localStorage.setItem(AUDIO_SETTINGS_KEY, JSON.stringify(audioSettings));
  } catch (err) {
    console.warn('audio settings not saved on this device', err);
  }
}

// A short human name for THIS screen, so the list of devices on another
// dashboard reads "iPad" and not a 120-character user-agent string.
// Guesswork, and only ever used as a label -- nothing branches on it.
function audioDeviceLabel() {
  const ua = navigator.userAgent || '';
  let device = 'this screen';
  if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) device = 'iPad';
  else if (/iPhone/.test(ua)) device = 'iPhone';
  else if (/Android/.test(ua)) device = 'Android';
  else if (/CrKey|TV|SmartTV|BRAVIA|AFT/.test(ua)) device = 'TV';
  else if (/Macintosh/.test(ua)) device = 'Mac';
  else if (/Windows/.test(ua)) device = 'Windows';
  else if (/Linux/.test(ua)) device = 'Linux';
  // indexOf rather than a regex literal, and not for style: matching
  // "Edg/" as a regex needs the slash escaped, and an escaped slash is
  // not a valid Python escape sequence. This whole script is a Python
  // f-string, so that is a SyntaxWarning on import (a SyntaxError from
  // 3.12 onwards) for a piece of JavaScript that was perfectly correct.
  // Python interprets every escape here before the browser ever sees
  // it, which the dashboard's own test suite exists partly to catch.
  // Demonstrated while writing this very comment, which had to be
  // reworded for the same reason.
  let browser = '';
  if (ua.indexOf('Edg/') >= 0) browser = 'Edge';
  else if (ua.indexOf('Chrome/') >= 0 && ua.indexOf('Chromium') < 0) browser = 'Chrome';
  else if (ua.indexOf('Firefox/') >= 0) browser = 'Firefox';
  else if (ua.indexOf('Safari/') >= 0) browser = 'Safari';
  return browser ? device + ' / ' + browser : device;
}

function audioContextCtor() {
  return window.AudioContext || window.webkitAudioContext || null;
}

// Created lazily, and ideally from inside a click handler: a context
// constructed during a user gesture starts 'running', while one
// constructed at page load starts 'suspended' and needs a gesture later
// anyway. Either way this never throws into a caller -- a browser with
// no Web Audio at all is a silent dashboard, not a broken one.
function ensureAudioContext() {
  if (audioCtx) return audioCtx;
  const Ctor = audioContextCtor();
  if (!Ctor) {
    audioLoadState = 'error';
    audioLoadDetail = 'this browser has no Web Audio support';
    return null;
  }
  try {
    audioCtx = new Ctor();
    audioGain = audioCtx.createGain();
    audioGain.gain.value = audioSettings.volume;
    audioGain.connect(audioCtx.destination);
    // The context can be suspended by the browser long after it was
    // first allowed -- iOS does it when a tab is backgrounded, and a
    // laptop does it on sleep. Watching statechange is what turns "the
    // TV stopped talking three hours ago" into something visible.
    // ASSIGNED FROM THE STATE, NOT ONLY CLEARED. This used to read
    // `if (state === 'running') audioBlocked = false`, which could only
    // ever turn the warning OFF. A context going the other way --
    // running -> interrupted on an iPad -- fired this handler, left the
    // flag stale at false, and rendered no banner. An iPad reported
    // exactly that on 2026-09-15: enabled, blocked=false,
    // state=interrupted, three phrases spoken and then silence, with
    // nothing on screen saying why. Silent failure is the one outcome
    // this feature exists to prevent, so the flag tracks the state.
    audioCtx.onstatechange = () => {
      audioBlocked = !!audioCtx && audioCtx.state !== 'running';
      // iOS hands back an 'interrupted' context after a call, Siri, a
      // screen lock, or another app taking the audio session. Recover
      // unprompted; the banner stays up until it works. Guarded against
      // re-entry, because recovery changes state and would re-trigger
      // this handler.
      if (audioCtx && audioCtx.state === 'interrupted' && !audioRecovering) {
        resumeAudioContext().then(() => { renderAudioPanel(); reportAudioState(); });
      }
      renderAudioPanel();
      reportAudioState();
    };
  } catch (err) {
    console.error('could not create an AudioContext', err);
    audioCtx = null;
    audioLoadState = 'error';
    audioLoadDetail = String(err && err.message ? err.message : err);
  }
  return audioCtx;
}

// THE AUTOPLAY GATE, which is the one genuinely new hazard this whole
// change introduced and the one thing that must never fail quietly.
//
// Browsers refuse audible playback until the document has had a user
// gesture. One gesture unlocks the page for as long as it stays loaded,
// and the audio On/Off toggle is a perfectly good one -- so the normal
// path just works. The case that bites is a TV mounted above the board
// that nobody can reach: it is unlocked by hand at setup, and then a
// reload (a crash, a Wi-Fi reconnect, an overnight browser update)
// silently takes the permission away. Nothing looks wrong. The board
// scores, the page renders, and it simply never speaks again.
//
// So: resume() is always awaited, the resulting state is always
// inspected, and anything short of 'running' raises a banner that covers
// the top of the page until someone touches it. Silent failure is the
// exact bug this whole change exists to remove; reintroducing it here
// would be the worst possible outcome.
// 'interrupted' IS A FOURTH STATE, and only Safari has it. The spec
// defines suspended / running / closed; iOS adds this one when the audio
// session is taken away -- a phone call, Siri, another app, the screen
// locking. It matters because `resume()` frequently resolves WITHOUT
// leaving it: the session is gone, and resuming a dead context cannot get
// one back. The only reliable recovery is a new context.
//
// Decoded AudioBuffers belong to the context that decoded them, so a
// rebuild drops them and reloads. The clips come from the HTTP cache, so
// that is a re-decode of about a megabyte, not a download.
async function rebuildAudioContext() {
  const old = audioCtx;
  audioCtx = null;
  audioGain = null;
  audioBuffers = null;
  audioBufferVoice = null;
  if (old) {
    try { await old.close(); } catch (err) { /* already dead; nothing owed */ }
  }
  return ensureAudioContext();
}

async function resumeAudioContext() {
  let ctx = ensureAudioContext();
  if (!ctx) { audioBlocked = false; return false; }
  if (ctx.state === 'running') { audioBlocked = false; return true; }
  // NEVER GUARD THE RESUME ITSELF. An earlier version wrapped this whole
  // function in the re-entry guard, and that broke the one path that
  // matters. A tap on iOS fires pointerdown, then touchend, then click,
  // and all three reach here: the first set the guard, and the other two
  // -- including the banner's own click handler -- returned immediately
  // WITHOUT calling resume(). So the banner could never be dismissed by
  // tapping it, which is precisely the bug the banner exists to let the
  // operator fix. Measured on an iPad, 2026-09-15: enabled, blocked,
  // 64/64 clips loaded, context stuck suspended through every tap.
  //
  // resume() is idempotent and cheap, so concurrent calls are harmless.
  // Only the REBUILD needs guarding, because rebuilding changes context
  // state, which fires onstatechange, which calls back into here.
  try {
    await ctx.resume();
  } catch (err) {
    // NotAllowedError is the documented rejection for "no gesture yet".
    // Logged at debug volume rather than error: it is an expected state
    // on a fresh load, not a fault.
    console.debug('AudioContext.resume() was refused', err);
  }
  if (ctx.state === 'interrupted' && !audioRecovering) {
    // Survived the resume, so the session really is gone. Rebuilding
    // inside the caller's gesture is what lets the new context reach
    // 'running' immediately on iOS.
    audioRecovering = true;
    try {
      ctx = await rebuildAudioContext();
      if (ctx) {
        try { await ctx.resume(); } catch (err) {
          console.debug('resume after rebuild was refused', err);
        }
      }
    } finally {
      audioRecovering = false;
    }
  }
  audioBlocked = !ctx || ctx.state !== 'running';
  // A rebuild drops the buffers, so reload now rather than discovering the
  // gap when the next dart lands.
  if (!audioBlocked && !audioBuffers && audioSettings.enabled) {
    audioChosenVoiceLoaded().then(loadVoiceSet).then(() => renderAudioPanel());
  }
  return !audioBlocked;
}

async function fetchAudioCatalog(force) {
  if (audioCatalog && !force) return audioCatalog;
  const resp = await fetch('/api/audio/voices');
  audioCatalog = await resp.json();
  return audioCatalog;
}

// Which voice this device should use: its own stored choice if that set
// is still installed, otherwise the server's default, otherwise whatever
// exists. A stored name that has been removed from the rig must not
// leave the device mute -- it should speak, and the panel says which
// voice it fell back to.
function audioChosenVoice() {
  const names = ((audioCatalog && audioCatalog.voices) || []).map((v) => v.voice);
  if (audioSettings.voice && names.indexOf(audioSettings.voice) >= 0) {
    return audioSettings.voice;
  }
  if (audioCatalog && names.indexOf(audioCatalog.default) >= 0) return audioCatalog.default;
  return names.length ? names[0] : '';
}

// audioChosenVoice(), but only AFTER the catalog it reads is here.
//
// The catalog is fetched lazily, so at page load it is still null and
// audioChosenVoice() answers '' -- and every caller then skipped loading
// the clips entirely. A screen that had been to the Config tab never
// noticed, because that tab fetches the catalog first. A screen that
// loads straight onto another tab with sound already on -- a kiosk via
// ?sound=on, or a TV that reloads overnight -- reported "on, running",
// loaded nothing, and stayed silent for every dart. Every path that LOADS
// clips goes through this; the ones that only display a name do not need
// to.
async function audioChosenVoiceLoaded() {
  try {
    await fetchAudioCatalog(false);
  } catch (err) {
    console.warn('voice catalog unavailable -- no clips can load yet', err);
  }
  return audioChosenVoice();
}

function decodeClip(ctx, bytes) {
  // Safari's decodeAudioData predates promises and still returns
  // undefined from the one-argument form, so the callback signature is
  // wrapped rather than assumed away. Cheap insurance for the exact
  // class of device -- an old iPad propped beside a board -- this
  // feature exists to serve.
  return new Promise((resolve, reject) => {
    const maybe = ctx.decodeAudioData(bytes, resolve, reject);
    if (maybe && typeof maybe.then === 'function') maybe.then(resolve, reject);
  });
}

// Fetch and DECODE the entire vocabulary. Both halves matter: a decoded
// AudioBuffer is what makes playback instant, and decoding is the
// expensive half (~64 short MP3s, tens of milliseconds each). Doing it
// here means the throw path never decodes anything.
//
// Every clip is requested at once and the browser's own per-host
// connection limit does the queueing. No progress bar: on a LAN this is
// well under a second, and a spinner for something that fast is noise.
async function loadVoiceSet(voice) {
  const ctx = ensureAudioContext();
  if (!ctx || !voice) return false;
  if (audioBufferVoice === voice && audioBuffers) return true;
  audioLoadState = 'loading';
  audioLoadDetail = 'loading ' + voice + '…';
  renderAudioPanel();
  try {
    const catalog = await fetchAudioCatalog(false);
    const clips = (catalog && catalog.clips) || {};
    const phrases = Object.keys(clips);
    const next = new Map();
    // BOUNDED, and deliberately not Promise.all over all of them. The
    // first version fired all 64 fetches at once, which is wrong twice
    // over on a page that also holds a never-ending MJPEG connection per
    // camera.
    //
    // Wrong once because the browser cannot run them anyway: WebKit
    // allows about six connections per host, three are already spent on
    // previews, so 64 parallel requests contend for what is left and iOS
    // begins failing them outright rather than queueing.
    //
    // Wrong twice because Promise.all rejects on the FIRST failure. A
    // single transient error discarded 63 clips that had downloaded
    // perfectly, left audioBuffers null, and the device stayed mute with
    // the enable-sound banner up -- and tapping that banner just ran the
    // same doomed load again. Measured on an iPad, 2026-09-15.
    //
    // So: a small worker pool, one retry per clip, and a partial set kept
    // rather than discarded. A voice missing two phrases still calls 62
    // darts correctly, and the coverage line reports the gap where the
    // voice is chosen. Silence on every dart is a far worse answer to
    // "one clip failed" than silence on one.
    const CLIP_FETCH_CONCURRENCY = 4;
    const failed = [];
    const queue = phrases.slice();
    const fetchOne = async (phrase) => {
      const url = '/api/audio/clips/' + encodeURIComponent(voice) + '/'
        + encodeURIComponent(clips[phrase]);
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(clips[phrase] + ': HTTP ' + resp.status);
      next.set(phrase, await decodeClip(ctx, await resp.arrayBuffer()));
    };
    const worker = async () => {
      while (queue.length) {
        const phrase = queue.shift();
        try {
          await fetchOne(phrase);
        } catch (err) {
          // One retry. The common failure here is socket contention,
          // which clears as the pool drains rather than being permanent.
          try {
            await fetchOne(phrase);
          } catch (err2) {
            failed.push(phrase);
          }
        }
      }
    };
    await Promise.all(
      Array.from({ length: Math.min(CLIP_FETCH_CONCURRENCY, phrases.length) },
                 () => worker())
    );
    if (!next.size) throw new Error('no clips could be loaded');
    if (failed.length) {
      // Named, not merely counted: "which dart will go unannounced" is the
      // question an operator actually has.
      console.warn('voice set loaded with gaps -- no clip for:', failed);
    }
    audioBuffers = next;
    audioBufferVoice = voice;
    audioLoadState = 'ready';
    audioLoadDetail = failed.length
      ? (next.size + '/' + phrases.length + ' phrases ready in ' + voice
         + ' (' + failed.length + ' failed)')
      : (next.size + ' phrases ready in ' + voice + '.');
    return true;
  } catch (err) {
    console.error('voice set failed to load', err);
    // The PREVIOUS set is deliberately left in place. A failed voice
    // change should leave a screen still speaking in the old voice, not
    // mute it -- the wrong voice is a cosmetic problem, silence is the
    // problem this feature exists to fix.
    audioLoadState = 'error';
    audioLoadDetail = 'could not load ' + voice + ': '
      + String(err && err.message ? err.message : err);
    return false;
  } finally {
    renderAudioPanel();
    reportAudioState();
  }
}

// Turning sound on: unlock the context (this runs inside the toggle's
// own change event, which IS the user gesture), then preload. Order
// matters -- resuming first means the gesture is spent on the thing that
// needs it, and the fetch happens on an already-permitted context.
async function armAudio() {
  const ok = await resumeAudioContext();
  const voice = await audioChosenVoiceLoaded();
  if (voice) await loadVoiceSet(voice);
  renderAudioPanel();
  reportAudioState();
  return ok;
}

// SPEAK ONE PHRASE. Everything expensive already happened; this is a
// buffer, a gain node and a start(). Never throws: it runs off a
// WebSocket message on the throw path, and a broken speaker must not be
// able to take the scoring display down with it.
function speakPhrase(phrase) {
  if (!audioSettings.enabled || !phrase) return;
  const ctx = audioCtx;
  if (!ctx || !audioBuffers) return;
  const buf = audioBuffers.get(phrase);
  if (!buf) {
    // The server said something this voice set has no clip for. Only
    // possible with an incomplete set, which is exactly what the
    // coverage line in the Config tab is there to warn about -- so say
    // it in the console too rather than being mysteriously silent for
    // one particular dart.
    console.warn('no clip for phrase', phrase, '-- set is incomplete');
    return;
  }
  if (ctx.state !== 'running') {
    // Blocked, mid-game. Do not silently drop it: raise the banner and
    // try to resume, so the next dart lands on a working screen.
    audioBlocked = true;
    renderAudioPanel();
    resumeAudioContext().then(() => { renderAudioPanel(); reportAudioState(); });
    return;
  }
  try {
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(audioGain);
    src.start();
    audioSpokenCount += 1;
    audioLastPhrase = phrase;
  } catch (err) {
    console.error('playback failed for', phrase, err);
  }
}

// -- The blocked banner --------------------------------------------------
//
// A <button> across the top of the page, not a toast and not a line in
// the Config tab. It has to be (a) impossible to miss from across a
// room, (b) hittable anywhere, because the person who can reach the TV
// is standing on a chair, and (c) a real gesture target -- which is why
// it is a button element and the click handler does the resume directly.
// DERIVED, NEVER CACHED. `audioBlocked` is a cache, and on 2026-09-15 an
// iPad proved it can be wrong in the one direction that matters: the flag
// initialises to false and is only corrected by a state CHANGE or a
// resume attempt, so a context that is BORN suspended and never
// transitions leaves it stale at false. The rig saw exactly that --
// enabled=True, blocked=False, ctx=suspended, 64/64 clips loaded, nothing
// spoken. No banner was rendered, because the banner keys off the flag,
// so there was nothing on screen to tap. The device looked like it was
// ignoring taps when in truth it had never offered the control.
//
// The context's own state is the truth and is free to read, so read it.
// See docs/DESIGN.md, "A health flag must be able to move both ways" --
// this is that rule applied to the case the rule was written about.
function audioIsBlocked() {
  if (!audioSettings.enabled) return false;
  // No Web Audio at all is a different message, handled in the panel: a
  // banner inviting a tap that cannot help would be a lie.
  if (!audioContextCtor()) return false;
  return !audioCtx || audioCtx.state !== 'running';
}

// Top-bar sound state. Deliberately reads the SAME two facts the panel
// does rather than tracking its own: a second source of truth for "is
// sound on" is one that can disagree with the first, and the disagreement
// would surface as a green badge over a silent room.
function renderAudioIndicator() {
  const el = document.getElementById('audio-indicator');
  if (!el) return;
  if (!audioSettings.enabled) {
    el.className = 'badge unknown';
    el.textContent = 'sound off';
    el.title = 'Spoken calls are off for this screen. Click to turn on.';
    return;
  }
  if (audioBlocked) {
    // NOT green. The user asked for sound and is not getting it, which is
    // worse than off -- off is at least what they chose.
    el.className = 'badge bad';
    el.textContent = 'sound blocked';
    el.title = 'Sound is on, but this browser will not play it yet -- tap the page once to allow audio, or click here to turn sound off.';
    return;
  }
  el.className = 'badge ok';
  el.textContent = 'sound on';
  el.title = 'Spoken calls will play on this screen. Click to turn off.';
}

function renderAudioPanel() {
  // The indicator rides the panel's own render rather than being poked
  // from each call site -- every path that changes audio state already
  // ends here, so anything that forgets to update the badge is a path
  // that also forgot to update the panel, and would be visibly broken.
  renderAudioIndicator();
  const enabledSel = document.getElementById('audio-enabled-select');
  if (enabledSel) enabledSel.value = audioSettings.enabled ? '1' : '0';
  const vol = document.getElementById('audio-volume');
  if (vol) vol.value = String(Math.round(audioSettings.volume * 100));
  const volVal = document.getElementById('audio-volume-value');
  if (volVal) volVal.textContent = Math.round(audioSettings.volume * 100) + '%';
  const test = document.getElementById('btn-audio-test');
  if (test) test.disabled = !(audioBuffers && audioBuffers.size);

  const note = document.getElementById('audio-coverage-note');
  if (note) {
    let text = '—';
    let cls = '';
    if (!audioContextCtor()) {
      text = 'This browser cannot play sound (no Web Audio).';
      cls = 'note-bad';
    } else if (!audioSettings.enabled) {
      text = 'Off on this screen. Every device decides for itself.';
    } else if (audioIsBlocked()) {
      // No banner any more, so this line is the only place that explains
      // it -- and it names BOTH causes, because they are
      // indistinguishable from here and the second one cost an afternoon.
      // A suspended context means no interaction yet; a silent switch
      // means the context will run and still make no sound, and nothing
      // in any API reports the switch.
      text = 'Waiting for a tap anywhere on this page — every browser '
           + 'requires one before it will play sound. On an iPad, also '
           + 'check the silent switch: it mutes this kind of audio while '
           + 'leaving video sound alone.';
      cls = 'note-bad';
    } else if (audioLoadState === 'error') {
      text = audioLoadDetail;
      cls = 'note-bad';
    } else if (audioLoadState === 'loading') {
      text = audioLoadDetail;
    } else if (audioLoadState === 'ready') {
      text = audioLoadDetail;
      cls = 'note-ok';
    }
    note.textContent = text;
    note.className = 'config-field-note' + (cls ? ' ' + cls : '');
  }
}

// -- Voice picker --------------------------------------------------------
//
// Fetched once from /api/audio/voices, which also carries per-set
// coverage and the phrase -> filename map. A set that is missing clips is
// marked in the option text itself: "61/64" beside a name is the only
// warning anyone gets before a particular dart goes unannounced, and it
// belongs where the choice is made rather than three lines below it.
let voicesLoaded = false;

async function refreshVoicePanel(force) {
  const sel = document.getElementById('audio-voice-select');
  const note = document.getElementById('audio-voice-note');
  if (!sel || (voicesLoaded && !force)) return;
  try {
    const v = await fetchAudioCatalog(!!force);
    voicesLoaded = true;
    const sets = (v && v.voices) || [];
    if (!sets.length) {
      sel.innerHTML = '<option value="">none installed</option>';
      sel.disabled = true;
      if (note) note.textContent = 'No voice sets on the rig — see assets/voices/.';
      return;
    }
    sel.disabled = false;
    sel.innerHTML = sets.map((s) => {
      const short = s.complete ? '' : ' — ' + s.present + '/' + s.total;
      const def = s.voice === v.default ? ' (default)' : '';
      return '<option value="' + escapeAttr(s.voice) + '">'
        + escapeAttr(s.voice) + escapeAttr(def + short) + '</option>';
    }).join('');
    sel.value = audioChosenVoice();
    const incomplete = sets.filter((s) => !s.complete);
    if (note) {
      note.textContent = incomplete.length
        ? incomplete.length + ' set(s) are missing clips — those darts stay silent.'
        : sets.length + ' voice set(s) on the rig, all complete.';
      note.className = 'config-field-note' + (incomplete.length ? ' note-bad' : '');
    }
  } catch (err) {
    console.error('voice list failed', err);
  }
}

// -- Reporting this screen's audio state back to the rig ------------------
//
// So that a person holding an iPad can see that the TV has gone quiet.
// Fire-and-forget: this is a courtesy to other dashboards, and a failed
// report must never disturb the screen that is actually working.
function reportAudioState() {
  const body = {
    client_id: debugTabId,
    label: audioDeviceLabel(),
    enabled: audioSettings.enabled,
    // Derived, so the rig's client list cannot disagree with the device's
    // own banner -- they now read the same source.
    blocked: audioIsBlocked(),
    voice: audioBufferVoice || audioChosenVoice(),
    volume: audioSettings.volume,
    ready: audioBuffers ? audioBuffers.size : 0,
    load_state: audioLoadState,
    context_state: audioCtx ? audioCtx.state : 'none',
    spoken: audioSpokenCount,
    last_phrase: audioLastPhrase,
  };
  fetch('/api/audio/clients', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).catch(() => {});
}

function audioClientLine(c) {
  let cls = 'unknown';
  let what = 'off';
  if (c.blocked) { cls = 'bad'; what = 'BLOCKED — needs a tap'; }
  else if (!c.enabled) { cls = 'unknown'; what = 'muted'; }
  else if (c.load_state === 'error') { cls = 'bad'; what = 'clips failed to load'; }
  else if (c.context_state !== 'running') { cls = 'warn'; what = escapeAttr(c.context_state); }
  else if (!c.ready) { cls = 'warn'; what = 'loading…'; }
  else { cls = 'ok'; what = 'speaking — ' + escapeAttr(c.voice || '?'); }
  const me = c.client_id === debugTabId ? ' (this screen)' : '';
  return '<div class="audio-client">'
    + '<span class="badge ' + cls + '">' + what + '</span>'
    + '<span class="audio-client-name">' + escapeAttr(c.label || 'unknown') + me + '</span>'
    + '<span class="audio-client-age">' + escapeAttr(String(c.age_s)) + 's ago</span>'
    + '</div>';
}

async function refreshAudioClients() {
  const box = document.getElementById('audio-clients-list');
  if (!box) return;
  try {
    const body = await (await fetch('/api/audio/clients')).json();
    audioClientRows = body.clients || [];
    box.innerHTML = audioClientRows.length
      ? audioClientRows.map(audioClientLine).join('')
      : '<div class="audio-client-empty">No dashboard has reported yet.</div>';
  } catch (err) {
    console.error('audio client list failed', err);
  }
}

// Kept as the name the rest of the page calls on every state render.
// It renders LOCAL state now -- there is no /api/audio to poll, because
// there is no server-side audio setting left to poll for.
async function refreshAudioPanel() {
  renderAudioPanel();
}

// Turn spoken calls on or off for THIS screen. Shared by the Config
// select and the top-bar badge, so both do exactly the same thing and
// stay in sync (renderAudioPanel, below, re-renders both).
async function setAudioEnabled(on) {
  audioSettings.enabled = on;
  writeAudioSettings();
  const line = logAction('Spoken calls', 'pending', (on ? 'enabling' : 'disabling') + '…');
  if (!on) {
    audioBlocked = false;
    updateActionLine(line, 'ok', 'Spoken calls', 'off on this screen');
    renderAudioPanel();
    reportAudioState();
    return;
  }
  // The caller is a user gesture (a select change or a tap on the badge),
  // which is what unlocks audio for the page. Doing the resume here,
  // synchronously off that gesture, is the whole reason the normal path
  // never sees the banner at all.
  const ok = await armAudio();
  updateActionLine(line, ok ? 'ok' : 'warn', 'Spoken calls',
    ok ? 'on for this screen only' : 'on — tap anywhere on the page to start sound');
}

document.getElementById('audio-enabled-select').addEventListener('change', (ev) => {
  setAudioEnabled(ev.target.value === '1');
});

// The top-bar sound badge is a shortcut for the same toggle: click it to
// flip sound on/off without opening Config. The click is itself the user
// gesture that unlocks audio, exactly like the select's change event.
const _audioBadge = document.getElementById('audio-indicator');
if (_audioBadge) {
  const toggleAudio = () => setAudioEnabled(!audioSettings.enabled);
  _audioBadge.addEventListener('click', toggleAudio);
  _audioBadge.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); toggleAudio(); }
  });
}

document.getElementById('audio-voice-select').addEventListener('change', async (ev) => {
  const voice = ev.target.value;
  audioSettings.voice = voice;
  writeAudioSettings();
  const line = logAction('Voice', 'pending', 'loading ' + voice + '…');
  // A voice change refetches and re-decodes the whole set. ~600 KB and
  // well under a second on a LAN, while the person is sitting in a
  // settings panel -- the right place to spend it.
  const ok = audioSettings.enabled ? await loadVoiceSet(voice) : true;
  updateActionLine(line, ok ? 'ok' : 'bad', 'Voice',
    ok ? voice + ' on this screen' : (audioLoadDetail || 'failed to load'));
});

document.getElementById('audio-volume').addEventListener('input', (ev) => {
  const pct = Math.max(0, Math.min(100, Number(ev.target.value) || 0));
  audioSettings.volume = pct / 100;
  if (audioGain) audioGain.gain.value = audioSettings.volume;
  const volVal = document.getElementById('audio-volume-value');
  if (volVal) volVal.textContent = pct + '%';
});

// Written on release, not on every drag frame: an input event fires per
// pixel of slider travel, and localStorage writes are synchronous.
document.getElementById('audio-volume').addEventListener('change', () => {
  writeAudioSettings();
  reportAudioState();
});

document.getElementById('btn-copy-diagnostics').onclick = copyDiagnostics;

// -- Maintenance: restart, and restart onto new code --------------------
//
// THE ONLY TWO CONTROLS ON THIS PAGE THAT TAKE THE RIG AWAY, so both are
// gated on a confirm() that says which code comes back. Until 2026-09-17
// the launcher pulled on EVERY relaunch, so "restart" and "update" were
// the same word and nobody could ask for one without the other -- a rig
// that crashed mid-match came back on whatever was on main. Now the
// launcher pulls only when data/config.json says to, and these buttons
// are how you say so (POST /api/restart {"update": true}).
//
// CAPTURE IS NOT STARTED AFTERWARDS, deliberately -- see the Maintenance
// markup in index.html for why. The page reloads into a stopped rig.

//: How long to keep asking before giving up on the rig coming back. The
//: launcher sleeps 3s, then may run `git pull` and `pip install
//: -r requirements.txt` before the process even starts, and on the
//: Windows rig that install has taken well over a minute on a cold
//: wheel cache. Four minutes is long enough to cover that and short
//: enough that a rig which is genuinely not coming back says so while
//: someone is still standing there.
const RESTART_POLL_TIMEOUT_MS = 240000;
const RESTART_POLL_INTERVAL_MS = 1000;

// Drawn from the CONFIG DOCUMENT, which carries both launcher flags as
// ordinary keys -- the same document the checkbox beside these buttons
// writes. GET /api/restart reports the same two flags and still exists
// (FlightDeck reads it), but this page deliberately does not ask twice:
// two sources for one fact is how a checkbox and the note under it end up
// disagreeing on screen.
//
// "Update and restart" is HIDDEN rather than disabled on an
// always_update rig: there is nothing to enable, and the plain Restart
// button already pulls there -- which the note says, because a button
// labelled "Restart" that also updates is exactly the surprise this
// feature exists to remove.
function renderMaintenanceNote() {
  const note = document.getElementById('restart-note');
  const updateBtn = document.getElementById('btn-update-restart');
  if (!note || !updateBtn) return;
  const always = configValue('always_update', null);
  const queued = configValue('update_on_next_restart', null);
  if (always === null) return;   // no document yet; leave "Checking..."
  updateBtn.hidden = !!always;
  if (always) {
    note.textContent = 'This rig follows main: Restart also pulls the latest code.';
  } else if (queued) {
    note.textContent = 'An update is already queued — the next restart will pull first.';
  } else {
    note.textContent = 'Restart comes back on the code this machine already has.';
  }
}

// THE PID IS THE WHOLE TRICK. /api/restart schedules its SIGTERM half a
// second out, so for that half second the process about to die still
// answers /api/health perfectly -- poll naively and the page reloads
// into a server that is mid-shutdown, which looks exactly like the
// restart failing. So poll until health answers with a DIFFERENT pid
// than the one the POST reported. A rig whose /api/health has no pid at
// all (an older build, which is entirely possible on the machine you are
// updating) falls back to "answered at all", which is what this did
// before the field existed.
async function waitForRigToComeBack(oldPid) {
  const deadline = Date.now() + RESTART_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, RESTART_POLL_INTERVAL_MS));
    try {
      const resp = await fetch('/api/health', {cache: 'no-store'});
      if (!resp.ok) continue;
      const body = await resp.json();
      if (body.pid === undefined || body.pid === null) return true;
      if (body.pid !== oldPid) return true;
    } catch (err) {
      // Expected, repeatedly: the rig is down. Keep asking.
    }
  }
  return false;
}

async function restartRig(update) {
  const label = update ? 'Update and restart' : 'Restart';
  const ok = window.confirm(
    update
      ? 'Pull the latest code and restart this rig?\n\n'
        + 'Scoring stops now. The launcher pulls the checked-out branch, '
        + 'then starts the product again — this page reloads when it '
        + 'answers. Capture is NOT started for you.'
      : 'Restart this rig?\n\n'
        + 'Scoring stops now. The same code starts again — this page '
        + 'reloads when it answers. Capture is NOT started for you.'
  );
  if (!ok) return;
  const restartBtn = document.getElementById('btn-restart');
  const updateBtn = document.getElementById('btn-update-restart');
  const note = document.getElementById('restart-note');
  if (restartBtn) restartBtn.disabled = true;
  if (updateBtn) updateBtn.disabled = true;
  const line = logAction(label, 'pending', 'restarting…');
  if (note) note.textContent = 'Restarting…';
  let body;
  try {
    // NO BODY AT ALL for a plain restart -- byte for byte the request
    // FlightDeck has always sent, and the request this route has always
    // accepted. An empty {} would behave identically today; sending one
    // anyway would make the plain path depend on body parsing that the
    // old path never touched.
    const resp = update
      ? await fetch('/api/restart', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({update: true}),
        })
      : await fetch('/api/restart', {method: 'POST'});
    body = await resp.json();
  } catch (err) {
    console.error('restart request failed', err);
    body = null;
  }
  // A REFUSAL IS NOT A RESTART. The endpoint declines when it could not
  // record the update request, precisely so the rig does not come back
  // looking fine on the old code -- so put the buttons back rather than
  // sitting on a "Restarting..." that will never resolve.
  if (!body || !body.ok) {
    updateActionLine(line, 'bad', label, (body && body.reason) || 'request failed');
    if (restartBtn) restartBtn.disabled = false;
    if (updateBtn) updateBtn.disabled = false;
    refreshConfig();
    return;
  }
  updateActionLine(line, 'pending', label, 'waiting for the rig to come back…');
  const back = await waitForRigToComeBack(body.pid);
  if (back) {
    updateActionLine(line, 'ok', label, 'back up — reloading');
    if (note) note.textContent = 'Back up — reloading…';
    location.reload();
    return;
  }
  updateActionLine(line, 'bad', label,
    'no answer after ' + Math.round(RESTART_POLL_TIMEOUT_MS / 1000) + 's');
  if (note) {
    note.textContent = 'The rig has not answered yet. It may still be installing '
      + 'dependencies — reload this page to check.';
  }
  if (restartBtn) restartBtn.disabled = false;
  if (updateBtn) updateBtn.disabled = false;
}

document.getElementById('btn-restart').onclick = () => restartRig(false);
document.getElementById('btn-update-restart').onclick = () => restartRig(true);

document.getElementById('btn-audio-test').onclick = async () => {
  const line = logAction('Test sound', 'pending', 'playing…');
  // Plays HERE, in this browser, through exactly the path a real dart
  // call takes -- which makes it an honest test of the thing that
  // matters: whether this screen, at this volume, can be heard from the
  // oche. The old Test button played on the rig, which told you nothing
  // about the device you were holding.
  const ok = await resumeAudioContext();
  if (!ok) {
    updateActionLine(line, 'bad', 'Test sound',
      'tap anywhere on the page first — and on an iPad check the silent switch');
    renderAudioPanel();
    return;
  }
  const wasEnabled = audioSettings.enabled;
  audioSettings.enabled = true;
  speakPhrase('treble 20');
  audioSettings.enabled = wasEnabled;
  updateActionLine(line, 'ok', 'Test sound', 'played on this screen');
};

// THE SILENT SWITCH, and why a dartboard needs a silent audio file to
// beat it. iOS puts every page in an audio SESSION CATEGORY. Media
// elements (<audio>, <video>) get "playback", which ignores the physical
// silent switch -- which is why every other site on the iPad kept making
// noise. The Web Audio API gets "ambient", which obeys it. We use Web
// Audio deliberately, because pre-decoded buffers make a dart call a
// memory lookup instead of a fetch, and that choice is what bought us a
// switch nobody remembered flicking.
//
// Playing one real media element promotes the whole page to "playback",
// and the AudioContext then follows it. The clip is 0.05s of encoded
// silence, looped so the session does not lapse back, and held in a
// variable so nothing garbage-collects it mid-session.
//
// Not guaranteed forever: this behaviour has moved across iOS releases.
// It is an attempt, not a promise, which is why the Config panel still
// names the silent switch rather than claiming the problem is impossible.
const SILENT_WAV = 'data:audio/wav;base64,UklGRkQDAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YSADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==';
let audioSessionEl = null;

function unlockAudioSession() {
  if (audioSessionEl) return;
  try {
    const el = document.createElement('audio');
    el.src = SILENT_WAV;
    el.loop = true;
    // playsinline keeps iOS from treating it as fullscreen-able video, and
    // a tiny non-zero volume because some iOS builds decline to honour an
    // element muted to exactly zero. The audio itself is silence, so this
    // is inaudible either way.
    el.setAttribute('playsinline', '');
    el.volume = 0.001;
    const played = el.play();
    if (played && played.catch) played.catch((err) => {
      console.debug('silent session unlock was refused', err);
    });
    audioSessionEl = el;
  } catch (err) {
    console.debug('silent session unlock failed', err);
  }
}

// ARM ON ANY INTERACTION, SILENTLY. There is no banner any more: a page
// that shouts at you to tap it, on every load, for a policy the browser
// applies to every site, is noise -- and it became furniture within an
// hour of shipping. The assumption instead is that a dashboard someone is
// using gets touched at least once, and the first touch anywhere arms
// everything: session category, context resume, and the clip set.
//
// Deliberately still attached after success rather than `once`: an early
// tap can land before the audio toggle is even on, and a context can be
// suspended again later by a screen lock. The handler costs a property
// read when there is nothing to do.
['pointerdown', 'keydown', 'touchend'].forEach((evt) => {
  document.addEventListener(evt, () => {
    if (!audioSettings.enabled) return;
    unlockAudioSession();
    if (audioCtx && audioCtx.state === 'running' && audioBuffers) return;
    resumeAudioContext().then(() => {
      if (!audioBuffers) return audioChosenVoiceLoaded().then(loadVoiceSet);
    }).then(() => { renderAudioPanel(); reportAudioState(); });
  }, true);
});

// A backgrounded tab gets its context suspended on iOS and on a sleeping
// laptop. Coming back is a chance to fix it without anyone noticing.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible' || !audioSettings.enabled) return;
  resumeAudioContext().then(() => { renderAudioPanel(); reportAudioState(); });
});

// KIOSK SOUND SWITCH: `?sound=on` / `?sound=off` in this page's URL.
//
// Sound is per screen by design -- the server says WHAT to call, each
// browser decides WHETHER to -- and the only way a screen opts in is its
// own toggle. That leaves a kiosk (a TV driven by a browser nobody can
// touch) with no way to opt in at all. The URL is the one thing whoever
// sets up a kiosk does control, so it may state the choice there: e.g.
// `http://host:8420/?sound=on#engines`. It applies on every load and is
// saved exactly as the toggle would save it, so the Config panel and the
// "screens watching" list show the truth. A later manual toggle still
// works for that session; the URL wins again on the next load, which is
// what a kiosk's URL is for.
//
// Deliberately NOT a remote control from another screen: each screen
// stays in charge of itself. On a normal desktop the browser still wants
// one gesture before it will play; a kiosk launched with Chromium's
// --autoplay-policy=no-user-gesture-required plays straight away.
//
// Returns true/false for a recognised value, null when absent or garbled
// -- garbled is ignored (and said so in the console), never guessed at.
function soundFromUrl(search) {
  let raw;
  try { raw = new URLSearchParams(search || '').get('sound'); } catch (err) { return null; }
  if (raw === null) return null;
  const v = raw.trim().toLowerCase();
  if (['on', '1', 'true', 'yes'].includes(v)) return true;
  if (['off', '0', 'false', 'no'].includes(v)) return false;
  console.warn('?sound=' + raw + ' is not on/off -- ignored');
  return null;
}
{
  const fromUrl = soundFromUrl(window.location.search);
  if (fromUrl !== null && fromUrl !== audioSettings.enabled) {
    audioSettings.enabled = fromUrl;
    writeAudioSettings();
  }
}

// Arm on load if this device already said yes. No gesture has happened
// yet, so this will usually end with audioBlocked true and the banner
// up -- which is the entire point: a TV that reloaded overnight shows a
// "tap to enable sound" bar instead of being mysteriously mute. The
// clips are preloaded regardless, so the tap is instant rather than the
// start of a download.
if (audioSettings.enabled) {
  armAudio();
}
renderAudioPanel();
reportAudioState();
refreshAudioClients();
audioReportTimer = setInterval(reportAudioState, AUDIO_REPORT_INTERVAL_MS);
// Only while the Config tab (where the list lives) is on screen.
setInterval(() => { if (tabShowing('config')) refreshAudioClients(); }, AUDIO_CLIENTS_REFRESH_MS);

// -- Autodarts comparison ------------------------------------------------
// Restored 2026-09-11 after a text substitution for the camera selector
// deleted this whole block: it sat between the two anchors that edit cut
// between. Removing a balanced block leaves syntactically valid
// JavaScript, so nothing failed -- the controls simply stopped doing
// anything, silently.
//
// Enabled and CONNECTED are reported separately. "On but not connected"
// is the state worth seeing: that is the case where nothing is being
// compared, and collapsing the two would hide exactly what an
// operator is trying to diagnose.
function renderAdConfig() {
  const note = document.getElementById('ad-config-note');
  const box = document.getElementById('ad-enabled-toggle');
  const url = document.getElementById('ad-base-url');
  if (!box || !url) return;
  const ad = configRuntime('ad');
  if (!lastConfig || !ad || ad.available === false) {
    box.disabled = true; url.disabled = true;
    if (note) {
      // Reset the class too: a stale colour from the last good poll would
      // otherwise outlive the state that earned it.
      note.className = 'config-field-note';
      note.textContent = (ad && ad.reason) || 'Autodarts is not wired in this process.';
    }
    return;
  }
  // `ad_enabled`/`ad_base_url` come off the document, which carries the
  // LIVE listener's own values whenever one is wired -- so this shows what
  // the process is doing, not merely what the file last said.
  const payload = {enabled: lastConfig.ad_enabled, base_url: lastConfig.ad_base_url,
                   connected: ad.connected};
  box.disabled = false; url.disabled = false;
  // Never stomp a control the operator is currently using.
  if (document.activeElement !== box) box.value = payload.enabled ? 'yes' : 'no';
  if (document.activeElement !== url) url.value = payload.base_url || '';
  if (note) {
    // This span IS the field's note -- there is no longer a static one
    // beside it, so the off-state text has to carry the virtual-camera
    // consequence that static note used to be the only place to say.
    note.className = 'config-field-note' + (!payload.enabled ? ''
      : (payload.connected ? ' note-ok' : ' note-bad'));
    note.textContent = !payload.enabled
      ? 'Off — Autodarts is never contacted, and on Windows the virtual cameras are removed.'
      : (payload.connected
          ? 'On and connected to ' + payload.base_url
          : 'On but NOT connected to ' + payload.base_url + ' — no comparison is being recorded.');
  }
}

// How long after an AD change to re-read the document once. See
// submitAdConfig().
const AD_CONFIG_RECHECK_MS = 2000;

function submitAdConfig(patch, label) {
  // A single re-read shortly after the change, as a SAFETY NET only. The
  // real mechanism is the AD_CONNECTION push (applyAdConnection()); this
  // covers the one case it cannot -- this tab's events socket happening
  // to be down while the listener connects, so the push never reaches it.
  // Goes through the same stamp guard as everything else, so arriving
  // late can never undo a newer push.
  setTimeout(() => refreshConfig(), AD_CONFIG_RECHECK_MS);
  return patchConfig(patch, label, (body) => {
    // The guarded copy, not body.runtime.ad: by the time this reply is
    // processed an AD_CONNECTION may already have said the socket is up,
    // and the log line should not contradict the note beside it.
    const ad = configRuntime('ad') || {};
    let detail = body.config.ad_enabled
      ? (ad.connected ? 'on, connected to ' + body.config.ad_base_url
                      : 'on, not connected yet to ' + body.config.ad_base_url)
      : 'off — Autodarts will not be contacted';
    // A registry write that failed is reported HERE or nowhere an
    // operator will look -- the log is on the rig, and this is the
    // control that caused it. `applicable` is what keeps this quiet on
    // macOS and Linux, where not-registering is the correct outcome
    // rather than a failure.
    const vc = body.virtual_cameras;
    if (vc && vc.applicable && !vc.ok) {
      detail += ' — virtual cameras NOT '
              + (vc.action === 'unregister' ? 'unregistered' : 'registered')
              + ': ' + (vc.reason || 'regsvr32 failed');
    }
    return detail;
  });
}

// Both controls save on change; there is no Save button. Switching the
// comparison off should take effect NOW, and a URL that needs a second click
// to commit only creates doubt about whether it took.
document.getElementById('ad-enabled-toggle').addEventListener('change', (ev) => {
  submitAdConfig({ad_enabled: ev.target.value === 'yes'}, 'Compare against Autodarts');
});

// 'change' on a text input fires on blur or Enter, never per keystroke --
// saving mid-URL would fire a request per character and briefly point the
// listener at a nonsense host.
document.getElementById('ad-base-url').addEventListener('change', (ev) => {
  const value = (ev.target.value || '').trim();
  if (!value) { refreshConfig(); return; }   // empty: restore what is live
  submitAdConfig({ad_base_url: value}, 'Autodarts URL');
});

// -- Camera assignment (2026-09-11: moved onto the preview cards) --------
// The selector sits on each camera card rather than in its own section.
// Choosing a device and checking what it shows were previously two places
// on one page, and the only question this control answers -- "is this slot
// pointed at the right camera" -- can only be answered while looking at
// the picture.
//
// Changing one SAVES AND APPLIES immediately. There is no Save button: the
// previews are the confirmation, and a separate save step for a control
// whose result you are already looking at is a step that only adds doubt.
const CAMERA_DEVICE_CHOICES = 10;   // fallback 0..9 when the platform cannot enumerate

// One option list shared by every slot's select, built from what the
// platform actually enumerated (see opendarts/live/camera_names.py)
// instead of a fixed ten anonymous numbers.
//
// No uncertainty marker on the name. A non-authoritative one (macOS)
// used to render as "dev 2 — FaceTime HD ?", which asked the operator to
// second-guess a name they cannot verify from here; the device NUMBER
// beside it is exact either way, and the preview above the select is the
// real confirmation. The payload still carries names_authoritative for
// anything that needs to reason about it.
// Option values are strings, and a URL must not collide with a device
// index, so remote options carry a prefix no device value can produce.
const URL_OPTION_PREFIX = 'url:';
const URL_OPTION_NEW = '__add_url__';

function cameraDeviceOptionsHtml(payload) {
  const names = (payload && payload.device_names) || [];
  const live = (payload && payload.devices) || [];
  let total = (payload && Number.isInteger(payload.device_count) && payload.device_count > 0)
    ? payload.device_count : CAMERA_DEVICE_CHOICES;
  // A slot can be assigned past the enumerated range (config written on
  // another machine, or a camera since unplugged). The option must still
  // exist, or the select silently renders blank while the slot keeps
  // reading that device.
  live.forEach((d) => { if (Number.isInteger(d) && d >= total) total = d + 1; });
  // This rig's own virtual cameras are left out: they carry what this
  // rig publishes, so reading one back is a loop. Still listed while a
  // slot is actually set to one, or the select would render blank.
  const own = new Set((payload && payload.own_virtual_devices) || []);
  let html = '';
  for (let d = 0; d < total; d++) {
    if (own.has(d) && !live.includes(d)) continue;
    let label = 'dev ' + d;
    if (names[d]) label += ' — ' + names[d];
    html += '<option value="' + d + '">' + escapeAttr(label) + '</option>';
  }
  // Every slot's CURRENT url is offered in every dropdown, not just its
  // own: the option has to exist before a select can hold that value, and
  // re-rendering happens on a poll that must not clear a slot mid-edit.
  const urls = (payload && payload.urls) || [];
  const seen = new Set();
  urls.forEach((u) => {
    if (!u || seen.has(u)) return;
    seen.add(u);
    html += '<option value="' + escapeAttr(URL_OPTION_PREFIX + u) + '">'
         + escapeAttr(prettyStreamLabel(u)) + '</option>';
  });
  html += '<option value="' + URL_OPTION_NEW + '">… read from a URL</option>';
  return html;
}

// A stream URL is far too long for a dropdown, and the useful part is
// "which machine, which camera" -- not the path, the scheme or the query.
// http://192.0.2.10:8420/api/cameras/2/stream.mjpg?full=1
//   becomes   stream  192.0.2.10 cam2
function prettyStreamLabel(url) {
  try {
    const u = new URL(url);
    // Split rather than a regex: every backslash in this file passes
    // through an f-string first, so a regex character class reaches
    // Python as an invalid escape long before JavaScript ever sees it.
    const after = u.pathname.split('/cameras/')[1];
    const cam = after ? after.split('/')[0] : null;
    const where = u.hostname + (u.port && u.port !== '80' ? ':' + u.port : '');
    return 'stream — ' + where + (cam ? '  cam' + cam : '');
  } catch (e) {
    // Not parseable: show it truncated rather than not at all, so a
    // hand-typed oddity is still recognisable in the list.
    return 'stream — ' + (url.length > 38 ? url.slice(0, 35) + '…' : url);
  }
}

function setCameraDeviceOptions(sel, html) {
  // Rebuild only when the option set actually changed; rewriting it on
  // every poll would fight the operator mid-selection.
  if (sel.dataset.optionsHtml !== html) {
    sel.innerHTML = html;
    sel.dataset.optionsHtml = html;
  }
}

function renderCameraDevices(payload) {
  // Populate the options FIRST, whatever the payload says. Returning
  // early on a failed fetch left every select with zero options -- a
  // dropdown you cannot open, which reads as "the control is broken"
  // rather than "the server could not answer".
  const optionsHtml = cameraDeviceOptionsHtml(payload);
  const selects = Array.from(document.querySelectorAll('.cam-device-select'));
  selects.forEach((sel) => setCameraDeviceOptions(sel, optionsHtml));

  if (!payload || !payload.ok) {
    // Say why, on the control itself, and stop it pretending to work.
    const why = (payload && payload.reason) || 'could not read the camera assignment';
    selects.forEach((sel) => { sel.disabled = true; sel.title = why; });
    return;
  }
  selects.forEach((sel) => { sel.disabled = false; });

  const live = payload.devices || [];
  const names = payload.device_names || [];
  const liveUrls = payload.urls || [];
  const savedUrls = payload.saved_urls || null;
  live.forEach((device, slot) => {
    const sel = document.getElementById('cam-device-' + slot);
    if (!sel) return;
    const url = liveUrls[slot] || null;
    // `urls` is what this PROCESS reads; `saved_urls` is what the file
    // says. They differ exactly while a change is saved and not yet
    // restarted into, and the select shows the SAVED one -- otherwise
    // picking a stream would visibly snap back to the device on the next
    // poll, which reads as the setting being rejected.
    const pending = savedUrls ? (savedUrls[slot] || null) : url;
    const shown = pending || url;
    if (document.activeElement !== sel) {
      sel.value = shown ? (URL_OPTION_PREFIX + shown) : String(device);
    }
    sel.dataset.lastValue = sel.value;
    if (!shown) sel.dataset.lastDevice = String(device);
    sel.classList.toggle('is-remote', !!shown);
    if (url) {
      sel.title = 'Camera ' + slot + ' is reading frames from ' + url;
    } else if (shown) {
      sel.title = 'Camera ' + slot + ' will read from ' + shown +
        ' after a restart. It is still on hardware device ' + device + '.';
    } else {
      const name = names[device] || '';
      sel.title = 'Slot ' + slot + ' is reading hardware device ' + device +
        (name ? ' (' + name + ')' : '');
    }
  });
}

// renderCameraDevices() still takes the shape the retired
// GET /api/camera-devices returned -- the option builder, the per-slot
// titles and the saved-vs-live distinction are all unchanged. This is the
// one adapter: the same fields now arrive under `runtime.cameras`, where
// "is there a hub at all" is spelled `available` rather than `ok`.
function cameraDevicePayload() {
  const cams = configRuntime('cameras');
  if (!cams) return null;
  return Object.assign({}, cams, {ok: cams.available !== false});
}

function slotSelectsInOrder() {
  return Array.from(document.querySelectorAll('.cam-device-select'))
    .sort((a, b) => Number(a.dataset.slot) - Number(b.dataset.slot));
}

// A slot is EITHER a device index or a URL. The device array still has to
// carry a number for every slot -- the server validates its shape, and a
// remote slot's device is simply unused -- so a remote slot keeps whatever
// index it had rather than inventing a placeholder that would become the
// live device the moment someone switched it back to local.
function currentSlotAssignment() {
  const devices = [];
  const urls = [];
  slotSelectsInOrder().forEach((el, slot) => {
    const v = String(el.value || '');
    if (v.startsWith(URL_OPTION_PREFIX)) {
      urls.push(v.slice(URL_OPTION_PREFIX.length));
      devices.push(Number(el.dataset.lastDevice || slot));
    } else {
      urls.push(null);
      devices.push(Number(v));
      el.dataset.lastDevice = v;
    }
  });
  return {devices: devices, urls: urls};
}

async function applyCameraDevices() {
  const assignment = currentSlotAssignment();
  const devices = assignment.devices;
  const urls = assignment.urls;
  // Only LOCAL slots can collide -- two slots reading the same stream is
  // a legitimate thing to want (two engines on one camera), and two slots
  // whose unused device index happens to match is not a conflict at all.
  const localDevices = devices.filter((d, i) => !urls[i]);
  if (new Set(localDevices).size !== localDevices.length) {
    updateActionLine(logAction('Camera assignment', 'pending', 'checking…'),
      'bad', 'Camera assignment', 'that device is already assigned to another slot');
    refreshConfig();
    return;
  }
  await patchConfig({camera_devices: devices, camera_urls: urls},
    'Camera assignment', (body, outcome) =>
      outcome.applied.length
        ? 'saved and applied' + (body.restarted_capture ? ' (capture restarted)' : '')
        : 'saved, NOT applied — ' + (outcome.note || ''));
  if (typeof refreshCameras === 'function') refreshCameras();
}

// -- "read from a URL" -------------------------------------------------
// The dropdown offers devices AND streams, so choosing a stream needs
// somewhere to type one. A modal rather than an inline field: the select
// is 200px wide inside a camera card, and a URL is not.

let camUrlSlot = null;       // which slot the modal is editing
let camUrlPrevValue = null;  // what to restore if it is cancelled

function openCamUrlModal(slot, sel) {
  camUrlSlot = slot;
  camUrlPrevValue = sel.dataset.lastValue || String(sel.dataset.lastDevice || slot);
  document.getElementById('cam-url-modal-sub').textContent =
    'Camera ' + slot + ' will read frames from this address instead of local hardware.';
  const input = document.getElementById('cam-url-input');
  // Prefill with this slot's current stream if it already has one, so
  // editing an address does not mean retyping it.
  const cur = String(sel.value || '');
  input.value = cur.startsWith(URL_OPTION_PREFIX) ? cur.slice(URL_OPTION_PREFIX.length) : '';
  document.getElementById('cam-url-error').textContent = '';
  document.getElementById('cam-url-modal').hidden = false;
  input.focus();
  input.select();
}

function closeCamUrlModal(restore) {
  if (restore && camUrlSlot !== null) {
    const sel = document.getElementById('cam-device-' + camUrlSlot);
    // Put the select back where it was, or the dropdown is left reading
    // "... read from a URL" for a slot that is still on local hardware.
    if (sel && camUrlPrevValue !== null) sel.value = camUrlPrevValue;
  }
  document.getElementById('cam-url-modal').hidden = true;
  camUrlSlot = null;
  camUrlPrevValue = null;
}

// A bare host is what someone actually pastes. Expand it to that slot's
// own camera on the publisher, since "point camera 2 at that rig" almost
// always means that rig's camera 2.
function normaliseStreamUrl(raw, slot) {
  let v = String(raw || '').trim();
  if (!v) return null;
  // No regex literals anywhere in this file: every backslash passes
  // through an f-string first, so a `/.../` pattern reaches Python as an
  // invalid escape long before JavaScript sees it.
  const lower = v.toLowerCase();
  if (!(lower.startsWith('http://') || lower.startsWith('https://'))) v = 'http://' + v;
  if (v.indexOf('/api/cameras/') === -1) {
    while (v.endsWith('/')) v = v.slice(0, -1);
    v = v + '/api/cameras/' + slot + '/stream.mjpg?full=1';
  }
  return v;
}

document.getElementById('cam-url-cancel').onclick = () => closeCamUrlModal(true);

// True when the address names a whole SERVER rather than one stream --
// the only case where "all 3 cameras" means anything, since a full stream
// URL already picked its camera.
function isBareServerAddress(raw) {
  const v = String(raw || '').trim();
  return v !== '' && v.indexOf('/api/cameras/') === -1;
}

function setSlotToStream(slot, url) {
  const sel = document.getElementById('cam-device-' + slot);
  if (!sel) return;
  const value = URL_OPTION_PREFIX + url;
  // The option must exist before the select can hold it -- the poll will
  // rebuild the list from the server's answer a moment later.
  if (!Array.from(sel.options).some((o) => o.value === value)) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = prettyStreamLabel(url);
    sel.insertBefore(opt, sel.options[sel.options.length - 1]);
  }
  sel.value = value;
  sel.dataset.lastValue = value;
}

document.getElementById('cam-url-save').onclick = () => {
  const slot = camUrlSlot;
  if (slot === null) return closeCamUrlModal(false);
  const raw = document.getElementById('cam-url-input').value;
  const url = normaliseStreamUrl(raw, slot);
  if (!url) {
    document.getElementById('cam-url-error').textContent = 'Enter an address.';
    return;
  }
  const all = document.getElementById('cam-url-all');
  // ONE APPLY, NOT THREE. Each slot is set locally first and
  // applyCameraDevices() posts the whole assignment once -- three separate
  // posts would each stop and restart capture, and the first two would
  // briefly run a half-applied mix.
  if (all && all.checked && isBareServerAddress(raw)) {
    // Slot count read off the DOM rather than a constant: the cards are
    // what actually exist, and a second source of truth for "how many
    // cameras" is one that can drift from them.
    const slots = document.querySelectorAll('.cam-device-select');
    for (let i = 0; i < slots.length; i++) {
      setSlotToStream(Number(slots[i].dataset.slot), normaliseStreamUrl(raw, Number(slots[i].dataset.slot)));
    }
  } else {
    setSlotToStream(slot, url);
  }
  closeCamUrlModal(false);
  applyCameraDevices();
};

// The checkbox is meaningless once the address names a single stream, so
// it hides rather than sitting there lying about what Save will do.
document.getElementById('cam-url-input').addEventListener('input', (ev) => {
  const row = document.getElementById('cam-url-all-row');
  if (row) row.hidden = !isBareServerAddress(ev.target.value);
});

document.getElementById('cam-url-input').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') document.getElementById('cam-url-save').click();
  if (ev.key === 'Escape') closeCamUrlModal(true);
});

// Delegated: the cards are re-rendered, so a listener bound to each select
// would be lost on the next refresh.
document.addEventListener('change', (ev) => {
  if (ev.target && ev.target.classList && ev.target.classList.contains('cam-device-select')) {
    if (ev.target.value === URL_OPTION_NEW) {
      openCamUrlModal(Number(ev.target.dataset.slot), ev.target);
      return; // nothing to save until the modal produces an address
    }
    ev.target.dataset.lastValue = ev.target.value;
    applyCameraDevices();
  }
});

// ONE first fetch for the whole Config tab. Every other control on it used
// to kick off its own on load.
refreshConfig();

document.getElementById('detection-time-select').addEventListener('change', (ev) => {
  const frames = Number(ev.target.value);
  // The nested literal is hoisted rather than written inline. A doubled
  // closing brace in this file is what an unrendered f-string brace left
  // behind for years, and a test still watches for one.
  const settings = {dart_stable_frames: frames};
  patchConfig({lifecycle_settings: settings}, 'Detection time',
    (body, outcome) => outcome.note || (frames + ' frames'));
});

// -- Maintenance: pull the latest code on the way back up ----------------
// Bound to the config document's `always_update`, which the LAUNCHER
// (run.sh / run.ps1) reads on each loop -- nothing in this process acts on
// it, which is why the note says what happens at the next restart rather
// than claiming anything about now.
function renderAlwaysUpdate() {
  const box = document.getElementById('always-update-toggle');
  const on = configValue('always_update', null);
  if (!box || on === null) return;
  if (document.activeElement === box) return;
  box.checked = !!on;
}

const alwaysUpdateBox = document.getElementById('always-update-toggle');
if (alwaysUpdateBox) {
  alwaysUpdateBox.addEventListener('change', (ev) => {
    const on = !!ev.target.checked;
    patchConfig({always_update: on}, 'Always update', () =>
      on ? 'on — every restart pulls the latest code first'
         : 'off — a restart relaunches the code already on this rig');
  });
}

// The Autodarts board-status indicator light (2026-08-14).
// `status` is one of the standardized
// opendarts.live.board_status.BOARD_STATUS_* names the server already
// classified into -- 'stopped'/'ready'/'takeout'/'unknown' -- never the
// source's own status spelling. `which` is always 'ad', matching the
// #board-status-dot-ad/#board-status-item-ad elements above -- the
// second external board's light was removed once it became a real
// registry engine ("only AD has status"). Colors set via the CSS custom
// property NAMES themselves (not literal hex), so red/green/yellow are
// the exact same tokens the header status-pill's own dot uses
// (.status-pill .dot, --ok-fg et al) -- one color meaning across the
// whole page, not a second palette.
// `unknown` deliberately maps to --bad-fg (red), NOT --unk-fg (grey) --
// 2026-08-14: the light uses only red, yellow and green. A 4th visual
// color for "no signal yet" would be ambiguous; red (fail-safe -- "not confirmed ready",
// same direction a real traffic light defaults to on power-up) is the
// honest 3-color choice, not a fabricated status -- the tooltip below
// still says "Unknown" on hover, this only changes which color an
// unresolved status borrows, never what it CLAIMS.
const BOARD_STATUS_COLOR_VAR = {
  stopped: 'var(--bad-fg)',
  ready: 'var(--ok-fg)',
  takeout: 'var(--warn-fg)',
  unknown: 'var(--bad-fg)',
};
const BOARD_STATUS_LABEL = {
  stopped: 'Stopped',
  ready: 'Ready',
  takeout: 'Takeout',
  unknown: 'Unknown',
};
// Takeout lasting longer than this means AD never saw the darts come out.
// A real takeout is a few seconds; 30s is well past any of them, and short
// enough that an operator is told during the session rather than after it.
const AD_TAKEOUT_STUCK_SEC = 30;

function renderBoardStatus(which, status, ageSec) {
  const dot = document.getElementById('board-status-dot-' + which);
  const item = document.getElementById('board-status-item-' + which);
  if (!dot || !item) return;
  const key = BOARD_STATUS_COLOR_VAR.hasOwnProperty(status) ? status : 'unknown';
  // STUCK IN TAKEOUT. AD enters takeout on a visit's last throw and leaves
  // it when it sees the darts removed. If it never does, it sits there
  // reporting no further throws while its buffer keeps serving the
  // finished visit -- on 2026-09-21 that silently attached 20-55s-old
  // answers to six live darts. The matcher now refuses those, so the
  // failure is no longer silent in the DATA; this makes it not silent on
  // the SCREEN either, while the darts are still being thrown.
  const stuck = (key === 'takeout' && typeof ageSec === 'number'
                 && ageSec >= AD_TAKEOUT_STUCK_SEC);
  dot.style.background = stuck ? 'var(--bad-fg)' : BOARD_STATUS_COLOR_VAR[key];
  dot.classList.toggle('stuck', stuck);
  item.title = stuck
    ? ('Autodarts is STUCK in takeout (' + Math.round(ageSec) + 's). It has '
       + 'not seen the darts removed, so it is reporting no new throws and '
       + 'cannot score. Remove the darts, or reset the board.')
    : ('Autodarts board status: ' + BOARD_STATUS_LABEL[key]);
  const warn = document.getElementById('ad-takeout-warning');
  if (warn) {
    warn.hidden = !stuck;
    if (stuck) {
      warn.textContent = 'Autodarts stuck in takeout for '
        + Math.round(ageSec) + 's — it is not scoring. Clear the board or reset it.';
    }
  }
}

function renderState(state) {
  // Package root belongs on the brand as a tooltip, not printed across
  // the masthead: it is an operator detail, and a filesystem path is the
  // single loudest "this is somebody's dev build" signal a product can
  // put in its header.
  const brandEl = document.querySelector('.brand');
  if (brandEl && state.package_root) brandEl.title = 'package root: ' + state.package_root;
  // Guarded like every other writer: a slow /api/state response must not
  // undo a newer push (see acceptCaptureStatus()). The trigger half is
  // still applied -- it has its own freshness and no Starting to undo.
  if (acceptCaptureStatus(state.capture_loop)) pillCaptureLoop = state.capture_loop;
  pillTrigger = state.trigger || {};
  if (state.capture_loop && state.capture_loop.last_start_error) {
    lastStartError = state.capture_loop.last_start_error;
  }
  renderPill();
  // Restores the Scoring tab's board across a refresh: /api/state's
  // `visit` section carries the same THROW_DETECTED payloads the server
  // accumulated for the turn still in progress, so a reload mid-visit
  // comes back with the darts already on the board rather than an empty
  // one that only repopulates on the NEXT throw.
  if (state.visit) {
    visitId = state.visit.visit_id || null;
    visitThrows = (state.visit.throws || []).slice();
    // `available` is false when this process has no live event queue
    // (standalone -- no capture loop). THROW_DETECTED can then never
    // arrive, so say so instead of leaving an empty board that looks
    // like it's merely waiting for a dart that is never coming.
    liveEventsAvailable = state.visit.available !== false;
    renderVisit();
  }
  renderCalibration(state.calibration);
  renderDiagnostics(state.config);
  renderRecordedDataNote(state.config);
  // Same tick as the diagnostics table: publishing state changes with
  // Start/Stop and with the Autodarts toggle, so it must not be a
  // load-once value.
  refreshPublishing();
  refreshAudioPanel();
  refreshVoicePanel(false);
  // ONE request for the whole Config tab, on the same tick as everything
  // else: the ring FILLS and a capture WRITES while this page is open, so
  // a load-once value would show a 22-second buffer as permanently empty
  // and a dump as permanently queued. This replaced four separate polls
  // (/api/port, /api/store-packages, /api/frame-ring and the two selects
  // fed off /api/state) with the single config document -- and the
  // Maintenance note too, whose two launcher flags are keys of that same
  // document, so a queued update changes the note on this tick as well.
  refreshConfig();
  // Static for the life of the process, so fetched only until it
  // arrives -- not re-fetched on every state tick.
  if (!lastBuild) refreshBuildInfo();
  // The engine PICKER is gone (2026-09-08, Zeus is always primary), but
  // the engine ORDER is still real data the Engines tab uses to sort each
  // throw's rows -- see engineOrderRank(). Kept as plain state with no
  // control attached, rather than deleted along with the widget.
  if (state.engine_config) {
    lastEngineConfig = state.engine_config;
    renderScoringTable();
  }
  renderBoardStatus('ad', state.ad_board_status, state.ad_board_status_age_sec);
}

async function loadInitial() {
  try {
    const [stateResp, pkgResp] = await Promise.all([fetch('/api/state'), fetch('/api/packages')]);
    renderState(await stateResp.json());
    ingestPackages(await pkgResp.json());
  } catch (err) {
    console.error('initial load failed', err);
  }
  refreshCameraStatus();
}

// The live events socket, module-level so page teardown can close it (see
// the pagehide handler) and so a reconnect always replaces the same ref.
let liveWs = null;

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(proto + '://' + location.host + '/api/events');
  liveWs = ws;
  const indicator = document.getElementById('ws-indicator');
  ws.onopen = () => {
    indicator.textContent = 'live';
    indicator.className = 'badge ok';
  };
  ws.onclose = () => {
    indicator.textContent = 'reconnecting';
    indicator.className = 'badge bad';
    // The events socket dropping is the reliable signal that OD went away
    // (a restart or deploy). A multipart MJPEG preview that ends
    // server-side does NOT fire img.onerror -- Chrome leaves the last
    // frame painted and the client socket lingers against the
    // ~6-per-origin pool until it times out. Across the many restarts a
    // rig sees, enough strand that every fetch to this origin queues
    // forever. So force the previews closed now, releasing their sockets;
    // updateCameraFeeds() reconnects them once the WS (hence OD) is back.
    stopAllCamStreams();
    setTimeout(connectWebSocket, 2000);
  };
  ws.onerror = () => { ws.close(); };
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.type === 'HELLO') {
      // A new build on the server means this page is out of date -- reload
      // rather than keep running old JS against a new backend.
      if (reloadIfPageIsStale(msg.page_version, OD_BOOTSTRAP.page_version,
                              window.sessionStorage, () => window.location.reload())) {
        return;
      }
      renderState(msg.state);
      ingestPackages(msg.packages);
      checkPackageCountAndResync(msg.count);
    } else if (msg.type === 'PACKAGES_UPDATED') {
      // Real live push (opendarts.live.run_product) or this server's own
      // polling fallback -- either way, a brand-new throw captured AFTER
      // viewFilterSinceMs still reaches ingestPackages()/renderScoringTable()
      // and is shown live, exactly like an unfiltered view -- the server
      // has no concept of this filter at all (see the design-decision
      // comment above ingestPackages()), so it can never suppress data
      // the filter would otherwise want to show.
      ingestPackages(msg.packages);
      checkPackageCountAndResync(msg.count);
    } else if (msg.type === 'CALIBRATION_STATUS') {
      renderCalibration({
        checked_at_utc: msg.ts,
        cameras: msg.cameras,
        ring_geometry_relearned: msg.ring_geometry_relearned,
      });
      if (msg.ring_geometry_relearned && msg.ring_geometry_relearned.note) {
        const movedLine = logAction(
          'Cameras moved', 'warn', movedShort(msg.ring_geometry_relearned));
        if (movedLine) movedLine.title = msg.ring_geometry_relearned.note;
      }
      // `source` is only ever set by the async, Start-triggered auto-
      // calibrate bootstrap (opendarts.live.capture_daemon.
      // run_capture_loop_body's "Calibration bootstrap" section) -- a
      // manual "Calibrate" click's own broadcast omits it entirely (that
      // click already logs its own outcome directly in its onclick
      // handler above), so this branch can never double-log. Real,
      // visible confirmation of bug #3 (Start gave no sign of whether it
      // calibrated).
      if (msg.source === 'startup' || msg.source === 'startup_reused') {
        const cams = msg.cameras || {};
        const okCams = Object.values(cams).filter((c) => c.ok).length;
        const totalCams = Object.keys(cams).length;
        startingCalibDetail = msg.source === 'startup'
          ? 'auto-calibration ran (' + okCams + '/' + totalCams + ' cam(s) ok)'
          : 'reused existing calibration (already valid, no recalibration needed)';
        logAction('Calibrate (auto, on Start)', okCams > 0 || msg.source === 'startup_reused' ? 'ok' : 'bad', startingCalibDetail);
        renderPill();
      }
    } else if (msg.type === 'DART_CALL') {
      // SPEAK IT. Sent by the server immediately before the
      // THROW_DETECTED below, as a separate frame so the sound starts
      // ahead of the payload that redraws the board -- sound is the
      // slowest thing a human notices.
      //
      // `msg.phrase` arrives already resolved to what a caller would
      // SAY ("treble 20", "double bullseye"), never as sector+ring. That
      // vocabulary lives in opendarts/live/audio.py, pinned by test
      // against the scoring layer's own; a copy of it here would be a
      // second thing to keep right, and the day the two disagreed this
      // page would show one number and say another.
      speakPhrase(msg.phrase);
    } else if (msg.type === 'THROW_DETECTED') {
      // The Scoring tab's only live input for a scored dart. Fires ahead
      // of PACKAGE_SAVED/PACKAGES_UPDATED for the same throw (see
      // handle_ready_to_capture) and carries everything the board needs,
      // so the board updates on the score itself rather than waiting on
      // the package write.
      onThrowDetected(msg);
    } else if (msg.type === 'VISIT_CLEARED') {
      onVisitCleared(msg);
    } else if (msg.type === 'THROW_CORRECTED') {
      // Patches the open visit's board/slots; the Engines tab picks the
      // same correction up via the PACKAGES_UPDATED that follows it.
      onThrowCorrectedLive(msg);
    } else if (msg.type === 'TRIGGER_STATE') {
      // Real push from opendarts.live.run_product's capture loop thread --
      // update the status pill immediately, no fetch round-trip needed.
      // The capture loop reaching its first real TRIGGER_STATE this
      // session is also the definitive "no longer starting" signal (see
      // AppState.capture_starting's own docstring) -- mirrored here on
      // the client so the pill flips from Starting the instant this
      // arrives, not on some separate timer.
      pillTrigger = {
        available: true,
        state: msg.state,
        dart_count: msg.dart_count,
        reason: 'live push',
        last_event_utc: msg.ts,
      };
      // Only if this push is not older than the capture-loop snapshot
      // already applied (acceptCaptureStatus()) -- a TRIGGER_STATE from
      // before a NEWER Start must not clear that Start's Starting.
      if (pillCaptureLoop && acceptCaptureStatus(msg)) pillCaptureLoop.starting = false;
      renderPill();
    } else if (msg.type === 'CAPTURE_LOOP_STATUS') {
      // Real push from AppState.start_capture()/stop_capture() -- every
      // connected dashboard tab (not just the one that clicked) sees
      // Start/Stop/idle-timeout reflected live in the pill and sidebar.
      // Direct fix for bug #2 ("start/stop buttons do something in the
      // background but the pill doesn't reflect it").
      //
      // Dropped whole when older than what this tab already applied (see
      // acceptCaptureStatus()) -- start_capture()'s final broadcast can
      // reach a tab after the TRIGGER_STATE that ended Starting.
      if (!acceptCaptureStatus(msg)) return;
      pillCaptureLoop = msg;
      if (msg.ok === false && msg.reason) {
        lastStartError = msg.reason;
        pillCaptureLoop.last_start_error = msg.reason;
      } else if (msg.ok) {
        lastStartError = null;
      }
      renderIdleTimeoutSelect(pillCaptureLoop.idle_timeout_sec);
      renderPill();
      // Both capture-state edges, without waiting on the 3s tick: a
      // start connects the preview streams immediately, a stop closes
      // them and shows the placeholder instead of a frozen last frame.
      updateCameraFeeds();
    } else if (msg.type === 'AD_BOARD_STATUS') {
      // Real push from AdWsListener's own on_status_change callback (see
      // opendarts/live/run_product.py's _make_board_status_pusher()) -- the
      // WS connection we already hold open for throw-matching also
      // carries this, no separate poll.
      renderBoardStatus('ad', msg.status);
    } else if (msg.type === 'AD_CONNECTION') {
      // The listener's socket to Autodarts came up or went down (server's
      // _handle_live_event AD_CONNECTION branch). Separate from
      // AD_BOARD_STATUS on purpose: a board status is not a connection
      // state -- a disconnect leaves the status wherever AD last put it.
      applyAdConnection(msg);
    }
  };
}

updateCameraFeeds();
refreshCalibrationOverlays();
// The overlay is deliberately NOT on this tick. It changes only when
// calibration is re-derived, and refreshCalibrationOverlays() is called
// from the places where that can happen (renderCalibration, tab and
// visibility edges, and a stream going live). The tick is left doing
// what it was always actually for: retrying and tearing down streams.
setInterval(updateCameraFeeds, CAMERA_FEED_TICK_MS);
// Camera status feeds the Config tab's detail rows and the Info tab's
// diagnostics text; a TV on the Scoring tab was polling it every 3 s for
// nothing (2026-09-17). Either tab refreshes it on entry.
setInterval(() => {
  if (tabShowing('config') || tabShowing('info')) refreshCameraStatus();
}, CAMERA_STATUS_REFRESH_MS);

// The Load block (cpu/memory/disk) is the one part of the diagnostics that
// is live -- it must tick, or a stale "37%" is worse than no number. Re-poll
// /api/state and re-render the block, but ONLY while the tab is visible, so
// a Scoring-tab TV never pays for it. The interval is also what gives cpu its
// sampling window: server-side cpu_load() diffs against the previous call.
const LOAD_REFRESH_MS = 2500;
async function refreshLoad() {
  try {
    const r = await fetch('/api/state');
    if (r.ok) renderDiagnostics((await r.json()).config);
  } catch (e) { /* the load block must never break the page */ }
}
setInterval(() => {
  if (tabShowing('config') || tabShowing('info')) refreshLoad();
}, LOAD_REFRESH_MS);
loadInitial();
connectWebSocket();
