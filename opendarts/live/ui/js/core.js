'use strict';
// ---------------------------------------------------------------------------
// core -- the shared ground every view stands on: the server's bootstrap,
// DOM helpers, formatting, the API client, one store with change events,
// the activity record, toasts and sheets.
//
// The page is one document with every script inlined (see
// opendarts/live/ui/__init__.py), so these files share one scope. Each
// later file reads what it needs from here and adds a namespace of its own.
// ---------------------------------------------------------------------------

const OD = JSON.parse(document.getElementById('bootstrap').textContent);
// "control" is the whole app; "display" is a screen that only shows, set up
// from a controller (display.js, opendarts/live/displays.py).
const ROLE = OD.role === 'display' ? 'display' : 'control';
const CAM_IDS = OD.cam_ids;
const BOARD_SECTORS = OD.board_sectors;          // clockwise from 20 at the top
const BOARD_RINGS = OD.board_rings;
const VIDEO_RECORD_MODE = OD.video_record_mode;

// Dart identity colours: the marker on the board, the row in the turn and
// the dot in the throw list are the same dart, so they share one colour.
const DART_COLORS = ['#4fc4b0', '#e39a52', '#e0665a'];

// ---- DOM ---------------------------------------------------------------

const $ = (sel, root) => (root || document).querySelector(sel);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

function esc(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

// Whether a key press belongs to a field or a sheet rather than to the
// page's shortcuts. The target can be the document itself (no .closest).
function typingIn(target) {
  return !!(target && target.closest && target.closest('input, select, textarea, dialog'));
}

// Per-device preferences. Storage can be missing or throw (private mode, a
// TV browser), and nothing here may break the page when it does.
const prefs = {
  get(key, fallback) {
    try {
      const raw = window.localStorage.getItem('opendarts.' + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (err) { return fallback; }
  },
  set(key, value) {
    try { window.localStorage.setItem('opendarts.' + key, JSON.stringify(value)); } catch (err) { /* per-session only */ }
  },
};

// ---- format ------------------------------------------------------------

const RING_POINTS = {
  bull: () => 50, outer_bull: () => 25, outside: () => 0, miss: () => 0,
  single_inner: (s) => s, single_outer: (s) => s, treble: (s) => s * 3, double: (s) => s * 2,
};

function points(sector, ring) {
  const fn = RING_POINTS[ring];
  if (!fn) return null;
  return fn(sector === null || sector === undefined || sector === '' ? 0 : Number(sector)) || 0;
}

// The call as a player says it. A single is its bare number -- inner and
// outer single score the same, and "S20" is jargon nobody at the oche
// uses. Throws review shows the inner/outer distinction where it matters.
function callLabel(sector, ring) {
  if (ring === 'bull') return 'BULL';
  if (ring === 'outer_bull') return '25';
  if (ring === 'outside' || ring === 'miss') return 'MISS';
  if (!ring || sector === null || sector === undefined || sector === '') return '?';
  if (ring === 'treble') return 'T' + sector;
  if (ring === 'double') return 'D' + sector;
  return String(sector);
}

function ringWord(ring) {
  return {
    bull: 'bull', outer_bull: 'outer bull', single_inner: 'inner single',
    single_outer: 'outer single', treble: 'treble', double: 'double', outside: 'miss', miss: 'miss',
  }[ring] || ring || 'unknown';
}

function sameCall(a, b) {
  if (!a || !b || !a.ring || !b.ring) return false;
  const sa = a.sector === null || a.sector === undefined ? '' : String(a.sector);
  const sb = b.sector === null || b.sector === undefined ? '' : String(b.sector);
  return a.ring === b.ring && sa === sb;
}

const pad2 = (n) => String(n).padStart(2, '0');
function clock(d, seconds) {
  d = d instanceof Date ? d : new Date(d);
  if (isNaN(d)) return '';
  return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + (seconds === false ? '' : ':' + pad2(d.getSeconds()));
}

function ago(iso) {
  const t = Date.parse(iso);
  if (isNaN(t)) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 45) return 'just now';
  if (s < 3600) return Math.round(s / 60) + ' min ago';
  if (s < 86400) return Math.round(s / 3600) + ' h ago';
  const d = new Date(t);
  return d.toLocaleDateString(undefined, {month: 'short', day: 'numeric'}) + ', ' + clock(d, false);
}

function bytes(n) {
  if (n === null || n === undefined || isNaN(n)) return '—';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
  return (i ? n.toFixed(n >= 100 ? 0 : 1) : Math.round(n)) + ' ' + u[i];
}

function duration(s) {
  if (s === null || s === undefined || isNaN(s)) return '—';
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), hh = Math.floor(s / 3600) % 24, m = Math.floor(s / 60) % 60;
  if (d) return d + 'd ' + hh + 'h';
  if (hh) return hh + 'h ' + m + 'm';
  if (m) return m + 'm ' + (s % 60) + 's';
  return s + 's';
}

function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }

// ---- store -------------------------------------------------------------
// One object, and change events by topic. Views subscribe to the topics
// they draw from and re-render; nothing reaches into another view.

const S = {
  connected: false,          // the events socket is open right now
  everConnected: false,
  state: null,               // last /api/state (or HELLO's copy)
  capture: null,             // capture_loop section, ordering-stamped
  trigger: {},
  startError: null,
  calibrating: false,        // a manual Calibrate is in flight from this page
  calibration: null,
  visit: {id: null, throws: [], photoVersion: null, available: true},
  packages: new Map(),       // path -> package record
  engineConfig: null,
  config: null,
  runtime: null,
  restartRequired: [],
  camRows: null,             // /api/cameras/status rows, or null
  camStatusReason: '',
  build: null,
  ad: null,                  // runtime.ad, ordering-stamped
  adBoard: {status: null, age: null},
  activity: [],
};

const _subs = new Map();
function on(topic, fn) {
  if (!_subs.has(topic)) _subs.set(topic, []);
  _subs.get(topic).push(fn);
}
function emit(topic, arg) {
  for (const fn of _subs.get(topic) || []) {
    try { fn(arg); } catch (err) { console.error('render failed for', topic, err); }
  }
}

// Ordering stamps (docs/LIVE_API.md): an HTTP reply and a socket push can
// overtake each other, so a stale snapshot must be dropped, not applied.
// A different epoch means the server restarted -- accept and recount.
function stampGate() {
  let epoch = null, seq = -1;
  return (snap, epochKey, seqKey) => {
    if (!snap || typeof snap[seqKey] !== 'number') return true;
    if (snap[epochKey] === epoch && snap[seqKey] < seq) return false;
    epoch = snap[epochKey];
    seq = snap[seqKey];
    return true;
  };
}
const _captureGate = stampGate();
const acceptCapture = (snap) => _captureGate(snap, 'status_epoch', 'status_seq');
const _adGate = stampGate();
const acceptAd = (snap) => _adGate(snap, 'connection_epoch', 'connection_seq');

// ---- API ---------------------------------------------------------------
// Network failures throw; application refusals come back as the server's
// own {ok:false, reason} body, which callers show verbatim -- the server's
// reason is always better than one invented here.

const api = {
  async req(method, path, body) {
    const init = {method, cache: 'no-store'};
    if (body !== undefined) {
      init.headers = {'Content-Type': 'application/json'};
      init.body = JSON.stringify(body);
    }
    const resp = await fetch(path, init);
    let data = null;
    try { data = await resp.json(); } catch (err) { data = null; }
    if (data === null) data = {ok: resp.ok, reason: resp.ok ? null : 'HTTP ' + resp.status};
    if (!resp.ok && data && data.ok === undefined) data.ok = false;
    return data;
  },
  get(path) { return api.req('GET', path); },
  post(path, body) { return api.req('POST', path, body); },
  patch(path, body) { return api.req('PATCH', path, body); },
  del(path) { return api.req('DELETE', path); },
  put(path, body) { return api.req('PUT', path, body); },
};

// ---- activity ------------------------------------------------------------
// What this page asked the rig to do, and what came of it -- the honest
// record the old sidebar log kept, now in the Session sheet. Each entry is
// updated in place, so one click is one line, never a trail of partials.

const ACTIVITY_MAX = 30;
function activity(label, status, detail) {
  const entry = {label, status: status || 'pending', detail: detail || '', started: new Date(), finished: null};
  S.activity.unshift(entry);
  S.activity.length = Math.min(S.activity.length, ACTIVITY_MAX);
  emit('activity');
  return {
    entry,
    done(status2, detail2) {
      entry.status = status2;
      if (detail2 !== undefined) entry.detail = detail2;
      entry.finished = status2 === 'pending' ? null : new Date();
      emit('activity');
      if (status2 === 'bad') toast(label + ': ' + (entry.detail || 'failed'), 'bad');
      else if (status2 === 'warn') toast(label + ': ' + entry.detail, 'warn');
      return entry;
    },
  };
}

// ---- toasts --------------------------------------------------------------

function toast(message, kind, ms) {
  const box = $('#toasts');
  if (!box) return;
  const el = h('div', {class: 'toast ' + (kind || ''), role: kind === 'bad' ? 'alert' : 'status'}, message);
  box.append(el);
  requestAnimationFrame(() => el.classList.add('in'));
  const life = ms || (kind === 'bad' ? 7000 : kind === 'warn' ? 5000 : 2600);
  setTimeout(() => {
    el.classList.remove('in');
    setTimeout(() => el.remove(), 260);
  }, life);
}

// ---- sheets --------------------------------------------------------------
// Every secondary surface is a native <dialog>: focus is trapped, Escape
// closes, the backdrop is real. On a phone it rises from the bottom; on a
// wide screen it is a centred card (or, for .sheet--side, a right pane).

const sheet = {
  open(id) {
    const d = document.getElementById(id);
    if (!d) return null;
    if (!d.open) {
      if (typeof d.showModal === 'function') d.showModal(); else d.setAttribute('open', '');
      // Focus the sheet itself, not its first button: otherwise the close
      // button opens wearing a focus ring, as if it were the suggestion.
      const box = d.querySelector('.sheet-box');
      if (box) { box.tabIndex = -1; box.focus({preventScroll: true}); }
    }
    emit('sheet:' + id, true);
    return d;
  },
  close(id) {
    const d = typeof id === 'string' ? document.getElementById(id) : id;
    if (!d || !d.open) return;
    if (typeof d.close === 'function') d.close(); else d.removeAttribute('open');
  },
};

document.addEventListener('click', (ev) => {
  // The dialog has no padding and its .sheet-box fills it, so a click
  // whose target is the <dialog> itself can only have landed on the
  // backdrop.
  if (typeof HTMLDialogElement !== 'undefined' && ev.target instanceof HTMLDialogElement
      && ev.target.classList.contains('sheet')) {
    sheet.close(ev.target);
  }
  const closer = ev.target.closest('[data-close]');
  if (closer) sheet.close(closer.closest('dialog'));
});

// A generic confirmation, for the few actions that cannot be undone.
function confirmSheet({title, body, confirm, danger}) {
  return new Promise((resolve) => {
    const d = $('#sheet-confirm');
    $('#confirm-title').textContent = title;
    $('#confirm-body').innerHTML = body;
    const ok = $('#confirm-ok');
    ok.textContent = confirm || 'Confirm';
    ok.className = 'btn ' + (danger ? 'danger' : 'signal');
    let settled = false;
    const finish = (v) => {
      if (settled) return;
      settled = true;
      sheet.close(d);
      resolve(v);
    };
    ok.onclick = () => finish(true);
    d.addEventListener('close', () => finish(false), {once: true});
    sheet.open('sheet-confirm');
  });
}
