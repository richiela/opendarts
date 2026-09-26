// ---------------------------------------------------------------------------
// display -- this page in its show-only role: a TV beside the board, in a
// kiosk with no keyboard (opendarts/live/displays.py has the model).
//
// It takes no input and has no settings of its own to change. It reports
// in to the rig, and the rig answers with what it should look like; any
// controller changes that, and the change arrives here over the events
// socket. So it can never be left in a state nobody can reach to undo.
// ---------------------------------------------------------------------------

const Display = (() => {
  const REPORT_MS = 20000;
  const params = new URLSearchParams(window.location.search);
  let id = null, name = '', settings = null, wake = null, identifyTimer = null;

  // ?id= pins the identity, so a kiosk that loses its storage is still the
  // same display; otherwise one is made up once and kept.
  function identity() {
    // ?display=tv names the id in one go; ?id=tv does too
    const fromUrl = (params.get('display') || params.get('id') || '').trim();
    if (/^[A-Za-z0-9_-]{1,64}$/.test(fromUrl)) return fromUrl;
    let saved = prefs.get('displayId', null);
    if (typeof saved !== 'string' || !/^[A-Za-z0-9_-]{1,64}$/.test(saved)) {
      saved = 'd_' + Math.random().toString(36).slice(2, 10);
      prefs.set('displayId', saved);
    }
    return saved;
  }

  async function report() {
    const snd = Sound.status();
    try {
      const body = await api.post('/api/displays/hello', {
        display_id: id,
        name: params.get('name') || undefined,
        info: {w: window.innerWidth, h: window.innerHeight, sound: snd.state, page: String(OD.page_version || '').slice(0, 12)},
      });
      if (body && body.display) apply(body.display);
    } catch (err) { /* the socket's reconnect will bring it back */ }
  }

  const LAYOUT_VIEW = {split: 'split', scoring: 'play', engines: 'throws'};

  function apply(rec) {
    const s = rec.settings || {};
    const was = settings || {};
    settings = s;
    name = rec.name || name;
    document.title = name + ' · opendarts';
    document.body.dataset.layout = s.layout || 'split';
    // Applied, not saved as this browser's preference: the rig holds them.
    if (s.theme !== was.theme || s.text_size !== was.text_size) Shell.applyTheme(s.theme === 'light' ? 'light' : 'dark', s.text_size || 'normal');
    if (s.board_view && s.board_view !== LiveBoard.view) LiveBoard.setView(s.board_view);
    if (s.layout !== was.layout) {
      Shell.go(LAYOUT_VIEW[s.layout] || 'split');
      Throws.watch();
    }
    if (typeof s.volume === 'number' && s.volume !== Sound.settings.volume) Sound.setVolume(s.volume, true);
    if ((s.voice || '') !== Sound.settings.voice) Sound.setVoice(s.voice || '');
    if (!!s.sound !== Sound.settings.enabled) Sound.setEnabled(!!s.sound);
  }

  // Which screen is this? A controller asks, and the name fills the TV.
  function identify(label) {
    let el = document.getElementById('display-identify');
    if (!el) {
      el = h('div', {id: 'display-identify', class: 'identify'}, h('span', {class: 'k'}, 'THIS IS'), h('strong', {}, ''));
      document.body.append(el);
    }
    el.querySelector('strong').textContent = label || name;
    el.hidden = false;
    clearTimeout(identifyTimer);
    identifyTimer = setTimeout(() => { el.hidden = true; }, 6000);
  }

  function onMessage(m) {
    if (m.type === 'DISPLAY_UPDATED' && m.display && m.display.id === id) apply(m.display);
    else if (m.type === 'DISPLAY_IDENTIFY' && m.display_id === id) identify(m.name);
    else if (m.type === 'DISPLAY_RELOAD' && m.display_id === id) window.location.reload();
    else if (m.type === 'DISPLAY_FORGOTTEN' && m.display_id === id) report();   // comes back, fresh
  }

  // A TV that sleeps is a TV nobody can wake without walking over to it.
  async function keepAwake() {
    if (!('wakeLock' in navigator) || document.hidden) return;
    try {
      wake = await navigator.wakeLock.request('screen');
      wake.addEventListener('release', () => { wake = null; });
    } catch (err) { wake = null; }
  }

  function start() {
    id = identity();
    document.body.classList.add('display');
    const hint = $('#board-veil .offair');
    if (hint) hint.dataset.hint = location.host;
    Shell.go('split');
    Throws.watch();
    report();
    setInterval(report, REPORT_MS);
    on('connected', report);
    // Tell the controllers when this screen's sound changes state -- "needs
    // a tap" is the one thing a person must walk over to fix.
    let heard = Sound.status().state;
    on('sound', () => { const now = Sound.status().state; if (now !== heard) { heard = now; report(); } });
    keepAwake();
    document.addEventListener('visibilitychange', () => { if (!document.hidden && !wake) keepAwake(); });
  }

  return {start, onMessage, get id() { return id; }, get name() { return name; }};
})();

// ---- the controller's side: Rig › Displays --------------------------------------
// One card per display. Each control saves straight to the rig, which pushes
// it to the display (and to every other controller, whose cards redraw).

const DisplayAdmin = (() => {
  const LAYOUTS = [['split', 'Scoring + Engines'], ['scoring', 'Scoring'], ['engines', 'Engines']];
  const SIZES = [['normal', 'Normal'], ['large', 'Large'], ['huge', 'Huge']];
  const THEMES = [['dark', 'Blueprint'], ['light', 'Paper']];
  const BOARDS = [['photo', 'Photo'], ['diagram', 'Diagram']];
  let list = [], voices = null, pending = false;

  const box = () => $('#displays-list');

  async function refresh() {
    if (ROLE === 'display') return;
    const body = await api.get('/api/displays').catch(() => null);
    if (!body || !Array.isArray(body.displays)) return;
    list = body.displays;
    render();
  }

  async function loadVoices() {
    if (voices) return voices;
    try {
      const cat = await Sound.fetchCatalog(false);
      voices = ((cat && cat.voices) || []).map((v) => [v.voice, v.voice + (v.voice === cat.default ? ' (default)' : '')]);
    } catch (err) { voices = []; }
    return voices;
  }

  async function change(d, patch, what) {
    const body = await api.patch('/api/displays/' + encodeURIComponent(d.id), patch).catch(() => null);
    if (!body || body.ok === false) { toast((body && body.reason) || 'Could not reach the rig', 'bad'); refresh(); return; }
    const i = list.findIndex((x) => x.id === d.id);
    if (i >= 0) list[i] = body.display;
    render();
    if (what) toast(body.display.name + ': ' + what + (body.display.online ? '' : ' — it will show when the screen is back'), 'ok');
  }

  function seg(d, key, options, locked) {
    return h('div', {class: 'seg', role: 'group'}, options.map(([v, label]) => h('button', {
      type: 'button', 'aria-pressed': String(d.settings[key] === v), disabled: locked,
      onclick: () => { if (d.settings[key] !== v) change(d, {settings: {[key]: v}}, label); },
    }, label)));
  }

  function row(label, control, note, wide) {
    return h('div', {class: 'setting' + (wide ? ' wide' : '')}, h('span', {class: 'k'}, label), h('span', {}), h('div', {class: 'c'}, control),
      note ? h('p', {class: 'note'}, note) : null);
  }

  function presence(d) {
    if (!d.online) {
      const age = d.age_s === null ? 'never seen' : 'last seen ' + ago(new Date(Date.now() - d.age_s * 1000).toISOString());
      return {tone: 'idle', text: 'Not showing · ' + age};
    }
    const bits = ['Showing'];
    if (d.info && d.info.w) bits.push(d.info.w + '×' + d.info.h);
    const snd = d.info && d.info.sound;
    if (snd === 'blocked') return {tone: 'warn', text: bits.concat('sound needs one tap on the screen').join(' · ')};
    if (snd === 'on') bits.push('speaking');
    return {tone: 'live', text: bits.join(' · ')};
  }

  function card(d) {
    const s = d.settings, locked = !!s.locked;
    const p = presence(d);
    const nameIn = h('input', {type: 'text', class: 'dname', value: d.name, maxlength: '40', 'aria-label': 'Display name', disabled: locked});
    nameIn.addEventListener('change', () => { const v = nameIn.value.trim(); if (v && v !== d.name) change(d, {name: v}, 'renamed'); });
    nameIn.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') nameIn.blur(); });
    const vol = h('input', {type: 'range', min: '0', max: '100', value: String(Math.round((s.volume || 0) * 100)), disabled: locked, 'aria-label': 'Volume'});
    const volVal = h('span', {class: 'vol-val'}, Math.round((s.volume || 0) * 100) + '%');
    vol.addEventListener('input', () => { volVal.textContent = vol.value + '%'; });
    vol.addEventListener('change', () => change(d, {settings: {volume: Number(vol.value) / 100}}));
    const voice = h('select', {disabled: locked, 'aria-label': 'Voice'},
      [['', 'Rig default']].concat(voices || []).map(([v, label]) => h('option', {value: v, selected: v === (s.voice || '')}, label)));
    voice.addEventListener('change', () => change(d, {settings: {voice: voice.value}}, 'voice ' + (voice.value || 'default')));
    const sound = h('input', {type: 'checkbox', class: 'switch', checked: !!s.sound, disabled: locked, 'aria-label': 'Speak each dart'});
    sound.addEventListener('change', () => change(d, {settings: {sound: sound.checked}}, sound.checked ? 'sound on' : 'sound off'));
    const act = (label, fn, cls) => h('button', {type: 'button', class: 'btn small' + (cls ? ' ' + cls : ''), onclick: fn}, label);
    return h('article', {class: 'display-card' + (locked ? ' locked' : ''), dataset: {id: d.id}},
      h('header', {class: 'dc-head'},
        h('span', {class: 'dot', dataset: {tone: p.tone}}),
        nameIn,
        h('span', {class: 'dc-state', dataset: {tone: p.tone}}, p.text)),
      row('Shows', seg(d, 'layout', LAYOUTS, locked), null, true),
      row('Text size', seg(d, 'text_size', SIZES, locked)),
      row('Engines paper', seg(d, 'theme', THEMES, locked)),
      row('Board', seg(d, 'board_view', BOARDS, locked)),
      row('Speak each dart', sound, s.sound && d.info && d.info.sound === 'blocked'
        ? 'The screen’s browser is holding sound back until someone taps it once — or start its kiosk browser with autoplay allowed.' : null),
      s.sound ? row('Voice', voice) : null,
      s.sound ? row('Volume', h('span', {class: 'volume'}, vol, volVal)) : null,
      h('footer', {class: 'dc-foot'},
        act('Identify', async () => {
          const r = await api.post('/api/displays/' + encodeURIComponent(d.id) + '/identify').catch(() => null);
          toast(r && r.online ? 'Its name is on the screen now' : 'That screen is not showing right now', r && r.online ? 'ok' : 'warn');
        }),
        act('Reload', () => api.post('/api/displays/' + encodeURIComponent(d.id) + '/reload').then(() => toast('Reloading ' + d.name, 'ok'))),
        act(locked ? 'Unlock' : 'Lock', () => change(d, {settings: {locked: !locked}}, locked ? 'unlocked' : 'locked — no changes until it is unlocked')),
        h('span', {class: 'grow'}),
        act('Forget…', () => confirmSheet({
          title: 'Forget ' + d.name + '?',
          body: '<p>Its settings are removed from the rig. If the screen is still open it comes back as a new display with the defaults.</p>',
          confirm: 'Forget it', danger: true,
        }).then((ok) => { if (ok) api.del('/api/displays/' + encodeURIComponent(d.id)).then(refresh); }), 'ghost')));
  }

  function render() {
    const el = box();
    if (!el) return;
    // Never rebuild under a hand: a half-typed name or a dragged slider
    // would be thrown away. The next refresh catches up.
    if (el.contains(document.activeElement) && document.activeElement !== document.body
        && /INPUT|SELECT/.test(document.activeElement.tagName)) { pending = true; return; }
    pending = false;
    $('#display-url').textContent = location.host + '/?display=tv';
    if (!list.length) {
      el.replaceChildren(h('div', {class: 'empty'}, h('strong', {}, 'No displays yet'),
        h('p', {}, 'Open ' + location.host + '/?display=tv on the TV — it shows up here within a few seconds.')));
      return;
    }
    el.replaceChildren(...list.map(card));
  }

  if (ROLE !== 'display') {
    on('view', () => { if (Shell.showing('rig')) loadVoices().then(refresh); });
    on('displays', () => { if (Shell.showing('rig')) refresh(); });
    document.addEventListener('focusout', () => { if (pending) setTimeout(render, 0); });
    setInterval(() => { if (Shell.showing('rig')) refresh(); }, 10000);
  }

  return {refresh};
})();
