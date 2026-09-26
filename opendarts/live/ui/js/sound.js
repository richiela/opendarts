// ---------------------------------------------------------------------------
// sound -- spoken dart calls, played in THIS browser.
//
// The machinery is the current dashboard's, ported intact: it encodes a
// long list of real device behaviour (autoplay policy, iOS 'interrupted'
// contexts that must be rebuilt, the silent media-session unlock, partial
// voice sets). Only its surface changed: it emits 'sound' and the views
// draw from Sound.status(), instead of reaching into the page itself.
//
// Per device by design (docs/LIVE_API.md, Audio): the TV above the board
// and the tablet in your hand are different listeners.
// ---------------------------------------------------------------------------

const Sound = (() => {
  const DEFAULTS = {enabled: false, volume: 0.8, voice: ''};
  const REPORT_MS = 15000;
  const clientId = 'tab_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);

  function read() {
    // Same key the current dashboard uses, so a screen keeps its choice.
    let raw = null;
    try { raw = window.localStorage.getItem('opendarts.audio.v1'); } catch (err) { raw = null; }
    if (!raw) return Object.assign({}, DEFAULTS);
    try {
      const got = JSON.parse(raw);
      return {
        enabled: got.enabled === true,
        volume: typeof got.volume === 'number' && got.volume >= 0 && got.volume <= 1 ? got.volume : DEFAULTS.volume,
        voice: typeof got.voice === 'string' ? got.voice : '',
      };
    } catch (err) { return Object.assign({}, DEFAULTS); }
  }
  const settings = read();
  const save = () => { try { window.localStorage.setItem('opendarts.audio.v1', JSON.stringify(settings)); } catch (err) { /* this session only */ } };

  let ctx = null, gain = null, buffers = null, bufferVoice = null, catalog = null;
  let loadState = 'idle', loadDetail = '', blocked = false, recovering = false;
  let spoken = 0, lastPhrase = '';

  // Keep the output awake. After a stretch of silence something between
  // here and the ear goes to sleep -- the browser's output stream, the OS
  // device, an HDMI TV or a soundbar or a Bluetooth speaker that detects
  // silence -- and the first sound after it is spent waking it up: the
  // first dart after a break came out as only the tail of its call. A tone
  // far below hearing (-80 dB, 30 Hz) is not digital silence, so nothing
  // down the chain decides to sleep. Only while calls are on.
  let keeper = null;

  // Calls go out as fast as the browser can play them: the smallest
  // buffers ('interactive'), started the instant the call arrives. A bigger
  // buffer would ride out a CPU spike on a rig that also runs its TV's
  // browser (measured on the Pi: ~6 s of saturated cores per dart, and
  // audio underruns) -- but it delays every call, and speed was chosen.
  const LATENCY_HINT = 'interactive';

  // Whether audio is actually keeping up: the audio clock should advance as
  // fast as the wall clock; when rendering is starved it falls behind. Each
  // shortfall over 5 ms in a one-second sample is counted and reported
  // (/api/audio/clients), so a clipped call can be seen, not guessed at.
  const stall = {count: 0, ms: 0, worst: 0, last: null};
  function sampleClock() {
    if (!ctx || ctx.state !== 'running') { stall.last = null; return; }
    const now = {wall: performance.now() / 1000, audio: ctx.currentTime};
    if (stall.last) {
      const behind = ((now.wall - stall.last.wall) - (now.audio - stall.last.audio)) * 1000;
      if (behind > 5 && behind < 5000) { stall.count += 1; stall.ms += behind; stall.worst = Math.max(stall.worst, behind); }
    }
    stall.last = now;
  }
  setInterval(sampleClock, 1000);
  function keepWarm() {
    if (keeper || !ctx || ctx.state !== 'running' || !settings.enabled) return;
    try {
      const osc = ctx.createOscillator();
      osc.frequency.value = 30;
      const g = ctx.createGain();
      g.gain.value = 0.0001;
      osc.connect(g); g.connect(ctx.destination);
      osc.start();
      keeper = {osc, g, ctx};
    } catch (err) { console.debug('could not keep the output awake', err); }
  }
  function coolDown() {
    if (!keeper) return;
    try { keeper.osc.stop(); keeper.g.disconnect(); } catch (err) { /* already gone */ }
    keeper = null;
  }

  const changed = () => { emit('sound'); report(); };
  const Ctor = () => window.AudioContext || window.webkitAudioContext || null;

  function ensure() {
    if (ctx) return ctx;
    const C = Ctor();
    if (!C) { loadState = 'error'; loadDetail = 'This browser has no Web Audio support.'; return null; }
    try {
      try { ctx = new C({latencyHint: LATENCY_HINT}); } catch (err) { ctx = new C(); }   // older Safari takes no options
      gain = ctx.createGain();
      gain.gain.value = settings.volume;
      gain.connect(ctx.destination);
      ctx.onstatechange = () => {
        // Assigned from the condition, so it can move both ways.
        blocked = !!ctx && ctx.state !== 'running';
        if (ctx && ctx.state === 'running') keepWarm();
        if (ctx && ctx.state === 'interrupted' && !recovering) resume().then(changed);
        changed();
      };
    } catch (err) {
      console.error('could not create an AudioContext', err);
      ctx = null; loadState = 'error'; loadDetail = String(err && err.message ? err.message : err);
    }
    return ctx;
  }

  async function rebuild() {
    const old = ctx;
    coolDown();
    ctx = null; gain = null; buffers = null; bufferVoice = null;
    if (old) { try { await old.close(); } catch (err) { /* already gone */ } }
    return ensure();
  }

  async function resume() {
    let c = ensure();
    if (!c) { blocked = false; return false; }
    if (c.state === 'running') { blocked = false; keepWarm(); return true; }
    try { await c.resume(); } catch (err) { console.debug('resume refused', err); }
    // iOS leaves a context 'interrupted' after a call or Siri and will not
    // resume it; only a fresh one plays again.
    if (c.state === 'interrupted' && !recovering) {
      recovering = true;
      try {
        c = await rebuild();
        if (c) { try { await c.resume(); } catch (err) { console.debug('resume after rebuild refused', err); } }
      } finally { recovering = false; }
    }
    blocked = !c || c.state !== 'running';
    if (!blocked) keepWarm();
    if (!blocked && !buffers && settings.enabled) chosenVoiceLoaded().then(loadVoice).then(changed);
    return !blocked;
  }

  async function fetchCatalog(force) {
    if (catalog && !force) return catalog;
    catalog = await (await fetch('/api/audio/voices')).json();
    return catalog;
  }

  function chosenVoice() {
    const names = ((catalog && catalog.voices) || []).map((v) => v.voice);
    if (settings.voice && names.indexOf(settings.voice) >= 0) return settings.voice;
    if (catalog && names.indexOf(catalog.default) >= 0) return catalog.default;
    return names.length ? names[0] : '';
  }

  async function chosenVoiceLoaded() {
    try { await fetchCatalog(false); } catch (err) { console.warn('voice catalog unavailable', err); }
    return chosenVoice();
  }

  function decode(c, bytes) {
    return new Promise((resolve, reject) => {
      const maybe = c.decodeAudioData(bytes, resolve, reject);
      if (maybe && typeof maybe.then === 'function') maybe.then(resolve, reject);
    });
  }

  async function loadVoice(voice) {
    const c = ensure();
    if (!c || !voice) return false;
    if (bufferVoice === voice && buffers) return true;
    loadState = 'loading'; loadDetail = 'Loading ' + voice + '…';
    emit('sound');
    try {
      const cat = await fetchCatalog(false);
      const clips = (cat && cat.clips) || {};
      const phrases = Object.keys(clips);
      const next = new Map();
      const failed = [];
      const queue = phrases.slice();
      const one = async (phrase) => {
        const resp = await fetch('/api/audio/clips/' + encodeURIComponent(voice) + '/' + encodeURIComponent(clips[phrase]));
        if (!resp.ok) throw new Error(clips[phrase] + ': HTTP ' + resp.status);
        next.set(phrase, await decode(c, await resp.arrayBuffer()));
      };
      // Four at a time, one retry each: the socket budget is shared with
      // everything else this page does.
      const worker = async () => {
        while (queue.length) {
          const phrase = queue.shift();
          try { await one(phrase); } catch (err) {
            try { await one(phrase); } catch (err2) { failed.push(phrase); }
          }
        }
      };
      await Promise.all(Array.from({length: Math.min(4, phrases.length)}, worker));
      if (!next.size) throw new Error('no clips could be loaded');
      buffers = next; bufferVoice = voice; loadState = 'ready';
      loadDetail = failed.length
        ? next.size + ' of ' + phrases.length + ' calls ready in ' + voice + ' — ' + failed.length + ' missing'
        : 'Ready — ' + voice + ', ' + next.size + ' calls.';
      return true;
    } catch (err) {
      console.error('voice set failed to load', err);
      loadState = 'error';
      loadDetail = 'Could not load ' + voice + ': ' + String(err && err.message ? err.message : err);
      return false;
    } finally { changed(); }
  }

  async function arm() {
    const ok = await resume();
    const voice = await chosenVoiceLoaded();
    if (voice) await loadVoice(voice);
    changed();
    return ok;
  }

  function play(phrase, force) {
    if ((!settings.enabled && !force) || !phrase) return false;
    const c = ctx;
    if (!c || !buffers) return false;
    const buf = buffers.get(phrase);
    if (!buf) { console.warn('no clip for', phrase); return false; }
    if (c.state !== 'running') {
      blocked = true;
      emit('sound');
      resume().then(changed);
      return false;
    }
    try {
      const src = c.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      src.start();
      spoken += 1; lastPhrase = phrase;
      return true;
    } catch (err) { console.error('playback failed for', phrase, err); return false; }
  }

  function isBlocked() {
    if (!settings.enabled || !Ctor()) return false;
    return !ctx || ctx.state !== 'running';
  }

  function deviceLabel() {
    const ua = navigator.userAgent || '';
    let device = 'this screen';
    if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) device = 'iPad';
    else if (/iPhone/.test(ua)) device = 'iPhone';
    else if (/Android/.test(ua)) device = 'Android';
    else if (/CrKey|TV|SmartTV|BRAVIA|AFT/.test(ua)) device = 'TV';
    else if (/Macintosh/.test(ua)) device = 'Mac';
    else if (/Windows/.test(ua)) device = 'Windows';
    else if (/Linux/.test(ua)) device = 'Linux';
    let browser = '';
    if (ua.indexOf('Edg/') >= 0) browser = 'Edge';
    else if (ua.indexOf('Chrome/') >= 0 && ua.indexOf('Chromium') < 0) browser = 'Chrome';
    else if (ua.indexOf('Firefox/') >= 0) browser = 'Firefox';
    else if (ua.indexOf('Safari/') >= 0) browser = 'Safari';
    return browser ? device + ' / ' + browser : device;
  }

  // A report, never a control: every screen says whether it can actually
  // be heard, so the one at the oche can see the TV has gone quiet.
  function report() {
    fetch('/api/audio/clients', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        client_id: clientId, label: deviceLabel(), enabled: settings.enabled, blocked: isBlocked(),
        latency_hint: LATENCY_HINT,
        base_latency_ms: ctx && typeof ctx.baseLatency === 'number' ? Math.round(ctx.baseLatency * 1000) : null,
        output_latency_ms: ctx && typeof ctx.outputLatency === 'number' ? Math.round(ctx.outputLatency * 1000) : null,
        stalls: stall.count, stall_ms: Math.round(stall.ms), worst_stall_ms: Math.round(stall.worst),
        voice: bufferVoice || chosenVoice(), volume: settings.volume, ready: buffers ? buffers.size : 0,
        load_state: loadState, context_state: ctx ? ctx.state : 'none', spoken, last_phrase: lastPhrase,
      }),
    }).catch(() => {});
  }

  async function setEnabled(onOff) {
    settings.enabled = !!onOff;
    save();
    if (!settings.enabled) { coolDown(); blocked = false; changed(); return true; }
    return arm();
  }

  async function setVoice(voice) {
    settings.voice = voice;
    save();
    const ok = settings.enabled ? await loadVoice(voice) : true;
    changed();
    return ok;
  }

  function setVolume(v, commit) {
    settings.volume = Math.max(0, Math.min(1, v));
    if (gain) gain.gain.value = settings.volume;
    if (commit) { save(); report(); }
    emit('sound');
  }

  async function test() {
    const ok = await resume();
    if (!ok) return 'blocked';
    if (!buffers) { const voice = await chosenVoiceLoaded(); if (voice) await loadVoice(voice); }
    return play('treble 20', true) ? 'played' : 'failed';
  }

  // What every view shows: one state, with the words to say about it.
  function status() {
    if (!Ctor()) return {state: 'unsupported', label: 'No sound', detail: 'This browser cannot play sound (no Web Audio).'};
    if (!settings.enabled) return {state: 'off', label: 'Sound off', detail: 'Calls are silent on this screen. Every screen decides for itself.'};
    if (isBlocked()) return {state: 'blocked', label: 'Tap to allow sound', detail: 'The browser needs one tap on this page before it will play sound. On an iPad, also check the silent switch.'};
    if (loadState === 'error') return {state: 'error', label: 'Sound failed', detail: loadDetail};
    if (loadState !== 'ready') return {state: 'loading', label: 'Sound loading', detail: loadDetail || 'Loading the voice…'};
    return {state: 'on', label: 'Sound on', detail: loadDetail};
  }

  // Autoplay: the first gesture anywhere unlocks this page's audio. The
  // clip is the current dashboard's -- real (silent) samples, because iOS
  // does not count a zero-length file as playback.
  const SILENT_WAV = 'data:audio/wav;base64,UklGRkQDAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YSADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==';
  let sessionEl = null;
  function unlockSession() {
    if (sessionEl) return;
    try {
      const el = document.createElement('audio');
      el.loop = true;
      el.setAttribute('playsinline', '');
      el.volume = 0.001;
      el.src = SILENT_WAV;
      const p = el.play();
      if (p && p.catch) p.catch((err) => console.debug('silent unlock refused', err));
      sessionEl = el;
    } catch (err) { console.debug('silent unlock failed', err); }
  }
  ['pointerdown', 'keydown', 'touchend'].forEach((evt) => {
    document.addEventListener(evt, () => {
      if (!settings.enabled) return;
      unlockSession();
      if (ctx && ctx.state === 'running' && buffers) return;
      resume().then(() => { if (!buffers) return chosenVoiceLoaded().then(loadVoice); }).then(changed);
    }, true);
  });
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && settings.enabled) resume().then(changed);
  });

  // ?sound=on|off -- for a kiosk nobody will touch.
  (function fromUrl() {
    let raw = null;
    try { raw = new URLSearchParams(window.location.search).get('sound'); } catch (err) { return; }
    if (raw === null) return;
    const v = raw.trim().toLowerCase();
    const want = ['on', '1', 'true', 'yes'].includes(v) ? true : ['off', '0', 'false', 'no'].includes(v) ? false : null;
    if (want !== null && want !== settings.enabled) { settings.enabled = want; save(); }
  })();

  function start() {
    if (settings.enabled) arm();
    report();
    setInterval(report, REPORT_MS);
    emit('sound');
  }

  return {
    start, play, setEnabled, setVoice, setVolume, test, status,
    fetchCatalog, chosenVoice, clientId,
    get settings() { return settings; },
  };
})();
