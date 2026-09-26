// ---------------------------------------------------------------------------
// board -- everything drawn in board millimetres: the live board (photo or
// diagram), the agreement plot in Throws, and the truth picker.
//
// Board space is +x right, +y UP, origin at the bull, in mm -- the space
// board_xy_mm arrives in. Canvas space is +y down, hence the negations.
// ---------------------------------------------------------------------------

// Regulation radii in mm, matching opendarts.geometry.board's own bands.
const BOARD_MM = {
  doubleOuter: 170.0, doubleInner: 162.0,
  trebleOuter: 107.0, trebleInner: 99.0,
  outerBull: 15.9, bull: 6.35,
};
// Must match board_photo.HALF_MM: the board photo spans +/- this many mm.
const PHOTO_HALF_MM = 235.0;
// How much of it the scoreboard shows: just past the number ring's outer
// wire (~218-225 mm on the rig's photos), so the board fills the volt rim
// instead of floating in a band of backboard.
const PHOTO_VIEW_MM = 228.0;
// Where the player's eye is, in board mm (+z out of the board): at the
// oche, 2.37 m back, a touch right of and below the bull.
const OCHE_EYE = [120.0, -50.0, 2370.0];
// Real darts lean 5-15 degrees; from 2.4 m that barely shows, yet in person
// the lean is what you notice -- so the measured lean is exaggerated.
const DART_LEAN_EXAGGERATION = 2.0;
const DEFAULT_FLIGHT_COLOR = '#262428';

// Three inks for one geometry: the painted board on the show, and the
// board as an engineering drawing in the lab -- on paper, or on blueprint.
const INKS = {
  show: {surround: '#0b0c0e', dark: '#17191c', light: '#e6dcc3', red: '#b3362d', green: '#1c7348',
    wire: 'rgba(8,9,11,0.9)', numbers: 'rgba(230,226,218,0.62)', bull: '#b3362d', outer: '#1c7348'},
  paper: {surround: '#e6e0d3', dark: '#ddd6c6', light: '#f5f1e8', red: '#e8b9ad', green: '#b9d6c3',
    wire: '#15140f', numbers: '#56534a', bull: '#e8b9ad', outer: '#b9d6c3'},
  blueprint: {surround: '#12294a', dark: '#16325a', light: '#1d3f6e', red: '#6b3a4f', green: '#245a5a',
    wire: '#cfe0ff', numbers: '#a9bbd6', bull: '#6b3a4f', outer: '#245a5a'},
};

// The lab's ink follows the lab's theme.
function labInk() {
  const t = document.documentElement.dataset.theme;
  const dark = t === 'dark' || (!t && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  return dark ? 'blueprint' : 'paper';
}
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function sectorAngles(i) {
  // Canvas angles: 0 = +x, -PI/2 = straight up. Sector i (clockwise from
  // 20 at the top) is centred at -PI/2 + i*step.
  const step = (Math.PI * 2) / 20;
  const a0 = -Math.PI / 2 + (i - 0.5) * step;
  return [a0, a0 + step];
}

// Board mm -> the call a point there scores. The picker's inverse of the
// drawing; ring bands are the same radii the scorer uses.
function hitTest(x, y) {
  const r = Math.hypot(x, y);
  let ring;
  if (r <= BOARD_MM.bull) ring = 'bull';
  else if (r <= BOARD_MM.outerBull) ring = 'outer_bull';
  else if (r <= BOARD_MM.trebleInner) ring = 'single_inner';
  else if (r <= BOARD_MM.trebleOuter) ring = 'treble';
  else if (r <= BOARD_MM.doubleInner) ring = 'single_outer';
  else if (r <= BOARD_MM.doubleOuter) ring = 'double';
  else ring = 'outside';
  if (ring === 'bull' || ring === 'outer_bull' || ring === 'outside') return {sector: null, ring};
  const step = (Math.PI * 2) / 20;
  const canvasAngle = Math.atan2(-y, x);
  const i = ((Math.round((canvasAngle + Math.PI / 2) / step) % 20) + 20) % 20;
  return {sector: BOARD_SECTORS[i], ring};
}

// The centre of a segment, for drawing a selection or a candidate marker.
function segmentCentre(sector, ring) {
  if (ring === 'bull' || ring === 'outer_bull') return [0, ring === 'bull' ? 0 : (BOARD_MM.bull + BOARD_MM.outerBull) / 2];
  const i = BOARD_SECTORS.indexOf(String(sector));
  if (i < 0) return null;
  const r = {
    single_inner: (BOARD_MM.outerBull + BOARD_MM.trebleInner) / 2,
    treble: (BOARD_MM.trebleInner + BOARD_MM.trebleOuter) / 2,
    single_outer: (BOARD_MM.trebleOuter + BOARD_MM.doubleInner) / 2,
    double: (BOARD_MM.doubleInner + BOARD_MM.doubleOuter) / 2,
  }[ring];
  if (!r) return null;
  const a = -Math.PI / 2 + i * (Math.PI * 2) / 20;   // canvas angle
  return [Math.cos(a) * r, -Math.sin(a) * r];
}

function wedge(c, cx, cy, r0, r1, a0, a1) {
  c.beginPath();
  c.arc(cx, cy, r1, a0, a1, false);
  c.arc(cx, cy, r0, a1, a0, true);
  c.closePath();
}

// The drawn board at any scale and centre -- the full board on Play, a
// 50 mm patch of it in the agreement plot. `ppm` is pixels per mm.
function drawDiagram(c, cx, cy, ppm, opts) {
  opts = opts || {};
  const K = INKS[opts.ink || 'show'];
  const mm = (v) => v * ppm;
  const lw = opts.wire || 1;
  if (opts.surround !== false) {
    c.beginPath();
    c.arc(cx, cy, mm(BOARD_MM.doubleOuter * 1.24), 0, Math.PI * 2);
    c.fillStyle = K.surround;
    c.fill();
  }
  for (let i = 0; i < 20; i++) {
    const a = sectorAngles(i);
    const dark = i % 2 === 0;
    c.fillStyle = dark ? K.dark : K.light;
    wedge(c, cx, cy, mm(BOARD_MM.outerBull), mm(BOARD_MM.trebleInner), a[0], a[1]); c.fill();
    wedge(c, cx, cy, mm(BOARD_MM.trebleOuter), mm(BOARD_MM.doubleInner), a[0], a[1]); c.fill();
    c.fillStyle = dark ? K.red : K.green;
    wedge(c, cx, cy, mm(BOARD_MM.trebleInner), mm(BOARD_MM.trebleOuter), a[0], a[1]); c.fill();
    wedge(c, cx, cy, mm(BOARD_MM.doubleInner), mm(BOARD_MM.doubleOuter), a[0], a[1]); c.fill();
  }
  if (opts.highlight) {
    const hl = opts.highlight;
    c.save();
    c.fillStyle = hl.color || 'rgba(242,230,201,0.55)';
    if (hl.ring === 'bull' || hl.ring === 'outer_bull') {
      c.beginPath();
      c.arc(cx, cy, mm(hl.ring === 'bull' ? BOARD_MM.bull : BOARD_MM.outerBull), 0, Math.PI * 2);
      c.fill();
    } else if (hl.ring === 'outside') {
      c.beginPath();
      c.arc(cx, cy, mm(BOARD_MM.doubleOuter * 1.24), 0, Math.PI * 2);
      c.arc(cx, cy, mm(BOARD_MM.doubleOuter), 0, Math.PI * 2, true);
      c.fill();
    } else {
      const i = BOARD_SECTORS.indexOf(String(hl.sector));
      const band = {
        single_inner: [BOARD_MM.outerBull, BOARD_MM.trebleInner], treble: [BOARD_MM.trebleInner, BOARD_MM.trebleOuter],
        single_outer: [BOARD_MM.trebleOuter, BOARD_MM.doubleInner], double: [BOARD_MM.doubleInner, BOARD_MM.doubleOuter],
      }[hl.ring];
      if (i >= 0 && band) {
        const a = sectorAngles(i);
        wedge(c, cx, cy, mm(band[0]), mm(band[1]), a[0], a[1]);
        c.fill();
      }
    }
    c.restore();
  }
  c.strokeStyle = K.wire;
  c.lineWidth = lw;
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
  c.fillStyle = opts.highlight && opts.highlight.ring === 'outer_bull' ? (opts.highlight.color || '#3a9a6a') : K.outer;
  c.fill(); c.stroke();
  c.beginPath(); c.arc(cx, cy, mm(BOARD_MM.bull), 0, Math.PI * 2);
  c.fillStyle = opts.highlight && opts.highlight.ring === 'bull' ? (opts.highlight.color || '#d5574d') : K.bull;
  c.fill(); c.stroke();
  if (opts.numbers !== false) {
    c.font = '600 ' + Math.max(10, Math.round(mm(13))) + 'px ' + (opts.ink && opts.ink !== 'show' ? '"IBM Plex Mono", monospace' : '"Barlow Condensed", sans-serif');
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillStyle = K.numbers;
    BOARD_SECTORS.forEach((n, i) => {
      const a = sectorAngles(i);
      const mid = (a[0] + a[1]) / 2;
      c.fillText(String(n), cx + Math.cos(mid) * mm(BOARD_MM.doubleOuter * 1.12), cy + Math.sin(mid) * mm(BOARD_MM.doubleOuter * 1.12));
    });
  }
}

// ---- darts, seen from the oche (ported unchanged) --------------------------

function ochePoint(P, eye) {
  const k = eye[2] / (eye[2] - P[2]);
  return [eye[0] + (P[0] - eye[0]) * k, eye[1] + (P[1] - eye[1]) * k, k];
}
function norm3(v) { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; }
function cross3(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

// A steel-tip dart as seen from `eye`, in board mm. Pure, so it can be
// tested without a canvas. `axis` is the shaft direction out of the board;
// a missing one means straight in.
function dartSilhouette(tipXY, axis, rollDeg, eye, lean) {
  let a = axis && axis.length === 3 ? axis.slice() : [0, 0, 1];
  a = norm3([a[0] * lean, a[1] * lean, a[2]]);
  const ref = Math.abs(a[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  let e1 = norm3(cross3(a, ref));
  let e2 = cross3(a, e1);
  const r = rollDeg * Math.PI / 180;
  [e1, e2] = [
    [0, 1, 2].map((i) => Math.cos(r) * e1[i] + Math.sin(r) * e2[i]),
    [0, 1, 2].map((i) => -Math.sin(r) * e1[i] + Math.cos(r) * e2[i]),
  ];
  const at = (s, e, w) => [
    tipXY[0] + a[0] * s + (e ? e[0] * w : 0),
    tipXY[1] + a[1] * s + (e ? e[1] * w : 0),
    a[2] * s + (e ? e[2] * w : 0),
  ];
  const seg = (s0, s1, wmm, part) => {
    const p0 = ochePoint(at(s0), eye), p1 = ochePoint(at(s1), eye);
    return {part, a: [p0[0], p0[1]], b: [p1[0], p1[1]], w: wmm * (p0[2] + p1[2]) / 2};
  };
  const vanes = [e1, e1.map((v) => -v), e2, e2.map((v) => -v)].map((e) => {
    const pts3 = [at(84), at(122), at(121, e, 28), at(110, e, 28), at(90, e, 5)];
    return {
      depth: pts3.reduce((acc, q) => acc + q[2], 0) / pts3.length,
      pts: pts3.map((q) => ochePoint(q, eye).slice(0, 2)),
    };
  }).sort((u, v) => u.depth - v.depth);
  const light = [-0.35, 0.6, 1.0];
  const drop = (P) => [P[0] - light[0] * P[2] / light[2], P[1] - light[1] * P[2] / light[2]];
  const shadow = [[0, 12, 2], [12, 58, 6.5], [58, 95, 4], [84, 118, 12]].map(([s0, s1, w]) => ({
    a: drop(at(s0)), b: drop(at(s1)), w,
  }));
  return {
    shaft: [seg(0, 14, 2.6, 'point-edge'), seg(12, 58, 8.0, 'barrel-edge'), seg(58, 95, 5.2, 'shaft-edge'),
            seg(0, 14, 1.6, 'point'), seg(12, 58, 6.5, 'barrel'), seg(14, 56, 1.4, 'highlight'),
            seg(58, 95, 4.0, 'shaft')],
    vanes,
    cap: seg(95, 112, 2.5, 'shaft'),
    shadow,
    depth: Math.hypot(tipXY[0] - eye[0], tipXY[1] - eye[1], eye[2]),
  };
}

const DART_PART_COLORS = {
  'point-edge': '#282828', 'barrel-edge': '#191919', 'shaft-edge': '#141414',
  point: '#9b9696', barrel: '#443e3e', highlight: '#968c8c', shaft: '#302d2d',
};

function shadeHex(hex, f) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || '');
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  const ch = (sh) => Math.max(0, Math.min(255, Math.round(((n >> sh) & 255) * f)));
  return '#' + [16, 8, 0].map((sh) => ch(sh).toString(16).padStart(2, '0')).join('');
}

function drawDartsFromOche(c, cx, cy, ppm, throws) {
  const X = (x) => cx + x * ppm;
  const Y = (y) => cy - y * ppm;
  const darts = [];
  throws.forEach((t, idx) => {
    if (!t || !t.board_xy_mm) return;
    const sil = dartSilhouette(t.board_xy_mm, t.dart_axis, 45 + 17 * idx, OCHE_EYE, DART_LEAN_EXAGGERATION);
    darts.push({sil, flight: t.flight_color || DEFAULT_FLIGHT_COLOR});
  });
  darts.sort((u, v) => v.sil.depth - u.sil.depth);   // farthest first
  c.lineCap = 'round';
  c.lineJoin = 'round';
  for (const {sil, flight} of darts) {
    c.save();
    if ('filter' in c) c.filter = 'blur(' + Math.max(1, 2.5 * ppm) + 'px)';
    c.strokeStyle = 'rgba(0,0,0,0.32)';
    for (const sh of sil.shadow) {
      c.lineWidth = Math.max(1, sh.w * ppm);
      c.beginPath(); c.moveTo(X(sh.a[0]), Y(sh.a[1])); c.lineTo(X(sh.b[0]), Y(sh.b[1])); c.stroke();
    }
    c.restore();
    for (const sg of sil.shaft) {
      c.strokeStyle = DART_PART_COLORS[sg.part];
      c.lineWidth = Math.max(1, sg.w * ppm);
      c.beginPath(); c.moveTo(X(sg.a[0]), Y(sg.a[1])); c.lineTo(X(sg.b[0]), Y(sg.b[1])); c.stroke();
    }
    c.fillStyle = flight;
    c.strokeStyle = shadeHex(flight, 0.55);
    c.lineWidth = Math.max(1, 0.8 * ppm);
    for (const vane of sil.vanes) {
      c.beginPath();
      vane.pts.forEach((pt, i) => (i ? c.lineTo(X(pt[0]), Y(pt[1])) : c.moveTo(X(pt[0]), Y(pt[1]))));
      c.closePath(); c.fill(); c.stroke();
    }
    const cap = sil.cap;
    c.strokeStyle = DART_PART_COLORS.shaft;
    c.lineWidth = Math.max(1, cap.w * ppm);
    c.beginPath(); c.moveTo(X(cap.a[0]), Y(cap.a[1])); c.lineTo(X(cap.b[0]), Y(cap.b[1])); c.stroke();
  }
}

// Size a canvas to its box at device resolution; returns the CSS size, or 0
// while the box is hidden (0x0) -- the caller simply draws nothing then, and
// the ResizeObserver brings it back when the box has a size again.
function fitCanvas(canvas, square) {
  const box = canvas.parentElement.getBoundingClientRect();
  if (box.width < 2 || box.height < 2) return null;
  const dpr = window.devicePixelRatio || 1;
  const w = square ? Math.min(box.width, box.height) : box.width;
  const hh = square ? w : box.height;
  const W = Math.floor(w), H = Math.floor(hh);
  if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
  }
  const c = canvas.getContext('2d');
  if (!c) return null;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {c, w: W, h: H};
}

// ---- the live board ---------------------------------------------------------

const LiveBoard = (() => {
  const canvas = document.getElementById('board');
  let view = prefs.get('boardView', 'photo') === 'diagram' ? 'diagram' : 'photo';
  let photo = null, photoVersion = null;
  let arrival = null;            // {index, t0} for the landing pulse
  // The board as it stands -- photo, darts, pins -- drawn once per change
  // into an offscreen canvas. The landing pulse repaints ~60 times a second
  // for most of a second, and redrawing the photo and the darts' blurred
  // shadows on every one of those frames tied up a modest TV's browser for
  // about as long as the spoken call.
  let base = null, baseKey = '';

  const photoMode = () => view === 'photo' && !!photo;
  const ppmFor = (w) => (photoMode() ? (w / 2) / PHOTO_VIEW_MM : (w / 2) / (BOARD_MM.doubleOuter * 1.24));

  function drawBase(c, w, throws) {
    const cx = w / 2, cy = w / 2, ppm = ppmFor(w);
    if (photoMode()) {
      c.save();
      c.beginPath();
      c.arc(cx, cy, (PHOTO_VIEW_MM - 1) * ppm, 0, Math.PI * 2);
      c.clip();
      const span = PHOTO_HALF_MM * 2 * ppm;
      c.drawImage(photo, cx - span / 2, cy - span / 2, span, span);
      c.restore();
      drawDartsFromOche(c, cx, cy, ppm, throws);
      drawRing(c, cx, cy, (PHOTO_VIEW_MM - 1) * ppm);
      drawPins(c, cx, cy, ppm, throws, w);
      return;
    }
    drawDiagram(c, cx, cy, ppm, {wire: Math.max(1, w / 700)});
    drawRing(c, cx, cy, BOARD_MM.doubleOuter * 1.24 * ppm - 1);
    drawPins(c, cx, cy, ppm, throws, w);
  }

  function draw() {
    if (!canvas) return;
    const fit = fitCanvas(canvas, true);
    if (!fit) return;
    const {c, w} = fit;
    const throws = S.visit.throws || [];
    const dpr = window.devicePixelRatio || 1;
    const key = [w, dpr, view, photoMode() ? photoVersion : '',
      JSON.stringify(throws.map((t) => t && [t.board_xy_mm, t.dart_axis, t.flight_color]))].join('|');
    if (!base || key !== baseKey) {
      base = document.createElement('canvas');
      base.width = Math.round(w * dpr); base.height = Math.round(w * dpr);
      const bc = base.getContext('2d');
      if (!bc) { base = null; return; }
      bc.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawBase(bc, w, throws);
      baseKey = key;
    }
    c.clearRect(0, 0, w, w);
    c.drawImage(base, 0, 0, w, w);
    drawArrival(c, w / 2, w / 2, ppmFor(w), throws);
  }

  // The volt rim: the board is the lit stage of the show.
  function drawRing(c, cx, cy, r) {
    c.save();
    c.strokeStyle = 'rgba(255,210,31,0.9)'; c.lineWidth = 3;
    c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.stroke();
    c.strokeStyle = 'rgba(255,210,31,0.08)'; c.lineWidth = 12;
    c.beginPath(); c.arc(cx, cy, r + 7, 0, Math.PI * 2); c.stroke();
    c.restore();
  }

  // Each dart's number, pinned at its tip, in its colour: the thread from
  // the board to its slab. The tip is on the board plane, so the pin sits
  // exactly where the dart went in, in both views.
  function drawPins(c, cx, cy, ppm, throws, w) {
    const r = Math.max(11, w * 0.019);
    c.save();
    c.font = '800 ' + Math.round(r * 1.15) + 'px "Barlow Condensed", sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    const placed = [];
    throws.forEach((t, idx) => {
      if (!t || !t.board_xy_mm) return;
      const tx = cx + t.board_xy_mm[0] * ppm, ty = cy - t.board_xy_mm[1] * ppm;
      // Darts grouped in one bed would stack their pins and hide the earlier
      // number: step a pin outward from the centre until it is clear, and
      // run a leader back to the tip it belongs to.
      let x = tx, y = ty;
      const ang = Math.atan2(ty - cy, tx - cx) || 0;
      for (let step = 1; step < 12 && placed.some((p) => Math.hypot(p[0] - x, p[1] - y) < r * 2.1); step++) {
        const a = ang + (step % 2 ? 1 : -1) * 0.9;
        x = tx + Math.cos(a) * r * 1.2 * step; y = ty + Math.sin(a) * r * 1.2 * step;
      }
      placed.push([x, y]);
      const col = DART_COLORS[idx % 3];
      if (x !== tx || y !== ty) {
        c.strokeStyle = col; c.lineWidth = 2;
        c.beginPath(); c.moveTo(tx, ty); c.lineTo(x, y); c.stroke();
        c.fillStyle = col; c.beginPath(); c.arc(tx, ty, 3, 0, Math.PI * 2); c.fill();
      }
      c.shadowColor = col; c.shadowBlur = 14;
      c.fillStyle = col;
      c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill();
      c.shadowBlur = 0;
      c.strokeStyle = '#06080f'; c.lineWidth = 3; c.stroke();
      c.fillStyle = '#06080f';
      c.fillText(String(idx + 1), x, y + 1);
    });
    c.restore();
  }

  // One expanding ring where the newest dart landed -- the only motion on
  // the Play screen, so from across the room it reads as "that one".
  function drawArrival(c, cx, cy, ppm, throws) {
    if (!arrival) return;
    const t = throws[arrival.index];
    const k = (performance.now() - arrival.t0) / 900;
    if (!t || !t.board_xy_mm || k >= 1) { arrival = null; return; }
    const px = cx + t.board_xy_mm[0] * ppm, py = cy - t.board_xy_mm[1] * ppm;
    c.save();
    c.globalAlpha = (1 - k) * 0.9;
    c.strokeStyle = DART_COLORS[arrival.index % 3];
    c.lineWidth = 3;
    c.beginPath(); c.arc(px, py, 6 + k * 46, 0, Math.PI * 2); c.stroke();
    c.restore();
    requestAnimationFrame(draw);
  }

  function landed(index) {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    arrival = {index, t0: performance.now()};
    requestAnimationFrame(draw);
  }

  function setPhotoVersion(version) {
    if (!version || version === photoVersion) return;
    photoVersion = version;
    const img = new Image();
    img.onload = () => {
      if (photoVersion !== version) return;   // a newer one overtook it
      photo = img;
      draw();
      emit('boardphoto');
    };
    img.onerror = () => console.warn('board photo', version, 'failed to load');
    img.src = '/api/board/photo?v=' + encodeURIComponent(version);
  }

  function setView(v) {
    view = v === 'diagram' ? 'diagram' : 'photo';
    prefs.set('boardView', view);
    draw();
    emit('boardview', view);
  }

  if (canvas && window.ResizeObserver) new ResizeObserver(() => draw()).observe(canvas.parentElement);
  // The pins' numbers are drawn in the embedded face; redraw once it is in.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { base = null; draw(); });
  return {draw, landed, setPhotoVersion, setView, get view() { return view; }, get hasPhoto() { return !!photo; }};
})();

// ---- the agreement plot -------------------------------------------------------
// A throw's disagreement happens ON the board, so it is shown there: a
// patch of the real geometry around the dart, with every engine's point,
// the oracle's, and the confirmed truth. A wire dispute is obvious at a
// glance where a column of millimetres is not.

const ENGINE_MARK = {Apollo: 'Ap', Talos: 'Ta', Athena: 'At', Ares: 'Ar', Zeus: 'Ze'};

function drawAgreement(canvas, pkg, sections) {
  const fit = fitCanvas(canvas, false);
  if (!fit) return;
  const {c, w: W, h: H} = fit;
  c.clearRect(0, 0, W, H);
  const INK = cssVar('--ink', '#15140f'), INK2 = cssVar('--ink2', '#56534a'), INK3 = cssVar('--ink3', '#8b8676');
  const SIGNAL = cssVar('--signal', '#ff4f00'), WARN = cssVar('--warn', '#b86e00'), OK = cssVar('--ok', '#1d7a4b');
  const pts = [];
  for (const s of sections) {
    if (s.board_xy_mm && s.ok !== false && !s.timed_out) pts.push({kind: s.primary ? 'primary' : 'engine', name: s.name, xy: s.board_xy_mm, call: s});
  }
  if (pkg.ad_tip_xy_mm && pkg.ad_matched) pts.push({kind: 'oracle', name: 'Autodarts', xy: pkg.ad_tip_xy_mm, call: {sector: pkg.ad_sector, ring: pkg.ad_ring}});
  // Type grows with the figure: this page is often read from across the room.
  const k = Math.max(1, Math.min(1.9, W / 480));
  const CARD = cssVar('--card', '#f6f2ea');
  if (!pts.length) {
    c.fillStyle = INK2; c.font = Math.round(13 * k) + 'px "IBM Plex Mono", monospace'; c.textAlign = 'center';
    c.fillText('NO ENGINE PLACED THIS DART ON THE BOARD', W / 2, H / 2);
    return;
  }
  const anchor = (pts.find((p) => p.kind === 'primary') || pts[0]).xy;
  const spread = Math.max(0, ...pts.map((p) => Math.hypot(p.xy[0] - anchor[0], p.xy[1] - anchor[1])));
  const legendW = Math.min(210 * k, W * 0.36);
  const plotW = W - legendW;
  const half = Math.min(70, Math.max(9, spread * 2.4 + 5));      // mm either side
  const ppm = Math.min(plotW, H) / (2 * half);
  const cx = plotW / 2 - anchor[0] * ppm, cy = H / 2 + anchor[1] * ppm;
  c.save();
  c.beginPath(); c.rect(0, 0, W, H); c.clip();
  drawDiagram(c, cx, cy, ppm, {ink: labInk(), numbers: false, wire: Math.max(1.4, Math.min(3, ppm * 0.12))});
  const X = (xy) => cx + xy[0] * ppm, Y = (xy) => cy - xy[1] * ppm;
  // the confirmed answer: its segment's centre, ringed
  if (pkg.ad_operator_confirmed_ring) {
    const ctr = segmentCentre(pkg.ad_operator_confirmed_sector, pkg.ad_operator_confirmed_ring);
    if (ctr) {
      c.strokeStyle = OK; c.lineWidth = 2.5; c.setLineDash([5, 4]);
      c.beginPath(); c.arc(X(ctr), Y(ctr), 12 * k, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
    }
  }
  // callouts: every mark gets a leader to its own line in the margin legend
  const order = {primary: 0, engine: 1, oracle: 2};
  pts.sort((a, b) => order[a.kind] - order[b.kind]);
  // The legend sits on its own card so it reads over any part of the board.
  c.fillStyle = CARD; c.globalAlpha = 0.92; c.fillRect(plotW + 8, 0, W - plotW - 8, H); c.globalAlpha = 1;
  const lx = plotW + 22, rowH = Math.min(34 * k, (H - 30) / pts.length), top = 8 + rowH / 2;
  c.textBaseline = 'middle';
  pts.forEach((p, i) => {
    const x = X(p.xy), y = Y(p.xy), ly = top + i * rowH;
    const col = p.kind === 'primary' ? SIGNAL : p.kind === 'oracle' ? WARN : INK;
    c.strokeStyle = INK3; c.lineWidth = 1;
    c.beginPath(); c.moveTo(x, y); c.lineTo(lx - 26, ly); c.lineTo(lx - 6, ly); c.stroke();
    if (p.kind === 'oracle') {
      c.strokeStyle = col; c.lineWidth = 2.5;
      const d = 8 * k;
      c.beginPath(); c.moveTo(x, y - d); c.lineTo(x + d, y); c.lineTo(x, y + d); c.lineTo(x - d, y); c.closePath(); c.stroke();
    } else {
      c.fillStyle = col; c.beginPath(); c.arc(x, y, (p.kind === 'primary' ? 7 : 4.5) * k, 0, Math.PI * 2); c.fill();
    }
    c.fillStyle = col; c.textAlign = 'left';
    c.font = '600 ' + Math.round(13 * k) + 'px "IBM Plex Mono", monospace';
    c.fillText(p.name.toUpperCase(), lx, ly);
    c.fillStyle = INK; c.textAlign = 'right';
    c.font = '700 ' + Math.round(20 * k) + 'px "IBM Plex Sans Condensed", sans-serif';
    c.fillText(callLabel(p.call.sector, p.call.ring), W - 6, ly);
  });
  // scale bar
  const bar = 5 * ppm, by = H - 18;
  c.strokeStyle = INK; c.lineWidth = 2;
  c.beginPath(); c.moveTo(14, by); c.lineTo(14 + bar, by); c.moveTo(14, by - 6); c.lineTo(14, by + 6); c.moveTo(14 + bar, by - 6); c.lineTo(14 + bar, by + 6); c.stroke();
  c.fillStyle = INK; c.textAlign = 'left'; c.font = '600 ' + Math.round(12 * k) + 'px "IBM Plex Mono", monospace';
  c.fillText('5 MM', 22 + bar, by);
  c.restore();
}

// ---- the truth picker ---------------------------------------------------------
// Tap where the dart actually is. Most corrections are "it was the other
// one", so candidate chips come first (see truth sheet); the board is for
// everything else, and the ring chips fix a finger that landed on a wire.

function makePicker(canvas, onPick) {
  let selected = null, hover = null, marks = [];
  function geometry() {
    const fit = fitCanvas(canvas, true);
    if (!fit) return null;
    const ppm = (fit.w / 2) / (BOARD_MM.doubleOuter * 1.24);
    return Object.assign(fit, {ppm, cx: fit.w / 2, cy: fit.w / 2});
  }
  function draw() {
    const g = geometry();
    if (!g) return;
    const {c, w, ppm, cx, cy} = g;
    c.clearRect(0, 0, w, w);
    drawDiagram(c, cx, cy, ppm, {
      ink: labInk(), wire: 1,
      highlight: selected ? Object.assign({color: 'rgba(255,79,0,0.6)'}, selected)
        : hover ? Object.assign({color: 'rgba(255,79,0,0.22)'}, hover) : null,
    });
    const INK = cssVar('--ink', '#15140f'), WARN = cssVar('--warn', '#b86e00');
    for (const m of marks) {
      if (!m.xy) continue;
      const x = cx + m.xy[0] * ppm, y = cy - m.xy[1] * ppm;
      if (m.oracle) {
        c.strokeStyle = WARN; c.lineWidth = 2.5;
        c.beginPath(); c.moveTo(x, y - 7); c.lineTo(x + 7, y); c.lineTo(x, y + 7); c.lineTo(x - 7, y); c.closePath(); c.stroke();
      } else {
        c.fillStyle = INK; c.beginPath(); c.arc(x, y, m.primary ? 5.5 : 3.5, 0, Math.PI * 2); c.fill();
      }
    }
  }
  function locate(ev) {
    const g = geometry();
    if (!g) return null;
    const r = canvas.getBoundingClientRect();
    const x = (ev.clientX - r.left - g.cx) / g.ppm, y = -(ev.clientY - r.top - g.cy) / g.ppm;
    return hitTest(x, y);
  }
  canvas.addEventListener('pointermove', (ev) => {
    if (ev.pointerType !== 'mouse') return;
    const hit = locate(ev);
    if (!sameCall(hit, hover)) { hover = hit; draw(); }
  });
  canvas.addEventListener('pointerleave', () => { hover = null; draw(); });
  canvas.addEventListener('click', (ev) => {
    const hit = locate(ev);
    if (hit) onPick(hit);
  });
  if (window.ResizeObserver) new ResizeObserver(draw).observe(canvas.parentElement);
  return {
    select(call) { selected = call && call.ring ? call : null; draw(); },
    setMarks(list) { marks = list || []; draw(); },
    draw,
  };
}
