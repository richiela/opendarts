// ---------------------------------------------------------------------------
// throws -- every saved throw: scan the list, study one.
//
// "Truth" for a throw is the answer a person confirmed where there is one,
// otherwise Autodarts' (when it was asked). The server grades every engine
// against exactly that (sector_match), and so does this page.
// ---------------------------------------------------------------------------

const Throws = (() => {
  const PAGE = 150;
  let filter = 'all';
  let sinceMs = null;          // "Clear view": visual only, resets on reload
  let selected = null;         // package path
  // A screen left open on Throws follows the newest dart into the detail
  // pane until someone picks another row; "Follow newest" hands it back.
  let following = true;
  let agreeObserver = null;
  let newest = null;           // path of the top row under the current filter
  let lastTop = null;          // ...as of the last render: a new one slides in
  let shown = PAGE;
  let stale = true;
  let kind = 'after';          // which frame the detail's camera strip shows
  let resyncing = false;
  // Engines is the default: side-by-side calls are what this page is kept open for.
  let mode = prefs.get('throwsMode', 'matrix') === 'list' ? 'list' : 'matrix';

  // One throw's engines in one shape. The primary's answer lives at the top
  // level of the record, the voters under p.engines (docs/ENGINES.md).
  function sections(p) {
    const out = [];
    if (p.primary_engine) {
      out.push({name: p.primary_engine, primary: true, ok: p.ok, sector: p.sector, ring: p.ring,
        board_xy_mm: p.board_xy_mm, reason: p.reason, timed_out: false,
        sector_match: p.sector_match, tip_distance_mm: p.tip_distance_mm});
    }
    const order = engineOrder();
    const names = Object.keys(p.engines || {});
    if (order) names.sort((a, b) => (order[a] === undefined ? 99 : order[a]) - (order[b] === undefined ? 99 : order[b]));
    for (const name of names) {
      const s = p.engines[name];
      out.push({name, ok: s.ok, sector: s.sector, ring: s.ring, board_xy_mm: s.board_xy_mm, reason: s.reason,
        timed_out: s.timed_out, sector_match: s.sector_match, tip_distance_mm: s.tip_distance_mm});
    }
    return out;
  }

  // Voters in the configured order, so every row reads the same way.
  function engineOrder() {
    const cfg = S.engineConfig;
    if (!cfg) return null;
    const names = [cfg.primary].concat(cfg.also_run || [], cfg.available_engines || []);
    const rank = {};
    names.forEach((n, i) => { if (n && rank[n] === undefined) rank[n] = i; });
    return rank;
  }

  function truthOf(p) {
    if (p.ad_operator_confirmed_ring) return {sector: p.ad_operator_confirmed_sector, ring: p.ad_operator_confirmed_ring, who: 'You'};
    if (p.ad_matched && p.ad_ring) return {sector: p.ad_sector, ring: p.ad_ring, who: 'Autodarts'};
    return null;
  }
  const noRead = (p) => p.ok === false || !p.ring;
  const ours = (p) => ({sector: p.sector, ring: p.ring});

  // Autodarts is an oracle, not the truth: a disagreement with it is a
  // DISPUTE until a person says what landed. Only then is our call right
  // or wrong -- and the list must not paint an unresolved dispute as our
  // mistake, when the oracle may be the one that missed.
  function verdict(p) {
    const truth = truthOf(p);
    if (noRead(p)) return {key: 'noread', truth};
    if (!truth) return {key: 'unknown', truth};
    if (sameCall(ours(p), truth)) return {key: 'right', truth};
    return {key: truth.who === 'You' ? 'wrong' : 'disputed', truth};
  }

  // What the truth sheet offers first: the oracle, then the engines.
  function candidatesFor(p) {
    const out = [];
    for (const s of sections(p)) {
      if (s.ok === false || s.timed_out || !s.ring) continue;
      out.push({sector: s.sector, ring: s.ring, who: s.name, source: s.name, xy: s.board_xy_mm, primary: !!s.primary});
    }
    if (p.ad_matched && p.ad_ring) out.splice(1, 0, {sector: p.ad_sector, ring: p.ad_ring, who: 'Autodarts', source: 'Autodarts', xy: p.ad_tip_xy_mm});
    return out;
  }

  function all() {
    return Array.from(S.packages.values())
      .sort((a, b) => (b.captured_at_utc || '').localeCompare(a.captured_at_utc || ''));
  }
  function visible(list) {
    let v = list;
    if (sinceMs !== null) v = v.filter((p) => p.captured_at_utc && Date.parse(p.captured_at_utc) >= sinceMs);
    return v;
  }
  function passes(p, k) {
    if (k === 'all') return true;
    if (k === 'corrected') return !!p.ad_operator_confirmed_ring;
    if (k === 'noread') return noRead(p);
    if (k === 'disagree') { const v = verdict(p).key; return v === 'wrong' || v === 'disputed'; }
    if (k === 'split') {
      return sections(p).some((s) => !s.primary && s.ok !== false && !s.timed_out && s.ring && !sameCall(s, ours(p)));
    }
    return true;
  }

  // ---- tally ------------------------------------------------------------------
  function tally(list) {
    const order = [], counts = new Map();
    const bump = (name, right) => {
      if (!counts.has(name)) { order.push(name); counts.set(name, {right: 0, total: 0}); }
      const c = counts.get(name);
      c.total += 1;
      if (right) c.right += 1;
    };
    for (const p of list) {
      for (const s of sections(p)) {
        if (s.sector_match === null || s.sector_match === undefined) continue;
        bump(s.name, s.sector_match === true);
      }
      const asked = p.ad_matched === true || p.ad_matched === false;
      if (asked && (p.ad_matched !== false || p.ad_operator_confirmed_ring)) bump('Autodarts', !p.ad_operator_marked_wrong);
    }
    return order.map((name) => Object.assign({name}, counts.get(name)));
  }

  // One gauge per engine, the oracle last: accuracy against the known
  // answer, the primary's bar in signal orange.
  function renderTally(list) {
    const el = $('#tally');
    const rows = tally(list);
    if (!rows.length) {
      el.replaceChildren(h('p', {class: 'muted'}, 'Accuracy appears once throws are graded — against Autodarts, or against what a person recorded.'));
      return;
    }
    const primary = S.engineConfig && S.engineConfig.primary;
    el.replaceChildren(...rows.map((t) => {
      const pct = t.total ? (100 * t.right) / t.total : 0;
      const role = t.name === primary ? 'primary' : t.name === 'Autodarts' ? 'oracle' : '';
      return h('div', {class: 'gauge' + (role === 'primary' ? ' primary' : ''), title: t.right + ' of ' + t.total + ' graded throws right'},
        h('div', {class: 'nm'}, h('span', {}, t.name), role ? h('em', {}, role) : null),
        h('div', {class: 'pc'}, pct.toFixed(1) + '%'),
        h('div', {class: 'br'}, h('i', {style: 'width:' + pct.toFixed(1) + '%'})),
        h('div', {class: 'ct'}, t.right + ' / ' + t.total + (role === 'primary' ? ' right' : '')));
    }));
  }

  // ---- list ---------------------------------------------------------------------
  const LAB_DART = ['var(--l-d1)', 'var(--l-d2)', 'var(--l-d3)'];
  const dartNum = (p) => h('span', {class: 'num', style: '--c:' + LAB_DART[(p.visit_index || 0) % 3]}, String((p.visit_index || 0) + 1));

  function pip(state, name, call) {
    const title = name + ': ' + (state === 'none' ? 'no result' : callLabel(call.sector, call.ring) + ' (' + ringWord(call.ring) + ')');
    return h('span', {class: 'pip' + (state === 'agree' ? '' : ' ' + state), title});
  }

  // The engine columns of the matrix, in configured order.
  function engineColumns(list) {
    const cfg = S.engineConfig;
    if (cfg && cfg.primary) return [cfg.primary].concat(cfg.also_run || []);
    const first = list.find((p) => p.primary_engine);
    return first ? [first.primary_engine].concat(Object.keys(first.engines || {})) : [];
  }

  function shownCall(p) {
    return p.ad_operator_confirmed_ring ? {sector: p.ad_operator_confirmed_sector, ring: p.ad_operator_confirmed_ring} : ours(p);
  }

  function callCell(p) {
    const call = shownCall(p);
    return h('span', {class: 'call'},
      noRead(p) && !p.ad_operator_confirmed_ring ? h('span', {class: 'noread'}, 'NO READ') : callLabel(call.sector, call.ring),
      p.ad_operator_confirmed_ring ? h('span', {class: 'fix', title: 'Recorded by a person — the engines scored ' + callLabel(p.sector, p.ring)}, '✎') : null);
  }

  function listRow(p, number) {
    const v = verdict(p);
    const call = shownCall(p);
    const pts = noRead(p) && !p.ad_operator_confirmed_ring ? '' : points(call.sector, call.ring);
    const voters = sections(p).filter((s) => !s.primary);
    const oracle = !p.ad_matched ? h('span', {class: 'ctr o-none', title: 'Autodarts was not asked, or did not answer'}, '—')
      : sameCall(ours(p), {sector: p.ad_sector, ring: p.ad_ring})
        ? h('span', {class: 'ctr o-ok', title: 'Autodarts agrees'}, '✓')
        : h('span', {class: 'ctr o-diff', title: 'Autodarts said ' + callLabel(p.ad_sector, p.ad_ring)}, callLabel(p.ad_sector, p.ad_ring));
    const tip = p.tip_distance_mm === null || p.tip_distance_mm === undefined ? '' : p.tip_distance_mm.toFixed(1) + ' mm';
    return h('button', {type: 'button', class: 'trow v-' + v.key + (passes(p, 'split') ? ' split' : '') + (p.path === selected ? ' sel' : ''), dataset: {path: p.path}, onclick: () => select(p.path, true)},
      h('span', {class: 'ix'}, String(number)), dartNum(p), callCell(p),
      h('span', {class: 'r'}, pts === null ? '' : String(pts)),
      h('span', {'aria-label': 'engine agreement'}, voters.map((s) => pip(
        s.ok === false || s.timed_out || !s.ring ? 'none' : sameCall(s, ours(p)) ? 'agree' : 'differ', s.name, s))),
      oracle,
      h('span', {class: 'r dim'}, tip),
      h('span', {class: 'r dim'}, clock(p.captured_at_utc)));
  }

  // Every engine's call, side by side -- the view for studying the engines.
  // Each cell is marked against the known answer: right, wrong (struck
  // through), or unknown when nothing has been graded.
  function matrixRow(p, number, cols) {
    const v = verdict(p);
    const truth = truthOf(p);
    const secs = new Map(sections(p).map((s) => [s.name, s]));
    const cells = cols.map((name) => {
      const s = secs.get(name);
      if (!s) return p._pending ? h('span', {class: 'mcell none pending', title: name + ': on its way'}, '…') : h('span', {class: 'mcell none'}, '—');
      if (s.timed_out) return h('span', {class: 'mcell none', title: name + ': timed out'}, 'TIME');
      if (s.ok === false || !s.ring) return h('span', {class: 'mcell none', title: name + ': ' + (s.reason || 'no result')}, '×');
      const cls = s.sector_match === true ? 'right' : s.sector_match === false ? 'wrong' : 'unk';
      return h('span', {class: 'mcell ' + cls, title: name + ': ' + ringWord(s.ring)}, callLabel(s.sector, s.ring));
    });
    let ad;
    if (p._pending) ad = h('span', {class: 'mcell none pending', title: 'Autodarts: on its way'}, '…');
    else if (!p.ad_matched) ad = h('span', {class: 'mcell none'}, '—');
    else {
      const adCall = {sector: p.ad_sector, ring: p.ad_ring};
      const cls = truth && truth.who === 'You' ? (sameCall(adCall, truth) ? 'right' : 'wrong') : 'ad';
      ad = h('span', {class: 'mcell ' + cls, title: 'Autodarts' + (p.ad_latency_ms ? ' · ' + Math.round(p.ad_latency_ms) + ' ms' : '')}, callLabel(adCall.sector, adCall.ring));
    }
    return h('button', {type: 'button', class: 'trow v-' + v.key + (passes(p, 'split') ? ' split' : '') + (p.path === selected ? ' sel' : ''), dataset: {path: p.path}, onclick: () => select(p.path, true)},
      h('span', {class: 'ix'}, String(number)), dartNum(p), callCell(p), cells, ad, h('span', {class: 'r dim'}, clock(p.captured_at_utc, false)));
  }

  function listHead(mode, cols) {
    const c = (t, cls) => h('span', {class: cls || ''}, t);
    return mode === 'matrix'
      ? h('div', {class: 'thead'}, c('#'), c(''), c('Call'), cols.map((n) => c(n.slice(0, 6), 'ctr')), c('AD', 'ctr'), c('Time', 'r'))
      : h('div', {class: 'thead'}, c('#'), c(''), c('Call'), c('Pts', 'r'), c('Vote'), c('Oracle', 'ctr'), c('Δ tip', 'r'), c('Time', 'r'));
  }

  function render() {
    if (!Shell.showing('throws')) { stale = true; return; }
    stale = false;
    const list = all();
    const vis = visible(list);
    const hidden = list.length - vis.length;
    renderTally(vis);
    renderBanner();
    const visits = new Set(vis.map((p) => p.visit_id || p.path)).size;
    const oldest = vis.length ? vis[vis.length - 1].captured_at_utc : null;
    $('#throws-sub').textContent = vis.length
      ? plural(vis.length, 'throw') + ' · ' + plural(visits, 'visit') + (oldest ? ' · since ' + clock(oldest, false) : '')
      : 'nothing saved yet';
    const filtered = vis.filter((p) => passes(p, filter));
    newest = filtered.length ? filtered[0].path : null;
    // The top row is the live one: holding it is following, however it came
    // to be picked (a throw opened from Play can land before its row does).
    if (selected && selected === newest) following = true;
    if (following) selected = newest;
    const numberOf = new Map();
    vis.forEach((p, i) => numberOf.set(p.path, vis.length - i));
    $$('#throw-filters [data-filter]').forEach((b) => {
      const k = b.dataset.filter;
      b.setAttribute('aria-pressed', String(k === filter));
      const n = vis.filter((p) => passes(p, k)).length;
      $('.n', b).textContent = n ? String(n) : (k === 'all' ? '0' : '');
    });
    $$('#throw-density button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === mode)));
    $('#view-note').textContent = sinceMs === null ? ''
      : 'Showing throws since ' + clock(new Date(sinceMs), false) + (hidden ? ' — ' + plural(hidden, 'earlier throw') + ' hidden, still on disk' : '');
    $('#show-all').hidden = sinceMs === null;
    const box = $('#throw-list');
    box.className = 'throw-list ' + (mode === 'matrix' ? 'tl-matrix' : 'tl-list');
    const cols = engineColumns(list);
    box.style.setProperty('--eng', String(Math.max(1, cols.length)));
    if (!filtered.length) {
      box.replaceChildren(h('div', {class: 'empty'},
        h('strong', {}, list.length ? 'Nothing here' : 'No throws yet'),
        h('p', {}, list.length
          ? (filter === 'all' ? 'No throws since the view was cleared.' : 'No throws match this filter.')
          : (S.config && S.config.store_packages === false
            ? 'Saving throws is off (Rig › Recording), so there is nothing to review.'
            : 'Every scored dart is saved and appears here.'))));
      renderDetail();
      return;
    }
    const groups = [];
    let cur = null;
    for (const p of filtered.slice(0, shown)) {
      const key = p.visit_id || p.path;
      if (!cur || cur.key !== key) { cur = {key, items: []}; groups.push(cur); }
      cur.items.push(p);
    }
    const frag = document.createDocumentFragment();
    frag.append(listHead(mode, cols));
    for (const g of groups) {
      const first = g.items[g.items.length - 1];
      let tot = 0;
      for (const p of g.items) { const c = shownCall(p); tot += points(c.sector, c.ring) || 0; }
      const isVisit = g.key.startsWith('visit_');
      frag.append(h('div', {class: 'tgroup'},
        h('div', {class: 'tgroup-head'},
          h('span', {}, h('b', {}, isVisit ? 'Visit' : 'Throw'), ' · ' + clock(first.captured_at_utc, false) + ' · ' + ago(first.captured_at_utc)),
          isVisit ? h('b', {}, tot + ' pts') : null),
        g.items.map((p) => (mode === 'matrix' ? matrixRow(p, numberOf.get(p.path), cols) : listRow(p, numberOf.get(p.path))))));
    }
    if (filtered.length > shown) {
      frag.append(h('button', {type: 'button', class: 'btn more', onclick: () => { shown += PAGE; render(); }},
        'Show ' + Math.min(PAGE, filtered.length - shown) + ' more'));
    }
    box.replaceChildren(frag);
    const topRow = box.querySelector('.trow');
    if (topRow && lastTop !== null && newest !== lastTop) topRow.classList.add('arrive');
    lastTop = newest;
    if (selected && !S.packages.has(selected)) selected = null;
    renderDetail();
  }

  // Autodarts stuck in takeout is not scoring, and a rig that has stopped
  // saving has nothing new to review: both are said at the top of the page
  // where the missing rows would otherwise be the only sign.
  function renderBanner() {
    const el = $('#throws-banner');
    const b = S.adBoard || {};
    const guard = S.state && S.state.config && S.state.config.disk && S.state.config.disk.guard;
    let msg = null;
    if (S.config && S.config.ad_enabled && b.status === 'takeout' && typeof b.age === 'number' && b.age >= 30) {
      msg = ['Autodarts is stuck in takeout', Math.round(b.age) + ' s — it is not scoring. Clear the board, or reset it.'];
    } else if (guard && guard.below_floor) {
      msg = ['Saving has stopped', 'the disk is below its ' + (guard.floor_label || '') + ' floor — ' + (guard.free_label || '') + ' free.'];
    } else if (S.config && S.config.store_packages === false) {
      msg = ['Saving throws is off', 'new darts are scored but not kept for review. Rig › Recording.'];
    }
    el.hidden = !msg;
    if (msg) el.replaceChildren(h('b', {}, msg[0]), h('span', {}, msg[1]));
  }

  // ---- detail ---------------------------------------------------------------------
  function select(path, fromClick) {
    selected = path;
    // Measured now, not from the last render: Play can open a throw before
    // this page has ever drawn.
    const top = visible(all()).find((q) => passes(q, filter));
    following = !!top && path === top.path;
    $$('#throw-list .trow').forEach((r) => r.classList.toggle('sel', r.dataset.path === path));
    renderDetail();
    const p = S.packages.get(path);
    if (p) Shell.setHash('throws/' + encodeURIComponent(p.throw_id), true);
    if (fromClick && window.matchMedia('(max-width: 1100px)').matches) $('#throws').classList.add('detail-open');
  }

  // A throw's record lands a moment before its frames do (the clip is
  // written just after the package, and meta.json then points at it), so
  // the newest dart's stills can 404 for a few tens of ms -- up to a few
  // seconds when clips wait for Autodarts. Wait and ask again rather than
  // show a broken picture; give up quietly after ~12 s.
  function stillImage(p, cam) {
    const img = h('img', {src: frameUrl(p, cam, kind), alt: 'camera ' + (cam + 1) + ' ' + (kind === 'bg' ? 'before' : 'after'), loading: 'lazy'});
    let tries = 0;
    img.addEventListener('error', () => {
      const box = img.closest('.still');
      if (!img.isConnected) return;
      if (tries >= 6) { if (box) { box.classList.remove('waiting'); box.classList.add('missing'); } return; }
      tries += 1;
      if (box) box.classList.add('waiting');
      setTimeout(() => { if (img.isConnected) img.src = frameUrl(p, cam, kind) + '&try=' + tries; }, 400 * tries);
    });
    img.addEventListener('load', () => { const box = img.closest('.still'); if (box) box.classList.remove('waiting', 'missing'); });
    return img;
  }

  function frameUrl(p, cam, which) {
    return '/api/packages/' + encodeURIComponent(p.session) + '/' + encodeURIComponent(p.throw_id)
      + '/frame/' + cam + '/' + which + '.png?fmt=jpeg';
  }
  function viewerUrl(p) {
    return '/packages/' + encodeURIComponent(p.session) + '/' + encodeURIComponent(p.throw_id) + '/viewer';
  }

  // "Zeus, Apollo and Talos say 8. Athena, Ares and Autodarts say 16." --
  // who said what, in words, grouped by the call.
  function whoSaidWhat(p) {
    const groups = new Map();
    const add = (who, call) => {
      const k = callLabel(call.sector, call.ring);
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(who);
    };
    for (const s of sections(p)) if (s.ok !== false && !s.timed_out && s.ring) add(s.name, s);
    if (p.ad_matched && p.ad_ring) add('Autodarts', {sector: p.ad_sector, ring: p.ad_ring});
    const join = (xs) => (xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1]);
    return Array.from(groups.entries()).sort((a, b) => b[1].length - a[1].length)
      .map(([k, who]) => join(who) + (who.length === 1 ? ' says ' : ' say ') + k + '.').join(' ');
  }

  const STAMP = {
    disputed: ['Disputed', 'var(--warn)'], wrong: ['Wrong', 'var(--bad)'], noread: ['No read', 'var(--ink3)'],
  };

  function renderDetail() {
    const pane = $('#throw-detail');
    if (ROLE === 'display' && Shell.view === 'split') return;   // hidden there: no frames, no plot
    const p = selected && S.packages.get(selected);
    if (!p) {
      pane.replaceChildren(h('div', {class: 'empty'}, h('strong', {}, 'Pick a throw'),
        h('p', {}, 'Its cameras, every engine’s answer, and where each one put the dart.')));
      return;
    }
    const secs = sections(p);
    const v = verdict(p);
    const call = ours(p);
    const when = p.captured_at_utc && !isNaN(Date.parse(p.captured_at_utc)) ? clock(p.captured_at_utc) + ' · ' + ago(p.captured_at_utc) : 'time not recorded';
    let stamp = STAMP[v.key] || null;
    if (!stamp && v.key === 'right' && p.ad_operator_confirmed_ring) stamp = ['Confirmed', 'var(--ok)'];
    let tail = '';
    if (v.key === 'disputed') tail = ' Nobody has said which is right yet.';
    else if (v.key === 'wrong') tail = ' You recorded ' + callLabel(v.truth.sector, v.truth.ring) + '.';
    else if (v.key === 'right' && p.ad_operator_confirmed_ring) tail = ' You confirmed it.';
    else if (!p.ad_matched && v.key !== 'noread') tail = ' Autodarts was not asked.';
    const canvas = h('canvas', {class: 'agree-canvas', 'aria-label': 'Where each engine placed the dart'});
    const stills = h('div', {class: 'stills'}, CAM_IDS.map((c) => h('a', {href: viewerUrl(p), target: '_blank', rel: 'noopener', class: 'still', title: 'Open the full viewer'},
      stillImage(p, c),
      h('p', {class: 'figcap'}, h('span', {}, 'Cam ' + (c + 1)), h('span', {}, kind === 'bg' ? 'Before' : 'After')))));
    const xy = (a) => (a ? a[0].toFixed(1) + ', ' + a[1].toFixed(1) : '');
    const engineRows = secs.map((s) => h('tr', {class: s.primary ? 'primary' : ''},
      h('td', {}, s.name),
      h('td', {class: 'c'}, s.timed_out ? '—' : s.ok === false ? '×' : callLabel(s.sector, s.ring)),
      h('td', {class: s.sector_match === true ? 'ok' : s.sector_match === false ? 'bad' : ''}, s.sector_match === true ? '✓' : s.sector_match === false ? '✕' : ''),
      h('td', {class: 'muted'}, s.timed_out ? 'timed out' : s.ring ? ringWord(s.ring) : 'no result'),
      h('td', {}, s.tip_distance_mm === null || s.tip_distance_mm === undefined ? '' : s.tip_distance_mm.toFixed(1) + ' mm'),
      h('td', {class: 'muted'}, xy(s.board_xy_mm)),
      h('td', {class: 'reason', title: s.reason || ''}, s.reason || '')));
    if (p.ad_matched) {
      engineRows.push(h('tr', {class: 'oracle-row'}, h('td', {}, 'Autodarts'),
        h('td', {class: 'c'}, callLabel(p.ad_sector, p.ad_ring)), h('td', {}, ''), h('td', {class: 'muted'}, ringWord(p.ad_ring)),
        h('td', {}, p.ad_latency_ms ? Math.round(p.ad_latency_ms) + ' ms' : ''), h('td', {class: 'muted'}, xy(p.ad_tip_xy_mm)), h('td', {}, '')));
    }
    const canSaveFrames = VIDEO_RECORD_MODE !== 'all' && S.runtime && S.runtime.frame_ring && S.runtime.frame_ring.attached;
    // replaceChildren would print an absent optional part as "null".
    pane.replaceChildren(...[
      h('div', {class: 'h detail-head'},
        h('button', {type: 'button', class: 'btn small back', onclick: () => $('#throws').classList.remove('detail-open')}, '‹ Throws'),
        h('span', {}, h('b', {}, 'Throw ' + (p.throw_id || '').split('-').slice(-2, -1)[0]),
          (p.visit_index !== null && p.visit_index !== undefined ? ' · dart ' + (p.visit_index + 1) : '') + ' · ' + when),
        following
          ? h('span', {class: 'follow on', title: 'The newest dart shows here as it lands'}, '● Newest')
          : h('button', {type: 'button', class: 'follow', onclick: () => { following = true; render(); }}, 'Follow newest'),
        h('a', {href: viewerUrl(p), target: '_blank', rel: 'noopener'}, h('b', {}, 'Viewer ↗'))),
      h('div', {class: 'detail-call'},
        h('span', {class: 'big'}, noRead(p) ? '—' : callLabel(call.sector, call.ring)),
        h('span', {class: 'pts'}, noRead(p) ? 'NO READ' : points(call.sector, call.ring) + ' PTS'),
        stamp ? h('span', {class: 'stamp', style: '--c:' + stamp[1]}, stamp[0]) : null),
      h('p', {class: 'verdict'}, (whoSaidWhat(p) || 'No engine placed this dart.') + tail),
      h('div', {class: 'detail-actions'},
        h('button', {type: 'button', class: 'btn signal', onclick: () => openTruth(p)}, 'What actually landed?'),
        canSaveFrames ? h('button', {type: 'button', class: 'btn', onclick: () => saveFrames(p),
          title: 'Write the frames around this throw out of the buffer, while it still reaches back that far'}, 'Save frames') : null),
      h('figure', {}, h('div', {class: 'cropbox agree-box'}, canvas),
        h('figcaption', {class: 'figcap'}, h('span', {}, 'Fig. — where each engine put it'), h('span', {}, p.primary_engine ? p.primary_engine + ' scored it' : ''))),
      h('div', {class: 'h'}, h('b', {}, 'Cameras'),
        h('div', {class: 'seg'},
          h('button', {type: 'button', 'aria-pressed': String(kind === 'bg'), onclick: () => { kind = 'bg'; renderDetail(); }}, 'Before'),
          h('button', {type: 'button', 'aria-pressed': String(kind === 'after'), onclick: () => { kind = 'after'; renderDetail(); }}, 'After'))),
      stills,
      h('div', {class: 'h'}, h('b', {}, 'Engines'), h('span', {}, 'call · vs known · ring · Δ tip · board xy mm')),
      h('div', {class: 'table-wrap'}, h('table', {class: 'engines'}, h('tbody', {}, engineRows))),
    ].filter(Boolean));
    requestAnimationFrame(() => drawAgreement(canvas, p, secs));
    // ...and again whenever its box changes: a window resized, a screen
    // rotated, the text size changed. The pane is rebuilt on every render,
    // so the old observer goes with the old canvas.
    if (agreeObserver) agreeObserver.disconnect();
    if (window.ResizeObserver) {
      agreeObserver = new ResizeObserver(() => drawAgreement(canvas, p, secs));
      agreeObserver.observe(canvas.parentElement);
    }
  }

  function openTruth(p) {
    Truth.open({
      visitId: p.visit_id || null, index: p.visit_index === undefined ? null : p.visit_index,
      session: p.session, throwId: p.throw_id,
      title: 'Throw ' + ((p.throw_id || '').split('-').slice(-2, -1)[0] || ''),
      when: p.captured_at_utc ? clock(p.captured_at_utc) : '',
      scored: ours(p), candidates: candidatesFor(p),
      confirmed: p.ad_operator_confirmed_ring ? {sector: p.ad_operator_confirmed_sector, ring: p.ad_operator_confirmed_ring} : null,
    });
  }

  async function saveFrames(p) {
    const act = activity('Save frames', 'pending', p.throw_id);
    const body = await api.post('/api/packages/' + encodeURIComponent(p.session) + '/' + encodeURIComponent(p.throw_id)
      + '/capture-misscore', {reason: 'operator asked for the frames around this throw'}).catch(() => null);
    if (!body || !body.ok) { act.done('warn', (body && body.reason) || 'request failed'); return; }
    act.done('ok', 'writing ' + (body.job ? body.job.mb_total + ' MB' : 'the frames'));
    toast('Saving the frames around this throw', 'ok');
  }

  // ---- data in ---------------------------------------------------------------------
  function ingest(packages) {
    for (const p of packages || []) if (p && p.path) S.packages.set(p.path, p);
    emit('packages');
  }

  // The row for a dart the moment it is called. THROW_DETECTED carries the
  // primary engine's answer; the voters and Autodarts are written into the
  // saved package a moment later, and arrive with it (PACKAGES_UPDATED
  // replaces this record by path). Until then their cells say "…", not
  // "—": still coming, not missing.
  function provisional(ev) {
    if (!ev || !ev.path || S.packages.has(ev.path)) return;
    S.packages.set(ev.path, {
      _pending: true, path: ev.path, session: ev.session, throw_id: ev.throw_id,
      captured_at_utc: ev.captured_at_utc, visit_id: ev.visit_id, visit_index: ev.visit_index,
      primary_engine: ev.primary_engine, ok: ev.ok, sector: ev.sector, ring: ev.ring,
      board_xy_mm: ev.board_xy_mm, reason: ev.reason, engines: {},
    });
    emit('packages');
  }
  const pendingCount = () => { let n = 0; for (const p of S.packages.values()) if (p._pending) n++; return n; };
  function find(session, throwId) {
    if (!session || !throwId) return null;
    for (const p of S.packages.values()) if (p.session === session && p.throw_id === throwId) return p;
    return null;
  }
  // Live pushes carry only the newest 20; when the true count says we are
  // missing some, fetch the lot once rather than show a short list.
  async function reconcile(count) {
    if (typeof count !== 'number' || count === S.packages.size - pendingCount() || resyncing) return;
    resyncing = true;
    try {
      const fresh = await api.get('/api/packages');
      if (Array.isArray(fresh)) {
        // a dart called but not saved yet keeps its row through the resync
        const waiting = Array.from(S.packages.values()).filter((p) => p._pending && !fresh.some((f) => f.path === p.path));
        S.packages.clear();
        ingest(fresh.concat(waiting));
      }
    } catch (err) {
      console.error('package resync failed', err);
    } finally { resyncing = false; }
  }

  async function deleteRecorded() {
    const snap = await api.get('/api/recorded-data').catch(() => null);
    if (!snap || snap.ok === false) { toast('Could not read what is on disk — nothing was deleted', 'bad'); return; }
    if (snap.busy) { toast('A capture is being written — try again when it finishes', 'warn'); return; }
    const t = snap.total || {};
    if (!t.count && !t.bytes) { toast('Nothing to delete — no throws or captures on this rig', 'ok'); return; }
    const pk = snap.packages || {}, cp = snap.captures || {};
    const yes = await confirmSheet({
      title: 'Delete recorded data?',
      body: '<p><strong>' + esc(plural(pk.count || 0, 'throw package')) + '</strong> (' + esc(pk.label || '0 B') + ') and <strong>'
        + esc(plural(cp.count || 0, 'capture')) + '</strong> (' + esc(cp.label || '0 B') + ') will be removed from this rig. This cannot be undone.</p>'
        + '<p class="muted mono">' + esc(pk.root || '') + '<br>' + esc(cp.root || '') + '</p>'
        + '<p class="muted">Anything you still need must be copied off the rig first.</p>',
      confirm: 'Delete everything', danger: true,
    });
    if (!yes) return;
    const act = activity('Delete recorded data', 'pending', 'deleting…');
    const body = await api.post('/api/packages/delete-all').catch(() => null);
    if (!body || !body.ok) { act.done('bad', (body && body.reason) || 'request failed'); return; }
    act.done('ok', plural((body.packages || {}).deleted || 0, 'throw') + ' and ' + plural((body.captures || {}).deleted || 0, 'capture') + ' deleted');
    S.packages.clear();
    sinceMs = null;
    selected = null;
    emit('packages');
    Rig.refreshConfig();
  }

  // ---- wiring ------------------------------------------------------------------------
  $$('#throw-filters [data-filter]').forEach((b) => b.addEventListener('click', () => {
    filter = b.dataset.filter; shown = PAGE; render();
  }));
  $$('#throw-density button').forEach((b) => b.addEventListener('click', () => {
    mode = b.dataset.v === 'matrix' ? 'matrix' : 'list'; prefs.set('throwsMode', mode); render();
  }));
  $('#throws-missed').addEventListener('click', () => Play.openMissed());
  $('#clear-view').addEventListener('click', () => { sinceMs = Date.now(); render(); });
  $('#show-all').addEventListener('click', () => { sinceMs = null; render(); });
  $('#delete-recorded').addEventListener('click', deleteRecorded);

  // j/k and the arrows walk the list; Enter says what landed.
  document.addEventListener('keydown', (ev) => {
    if (!Shell.showing('throws') || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (typingIn(ev.target)) return;
    const rows = $$('#throw-list .trow');
    if (!rows.length) return;
    const i = rows.findIndex((r) => r.dataset.path === selected);
    if (ev.key === 'ArrowDown' || ev.key === 'j') {
      ev.preventDefault();
      const r = rows[Math.min(rows.length - 1, i + 1)];
      select(r.dataset.path); r.scrollIntoView({block: 'nearest'});
    } else if (ev.key === 'ArrowUp' || ev.key === 'k') {
      ev.preventDefault();
      const r = rows[Math.max(0, i - 1)];
      select(r.dataset.path); r.scrollIntoView({block: 'nearest'});
    } else if (ev.key === 'Enter' && selected) {
      const p = S.packages.get(selected);
      if (p) { ev.preventDefault(); openTruth(p); }
    }
  });

  // The missed-dart button, and the writer's progress, on the page you are
  // on when a dart gets missed.
  function renderMissedJob() {
    const fr = S.runtime && S.runtime.frame_ring;
    const btn = $('#throws-missed');
    btn.disabled = !(fr && fr.attached && fr.configured_seconds > 0);
    btn.title = btn.disabled ? 'The throw buffer is off (Rig › Recording), so there are no frames to save' : 'Save the last seconds of every camera, for a dart that never scored';
    const w = (fr && fr.writer) || {};
    const job = w.current || w.last;
    const el = $('#missed-job');
    if (!job) { el.textContent = ''; return; }
    el.textContent = job.state === 'writing' || job.state === 'queued'
      ? 'Saving ' + String(job.kind || '').replace('_', ' ') + ' capture — ' + job.mb_written + ' of ' + job.mb_total + ' MB'
      : job.state === 'failed' ? 'Last capture failed: ' + (job.error || 'no reason recorded')
        : 'Last capture: ' + String(job.kind || '').replace('_', ' ') + ', ' + job.mb_total + ' MB';
  }
  on('config', () => { renderMissedJob(); if (Shell.showing('throws')) renderBanner(); });
  on('adboard', () => { if (Shell.showing('throws')) renderBanner(); });
  on('theme', () => { if (Shell.showing('throws')) renderDetail(); });
  // A dart brings two or three package messages within a fraction of a
  // second; they are drawn once, on the next frame, not once each.
  let queued = false;
  function soon() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; render(); });
  }
  on('packages', soon);
  on('engines', soon);
  on('view', () => { if (Shell.showing('throws') && stale) render(); });

  // A display watches: every engine's call, all throws, the newest dart in
  // the pane -- and nothing on it to press. Not saved as this browser's
  // preference, so a kiosk that was once a controller does not keep it.
  function watch() {
    mode = 'matrix'; filter = 'all'; sinceMs = null; following = true; shown = 40;
    render();
  }

  return {
    ingest, provisional, find, reconcile, candidatesFor, render, watch,
    selectById(throwId) {
      for (const p of S.packages.values()) if (p.throw_id === throwId) { select(p.path); return true; }
      return false;
    },
  };
})();
