// ---------------------------------------------------------------------------
// truth -- "What actually landed?", the one way to record a correct answer.
//
// It posts to /api/visits/{visit}/throws/{index}/correct, which records the
// operator's answer BESIDE the engine call (never over it), derives whether
// Autodarts was wrong from what is on disk, and is idempotent
// (opendarts.capture.throw_package.record_throw_correction). The same sheet
// serves the player at the oche and the builder in Throws: to both, it is
// the same question.
// ---------------------------------------------------------------------------

const Truth = (() => {
  const dlg = document.getElementById('sheet-truth');
  let target = null;      // {visitId, index, session, throwId, scored, candidates, confirmed}
  let choice = null;      // {sector, ring, source}
  let picker = null;

  const SECTORLESS = new Set(['bull', 'outer_bull', 'outside']);

  function render() {
    if (!target) return;
    const scored = target.scored;
    $('#truth-sub').textContent = (target.title || 'This dart') + ' · scored '
      + (scored && scored.ring ? callLabel(scored.sector, scored.ring) : 'nothing')
      + (target.when ? ' · ' + target.when : '');
    // Candidates: what the engines and the oracle said, de-duplicated,
    // most-trusted first. Picking one is the whole correction, usually.
    const seen = new Set();
    const chips = [];
    for (const cand of target.candidates || []) {
      if (!cand.ring) continue;
      const key = cand.ring + ':' + (cand.sector || '');
      if (seen.has(key)) {
        const prev = chips.find((ch) => ch.key === key);
        if (prev && prev.who.indexOf(cand.who) < 0) prev.who.push(cand.who);
        continue;
      }
      seen.add(key);
      chips.push({key, sector: cand.sector, ring: cand.ring, who: [cand.who], source: cand.source});
    }
    const box = $('#truth-candidates');
    box.replaceChildren(...chips.map((ch) => h('button', {
      type: 'button', class: 'chip chip-cand' + (choice && sameCall(choice, ch) ? ' on' : ''),
      onclick: () => choose({sector: ch.sector, ring: ch.ring, source: ch.source}),
    }, h('b', {}, callLabel(ch.sector, ch.ring)), h('span', {}, ch.who.join(' · ')))));
    box.hidden = chips.length === 0;
    // Rings for the chosen sector, and the three sectorless answers.
    const rings = $('#truth-rings');
    const sector = choice && !SECTORLESS.has(choice.ring) ? choice.sector : null;
    const ringChip = (label, call) => h('button', {
      type: 'button', class: 'chip' + (choice && sameCall(choice, call) ? ' on' : ''),
      onclick: () => choose(Object.assign({source: 'manual'}, call)),
    }, label);
    const kids = [];
    if (sector) {
      kids.push(ringChip(sector + ' inner', {sector, ring: 'single_inner'}),
        ringChip('T' + sector, {sector, ring: 'treble'}),
        ringChip(sector + ' outer', {sector, ring: 'single_outer'}),
        ringChip('D' + sector, {sector, ring: 'double'}),
        h('span', {class: 'chip-gap'}));
    }
    kids.push(ringChip('25', {sector: null, ring: 'outer_bull'}),
      ringChip('Bull', {sector: null, ring: 'bull'}),
      ringChip('Miss', {sector: null, ring: 'outside'}));
    rings.replaceChildren(...kids);
    // The answer, spelled out, and everything saving it will change: the
    // dart, the visit's total, and what it says about the oracle.
    const ans = $('#truth-answer');
    if (!choice) {
      ans.innerHTML = '<span class="muted">Tap a suggestion, or where the dart is on the board.</span>';
    } else {
      const pts = points(choice.sector, choice.ring);
      const same = scored && sameCall(scored, choice);
      const said = [same ? 'Confirms the call as scored.'
        : 'Replaces ' + (scored && scored.ring ? callLabel(scored.sector, scored.ring) : 'the missing call') + ' for this dart.'];
      if (target.visitTotal !== undefined && !same) {
        said.push('The visit becomes ' + (target.visitTotal - (target.scoredPts || 0) + (pts || 0)) + '.');
      }
      const oracle = (target.candidates || []).find((cd) => cd.who === 'Autodarts');
      if (oracle) said.push('Autodarts is recorded as ' + (sameCall(oracle, choice) ? 'right.' : 'wrong.'));
      ans.innerHTML = '<strong class="truth-call">' + esc(callLabel(choice.sector, choice.ring)) + '</strong>'
        + '<span class="truth-pts">' + esc(pts) + ' PTS</span>'
        + '<span class="truth-note">' + esc(said.join(' ')) + '</span>';
    }
    $('#truth-save').disabled = !choice;
    $('#truth-save').textContent = choice ? 'Record ' + callLabel(choice.sector, choice.ring) : 'Record it';
    $('#truth-clear').hidden = !target.confirmed;
    // From the scoreboard, one step to the throw's full analysis.
    $('#truth-review').hidden = !(target.fromPlay && target.throwId && Throws.find(target.session, target.throwId));
    if (picker) {
      picker.select(choice);
      picker.setMarks((target.candidates || []).filter((cd) => cd.xy && (cd.primary || cd.who === 'Autodarts')).map((cd) => ({
        xy: cd.xy, oracle: cd.who === 'Autodarts', primary: !!cd.primary,
      })));
    }
  }

  function choose(call) {
    choice = {sector: SECTORLESS.has(call.ring) ? null : String(call.sector), ring: call.ring, source: call.source || 'manual'};
    render();
  }

  // t: {visitId, index, session, throwId, title, scored:{sector,ring},
  //     candidates:[{sector, ring, who, source, xy, primary}], confirmed:{sector,ring}|null}
  function open(t) {
    target = t;
    choice = t.confirmed ? Object.assign({source: 'manual'}, t.confirmed) : null;
    $('#truth-error').textContent = '';
    sheet.open('sheet-truth');
    if (!picker) picker = makePicker($('#truth-canvas'), (hit) => choose(Object.assign({source: 'manual'}, hit)));
    render();
    requestAnimationFrame(() => picker.draw());
  }

  async function save() {
    if (!target || !choice) return;
    const btn = $('#truth-save');
    btn.disabled = true;
    const label = callLabel(choice.sector, choice.ring);
    const act = activity('Correct', 'pending', (target.title || 'dart') + ' → ' + label);
    let body;
    try {
      if (target.visitId !== null && target.visitId !== undefined && target.index !== null && target.index !== undefined) {
        const payload = {ring: choice.ring, source: choice.source || 'manual', note: 'dashboard'};
        if (!SECTORLESS.has(choice.ring)) payload.sector = String(choice.sector);
        body = await api.post('/api/visits/' + encodeURIComponent(target.visitId) + '/throws/'
          + encodeURIComponent(target.index) + '/correct', payload);
      } else {
        // A throw saved before the visit model has no visit to address; its
        // package still takes the same annotation through mark-ad-wrong.
        body = await api.post('/api/packages/' + encodeURIComponent(target.session) + '/'
          + encodeURIComponent(target.throwId) + '/mark-ad-wrong', {
          wrong: true, confirmed_source: choice.source || 'manual',
          confirmed_sector: SECTORLESS.has(choice.ring) ? null : String(choice.sector), confirmed_ring: choice.ring,
        });
      }
    } catch (err) {
      body = {ok: false, reason: 'could not reach the rig'};
    }
    btn.disabled = false;
    if (!body || !body.ok) {
      $('#truth-error').textContent = (body && body.reason) || 'Not recorded.';
      act.done('bad', (body && body.reason) || 'not recorded');
      return;
    }
    if (body.package) Throws.ingest([body.package]);
    act.done('ok', 'recorded ' + label);
    toast('Recorded ' + label, 'ok');
    sheet.close(dlg);
  }

  async function clear() {
    if (!target || !target.session || !target.throwId) return;
    const act = activity('Correction removed', 'pending', target.title || 'dart');
    const body = await api.post('/api/packages/' + encodeURIComponent(target.session) + '/'
      + encodeURIComponent(target.throwId) + '/mark-ad-wrong', {wrong: false}).catch(() => null);
    if (!body || !body.ok) { act.done('bad', (body && body.reason) || 'request failed'); return; }
    if (body.package) Throws.ingest([body.package]);
    act.done('ok', 'back to the scored call');
    sheet.close(dlg);
  }

  $('#truth-review').addEventListener('click', () => {
    if (!target) return;
    const id = target.throwId;
    sheet.close(dlg);
    Shell.go('throws');
    Throws.selectById(id);
  });
  $('#truth-save').addEventListener('click', save);
  $('#truth-clear').addEventListener('click', clear);
  return {open};
})();
