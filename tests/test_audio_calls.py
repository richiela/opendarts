"""Spoken dart calls -- phrase vocabulary, clip serving, and the API.

Deliberately does NOT assert that sound comes out. It never could -- CI
has no speaker -- but as of 2026-09-15 it could not even in principle:
playback happens in a BROWSER now, and the thing that decides whether a
noise is made is an autoplay policy on someone's iPad. What IS pinned
here is everything that can be wrong silently on this side of the wire:

  * a phrase no clip set contains,
  * a filename that disagrees with what is on disk,
  * a clip route that could be talked into serving a file outside the
    voice directories,
  * a stale client report presented as a device that is still speaking.

REWRITTEN TWICE IN ONE DAY. The first rewrite dropped the render
pipeline (three platform renderers, a voice-name resolver, a PowerShell
injection guard, a prerender progress modal). This one drops the
PLAYBACK half for the same reason -- `speak()`, `can_play()`,
`audio_available()` and `resolve_voice()` no longer exist, so the tests
that pinned them are gone rather than being kept limping.

What survived both rewrites, and why:

  * The phrase set and `phrase_for()` must agree, or a real dart falls
    through to silence. Originally pinned against what prerender() would
    render, then against what a shipped set must contain; unchanged in
    substance since it was written.
  * "Audio is on" must never be able to mean "on with nothing to play".
    `audio_available()` used to carry that; `coverage()` carries the
    half that still has a server-side answer, and the other half -- can
    this screen actually be heard -- is what /api/audio/clients is for.
"""


import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendarts.live import audio
from opendarts.live.server import create_app


@pytest.fixture()
def package_root(tmp_path):
    """A throwaway package root. Nothing here writes packages; it exists
    because create_app requires one."""
    d = tmp_path / "packages"
    d.mkdir()
    return d


@pytest.fixture()
def voices_dir(tmp_path, monkeypatch):
    """An isolated `assets/voices/` so tests do not depend on which sets
    happen to be committed, nor break when one is added."""
    root = tmp_path / "voices"
    root.mkdir()
    monkeypatch.setattr(audio, "SHIPPED_VOICES_DIR", root)
    return root


def _make_set(root: Path, name: str, phrases=None, marker: str = "") -> Path:
    """A voice set on disk. Full unless `phrases` names a subset."""
    d = root / name
    d.mkdir()
    for text in (phrases if phrases is not None else audio.phrases()):
        (d / audio.clip_name(text)).write_bytes(b"ID3\x04\x00\x00fake-mp3")
    if marker:
        (d / audio.VOICE_MARKER).write_text(marker + "\n")
    return d


# ---------------------------------------------------------------------------
# Vocabulary. The phrase set is the CONTRACT a shipped clip set must meet.
# ---------------------------------------------------------------------------

def test_every_phrase_for_result_is_one_a_clip_set_contains():
    """The failure this prevents is silent: a dart is scored, the phrase
    has no clip, and nothing is announced. Every value phrase_for() can
    return has to be in phrases(), which is what a set is rendered from."""
    known = set(audio.phrases())
    seen = set()
    for sector in list(range(1, 21)) + [None, 0, 21, "x"]:
        for ring in ("single_inner", "single_outer", "treble", "double",
                     "bull", "outer_bull", "outside", None, "nonsense"):
            for ok in (True, False):
                got = audio.phrase_for(sector, ring, ok)
                if got is not None:
                    seen.add(got)
    assert seen, "the sweep produced nothing -- the test is not exercising phrase_for"
    assert seen <= known, f"phrase_for can say what no set contains: {sorted(seen - known)}"


def test_bull_rings_map_to_the_two_distinct_centre_calls():
    assert audio.phrase_for(25, "outer_bull") == "bullseye"
    assert audio.phrase_for(50, "bull") == "double bullseye"


def test_a_dart_that_did_not_score_is_announced_not_skipped():
    """A dart that landed and could not be read is exactly the case an
    operator most wants to hear about."""
    assert audio.phrase_for(20, "treble", ok=False) == "no score"
    assert audio.phrase_for(None, None) == "no score"


def test_singles_are_called_as_the_bare_number():
    """A caller announces "twenty", never "inner twenty"."""
    assert audio.phrase_for(20, "single_inner") == "20"
    assert audio.phrase_for(20, "single_outer") == "20"


def test_phrase_set_is_closed_and_complete():
    got = list(audio.phrases())
    assert len(got) == len(set(got)), "duplicate phrase -- a set would render it twice"
    assert len(got) == 64


def test_clip_name_is_the_readable_phrase():
    """Named by phrase, so a human can find and replace one call."""
    assert audio.clip_name("treble 20") == "treble-20.mp3"
    assert audio.clip_name("double bullseye") == "double-bullseye.mp3"


def test_clips_are_mp3_not_wav():
    """WAV was forced by `winsound` reading nothing else and `aplay`
    refusing MP3. Both went with server-side playback on 2026-09-15, and
    the set shrank from 5.2 MB to under a megabyte -- which matters,
    because the browser fetches the WHOLE vocabulary each time a screen
    arms itself."""
    assert audio.CLIP_SUFFIX == ".mp3"
    assert all(n.endswith(".mp3") for n in audio.clip_names().values())


def test_clip_names_are_unique_and_filesystem_safe():
    names = [audio.clip_name(t) for t in audio.phrases()]
    assert len(names) == len(set(names)), "two phrases collide on one filename"
    for n in names:
        assert not (set(n) & set('/\\:*?"<>| ')), f"unsafe filename: {n}"


def test_clip_names_map_covers_the_whole_vocabulary():
    """The browser is handed this map instead of a slugification rule, so
    that `clip_name()` is never reimplemented in JavaScript. A phrase
    missing from it is a dart the browser cannot find a URL for."""
    m = audio.clip_names()
    assert set(m) == set(audio.phrases())
    assert m["treble 20"] == "treble-20.mp3"


# ---------------------------------------------------------------------------
# Voice sets. A directory IS a voice -- adding one is dropping in a folder.
# ---------------------------------------------------------------------------

def test_a_directory_of_clips_is_a_voice(voices_dir):
    _make_set(voices_dir, "Bella")
    _make_set(voices_dir, "Daniel")
    assert audio.voices() == ["Bella", "Daniel"]


def test_an_empty_directory_is_not_offered_as_a_voice(voices_dir):
    """It would be offered as a voice that plays nothing -- worse than
    not being listed, because the operator selects it and hears silence."""
    (voices_dir / "Ghost").mkdir()
    _make_set(voices_dir, "Bella")
    assert audio.voices() == ["Bella"]


def test_voice_lookup_is_case_insensitive(voices_dir):
    """The value arrives from a browser's localStorage or a URL path."""
    _make_set(voices_dir, "Bella")
    assert audio.voice_dir("bella") is not None
    assert audio.voice_dir("BELLA") is not None
    assert audio.voice_dir("nope") is None


def test_voice_dir_cannot_be_talked_out_of_the_voices_directory(voices_dir):
    """The whole traversal defence, in one function. `voice_dir` matches
    against enumerated names rather than joining the caller's string onto
    a path, so an attempt to escape is not something that has to be
    spotted -- it is simply a name nothing matches."""
    _make_set(voices_dir, "Bella")
    for attempt in ("..", "../..", "../Bella", "/etc", "Bella/../..",
                    "..%2f..", "\\..\\.."):
        assert audio.voice_dir(attempt) is None, attempt


def test_the_kokoro_voice_a_set_was_rendered_from_is_readable(voices_dir):
    """Filenames carry the phrase and not the voice, so the `.voice`
    marker is the only thing on disk that says what "Bella" actually is.

    The constant behind it was deleted on 2026-09-15 while
    tools/voice/generate_kokoro_clips.py kept writing it, so the
    generator crashed on its last line -- after rendering all 64 clips
    correctly. Pinned here because that failure is invisible from the
    product side: the set looks finished."""
    _make_set(voices_dir, "Bella", marker="af_bella")
    _make_set(voices_dir, "Nameless")
    assert audio.VOICE_MARKER == ".voice"
    assert audio.rendered_voice("Bella") == "af_bella"
    assert audio.rendered_voice("Nameless") is None
    assert audio.rendered_voice("NotInstalled") is None


def test_the_marker_is_not_mistaken_for_a_voice(voices_dir):
    """A dotfile inside a set must not make an otherwise-empty directory
    look like an installed voice."""
    d = voices_dir / "MarkerOnly"
    d.mkdir()
    (d / audio.VOICE_MARKER).write_text("af_bella\n")
    assert audio.voices() == []


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def test_clip_path_finds_the_clip_in_the_named_voice(voices_dir):
    _make_set(voices_dir, "Bella")
    p = audio.clip_path("treble 20", "Bella")
    assert p is not None and p.name == "treble-20.mp3"
    assert p.parent.name == "Bella"


def test_clip_path_is_none_for_an_unknown_voice_or_phrase(voices_dir):
    _make_set(voices_dir, "Bella", phrases=["miss"])
    assert audio.clip_path("miss", "Nobody") is None
    assert audio.clip_path("treble 20", "Bella") is None


# ---------------------------------------------------------------------------
# Coverage -- "audio is on" must never mean "on but nothing to play"
# ---------------------------------------------------------------------------

def test_coverage_reports_a_complete_set(voices_dir):
    _make_set(voices_dir, "Bella")
    c = audio.coverage("Bella")
    assert (c["present"], c["total"], c["complete"]) == (64, 64, True)
    assert c["voice"] == "Bella"
    assert c["installed"] is True


def test_coverage_reports_a_partial_set_honestly(voices_dir):
    """A set three clips short is silent on exactly the darts nobody
    throws while testing, which is the whole reason this is surfaced."""
    _make_set(voices_dir, "Bella", phrases=["miss", "no score"])
    c = audio.coverage("Bella")
    assert c["present"] == 2
    assert c["complete"] is False


def test_coverage_says_uninstalled_rather_than_substituting(voices_dir):
    """`resolve_voice()` used to silently answer about a DIFFERENT set.
    That was right when the voice was a rig-wide config value a typo
    could mute; the picker now only offers names the server just
    enumerated, so a miss means something genuinely went away and
    quietly reporting on another set would hide it."""
    _make_set(voices_dir, "Bella")
    c = audio.coverage("Zaphod")
    assert c["installed"] is False
    assert c["present"] == 0 and c["complete"] is False
    assert c["voice"] == "Zaphod", "must report what was ASKED for, not a substitute"


def test_coverage_reports_the_canonical_spelling(voices_dir):
    """"bella" resolves to the "Bella" the URLs and the picker use."""
    _make_set(voices_dir, "Bella")
    assert audio.coverage("bella")["voice"] == "Bella"


def test_the_server_no_longer_has_any_way_to_play_a_sound():
    """The point of the whole change, pinned. A `speak()` growing back
    here would put playback on a headless box in an empty room again --
    and, worse, would do it silently alongside the browser's."""
    for gone in ("speak", "can_play", "audio_available", "resolve_voice"):
        assert not hasattr(audio, gone), f"audio.{gone} came back"
    assert not hasattr(audio, "subprocess"), "audio can spawn a process again"


# ---------------------------------------------------------------------------
# API: the voice catalogue
# ---------------------------------------------------------------------------

def test_voices_endpoint_lists_sets_with_their_coverage(package_root, voices_dir):
    _make_set(voices_dir, "Bella")
    _make_set(voices_dir, "Daniel", phrases=["miss"])
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).get("/api/audio/voices").json()
    assert body["ok"] is True
    assert [v["voice"] for v in body["voices"]] == ["Bella", "Daniel"]
    assert body["voices"][0]["complete"] is True
    assert body["voices"][1]["complete"] is False
    assert body["default"] == audio.DEFAULT_VOICE


def test_voices_endpoint_carries_the_phrase_to_filename_map(package_root, voices_dir):
    """Served rather than derived client-side, so the naming rule exists
    in one language only."""
    _make_set(voices_dir, "Bella")
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).get("/api/audio/voices").json()
    assert body["clips"]["treble 20"] == "treble-20.mp3"
    assert len(body["clips"]) == 64


def test_the_old_settings_and_test_routes_are_gone(package_root):
    """`GET`/`POST /api/audio` and `POST /api/audio/test` went with
    server-side playback. Pinned because a client still calling them
    fails at the click, not at startup."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    assert client.get("/api/audio").status_code == 404
    assert client.post("/api/audio", json={"enabled": True}).status_code == 404
    assert client.post("/api/audio/test", json={}).status_code == 404


# ---------------------------------------------------------------------------
# API: serving the clips
# ---------------------------------------------------------------------------

def test_a_clip_is_served_as_audio(package_root, voices_dir):
    _make_set(voices_dir, "Bella")
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).get("/api/audio/clips/Bella/treble-20.mp3")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content.startswith(b"ID3")
    # Cacheable, because a reload should not refetch a megabyte -- which
    # is exactly what a TV does at 3am after a browser update.
    assert "max-age" in resp.headers.get("cache-control", "")


def test_a_clip_route_is_case_insensitive_in_the_voice(package_root, voices_dir):
    _make_set(voices_dir, "Bella")
    app = create_app(package_root=package_root, enable_background_poll=False)
    assert TestClient(app).get("/api/audio/clips/bella/miss.mp3").status_code == 200


def test_an_unknown_voice_or_clip_is_a_404(package_root, voices_dir):
    _make_set(voices_dir, "Bella", phrases=["miss"])
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    assert client.get("/api/audio/clips/Nobody/miss.mp3").status_code == 404
    # In the set's name list, but not on disk.
    assert client.get("/api/audio/clips/Bella/treble-20.mp3").status_code == 404


@pytest.mark.parametrize("name", [
    "..%2f..%2f..%2fetc%2fpasswd",
    "....//....//etc/passwd",
    "%2e%2e%2f%2e%2e%2fconfig.json",
    "treble-20.mp3.bak",
    ".voice",
])
def test_the_clip_route_serves_nothing_outside_the_vocabulary(
    package_root, voices_dir, name
):
    """A route that takes a filename and returns a file is the classic
    shape for a traversal bug. There is no sanitiser here to get wrong:
    the name must be one of the 64 GENERATED filenames or nothing is
    read at all -- including the set's own `.voice` marker."""
    _make_set(voices_dir, "Bella", marker="af_bella")
    (voices_dir.parent / "secret.txt").write_text("not yours")
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).get(f"/api/audio/clips/Bella/{name}")
    assert resp.status_code == 404
    assert b"not yours" not in resp.content


# ---------------------------------------------------------------------------
# API: which screens are actually speaking
# ---------------------------------------------------------------------------

def test_a_client_reports_itself_and_is_listed_back(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    assert client.post("/api/audio/clients", json={
        "client_id": "tab_1", "label": "TV", "enabled": True, "blocked": False,
    }).json()["ok"] is True
    body = client.get("/api/audio/clients").json()
    assert [c["client_id"] for c in body["clients"]] == ["tab_1"]
    assert body["clients"][0]["label"] == "TV"
    assert body["clients"][0]["age_s"] >= 0


def test_a_report_without_an_id_is_a_400_not_a_silent_no_op(package_root):
    """The endpoint exists to catch a screen that has gone quiet.
    Swallowing a malformed report would hide the client-side bug it is
    supposed to reveal."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).post("/api/audio/clients", json={"label": "TV"})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_a_blocked_screen_is_reported_as_blocked(package_root):
    """The state this whole endpoint exists for: a TV whose autoplay
    permission was lost on a reload, visible from the iPad in someone's
    hand."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    client.post("/api/audio/clients", json={
        "client_id": "tv", "label": "TV", "enabled": True, "blocked": True,
        "context_state": "suspended",
    })
    got = client.get("/api/audio/clients").json()["clients"][0]
    assert got["blocked"] is True and got["context_state"] == "suspended"


def test_a_screen_that_stopped_checking_in_drops_off(package_root, monkeypatch):
    """"The TV is speaking", reported by a tab that closed forty minutes
    ago, is worse than no line at all -- the panel exists to show a
    screen that has gone quiet."""
    from opendarts.live import server as server_mod

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    client.post("/api/audio/clients", json={"client_id": "ghost", "enabled": True})
    assert len(client.get("/api/audio/clients").json()["clients"]) == 1

    real = server_mod.time.monotonic
    monkeypatch.setattr(server_mod.time, "monotonic",
                        lambda: real() + server_mod.AUDIO_CLIENT_STALE_S + 1)
    assert client.get("/api/audio/clients").json()["clients"] == []


def test_the_client_list_is_capped(package_root):
    """A long-running rig with a tab opened per visit must not grow this
    dict forever. Oldest-reporting evicted first."""
    from opendarts.live.server import AUDIO_CLIENT_MAX

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    for i in range(AUDIO_CLIENT_MAX + 5):
        client.post("/api/audio/clients", json={"client_id": f"tab_{i}"})
    ids = [c["client_id"] for c in client.get("/api/audio/clients").json()["clients"]]
    assert len(ids) == AUDIO_CLIENT_MAX
    assert "tab_0" not in ids and f"tab_{AUDIO_CLIENT_MAX + 4}" in ids


def test_a_returning_client_is_not_evicted_as_the_oldest(package_root):
    """A tab that keeps reporting is the opposite of stale. A plain dict
    assignment does not reorder an existing key, so this is a real thing
    to get wrong."""
    from opendarts.live.server import AUDIO_CLIENT_MAX

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    client.post("/api/audio/clients", json={"client_id": "tv"})
    for i in range(AUDIO_CLIENT_MAX):
        client.post("/api/audio/clients", json={"client_id": f"tab_{i}"})
        client.post("/api/audio/clients", json={"client_id": "tv", "enabled": True})
    ids = [c["client_id"] for c in client.get("/api/audio/clients").json()["clients"]]
    assert "tv" in ids


# ---------------------------------------------------------------------------
# The dashboard's own half
# ---------------------------------------------------------------------------

def test_the_audio_controls_are_one_box(package_root):
    """On/off, voice, volume and Test are a single subject -- someone
    setting up sound on a screen does all of it in one pass."""
    import re as _re

    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/").text
    at = html.index('id="audio-enabled-select"')
    starts = [m.start() for m in
              _re.finditer(r'<(?:label|div) class="config-field(?:\s[^"]*)?"', html)
              if m.start() < at]
    box_start = starts[-1]
    depth, box = 0, None
    for m in _re.finditer(r'<(?:label|div)\b[^>]*>|</(?:label|div)>', html[box_start:]):
        depth += -1 if m.group(0).startswith("</") else 1
        if depth == 0:
            box = html[box_start:box_start + m.end()]
            break
    assert box is not None
    for control in ("audio-enabled-select", "audio-voice-select",
                    "audio-volume", "btn-audio-test"):
        assert f'id="{control}"' in box, f"{control} is outside the audio box"


def test_there_is_no_blocked_banner_to_dismiss(package_root):
    """The banner is gone, deliberately, and must not come back.

    It existed to tell an operator that the browser had muted the page.
    In practice it appeared on EVERY load -- a gesture is required by
    every browser, every time -- so it was permanent furniture within an
    hour, and on 2026-09-15 it was also the thing that could not be
    dismissed on an iPad for three separate reasons in a row.

    What replaced it assumes the truth instead: a dashboard someone is
    using gets touched at least once, and the first touch anywhere arms
    everything. A page that shouts at you to tap it, before you have even
    tried to use it, is noise standing in for a design decision.
    """
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/").text
    assert "audio-blocked-banner" not in html
    assert "audio-blocked" not in html, (
        "the blocked-sound banner is back. Arming happens on the first "
        "interaction instead; if that has regressed, fix the arming rather "
        "than reinstating a bar on every page load."
    )


def test_the_ios_silent_switch_workaround_is_present(package_root):
    """A silent media element, or the iPad stays mute with the switch on.

    iOS assigns every page an audio session CATEGORY. Media elements get
    "playback", which ignores the physical silent switch; the Web Audio
    API gets "ambient", which obeys it. We use Web Audio for pre-decoded
    low-latency clips, so without this the rig is silent whenever that
    switch is flipped -- while every other site on the same iPad keeps
    making noise, which is exactly how it stayed hidden for hours.

    Playing one real media element promotes the page to "playback" and
    the AudioContext follows it.
    """
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/").text
    assert "unlockAudioSession" in html
    assert "data:audio/wav;base64," in html, (
        "the silent unlock clip is gone -- an iPad with its silent switch "
        "on will be mute while every other site still plays"
    )
    assert "audioSessionEl" in html, (
        "the unlock element is not retained; letting it be garbage "
        "collected drops the session back to the ambient category"
    )


def test_nothing_in_the_page_still_calls_the_deleted_audio_routes(package_root):
    """Removing a route and leaving the fetch is valid JavaScript and a
    control that silently does nothing -- this page's own recurring bug."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/").text
    for gone in ("'/api/audio'", "/api/audio/test", "btn-audio-prerender",
                 "render-modal", "/api/audio/prerender"):
        assert gone not in html, f"{gone} survived the move into the browser"


# ---------------------------------------------------------------------------
# The wire: what a scored dart puts on /api/events
# ---------------------------------------------------------------------------

def _throw(sector="20", ring="treble", ok=True):
    return {
        "type": "THROW_DETECTED", "session": "s", "throw_id": "t",
        "visit_id": "visit_1", "visit_index": 0,
        "ok": ok, "sector": sector, "ring": ring,
    }


def _broadcast_for(package_root, event):
    """Run one live event through a real AppState and return what went
    out to WebSocket clients, in order."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    sent = []

    class _Sock:
        async def send_text(self, data):
            import json
            sent.append(json.loads(data))

    state.clients.add(_Sock())
    asyncio.run(state._handle_live_event(event))  # noqa: SLF001
    return state, sent


def test_a_scored_dart_broadcasts_the_phrase_before_the_throw(package_root):
    """The browser is told what to SAY, not what was scored, and it is
    told first. Both halves are deliberate -- see the DART_CALL branch in
    AppState._handle_live_event."""
    _, sent = _broadcast_for(package_root, _throw())
    assert [m["type"] for m in sent] == ["DART_CALL", "THROW_DETECTED"]
    assert sent[0]["phrase"] == "treble 20"
    assert "sector" not in sent[0] and "ring" not in sent[0]


def test_a_dart_that_could_not_be_read_is_still_called(package_root):
    """"No score" is the case an operator most wants to hear about, so it
    must not fall out of the audio path just because it scored nothing."""
    _, sent = _broadcast_for(package_root, _throw(ok=False))
    assert sent[0]["phrase"] == "no score"


def test_the_call_is_sent_whether_or_not_anyone_wants_it(package_root):
    """No server-side enable flag is left to gate on. Every device
    decides for itself, from its own localStorage, and a muted dashboard
    drops a ~40-byte message -- cheaper than the server holding an
    opinion it would have to reconcile with three screens."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    assert not hasattr(state, "audio_enabled")
    assert not hasattr(state, "audio_voice")
