"""Spoken dart calls: the vocabulary, and the clips that speak it.

NOTHING IN THIS MODULE PLAYS A SOUND, as of 2026-09-15. It used to: a
`speak()` that shelled out to `afplay`/`aplay`/`paplay` or called
`winsound`, gated by a `can_play()` capability probe. All of that is
gone, and the browser plays the clips instead.

WHY THE SERVER STOPPED PLAYING. It was playing into an empty room. The
rig is a headless box wherever the cables reach -- a shelf, a cupboard,
under the board -- and the person throwing is looking at a dashboard on
an iPad beside them or a TV above the board. Sound has to come out of
the thing they are looking at, and the server is never that thing. On
the Linux rig this was worse than useless: a default Ubuntu Server
install ships no userspace audio at all, so `setup_linux.sh` installed
`alsa-utils` and added the operator to the `audio` group purely so a
machine nobody can hear could talk to itself.

WHAT THE SERVER DOES NOW, which is much less:

  * owns the VOCABULARY (`phrases()`, `phrase_for()`, `clip_name()`) --
    unchanged, and deliberately so. What a dart is CALLED is scoring
    vocabulary, it has to agree with `server.retail_dart_fields()`, and
    a second copy of it in JavaScript would be a second thing to keep
    right. The browser is handed the finished phrase string.
  * serves the clip files as static assets, and lists what sets are
    installed (`voices()`, `coverage()`).
  * broadcasts the phrase over the live-events WebSocket the dashboard
    already holds open.

Which device is on, how loud, and in which voice are all PER-DEVICE
settings living in each browser's localStorage -- not here, and not in
config.json. Two screens in the same room will both speak and be a few
tens of milliseconds out of step; that is expected, and it is exactly
why each one needs its own mute rather than one server-wide switch.

WHY PRE-RENDERED CLIPS AT ALL -- the original reason, and it still
holds: live synthesis spends roughly a second before any sound comes
out, which would put a full second between a dart landing and being
called. Clip sets are generated OFFLINE by a maintainer with
`tools/voice/generate_kokoro_clips.py` and committed, so a rig renders
nothing and a headless Linux box with no TTS installed still talks.

Why the clips can be committed at all: they are generated with Kokoro
(Apache-2.0, permissively licensed voice packs). An earlier set of
Apple-`say` clips lived here until 2026-09-13 and was deleted precisely
because redistributing Apple voice output is a licensing risk. That
objection does not apply to these.

ADDING A VOICE is dropping a directory into `assets/voices/`. The
directory name IS the voice as far as the product and its dropdown are
concerned; nothing here enumerates a fixed list. `phrases()` is the set
of terms a set must contain -- render those, name the files with
`clip_name()`, and it works.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger("opendarts.live.audio")

#: MP3, since 2026-09-15. It was WAV for exactly one reason, now gone:
#: `winsound` -- the only thing on Windows that played a file with no
#: external tool -- reads WAV only, and `aplay`/`paplay` on Linux would
#: not take MP3 either. No browser has that constraint; every engine
#: that can render this dashboard has decoded MP3 for twenty years. The
#: committed set went from 5.2 MB to ~0.6 MB on the change, which
#: matters more than it sounds: the browser PREFETCHES the whole
#: vocabulary the moment audio is armed (see the dashboard's
#: `loadVoiceSet`), so this is bytes over someone's Wi-Fi, not bytes on
#: a disk.
CLIP_SUFFIX = ".mp3"

#: Shipped clip sets, one directory per voice, committed to the repo.
#: Multiple sets coexist; the directory name is the selectable voice.
SHIPPED_VOICES_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "voices"

#: Which Kokoro voice id a directory was rendered from, recorded by the
#: generator as a one-line file inside the set.
#:
#: Load-bearing because filenames carry the PHRASE and not the voice:
#: `treble-20.mp3` looks identical in every set, so without this there
#: is nothing on disk that says what "Bella" actually is, and a
#: half-re-rendered directory is indistinguishable from a finished one.
#: It was originally added (2026-09-09) to fix exactly that failure --
#: changing the voice re-rendered nothing because every filename already
#: existed, and the old voice played forever.
#:
#: This constant was DELETED on 2026-09-15 with the on-rig renderer,
#: while `generate_kokoro_clips.py` kept writing it -- so the generator
#: raised AttributeError on its very last line, after spending a minute
#: rendering 64 clips correctly. Restored here, with a reader
#: (`rendered_voice()`), so the marker is something the product can show
#: rather than a file only one tool ever touches.
VOICE_MARKER = ".voice"

#: Used when a browser has no voice stored yet. A name rather than
#: "first one found", so two devices with the same sets available start
#: out sounding the same.
DEFAULT_VOICE = "Bella"


def voices() -> "list[str]":
    """Every installed clip set, by name.

    A directory counts only if it actually holds clips -- an empty folder
    left behind by a deleted set would otherwise be offered as a voice
    that plays nothing.
    """
    if not SHIPPED_VOICES_DIR.is_dir():
        return []
    return sorted(d.name for d in SHIPPED_VOICES_DIR.iterdir()
                  if d.is_dir() and any(d.glob("*" + CLIP_SUFFIX)))


def voice_dir(voice: "str | None") -> "Path | None":
    """The directory for `voice`, or None if there is no such set.

    Matched case-insensitively: the value arrives from a browser's
    localStorage or a URL path segment, and "bella" should find "Bella".

    This is also the ONLY way a request-supplied string becomes a path in
    this product. It resolves against the enumerated set names rather
    than joining the caller's string onto a directory, so `../../etc` is
    not a traversal attempt that has to be spotted and rejected -- it is
    simply a name no installed set has.
    """
    if not voice:
        return None
    wanted = voice.strip().lower()
    for name in voices():
        if name.lower() == wanted:
            return SHIPPED_VOICES_DIR / name
    return None


def phrases() -> Iterator[str]:
    """Every phrase this module can ever be asked to speak.

    Deliberately a closed set, and the contract a clip set has to meet:
    a voice directory must contain a clip for each of these, named by
    `clip_name()`. Anything `phrase_for()` can return must appear here or
    it would fall through to silence at throw time. The two are pinned
    together by test.
    """
    for n in range(1, 21):
        yield str(n)
        yield f"treble {n}"
        yield f"double {n}"
    # Board centre. The green outer ring (25) is "bullseye"; the red
    # inner (50) is "double bullseye" -- the same naming the QA harness
    # rendered, so the two caches agree.
    yield "bullseye"
    yield "double bullseye"
    yield "miss"
    yield "no score"


def phrase_for(sector: Any, ring: Any, ok: Any = True) -> "str | None":
    """The phrase for one scored dart, or None when there is nothing
    honest to say.

    Mirrors `server.retail_dart_fields()`'s vocabulary rather than
    inventing a second one: `outside` is a miss, `bull`/`outer_bull` are
    the two centre rings, and anything that did not score is "no score"
    instead of being silently skipped -- a dart that landed and could not
    be read is exactly the case an operator most wants to hear about.
    """
    if not ok or ring is None:
        return "no score"
    ring_s = str(ring)
    if ring_s == "outside":
        return "miss"
    if ring_s == "bull":
        return "double bullseye"
    if ring_s == "outer_bull":
        return "bullseye"
    try:
        sector_i = int(sector)
    except (TypeError, ValueError):
        return "no score"
    if not 1 <= sector_i <= 20:
        return "no score"
    if ring_s == "treble":
        return f"treble {sector_i}"
    if ring_s == "double":
        return f"double {sector_i}"
    # single_inner / single_outer both read as the bare number: a caller
    # announces "twenty", never "inner twenty".
    return str(sector_i)


def clip_name(text: str) -> str:
    """The filename for one phrase: the phrase itself, slugified.

    Lowercased with spaces as hyphens, so `treble 20` is `treble-20.mp3`
    and a human can find, play, or replace a specific call by name.

    Deliberately not sanitised beyond that: `phrases()` is a closed set of
    plain words and digits, pinned by test, so there is nothing here that
    needs escaping. A defensive slugifier would only hide the day someone
    adds a phrase that does need it.
    """
    return text.strip().lower().replace(" ", "-") + CLIP_SUFFIX


def clip_names() -> "dict[str, str]":
    """The whole vocabulary as phrase -> filename.

    THIS is what the browser is handed, and the reason there is no
    slugifier in the dashboard's JavaScript. The client receives a
    phrase on the wire ("treble 20") and needs a URL; giving it the map
    means the naming rule stays in one language, in one function, pinned
    by one set of tests. The same map serves every voice -- filenames
    carry the phrase and not the voice -- so it is fetched once.
    """
    return {text: clip_name(text) for text in dict.fromkeys(phrases())}


def clip_path(text: str, voice: "str | None") -> "Path | None":
    """The clip to play for `text` in `voice`, or None."""
    d = voice_dir(voice)
    if d is None:
        return None
    p = d / clip_name(text)
    return p if p.is_file() else None


def rendered_voice(voice: "str | None") -> "str | None":
    """The Kokoro voice id a set was rendered from, per its `.voice`
    marker, or None if the set has none.

    Reported rather than acted on. An older set predating the marker is
    perfectly playable, so its absence is a blank in a diagnostic line,
    never a reason to refuse to serve a directory full of working clips.
    """
    d = voice_dir(voice)
    if d is None:
        return None
    try:
        return (d / VOICE_MARKER).read_text().strip() or None
    except OSError:
        return None


def coverage(voice: "str | None" = None) -> "dict[str, Any]":
    """How much of the vocabulary one voice set actually contains.

    The surviving half of what this function used to do. It answered two
    questions at once -- "is this set complete" and "can this machine
    play a sound" -- and the second one no longer has a server-side
    answer to give: whether a sound is audible is a fact about a browser
    on someone's iPad, which reports it back separately (see
    `POST /api/audio/clients`). What is left is the diagnostic that was
    always the useful one: **a voice set with 61 of 64 clips is silent on
    exactly the three darts nobody throws in testing**, and an operator
    should be able to see that before it happens rather than after.

    An unknown voice is reported as `installed: false` rather than
    substituted for one that exists. Substitution made sense when the
    voice was a rig-wide config value that a fat-fingered edit could
    render mute; now the picker only ever offers names this server just
    enumerated, so a miss means something has genuinely gone away, and
    quietly answering about a DIFFERENT set would hide it.
    """
    wanted = list(dict.fromkeys(phrases()))
    d = voice_dir(voice)
    present = (0 if d is None
               else sum(1 for t in wanted if (d / clip_name(t)).is_file()))
    return {
        # The canonical spelling, not the caller's: "bella" resolves to
        # the "Bella" the URLs and the picker use.
        "voice": d.name if d is not None else voice,
        "installed": d is not None,
        "present": present,
        "total": len(wanted),
        "complete": d is not None and present == len(wanted),
        "dir": str(d) if d is not None else None,
        "rendered_from": rendered_voice(voice),
    }
