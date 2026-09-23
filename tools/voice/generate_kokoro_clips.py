#!/usr/bin/env python3
"""Render a full clip set with Kokoro TTS into assets/voices/<Name>/.

WHY KOKORO, AND WHY THESE CAN BE SHIPPED. A set of 64 Apple-`say` clips
used to live in `assets/voices/` and was deleted on 2026-09-13, because
redistributing Apple voice output in a public repo is a licensing risk.
Kokoro is Apache-2.0 and its voice packs are permissively licensed, so a
generated set can be committed and shipped -- which is the whole point:
a rig that cannot render (a headless Linux box with no TTS installed)
still gets spoken calls.

NOT RUN AT BUILD TIME, and not a dependency of the product. Kokoro pulls
in torch, which is gigabytes -- nothing a dart scorer should require to
start. This is a maintainer tool: run it once per voice, commit the
result, and the product only ever reads the files.

MP3, AS OF 2026-09-15. This wrote WAV for one reason: `winsound` on
Windows reads WAV only, and `aplay`/`paplay` on Linux would not take MP3
either, so WAV was the one format every player could manage. Spoken
calls now play in the BROWSER, and no browser has that constraint -- so
the format is chosen on merit instead, and the merit is size. The two
committed sets went from 5.2 MB of WAV to about 0.6 MB of MP3.

Size matters more here than it did: the browser prefetches the WHOLE
64-clip vocabulary the moment a screen turns sound on, so this is bytes
crossing someone's Wi-Fi every time a device is armed, not bytes sitting
on a disk. libsndfile encodes MP3 directly (>= 1.1), so this needs no
encoder beyond the `soundfile` already imported.

Usage, with the venv that has kokoro installed:

    ~/kokoro/bin/python tools/voice/generate_kokoro_clips.py \\
        --voice af_bella --name Bella

    ~/kokoro/bin/python tools/voice/generate_kokoro_clips.py \\
        --voice bm_daniel --name Daniel

Existing files are skipped unless --force is passed, so a re-run fills
gaps rather than rewriting the set.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

#: Kokoro emits 24 kHz float audio, and that rate is kept end to end --
#: resampling speech this short to a "nicer" 44.1 kHz would add bytes and
#: an interpolation pass to reproduce exactly the information the model
#: generated. 24 kHz mono MP3 lands around 42 kbps at libsndfile's
#: default quality, measured on the shipped set.
SAMPLE_RATE = 24000

#: libsndfile's own MP3 encoder (LAME under the hood). Named as constants
#: rather than inlined because they are the two things anyone re-rendering
#: a set would want to change, and because `format=` is not inferable
#: from the extension when the path is passed as a string.
CLIP_FORMAT = "MP3"
CLIP_SUBTYPE = "MPEG_LAYER_III"

#: Silence either side of the speech, in seconds. Kokoro pads generously
#: -- measured on a first pass, "treble 20" came out 1.75s of which 0.39s
#: was leading silence and 0.59s trailing, for 0.77s of actual speech.
#:
#: The leading pad is the one that matters: it is dead time between the
#: dart being scored and the call being heard, on a throw path this whole
#: module exists to keep short. Trimming it also halves the committed
#: size. A small pad is kept so the attack is not clipped.
TRIM_THRESHOLD = 0.02        # of peak amplitude
KEEP_PAD_S = 0.02


def trim_silence(samples, sample_rate: int):
    """Drop leading/trailing silence, keeping a short pad.

    Returns the samples unchanged if the clip is silent throughout, which
    would otherwise trim to nothing and write an empty file.
    """
    import numpy as np

    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if peak <= 0.0:
        return samples
    loud = np.where(np.abs(samples) > peak * TRIM_THRESHOLD)[0]
    if loud.size == 0:
        return samples
    pad = int(KEEP_PAD_S * sample_rate)
    start = max(0, int(loud[0]) - pad)
    end = min(len(samples), int(loud[-1]) + pad)
    return samples[start:end]


def spoken(phrase: str) -> str:
    """The phrase as Kokoro should say it.

    `phrases()` yields the product's own vocabulary -- "treble 20",
    "double bullseye" -- which is also what the filenames are built from.
    Kokoro reads digits fine, but two phrasings need help:

      * "bullseye" as one word comes out flat on the back half. The
        apostrophe-hyphen spelling gives "eye" its own stress. Same fix
        the FlightDeck project landed on by ear.
      * "no score" is read as two disconnected words without the comma
        pause that makes it sound like a call rather than a label.
    """
    if phrase == "bullseye":
        return "Bull's-eye"
    if phrase == "double bullseye":
        return "Double bull's-eye"
    if phrase == "no score":
        return "No score"
    return phrase


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True,
                    help="Kokoro voice id, e.g. af_bella or bm_daniel")
    ap.add_argument("--name", required=True,
                    help="Directory name under assets/voices/, e.g. Bella")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--force", action="store_true",
                    help="re-render clips that already exist")
    args = ap.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as exc:
        print(f"error: {exc}\n\nRun this with a python that has kokoro "
              f"installed, e.g. ~/kokoro/bin/python", file=sys.stderr)
        return 2

    from opendarts.live import audio

    out_dir = REPO_ROOT / "assets" / "voices" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # lang_code is the first letter of the voice id: 'a' American, 'b' British.
    pipeline = KPipeline(lang_code=args.voice[0])

    phrases = list(audio.phrases())
    written = skipped = 0
    for phrase in phrases:
        dest = out_dir / audio.clip_name(phrase)
        if dest.is_file() and not args.force:
            skipped += 1
            continue
        chunks = [chunk.audio for chunk in
                  pipeline(spoken(phrase), voice=args.voice, speed=args.speed)]
        if not chunks:
            print(f"  !! no audio produced for {phrase!r}", file=sys.stderr)
            continue
        samples = np.concatenate([np.asarray(c, dtype="float32") for c in chunks])
        samples = trim_silence(samples, SAMPLE_RATE)
        sf.write(str(dest), samples, SAMPLE_RATE,
                 format=CLIP_FORMAT, subtype=CLIP_SUBTYPE)
        written += 1
        print(f"  {dest.name}")

    # The product records which voice a directory holds, because filenames
    # carry the phrase and not the voice -- without this, changing voice
    # and re-rendering silently kept the old one.
    #
    # THIS LINE CRASHED between 2026-09-15 and its repair the same day.
    # `audio.VOICE_MARKER` was deleted along with the on-rig renderer
    # while this tool kept writing it, so every run raised AttributeError
    # here -- AFTER spending a minute rendering all 64 clips correctly,
    # which is the worst possible place for it: the set on disk looked
    # finished, the exit code said failure, and the one file that records
    # what the set actually IS was the one that never got written. The
    # constant is back in opendarts/live/audio.py with a reader
    # (`rendered_voice()`), so it is now something the product reports
    # rather than a file only this script ever touches.
    (out_dir / audio.VOICE_MARKER).write_text(args.voice + "\n")

    total = sum(1 for p in phrases if (out_dir / audio.clip_name(p)).is_file())
    print(f"\n{args.name}: {written} written, {skipped} skipped, "
          f"{total}/{len(phrases)} present in {out_dir}")
    return 0 if total == len(phrases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
