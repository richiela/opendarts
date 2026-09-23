# Contributing

Issues and bug reports are welcome; see below for pull requests.

## Reporting a problem

The **Info** tab has an **About this rig** section with a **Copy
diagnostics** button. Press it and paste the result into your issue — it
carries the build, platform, Python and OpenCV versions, camera
resolutions, the Autodarts and audio state, and any last start error.
Everything a maintainer would otherwise have to ask for, in one paste.

If the rig will not start at all, `GET /api/health` returns the same build
block, and `GET /api/logs/run_product?n=200` tails the log.

## Before opening a pull request

Run the suite. It needs a few packages the product itself does not, so
install `requirements-dev.txt` first:

```sh
pip install -r requirements-dev.txt
pytest tests/
```

A few things this project is strict about:

- **Measure, do not assume.** Any accuracy or performance claim is made by
  replaying saved throw packages or by measuring on a rig, never by
  reasoning about the code. `docs/DESIGN.md` has the standing constraints.
- **The dashboard is three plain files** in `opendarts/live/dashboard/`:
  `index.html`, `app.css` and `app.js`. The server reads them once at
  startup and serves the assembled page from `GET /`; there is no build
  step. Edit the files, not `server.py`. Install
  [Node.js](https://nodejs.org) before touching the JavaScript: one test
  parses the rendered script with `node --check`, and it is the only check
  that catches an unterminated string literal there.
- **New tests should fail when the fix is reverted.** If a test passes
  either way it is not pinning anything.

## Patches, and why I can't merge them yet

**Issues are very welcome. Pull requests I can't merge right now** — and
the reason is specific rather than bureaucratic.

This project is [licensed](LICENSE) under the GNU AGPL v3, with
commercial exceptions available by separate arrangement. Selling an
exception to the copyleft is only possible while one party holds the
rights to all of the code. Copyright works per-author: a merged
contribution stays owned by whoever wrote it, and from that moment no
exception can be sold without tracking down every contributor for
permission. One refusal, or one person who has changed email address,
ends it permanently.

Note what this does *not* restrict. The AGPL itself is granted to
everyone, free, for any purpose including commercial use — nothing about
the merge policy narrows what you may do with the code. The only thing
held in reserve is the right to sell someone relief from copyleft, and
that right evaporates the moment the copyright is shared.

So the position is not "contributions are unwelcome". It is that the
paperwork which makes them safe does not exist yet, and merging without
it is the one move here that cannot be undone.

### What to do instead

**Open an issue.** Describe what you saw, what you expected, and how to
reproduce it. A clear bug report carries no licensing question at all and
is genuinely the more useful half of most fixes — finding the problem is
usually harder than fixing it.

**If you have already written the fix, say so in the issue.** Describing
what needs changing is fine; ideas are not copyrightable, only their
expression is. You will get credit in the commit message.

**If you have something substantial**, open an issue and say so. A
contributor licence agreement can be set up at that point — it takes
about ten minutes and only needs doing once.

A pull request will get a reply either way. Leaving one to rot is worse
than declining it.
