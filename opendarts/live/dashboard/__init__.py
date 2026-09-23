"""The dashboard page, as three real files instead of one Python f-string.

`index.html` is the whole page skeleton, `app.css` the `<style>` block and
`app.js` the `<script>` block; `opendarts.live.server._render_dashboard_html`
reads all three once at import and inlines them into the single document
`GET /` has always served. There is no build step and no second request:
this is a text relocation, not a change of how the page is delivered.

Why they moved out of `server.py`: the page used to be one ~6,000-line
f-string, which meant every CSS and JS brace had to be written doubled
(`{{`/`}}`). Three separate times a text substitution got that wrong and
shipped a page that loaded, rendered, and then died at the first JavaScript
error -- taking every control with it, including the tab switcher.
`tests/test_dashboard_js_syntax.py` exists because of those three.

The four placeholders `index.html` carries, all filled server-side:

  ``@@OD_APP_CSS@@``        contents of app.css
  ``@@OD_APP_JS@@``         contents of app.js
  ``@@OD_HOST_LABEL@@``     this machine's name, for the tab title (escaped)
  ``@@OD_CAM_CARDS@@``      one camera card per slot, built from n_cameras
  ``@@OD_BOOTSTRAP_JSON@@`` every other server-computed value, as one JSON
                            object the script reads once (``OD_BOOTSTRAP``)

app.css and app.js carry no placeholders at all -- they are static assets a
browser, an editor or `node --check` can read directly.

A trailing newline on app.css/app.js is a file convention, not page content:
the reader strips exactly one before inlining, so the served bytes are the
same as when the text lived inside the f-string.
"""
