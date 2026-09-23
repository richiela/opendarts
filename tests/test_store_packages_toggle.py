"""store_packages=False must skip every on-disk write and nothing else."""

from opendarts.live import capture_daemon


def _has_param(fn, name):
    import inspect
    return name in inspect.signature(fn).parameters


def test_both_entry_points_accept_the_flag():
    assert _has_param(capture_daemon.handle_ready_to_capture, "store_packages")
    assert _has_param(capture_daemon.run_capture_loop_body, "store_packages")


def test_default_is_to_store():
    import inspect
    for fn in (capture_daemon.handle_ready_to_capture, capture_daemon.run_capture_loop_body):
        assert inspect.signature(fn).parameters["store_packages"].default is True, (
            "a rig that silently discards its own evidence is a bad default"
        )


def test_skip_happens_before_the_save_dispatch():
    """The guard must sit ahead of both the threaded and inline save
    paths -- skipping only one would still write packages half the time."""
    import inspect
    src = inspect.getsource(capture_daemon.handle_ready_to_capture)
    guard = src.index("if not store_packages:")
    thread = src.index('name="throw-package-save"')
    assert guard < thread, "the storage guard must precede the save dispatch"


def test_throw_detected_is_emitted_before_the_storage_guard():
    """The live feed and retail channel must behave identically with
    storage on or off -- that is the whole point."""
    import inspect
    src = inspect.getsource(capture_daemon.handle_ready_to_capture)
    assert src.index("THROW_DETECTED") < src.index("if not store_packages:")


def test_config_reader_defaults_safely(monkeypatch):
    from opendarts.live import run_product as rp
    monkeypatch.setattr("opendarts.live.config.read_config_section",
                        lambda *a, **k: None)
    assert rp._read_store_packages() is True

    monkeypatch.setattr("opendarts.live.config.read_config_section",
                        lambda *a, **k: "yes please")
    assert rp._read_store_packages() is True, "a bad value must fall back to storing"

    monkeypatch.setattr("opendarts.live.config.read_config_section",
                        lambda *a, **k: False)
    assert rp._read_store_packages() is False

    def boom(*a, **k):
        raise OSError("unreadable config")
    monkeypatch.setattr("opendarts.live.config.read_config_section", boom)
    assert rp._read_store_packages() is True, "a broken config must not stop a session"
