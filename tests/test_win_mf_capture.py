"""The Media Foundation MJPG capture.

The real work only happens on Windows against a real camera. What can be
pinned here is the part that must hold everywhere: the module imports and
answers politely off Windows, and the subtype decoding is right.
"""
from __future__ import annotations

from types import SimpleNamespace

from opendarts.live import win_mf_capture
from opendarts.live.camera_names import _guid


def test_off_windows_it_declines_rather_than_touching_windll(monkeypatch):
    monkeypatch.setattr(win_mf_capture, "platform", SimpleNamespace(system=lambda: "Linux"))
    assert win_mf_capture.list_devices() == []
    cap = win_mf_capture.MfJpegCapture(0)
    assert cap.isOpened() is False
    assert cap.error == "not Windows"


def test_video_subtypes_decode_to_their_fourcc():
    mjpg = _guid("{47504A4D-0000-0010-8000-00AA00389B71}")
    nv12 = _guid("{3231564E-0000-0010-8000-00AA00389B71}")
    rgb24 = _guid("{00000014-0000-0010-8000-00AA00389B71}")
    assert win_mf_capture._fourcc_of(mjpg) == "MJPG"
    assert win_mf_capture._fourcc_of(nv12) == "NV12"
    assert win_mf_capture._fourcc_of(rgb24) == "20"
    assert mjpg.Data1 == win_mf_capture.FOURCC_MJPG


def test_hresult_constants_are_the_signed_values_ctypes_returns():
    assert win_mf_capture.RPC_E_CHANGED_MODE == 0x80010106 - 2 ** 32
    assert (win_mf_capture.MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED
            == 0xC00D3EA3 - 2 ** 32)
