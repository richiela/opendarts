"""Media Foundation camera access that asks for MJPG and keeps the bytes.

WHY THIS EXISTS. On Linux, OpenCV hands over a camera's own JPEG when
CAP_PROP_CONVERT_RGB is off (measured 2026-09-17), so every
stream and loopback can carry the camera's bytes unchanged instead of a
second-generation re-encode. On Windows, OpenCV's wrappers never get
there:

  * CAP_MSMF with conversion off does select a NATIVE media type, but it
    ranks native types by resolution and frame rate only -- the subtype
    is never considered -- so on a Windows rig it picked NV12 over MJPG and returned
    uncompressed bytes.
  * CAP_DSHOW can negotiate MJPG on the pin, but its sample grabber is
    always configured for decoded pixels.

Neither is a Windows limitation. Media Foundation's Source Reader will
deliver the compressed payload if it is asked for the MJPG native type
explicitly, with converters disabled. This module does exactly that,
through ctypes COM (the same no-new-dependency approach as
camera_names.py), and nothing else.

IMPORT-SAFE ON EVERY PLATFORM. Nothing Windows-specific is touched at
import time -- `ctypes.windll` and `WINFUNCTYPE` only exist on Windows,
and tests/test_windows_import_safety.py imports every module with the
UNIX-only stdlib hidden, so the reverse (Windows-only names at module
scope) would be the same class of bug.
"""
from __future__ import annotations

import ctypes
import logging
import platform
import threading
import time
from typing import Any

from opendarts.live.camera_names import _GUID, _guid

log = logging.getLogger("opendarts.live.win_mf_capture")

# --- constants --------------------------------------------------------------

MF_VERSION = 0x00020070                       # MF_SDK_VERSION << 16 | MF_API_VERSION
MFSTARTUP_FULL = 0
COINIT_MULTITHREADED = 0x0
RPC_E_CHANGED_MODE = -2147417850              # 0x80010106
MF_SOURCE_READER_FIRST_VIDEO_STREAM = 0xFFFFFFFC
MF_SOURCE_READERF_ERROR = 0x1
MF_SOURCE_READERF_ENDOFSTREAM = 0x2

MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE = "{c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3}"
MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID = "{8ac3587a-4ae7-42d8-99e0-0a6013eef90f}"
MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME = "{60d0e559-52f8-4fa2-bbce-acdb34a8ec01}"
MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK = "{58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f}"
IID_IMFMediaSource = "{279a808d-aec7-40c8-9c6b-a6b492c78a66}"
MF_MT_SUBTYPE = "{f7e34c9a-42e8-4714-b74b-cb29d72c35e5}"
MF_MT_FRAME_SIZE = "{1652c33d-d6b2-4012-b834-72030849a37d}"
MF_MT_FRAME_RATE = "{c459a2e8-3d2c-4e44-b132-fee5156c7bb0}"
MF_READWRITE_DISABLE_CONVERTERS = "{98d5b065-1374-4847-8d5d-31520fee7156}"

#: Video subtype GUIDs are FOURCC-based: Data1 is the FOURCC, the rest is
#: this fixed tail. MJPG's Data1 is 0x47504A4D.
FOURCC_MJPG = 0x47504A4D

# Vtable slots. IUnknown is 0-2; IMFAttributes adds 3-32.
_RELEASE = 2
_ATTR_GET_UINT64 = 8
_ATTR_GET_GUID = 10
_ATTR_GET_ALLOCATED_STRING = 13
_ATTR_SET_UINT32 = 21
_ATTR_SET_GUID = 24
_ACTIVATE_ACTIVATE_OBJECT = 33                # IMFActivate : IMFAttributes
_READER_GET_NATIVE_MEDIA_TYPE = 5             # IMFSourceReader : IUnknown
_READER_SET_CURRENT_MEDIA_TYPE = 7
_READER_READ_SAMPLE = 9
_SAMPLE_CONVERT_TO_CONTIGUOUS = 41            # IMFSample : IMFAttributes
_BUFFER_LOCK = 3                              # IMFMediaBuffer : IUnknown
_BUFFER_UNLOCK = 4
_SOURCE_SHUTDOWN = 12                         # IMFMediaSource


def _hr(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08X}"


def _fourcc_of(guid: _GUID) -> str:
    """A video subtype GUID as its FOURCC, or its Data1 number."""
    raw = int(guid.Data1).to_bytes(4, "little")
    if all(32 <= c < 127 for c in raw):
        return raw.decode("ascii")
    return str(int(guid.Data1))


class _Com:
    """Just enough ctypes COM plumbing, bound lazily to Windows."""

    def __init__(self) -> None:
        self.ole32 = ctypes.windll.ole32              # type: ignore[attr-defined]
        self.mfplat = ctypes.windll.mfplat            # type: ignore[attr-defined]
        self.mf = ctypes.windll.mf                    # type: ignore[attr-defined]
        self.mfreadwrite = ctypes.windll.mfreadwrite  # type: ignore[attr-defined]
        self.ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        self.ole32.CoTaskMemFree.restype = None

    @staticmethod
    def call(obj: ctypes.c_void_p, index: int, restype: Any,
             argtypes: "tuple[Any, ...]", *args: Any) -> Any:
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined]
        return proto(vtable[index])(obj, *args)

    def release(self, obj: ctypes.c_void_p) -> None:
        if obj and obj.value:
            self.call(obj, _RELEASE, ctypes.c_ulong, ())

    def get_guid(self, attrs: ctypes.c_void_p, key: str) -> "_GUID | None":
        out = _GUID()
        k = _guid(key)
        hr = self.call(attrs, _ATTR_GET_GUID, ctypes.c_long,
                       (ctypes.POINTER(_GUID), ctypes.POINTER(_GUID)),
                       ctypes.byref(k), ctypes.byref(out))
        return out if hr >= 0 else None

    def get_uint64(self, attrs: ctypes.c_void_p, key: str) -> "int | None":
        out = ctypes.c_uint64()
        k = _guid(key)
        hr = self.call(attrs, _ATTR_GET_UINT64, ctypes.c_long,
                       (ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_uint64)),
                       ctypes.byref(k), ctypes.byref(out))
        return int(out.value) if hr >= 0 else None

    def get_string(self, attrs: ctypes.c_void_p, key: str) -> "str | None":
        out = ctypes.c_wchar_p()
        length = ctypes.c_uint32()
        k = _guid(key)
        hr = self.call(attrs, _ATTR_GET_ALLOCATED_STRING, ctypes.c_long,
                       (ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_wchar_p),
                        ctypes.POINTER(ctypes.c_uint32)),
                       ctypes.byref(k), ctypes.byref(out), ctypes.byref(length))
        if hr < 0:
            return None
        try:
            return out.value
        finally:
            self.ole32.CoTaskMemFree(ctypes.cast(out, ctypes.c_void_p))

    def set_guid(self, attrs: ctypes.c_void_p, key: str, value: str) -> int:
        k, v = _guid(key), _guid(value)
        return self.call(attrs, _ATTR_SET_GUID, ctypes.c_long,
                         (ctypes.POINTER(_GUID), ctypes.POINTER(_GUID)),
                         ctypes.byref(k), ctypes.byref(v))

    def set_uint32(self, attrs: ctypes.c_void_p, key: str, value: int) -> int:
        k = _guid(key)
        return self.call(attrs, _ATTR_SET_UINT32, ctypes.c_long,
                         (ctypes.POINTER(_GUID), ctypes.c_uint32),
                         ctypes.byref(k), value)

    def create_attributes(self) -> ctypes.c_void_p:
        obj = ctypes.c_void_p()
        hr = self.mfplat.MFCreateAttributes(ctypes.byref(obj), ctypes.c_uint32(2))
        if hr < 0:
            raise OSError(f"MFCreateAttributes failed {_hr(hr)}")
        return obj


def _describe_type(com: _Com, media_type: ctypes.c_void_p) -> "dict[str, Any]":
    sub = com.get_guid(media_type, MF_MT_SUBTYPE)
    size = com.get_uint64(media_type, MF_MT_FRAME_SIZE)
    rate = com.get_uint64(media_type, MF_MT_FRAME_RATE)
    fps = None
    if rate:
        num, den = rate >> 32, rate & 0xFFFFFFFF
        fps = round(num / den, 2) if den else None
    return {
        "subtype": _fourcc_of(sub) if sub is not None else None,
        "is_mjpg": bool(sub is not None and sub.Data1 == FOURCC_MJPG),
        "width": (size >> 32) if size else None,
        "height": (size & 0xFFFFFFFF) if size else None,
        "fps": fps,
    }


def device_key(path: "str | None") -> "str | None":
    """A camera's device path, reduced to the part that names the DEVICE.

    Media Foundation and DirectShow report the same camera's path with a
    different interface-class GUID on the end (`#{e5323777-...}` versus
    `#{65e8773d-...}`); everything before that is the same instance ID.
    Lower-cased because the two APIs do not agree on case either.
    """
    if not path:
        return None
    p = path.lower()
    cut = p.rfind("#{")
    return p[:cut] if cut > 0 else p


def list_devices() -> "list[tuple[str, str | None]]":
    """(friendly name, symbolic link) for every camera, in Media
    Foundation's order -- the order MfJpegCapture's `device` counts in.
    [] off Windows or when enumeration fails.

    Media Foundation lists only real capture devices, never a DirectShow
    filter such as this product's own virtual cameras.
    """
    if platform.system() != "Windows" or not hasattr(ctypes, "WINFUNCTYPE"):
        return []
    try:
        com = _Com()
        _ensure_com(com)
    except Exception:  # noqa: BLE001 -- a listing must answer, not raise
        return []
    hr = com.mfplat.MFStartup(ctypes.c_ulong(MF_VERSION), ctypes.c_ulong(MFSTARTUP_FULL))
    if hr < 0:
        return []
    attrs = ctypes.c_void_p()
    devices_ptr = ctypes.POINTER(ctypes.c_void_p)()
    count = ctypes.c_uint32()
    out: "list[tuple[str, str | None]]" = []
    try:
        attrs = com.create_attributes()
        com.set_guid(attrs, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                     MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
        if com.mf.MFEnumDeviceSources(attrs, ctypes.byref(devices_ptr),
                                      ctypes.byref(count)) < 0:
            return []
        for i in range(count.value):
            act = ctypes.c_void_p(devices_ptr[i])
            out.append((
                com.get_string(act, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME) or "",
                com.get_string(act, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK),
            ))
        return out
    except Exception:  # noqa: BLE001
        return out
    finally:
        if devices_ptr:
            for i in range(count.value):
                com.release(ctypes.c_void_p(devices_ptr[i]))
            com.ole32.CoTaskMemFree(ctypes.cast(devices_ptr, ctypes.c_void_p))
        com.release(attrs)
        com.mfplat.MFShutdown()


# --- a capture that reads like cv2.VideoCapture ------------------------------

#: How long a lost camera waits before the next reopen attempt. Media
#: Foundation hands the camera to whichever controller opened it last, and
#: the one it took it from gets MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED; a
#: tight retry loop would just fight the other process for it.
REOPEN_AFTER_S = 2.0

MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED = -1072873821   # 0xC00D3EA3

#: Empty ReadSample results tolerated in one read() before giving up.
MAX_EMPTY_SAMPLES = 30

_CV_FRAME_WIDTH = 3        # cv2.CAP_PROP_* numbers, so this module needs no cv2
_CV_FRAME_HEIGHT = 4
_CV_FPS = 5
_CV_FOURCC = 6


def _read_sample_bytes(com: _Com, reader: ctypes.c_void_p) -> "tuple[int, int, bytes | None]":
    """One ReadSample. Returns (hr, stream flags, payload or None)."""
    sample = ctypes.c_void_p()
    actual = ctypes.c_uint32()
    flags = ctypes.c_uint32()
    ts = ctypes.c_longlong()
    hr = com.call(reader, _READER_READ_SAMPLE, ctypes.c_long,
                  (ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                   ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_longlong),
                   ctypes.POINTER(ctypes.c_void_p)),
                  MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, ctypes.byref(actual),
                  ctypes.byref(flags), ctypes.byref(ts), ctypes.byref(sample))
    if hr < 0 or not sample.value:
        com.release(sample)
        return hr, flags.value, None
    buffer = ctypes.c_void_p()
    try:
        hr = com.call(sample, _SAMPLE_CONVERT_TO_CONTIGUOUS, ctypes.c_long,
                      (ctypes.POINTER(ctypes.c_void_p),), ctypes.byref(buffer))
        if hr < 0:
            return hr, flags.value, None
        ptr = ctypes.POINTER(ctypes.c_ubyte)()
        max_len = ctypes.c_uint32()
        cur_len = ctypes.c_uint32()
        hr = com.call(buffer, _BUFFER_LOCK, ctypes.c_long,
                      (ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
                       ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)),
                      ctypes.byref(ptr), ctypes.byref(max_len), ctypes.byref(cur_len))
        if hr < 0:
            return hr, flags.value, None
        try:
            return hr, flags.value, ctypes.string_at(ptr, cur_len.value)
        finally:
            com.call(buffer, _BUFFER_UNLOCK, ctypes.c_long, ())
    finally:
        com.release(buffer)
        com.release(sample)


_thread_com = threading.local()


def _ensure_com(com: _Com) -> None:
    """Join the multithreaded apartment on THIS thread, once.

    The hub reads from a pool of worker threads, and a COM call from a
    thread that never initialised COM fails. Never uninitialised: the
    pool's threads live as long as the hub, and leaving the apartment
    under a live reader would be the worse bug.
    """
    if getattr(_thread_com, "ready", False):
        return
    hr = com.ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    if hr < 0 and hr != RPC_E_CHANGED_MODE:
        raise OSError(f"CoInitializeEx {_hr(hr)}")
    _thread_com.ready = True


class MfJpegCapture:
    """One camera through a Media Foundation Source Reader, delivering the
    camera's own JPEG.

    Shaped like the part of cv2.VideoCapture the capture hub uses, so the
    hub opens it in the same backend loop as everything else:

      * isOpened()  the device was found and activated
      * set()       records width / height / fps; the stream is configured
                    on the first read(), once all three are known, the way
                    a cv2 backend applies them
      * read()      (True, 1-D uint8 array of JPEG bytes) -- what OpenCV's
                    V4L2 backend returns with CONVERT_RGB off -- or
                    (False, None)
      * get()       the negotiated size / rate / FOURCC
      * release()

    `device` is the Media Foundation enumeration index, which is the same
    order CAP_MSMF uses -- both enumerate with MFEnumDeviceSources.

    A camera lost to another controller (PREEMPTED) is reopened after
    REOPEN_AFTER_S rather than given up on: that is what happens when
    another app briefly takes the camera and lets it go.
    """

    def __init__(self, device: Any) -> None:
        self.device = int(device)
        self.width = 1280
        self.height = 720
        self.fps = 30.0
        self.error: "str | None" = None
        self.negotiated: "dict[str, Any] | None" = None
        self._com: "_Com | None" = None
        self._source = ctypes.c_void_p()
        self._reader = ctypes.c_void_p()
        self._configured = False
        self._retry_at = 0.0
        self._released = False
        self._opened = False
        if platform.system() != "Windows" or not hasattr(ctypes, "WINFUNCTYPE"):
            self.error = "not Windows"
            return
        try:
            self._com = _Com()
            _ensure_com(self._com)
            hr = self._com.mfplat.MFStartup(ctypes.c_ulong(MF_VERSION),
                                            ctypes.c_ulong(MFSTARTUP_FULL))
            if hr < 0:
                raise OSError(f"MFStartup {_hr(hr)}")
            self._activate()
            self._opened = True
        except Exception as exc:  # noqa: BLE001 -- an open must answer, not raise
            self.error = f"{type(exc).__name__}: {exc}"
            self._close_reader()

    # -- cv2.VideoCapture surface ---------------------------------------

    def isOpened(self) -> bool:  # noqa: N802 -- matches cv2
        # Stays True while a lost camera is waiting to be reopened: the hub
        # stops calling read() on a capture that says it is closed, and
        # read() is where the reopen happens.
        return self._opened and not self._released

    def set(self, prop: int, value: float) -> bool:
        if prop == _CV_FRAME_WIDTH:
            self.width = int(value)
        elif prop == _CV_FRAME_HEIGHT:
            self.height = int(value)
        elif prop == _CV_FPS:
            self.fps = float(value)
        elif prop == _CV_FOURCC:
            # Only MJPG is ever read, so asking for it is agreeing.
            return int(value) == FOURCC_MJPG
        else:
            return False
        self._configured = False
        return True

    def get(self, prop: int) -> float:
        got = self.negotiated or {}
        if prop == _CV_FRAME_WIDTH:
            return float(got.get("width") or 0)
        if prop == _CV_FRAME_HEIGHT:
            return float(got.get("height") or 0)
        if prop == _CV_FPS:
            return float(got.get("fps") or 0)
        if prop == _CV_FOURCC:
            return float(FOURCC_MJPG) if got else 0.0
        return 0.0

    def read(self) -> "tuple[bool, Any]":
        import numpy as np

        if self._released or self._com is None:
            return False, None
        try:
            _ensure_com(self._com)
            if not self._reader.value:
                if time.monotonic() < self._retry_at:
                    time.sleep(0.05)
                    return False, None
                if not self._source.value:
                    self._activate()
                self._open_reader()
            if not self._configured:
                self._configure()
            # BLOCK FOR A FRAME, as cv2's read() does. ReadSample can
            # succeed with no sample -- a stream tick, or the first calls
            # after the type is set -- and on a Windows rig the very first read did,
            # which the hub took as "no usable JPEG" and fell back to
            # CAP_MSMF. Bounded so a camera that goes quiet still returns.
            for _ in range(MAX_EMPTY_SAMPLES):
                hr, flags, data = _read_sample_bytes(self._com, self._reader)
                if hr < 0 or data is not None or flags & (
                        MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM):
                    break
        except Exception as exc:  # noqa: BLE001 -- one bad read must not kill the pump
            self._lost(f"{type(exc).__name__}: {exc}")
            return False, None
        if hr < 0 or flags & (MF_SOURCE_READERF_ERROR | MF_SOURCE_READERF_ENDOFSTREAM):
            reason = ("taken by another app (PREEMPTED)"
                      if hr == MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED
                      else f"ReadSample {_hr(hr)} flags 0x{flags:X}")
            self._lost(reason)
            return False, None
        if data is None:
            self.error = f"no frame after {MAX_EMPTY_SAMPLES} empty reads"
            return False, None
        return True, np.frombuffer(data, dtype=np.uint8)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._close_reader()
        self._close_source()
        if self._com is not None:
            self._com.mfplat.MFShutdown()     # balances the MFStartup above

    # -- internals -------------------------------------------------------

    def _activate(self) -> None:
        com = self._com
        assert com is not None
        enum_attrs = com.create_attributes()
        devices_ptr = ctypes.POINTER(ctypes.c_void_p)()
        count = ctypes.c_uint32()
        try:
            com.set_guid(enum_attrs, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                         MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
            hr = com.mf.MFEnumDeviceSources(enum_attrs, ctypes.byref(devices_ptr),
                                            ctypes.byref(count))
            if hr < 0:
                raise OSError(f"MFEnumDeviceSources {_hr(hr)}")
            if not 0 <= self.device < count.value:
                raise OSError(f"no Media Foundation device {self.device} "
                              f"(have {count.value})")
            activate = ctypes.c_void_p(devices_ptr[self.device])
            iid = _guid(IID_IMFMediaSource)
            source = ctypes.c_void_p()
            hr = com.call(activate, _ACTIVATE_ACTIVATE_OBJECT, ctypes.c_long,
                          (ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)),
                          ctypes.byref(iid), ctypes.byref(source))
            if hr < 0 or not source.value:
                raise OSError(f"ActivateObject {_hr(hr)}")
            self._source = source
        finally:
            if devices_ptr:
                for i in range(count.value):
                    com.release(ctypes.c_void_p(devices_ptr[i]))
                com.ole32.CoTaskMemFree(ctypes.cast(devices_ptr, ctypes.c_void_p))
            com.release(enum_attrs)

    def _open_reader(self) -> None:
        com = self._com
        assert com is not None
        attrs = com.create_attributes()
        try:
            # Converters OFF, or the reader may decode to NV12 for us.
            com.set_uint32(attrs, MF_READWRITE_DISABLE_CONVERTERS, 1)
            reader = ctypes.c_void_p()
            hr = com.mfreadwrite.MFCreateSourceReaderFromMediaSource(
                self._source, attrs, ctypes.byref(reader))
            if hr < 0 or not reader.value:
                raise OSError(f"MFCreateSourceReaderFromMediaSource {_hr(hr)}")
            self._reader = reader
            self._configured = False
        finally:
            com.release(attrs)

    def _configure(self) -> None:
        """Select the MJPG native type closest to the requested geometry:
        exact size required, nearest frame rate wins."""
        com = self._com
        assert com is not None
        best_index = best_score = None
        index = 0
        while True:
            mt = ctypes.c_void_p()
            hr = com.call(self._reader, _READER_GET_NATIVE_MEDIA_TYPE, ctypes.c_long,
                          (ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)),
                          MF_SOURCE_READER_FIRST_VIDEO_STREAM, index, ctypes.byref(mt))
            if hr < 0:
                break
            try:
                desc = _describe_type(com, mt)
            finally:
                com.release(mt)
            if (desc["is_mjpg"] and desc["width"] == self.width
                    and desc["height"] == self.height and desc["fps"]):
                score = abs(desc["fps"] - self.fps)
                if best_score is None or score < best_score:
                    best_score, best_index = score, index
            index += 1
        if best_index is None:
            raise OSError(f"no MJPG {self.width}x{self.height} type "
                          f"({index} native types)")
        chosen = ctypes.c_void_p()
        hr = com.call(self._reader, _READER_GET_NATIVE_MEDIA_TYPE, ctypes.c_long,
                      (ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)),
                      MF_SOURCE_READER_FIRST_VIDEO_STREAM, best_index, ctypes.byref(chosen))
        if hr < 0:
            raise OSError(f"GetNativeMediaType {_hr(hr)}")
        try:
            hr = com.call(self._reader, _READER_SET_CURRENT_MEDIA_TYPE, ctypes.c_long,
                          (ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p),
                          MF_SOURCE_READER_FIRST_VIDEO_STREAM, None, chosen)
            if hr < 0:
                raise OSError(f"SetCurrentMediaType {_hr(hr)}")
            self.negotiated = _describe_type(com, chosen)
        finally:
            com.release(chosen)
        self._configured = True
        self.error = None

    def _lost(self, reason: str) -> None:
        if self.error != reason:
            log.warning("Media Foundation camera %d: %s -- reopening in %.0fs",
                        self.device, reason, REOPEN_AFTER_S)
        self.error = reason
        # The whole device, not just the reader: a source that was
        # preempted does not come back by asking it for a new reader.
        self._close_reader()
        self._close_source()
        self._retry_at = time.monotonic() + REOPEN_AFTER_S

    def _close_source(self) -> None:
        if self._source.value and self._com is not None:
            try:
                self._com.call(self._source, _SOURCE_SHUTDOWN, ctypes.c_long, ())
            except Exception:  # noqa: BLE001 -- a dead source may refuse
                pass
            self._com.release(self._source)
        self._source = ctypes.c_void_p()

    def _close_reader(self) -> None:
        if self._reader.value and self._com is not None:
            self._com.release(self._reader)
        self._reader = ctypes.c_void_p()
        self._configured = False
