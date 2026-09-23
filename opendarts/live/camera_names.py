"""opendarts/live/camera_names.py -- friendly names for the camera device
indexes the dashboard's assignment picker offers.

WHY THIS EXISTS. The picker used to offer a hardcoded `dev 0` .. `dev 9`
on every platform: ten entries on a machine with four cameras, and no way
to tell a board camera from the laptop's built-in webcam short of trial
and error. OpenCV itself cannot help -- `cv2.VideoCapture(index)` takes a
bare number and exposes no device-name API on any backend -- so the names
have to come from the platform underneath.

THE PROPERTY THAT ACTUALLY MATTERS: index N in `names` must be the device
`cv2.VideoCapture(N, <backend>)` opens, for the backend local_capture.py
pins on that platform (CAP_MSMF on Windows, CAP_V4L2 on Linux,
CAP_AVFOUNDATION on macOS). A confidently mislabelled dropdown is WORSE
than `dev 2` -- it sends the operator to the wrong camera with certainty
-- so each platform path is honest about whether it can guarantee that
mapping (`DeviceEnumeration.authoritative`):

- Windows: DirectShow's system device enumerator, walked via raw-ctypes
  COM. This is the same enumeration DirectShow/MSMF device indexes are
  defined by, so the mapping is correct by construction. Authoritative.
- Linux: /sys/class/video4linux/videoN/name. OpenCV's V4L2 backend opens
  literally "/dev/video<index>", so node number == OpenCV index, also by
  construction. Authoritative.
- macOS: there is no cheap authoritative source without PyObjC (which we
  deliberately do not add). `system_profiler SPCameraDataType` (built into
  macOS, no ffmpeg) matches AVFoundation ordering in practice -- verified
  on the rigs by unique id -- but by observation, not Apple contract, so
  it is reported NON-authoritative and the UI presents it as hints.

An entry may be "" when a device exists at that index but has no readable
name (or is a V4L2 node that cannot capture). It is never DROPPED --
removing one entry would shift every later name onto the wrong camera,
which is exactly the failure this module exists to avoid.

Never fatal, by design: names are cosmetic, and this module is imported
on every platform. Any failure returns the empty enumeration and logs
once; nothing here may take the dashboard down. The result is cached
(enumeration binds COM objects / spawns system_profiler and must not run on every
dashboard poll); pass refresh=True after plugging/unplugging hardware.

Nothing here opens a camera for CAPTURE. The capture pump owns the
cameras, and a second opener is the exact contention this system fights.
The one device-file touch is Linux's VIDIOC_QUERYCAP ioctl, which V4L2
requires to work on any open file handle without negotiating formats or
starting a stream -- V4L2 opens are non-exclusive; contention starts at
STREAMON, which this never issues.
"""
from __future__ import annotations

import ctypes
import json
import logging
import platform
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


log = logging.getLogger("opendarts.live.camera_names")


@dataclass(frozen=True)
class DeviceEnumeration:
    """What the platform said about its video capture devices.

    `names[N]` labels the device OpenCV index N opens -- only trust that
    when `authoritative` is True. `count` is how many indexes exist (the
    picker should offer 0..count-1), or None when the platform could not
    say. `source` records which mechanism answered, because on macOS the
    answer depends on what happened to be installed and the operator
    deserves to know which one they are trusting.
    """

    names: list[str] = field(default_factory=list)
    count: Optional[int] = None
    authoritative: bool = False
    source: str = "none"


_EMPTY = DeviceEnumeration()

_lock = threading.Lock()
_cached: Optional[DeviceEnumeration] = None
# Log the first failure only. Enumeration failing is a one-line curiosity,
# not a per-poll event stream -- and with caching plus explicit refresh it
# could otherwise fire on every refresh click against a broken platform.
_warned = False


def enumerate_devices(refresh: bool = False) -> DeviceEnumeration:
    """The cached platform enumeration; refresh=True re-asks the platform.

    NEVER WAITS ON AN ENUMERATION ALREADY IN FLIGHT. The first call can
    take seconds (system_profiler spawn, COM bind) and its caller is the dashboard
    polling `/api/camera-devices` every few seconds from
    `asyncio.to_thread` -- i.e. the SAME default executor
    `fetch_snapshot` uses. A blocking lock would stack one parked worker
    thread per poll for the whole duration, spending the capture path's
    thread pool on a cosmetic lookup. A caller that arrives mid-flight
    gets the previous answer (or none yet) instead; names are a label on
    a dropdown, and one poll's worth of staleness costs nothing.
    """
    global _cached, _warned
    if not _lock.acquire(blocking=False):
        return _cached if _cached is not None else _EMPTY
    try:
        if _cached is None or refresh:
            try:
                _cached = _enumerate_uncached()
            except Exception: # noqa: BLE001 -- cosmetic feature, never fatal
                if not _warned:
                    _warned = True
                    log.warning("camera name enumeration failed; the picker "
                                "will show bare device numbers", exc_info=True)
                _cached = _EMPTY
        return _cached
    finally:
        _lock.release()


#: What this rig's own DirectShow virtual cameras are called
#: (tools/winvcam/vcam_probe.cpp, kNames). They feed other software; reading
#: one back into our own capture is a loop -- OpenCV opens it happily through
#: DirectShow and the slot scores its own output (seen on a Windows rig,
#: 2026-09-17).
OWN_VIRTUAL_CAMERA_PREFIX = "OpenDarts Probe Cam"


def own_virtual_devices(names: "list[str]") -> list[int]:
    """Indices in `names` that are this rig's own virtual cameras."""
    return [i for i, n in enumerate(names)
            if isinstance(n, str) and n.startswith(OWN_VIRTUAL_CAMERA_PREFIX)]


def dshow_index_for(mf_index: int) -> Optional[int]:
    """The DirectShow index of the camera Media Foundation calls `mf_index`,
    matched by device path -- or None when there is no such camera or no way
    to be sure. Never a guess by number: the two lists are ordered
    differently, and a wrong guess opens a different camera (or one of our
    own virtual cameras) while looking like success."""
    from opendarts.live import win_mf_capture

    mf = win_mf_capture.list_devices()
    if not 0 <= mf_index < len(mf):
        return None
    want = win_mf_capture.device_key(mf[mf_index][1])
    if not want:
        return None
    for i, (_name, path) in enumerate(dshow_devices()):
        if win_mf_capture.device_key(path) == want:
            return i
    return None


def device_names() -> list[str]:
    """Friendly names in OpenCV index order; [] when unavailable."""
    return list(enumerate_devices().names)


def device_count() -> Optional[int]:
    """How many device indexes exist, or None if the platform can't say."""
    return enumerate_devices().count


def label_for(index: int) -> str:
    """Picker text for one device index.

    The number is ALWAYS present -- it is what the config stores and what
    the logs report, so a label without it would strand the operator the
    first time they need to cross-reference either.

    No uncertainty marker. A non-authoritative name (macOS, where the
    enumeration order is a strong correlation rather than a proof) used
    to get a trailing '?'. Dropped 2026-09-12: the number beside it is
    always exact, the name is the best this rig can establish, and
    annotating every macOS label with doubt only asked the operator to
    resolve something they have no way to resolve. `authoritative` is
    still reported by enumerate_devices() and by /api/camera-devices for
    anything that needs to reason about it -- it just no longer shows up
    as punctuation in a picker.
    """
    enum = enumerate_devices()
    name = enum.names[index] if 0 <= index < len(enum.names) else ""
    if not name:
        return f"dev {index}"
    return f"dev {index} — {name}"


def _enumerate_uncached() -> DeviceEnumeration:
    system = platform.system()
    if system == "Windows":
        return _enumerate_windows()
    if system == "Linux":
        return _enumerate_linux()
    if system == "Darwin":
        return _enumerate_macos()
    return _EMPTY


# --------------------------------------------------------------------------
# Linux: sysfs + VIDIOC_QUERYCAP.
#
# OpenCV's V4L2 backend turns index N into the literal path "/dev/videoN",
# so /sys/class/video4linux/videoN/name is the name of exactly the device
# OpenCV index N opens -- no ordering heuristics involved. The list is
# built DENSE over 0..max(node number): node numbers can be sparse
# (video0, video2 with no video1), and a compacted list would shift every
# name after the gap onto the wrong index.

# struct v4l2_capability: driver[16] card[32] bus_info[32] version:u32
# capabilities:u32 device_caps:u32 reserved[3]:u32 -- 104 bytes; the ioctl
# number encodes that size (_IOR('V', 0, struct v4l2_capability)).
_V4L2_CAPABILITY_SIZE = 104
_VIDIOC_QUERYCAP = 0x80000000 | (_V4L2_CAPABILITY_SIZE << 16) | (ord("V") << 8)
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000
_V4L2_CAP_DEVICE_CAPS = 0x80000000


#: The `driver` string v4l2loopback reports through QUERYCAP. Measured on
#: the Linux rig, 2026-09-15 -- not guessed, because the whole point of
#: this constant is to be matched exactly.
_V4L2LOOPBACK_DRIVER = "v4l2 loopback"


@dataclass(frozen=True)
class _NodeInfo:
    """What QUERYCAP said about one /dev/videoN."""

    is_capture: bool
    driver: str
    card: str


def _v4l2_query(dev_path: str) -> "Optional[_NodeInfo]":
    """QUERYCAP one node, or None when we could not ask.

    Safe against the capture pump: V4L2 file opens are non-exclusive and
    QUERYCAP neither negotiates a format nor starts streaming -- the
    kernel requires it to succeed on any open handle. Contention on a
    camera starts at VIDIOC_STREAMON, which this never issues.
    """
    import fcntl
    import os

    try:
        fd = os.open(dev_path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(_V4L2_CAPABILITY_SIZE)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
    except OSError:
        return None
    finally:
        os.close(fd)
    capabilities = int.from_bytes(buf[84:88], "little")
    device_caps = int.from_bytes(buf[88:92], "little")
    # device_caps describes THIS node; capabilities describes the whole
    # physical device, which for a UVC camera includes the metadata node
    # that cannot capture. Use the per-node field whenever the driver
    # provides it, or metadata nodes would pass as cameras.
    caps = device_caps if capabilities & _V4L2_CAP_DEVICE_CAPS else capabilities
    return _NodeInfo(
        is_capture=bool(caps & (_V4L2_CAP_VIDEO_CAPTURE
                                | _V4L2_CAP_VIDEO_CAPTURE_MPLANE)),
        driver=buf[0:16].split(b"\0")[0].decode("utf-8", "replace"),
        card=buf[16:48].split(b"\0")[0].decode("utf-8", "replace"),
    )


def _v4l2_node_is_capture(dev_path: str) -> Optional[bool]:
    """True/False when QUERYCAP answered, None when we could not ask."""
    info = _v4l2_query(dev_path)
    return None if info is None else info.is_capture


def suggest_camera_devices(
    count: int,
    sysfs_root: Path = Path("/sys/class/video4linux"),
    query: "Callable[[str], Optional[_NodeInfo]]" = _v4l2_query,
) -> "Optional[list[int]]":
    """Device indexes to use when the config names none; None to fall back.

    WHY THIS EXISTS. `DEFAULT_CAMERA_DEVICES` is [0, 1, 2], which is right
    on macOS and Windows and wrong on Linux for a reason no first-time
    operator can be expected to know: a UVC camera registers TWO nodes,
    one for capture and one for metadata, so three cameras occupy
    /dev/video0..5 and only the even ones deliver frames. The out-of-box
    default therefore picks two cameras and a metadata node, and the
    failure ("can't open camera by index") names none of that.

    TWO THINGS ARE EXCLUDED, and the second is the one that matters.
    Metadata nodes fail a capture check, which is the easy half. Our OWN
    v4l2loopback devices PASS one -- measured on the rig, /dev/video10-12
    report capture=True -- so a capture test alone would happily select a
    virtual camera and feed it its own output, a loop that would look like
    a frozen or duplicated image rather than a configuration error. They
    are excluded by driver string, which is exact, rather than by device
    number, which is only conventional.

    Returns None rather than a short list when it cannot find `count`
    usable nodes: a rig with one camera unplugged should fall back to the
    documented default and report a missing camera, not silently
    reconfigure itself to run with two.
    """
    if platform.system() != "Linux":
        # No metadata nodes to trip over; index 0..N-1 is already correct.
        return None
    try:
        entries = sorted(sysfs_root.iterdir())
    except OSError:
        return None

    usable: list[int] = []
    for entry in entries:
        m = re.fullmatch(r"video(\d+)", entry.name)
        if not m:
            continue
        info = query(f"/dev/{entry.name}")
        if info is None or not info.is_capture:
            continue
        if info.driver.strip().lower() == _V4L2LOOPBACK_DRIVER:
            continue
        usable.append(int(m.group(1)))

    usable.sort()
    if len(usable) < count:
        log.debug("camera autodetect found %d usable capture node(s), "
                  "need %d -- leaving the default alone", len(usable), count)
        return None
    chosen = usable[:count]
    if chosen != list(range(count)):
        # Only worth a line when it differs from what the default would
        # have done; on a machine where 0..N-1 was already right this is
        # noise.
        log.info("camera autodetect: using devices %s (capture-capable "
                 "non-loopback nodes); the [0, 1, 2] default would have "
                 "included a metadata node", chosen)
    return chosen


def _enumerate_linux(
    sysfs_root: Path = Path("/sys/class/video4linux"),
    is_capture: Callable[[str], Optional[bool]] = _v4l2_node_is_capture,
) -> DeviceEnumeration:
    try:
        entries = list(sysfs_root.iterdir())
    except OSError:
        # sysfs class missing entirely (no v4l2 in this kernel/container):
        # genuinely unknown, not "zero cameras".
        return _EMPTY

    found: dict[int, str] = {}
    for entry in entries:
        m = re.fullmatch(r"video(\d+)", entry.name)
        if not m:
            continue
        index = int(m.group(1))
        try:
            name = (entry / "name").read_text(encoding="utf-8",
                                              errors="replace").strip()
        except OSError:
            name = ""
        verdict = is_capture(f"/dev/{entry.name}")
        if verdict is False:
            # A metadata/output node. The index slot must survive (OpenCV
            # will still map index N to this node and fail to open it) but
            # the name must not: a UVC metadata node carries the SAME name
            # as its camera, and labelling it would lure the operator to
            # an index that cannot stream. When the check itself failed
            # (None -- permissions, container), keep the name: a possibly-
            # unopenable label degrades better than blanking real cameras.
            name = ""
        found[index] = name

    if not found:
        # The class exists and lists nothing camera-shaped: authoritative
        # zero, so the picker can say "no devices" instead of guessing 10.
        return DeviceEnumeration([], 0, True, "v4l2-sysfs")
    names = ["" for _ in range(max(found) + 1)]
    for index, name in found.items():
        names[index] = name
    return DeviceEnumeration(names, len(names), True, "v4l2-sysfs")


# --------------------------------------------------------------------------
# Windows: DirectShow's system device enumerator over raw ctypes COM.
#
# CoCreateInstance(CLSID_SystemDeviceEnum) -> ICreateDevEnum::
# CreateClassEnumerator(CLSID_VideoInputDeviceCategory) -> IEnumMoniker ->
# per moniker BindToStorage(IID_IPropertyBag) -> Read("FriendlyName").
# Deliberately no pywin32/comtypes: this repo runs on rigs where adding a
# dependency for one cosmetic string is not worth the install surface, so
# the vtables are walked by hand. Enumerating monikers binds property
# bags, never the capture filter itself, so no camera is opened.

_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}"
_IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_CLSID_VIDEO_INPUT_DEVICE_CATEGORY = "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"
_IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"

_S_OK = 0
_S_FALSE = 1
# COM already initialised on this thread in the other apartment model.
# Still usable for this enumeration -- but the init did not take, so it
# must not be balanced with CoUninitialize.
_RPC_E_CHANGED_MODE = -2147417850 # 0x80010106 as a signed HRESULT
_VT_BSTR = 8


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid_fields(text: str) -> tuple[int, int, int, tuple[int, ...]]:
    """Registry-format GUID string -> the four GUID struct fields.

    Pure and platform-free so the one part of the Windows path that CAN
    be unit-tested off-Windows, is.
    """
    parts = text.strip().strip("{}").split("-")
    if len(parts) != 5 or (len(parts[3]) + len(parts[4])) != 16:
        raise ValueError(f"not a GUID: {text!r}")
    tail = parts[3] + parts[4]
    return (
        int(parts[0], 16),
        int(parts[1], 16),
        int(parts[2], 16),
        tuple(int(tail[i:i + 2], 16) for i in range(0, 16, 2)),
    )


def _guid(text: str) -> _GUID:
    d1, d2, d3, d4 = _guid_fields(text)
    return _GUID(d1, d2, d3, (ctypes.c_ubyte * 8)(*d4))


class _VARIANT(ctypes.Structure):
    # Only enough of VARIANT to carry a BSTR out of IPropertyBag::Read:
    # vt + three reserved words, then the union, whose first pointer-sized
    # slot IS bstrVal when vt == VT_BSTR. Two pointer fields cover the
    # union's full 16 bytes on 64-bit (8+8) and overshoot harmlessly on
    # 32-bit -- VariantInit/VariantClear only touch what vt says is live.
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("data", ctypes.c_void_p),
        ("data_tail", ctypes.c_void_p),
    ]


def _enumerate_windows() -> DeviceEnumeration: # pragma: no cover - Windows only
    """Names in MEDIA FOUNDATION order, which is the order a Windows slot's
    device number counts in (win_mf_capture reads cameras by it, and OpenCV's
    MSMF used the same list). DirectShow's order is different -- it also
    lists virtual cameras, this product's own among them -- so labelling MF
    numbers with DirectShow names put the wrong name on a device (on a Windows
    rig, 2026-09-17). DirectShow names are only the fallback when MF lists
    nothing."""
    from opendarts.live import win_mf_capture

    mf = win_mf_capture.list_devices()
    if mf:
        names = [name for name, _path in mf]
        return DeviceEnumeration(names, len(names), True, "media-foundation")
    names = [name for name, _path in dshow_devices()]
    return DeviceEnumeration(names, len(names), True, "dshow-com")


def dshow_devices() -> "list[tuple[str, Optional[str]]]":
    """(friendly name, device path) for every DirectShow video input, in
    DirectShow's order -- the index OpenCV's CAP_DSHOW opens. [] off
    Windows or on failure."""
    if platform.system() != "Windows" or not hasattr(ctypes, "WINFUNCTYPE"):
        return []
    try:
        return _dshow_devices()
    except Exception:  # noqa: BLE001 -- cosmetic and fallback use only
        return []


def _dshow_devices() -> "list[tuple[str, Optional[str]]]": # pragma: no cover - Windows only
    ole32 = ctypes.windll.ole32 # type: ignore[attr-defined]
    oleaut32 = ctypes.windll.oleaut32 # type: ignore[attr-defined]

    def com_call(obj: ctypes.c_void_p, index: int, restype, argtypes, *args):
        # COM interface pointer -> vtable -> slot `index`, called with the
        # interface pointer as the implicit `this`. WINFUNCTYPE gives the
        # stdcall convention COM requires on 32-bit; on 64-bit it is the
        # one calling convention anyway.
        vtable = ctypes.cast(
            obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
        ).contents
        proto = ctypes.WINFUNCTYPE( # type: ignore[attr-defined]
            restype, ctypes.c_void_p, *argtypes)
        return proto(vtable[index])(obj, *args)

    def release(obj: ctypes.c_void_p) -> None:
        # IUnknown::Release is vtable slot 2 on every COM interface.
        if obj and obj.value:
            com_call(obj, 2, ctypes.c_ulong, ())

    clsid_devenum = _guid(_CLSID_SYSTEM_DEVICE_ENUM)
    iid_create = _guid(_IID_ICREATE_DEV_ENUM)
    clsid_category = _guid(_CLSID_VIDEO_INPUT_DEVICE_CATEGORY)
    iid_propbag = _guid(_IID_IPROPERTY_BAG)

    # DirectShow is a COM citizen of the apartment-threaded world; MSMF's
    # enumeration (what OpenCV actually calls at open time) does not care
    # which apartment lists the devices, only that the same system list
    # is being read.
    hr = ole32.CoInitializeEx(None, 0x2) # COINIT_APARTMENTTHREADED
    must_uninit = hr in (_S_OK, _S_FALSE)
    if hr < 0 and hr != _RPC_E_CHANGED_MODE:
        return []

    dev_enum = ctypes.c_void_p()
    enum_moniker = ctypes.c_void_p()
    devices: "list[tuple[str, Optional[str]]]" = []
    try:
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid_devenum), None, 1, # CLSCTX_INPROC_SERVER
            ctypes.byref(iid_create), ctypes.byref(dev_enum))
        if hr < 0 or not dev_enum.value:
            return []

        # ICreateDevEnum::CreateClassEnumerator is slot 3 (after IUnknown).
        hr = com_call(
            dev_enum, 3, ctypes.c_long,
            (ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
             ctypes.c_uint32),
            ctypes.byref(clsid_category), ctypes.byref(enum_moniker), 0)
        if hr == _S_FALSE or not enum_moniker.value:
            # Documented contract for "the category is empty": S_FALSE and
            # a NULL enumerator. Zero cameras, known with certainty.
            return []
        if hr < 0:
            return []

        while True:
            moniker = ctypes.c_void_p()
            fetched = ctypes.c_ulong(0)
            # IEnumMoniker::Next is slot 3 (IUnknown, then Next).
            hr = com_call(
                enum_moniker, 3, ctypes.c_long,
                (ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                 ctypes.POINTER(ctypes.c_ulong)),
                1, ctypes.byref(moniker), ctypes.byref(fetched))
            if hr != _S_OK or fetched.value != 1 or not moniker.value:
                break
            try:
                # IMoniker::BindToStorage is slot 9: IUnknown(3) +
                # IPersist::GetClassID + IPersistStream(4) + BindToObject.
                prop_bag = ctypes.c_void_p()
                hr = com_call(
                    moniker, 9, ctypes.c_long,
                    (ctypes.c_void_p, ctypes.c_void_p,
                     ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)),
                    None, None, ctypes.byref(iid_propbag),
                    ctypes.byref(prop_bag))
                name = ""
                path: Optional[str] = None

                def read(prop: str) -> Optional[str]:
                    var = _VARIANT()
                    oleaut32.VariantInit(ctypes.byref(var))
                    # IPropertyBag::Read is slot 3.
                    rhr = com_call(
                        prop_bag, 3, ctypes.c_long,
                        (ctypes.c_wchar_p, ctypes.POINTER(_VARIANT),
                         ctypes.c_void_p),
                        prop, ctypes.byref(var), None)
                    value = (ctypes.wstring_at(var.data)
                             if rhr >= 0 and var.vt == _VT_BSTR and var.data else None)
                    oleaut32.VariantClear(ctypes.byref(var))
                    return value

                if hr >= 0 and prop_bag.value:
                    try:
                        name = read("FriendlyName") or ""
                        path = read("DevicePath")
                    finally:
                        release(prop_bag)
                # A device with no readable name still HOLDS its index:
                # append "" rather than skip, or every later name shifts
                # onto the wrong camera.
                devices.append((name, path))
            finally:
                release(moniker)
    finally:
        release(enum_moniker)
        release(dev_enum)
        if must_uninit:
            ole32.CoUninitialize()

    return devices


# --------------------------------------------------------------------------
# macOS: best effort, and labelled as such.
#
# CAP_AVFOUNDATION indexes AVCaptureDevice's device list. Camera names come
# from `system_profiler SPCameraDataType` -- built into every macOS, so no
# ffmpeg, no PyObjC, nothing to install or spawn opportunistically.
# Verified on the rigs (2026-09-20): system_profiler enumerates cameras in
# the same order as AVFoundation (matched by unique id across all three),
# i.e. the index space OpenCV's CAP_AVFOUNDATION uses -- so the names line
# up with camera indices in practice. Still authoritative=False: Apple
# documents no ordering guarantee, so the UI shows these as hints. The
# contractually-authoritative route would be AVFoundation via PyObjC, a
# dependency this repo has declined for a cosmetic label.
#
# (ffmpeg's `avfoundation -list_devices` was the old primary source;
# dropped 2026-09-20 so the product never shells out to ffmpeg anywhere.)
_SYSTEM_PROFILER_CMD = ("system_profiler", "SPCameraDataType", "-json")


def _parse_system_profiler_cameras(text: str) -> Optional[list[str]]:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    cameras = data.get("SPCameraDataType") if isinstance(data, dict) else None
    if not isinstance(cameras, list):
        return None
    return [str(cam.get("_name") or "") for cam in cameras
            if isinstance(cam, dict)]


def _enumerate_macos(
    run: "Callable[..., subprocess.CompletedProcess[str]] | None" = None,
) -> DeviceEnumeration:
    # `system_profiler SPCameraDataType` is the only source -- built into
    # macOS, no ffmpeg to spawn or even probe for. It enumerates cameras
    # in AVFoundation's order (verified on the rigs, see the block comment
    # above), which is OpenCV's index space, so names align with indices;
    # authoritative=False all the same because Apple guarantees no order.
    if run is None:
        run = subprocess.run
    try:
        proc = run(_SYSTEM_PROFILER_CMD, capture_output=True, text=True,
                   timeout=15)
    except (OSError, subprocess.SubprocessError):
        return _EMPTY
    if proc.returncode != 0:
        return _EMPTY
    names = _parse_system_profiler_cameras(proc.stdout or "")
    if not names:
        # Empty here is "nothing visible to this process", not "this Mac
        # has no cameras" -- SPCameraDataType comes back empty for a
        # process without a camera grant exactly as it does for a Mac
        # with none attached. Reporting a count of zero would shrink the
        # picker on a rig whose cameras are working fine; unknown leaves
        # it on the bare-number fallback.
        return _EMPTY
    return DeviceEnumeration(names, len(names), False, "system-profiler")
