"""Facts about a camera's own JPEG frames, read from the bytes.

Used by the capture probes to answer the questions that decide whether a
camera's JPEG can be forwarded untouched:

  * how compressed is it -- estimated from its quantization tables
  * is the frame well formed -- does it end at its End-Of-Image marker,
    or carry bytes past it, or have no marker at all

The shape question matters because cameras differ. the scoring
cameras (2026-09-17) end every frame exactly at EOI. A Scolia camera
(Sonix 0c45:6340) on a Windows rig ended only 22 of 150 frames at EOI -- the
remainder either pad past the marker or omit it, and which one decides
whether a frame can be forwarded as-is, trimmed first, or not at all.

Pure functions, no OpenCV, importable anywhere.
"""
from __future__ import annotations

import struct
from typing import Any

# IJG (libjpeg) baseline tables at quality 50, in NATURAL order. A DQT
# segment stores its table in zig-zag order, but quality is estimated from
# the SUM of the table, which does not depend on the order.
_STD_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)
_STD_CHROMA = (
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
) + (99,) * 32

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"


def quantization_tables(data: bytes) -> "dict[int, list[int]]":
    """DQT tables by table id, read from the header up to Start-Of-Scan."""
    tables: "dict[int, list[int]]" = {}
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xDA:                    # start of scan: header is over
            break
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        if marker == 0xDB:
            j, end = i + 4, i + 2 + length
            while j < end:
                precision, table_id = data[j] >> 4, data[j] & 0x0F
                if precision:
                    values = list(struct.unpack(">64H", data[j + 1:j + 129]))
                    j += 129
                else:
                    values = list(data[j + 1:j + 65])
                    j += 65
                tables[table_id] = values
        i += 2 + length
    return tables


def _ijg_quality(table: "list[int]", reference: "tuple[int, ...]") -> "int | None":
    if len(table) != 64 or not sum(reference):
        return None
    scale = 100.0 * sum(table) / sum(reference)
    quality = 5000.0 / scale if scale > 100 else (200.0 - scale) / 2.0
    return int(round(max(1.0, min(100.0, quality))))


def estimated_quality(data: bytes) -> "dict[str, int | None]":
    """IJG-equivalent quality of the brightness and colour tables.

    An estimate: cameras use their own tables, and this reports the IJG
    quality whose table has the same total weight. Measured values on
    real hardware: ~45 for one rig's scoring cameras, ~93 for a Logitech
    BRIO.
    """
    tables = quantization_tables(data)
    return {
        "luma": _ijg_quality(tables[0], _STD_LUMA) if 0 in tables else None,
        "chroma": _ijg_quality(tables[1], _STD_CHROMA) if 1 in tables else None,
    }


def dimensions(data: bytes) -> "tuple[int, int, int] | None":
    """(height, width, components) from the frame header, without
    decoding. None if there is no start-of-frame marker before the scan."""
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xDA:
            return None
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        # SOF0..SOF15, less DHT (C4), JPG (C8) and DAC (CC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 10 > len(data):
                return None
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return height, width, data[i + 9]
        i += 2 + length
    return None


def frame_shape(data: bytes) -> "dict[str, Any]":
    """Where this frame's End-Of-Image marker is, relative to its end.

    kind:
      clean     ends exactly at EOI
      padded    EOI found earlier; `trailing` bytes follow it
      no_eoi    no EOI marker anywhere after SOI
      not_jpeg  does not start with SOI

    `trailing_all_zero` tells padding (safe to trim) from something that
    looks like the start of more data.
    """
    if data[:2] != SOI:
        return {"kind": "not_jpeg"}
    if data[-2:] == EOI:
        return {"kind": "clean", "trailing": 0}
    last = data.rfind(EOI, 2)
    if last < 0:
        return {"kind": "no_eoi"}
    tail = data[last + 2:]
    return {
        "kind": "padded",
        "trailing": len(tail),
        "trailing_all_zero": not any(tail),
        "trailing_first_bytes": tail[:8].hex(),
    }


def trim_to_eoi(data: bytes) -> bytes:
    """The frame up to and including its last EOI, or unchanged."""
    if data[-2:] == EOI:
        return data
    last = data.rfind(EOI, 2)
    return data[:last + 2] if last >= 0 else data


def repaired(data: bytes) -> "bytes | None":
    """A camera frame made well formed, or None if it is not a JPEG.

    Padded frames are cut at their last EOI. Frames with no EOI get one
    appended. Measured on a Scolia (Sonix 0c45:6340),
    2026-09-17, 300 frames per size: 77/300 at 1280x720 and 48/300 at
    320x240 failed `cv2.imdecode` as delivered. After this, 0/300 failed,
    and none decoded with a grey-filled bottom -- the camera sends the
    whole scan and only leaves off the marker.

    Safe because FFD9 cannot occur inside entropy-coded data: the last
    one in the buffer is the real end of the image.
    """
    if data[:2] != SOI:
        return None
    if data[-2:] == EOI:
        return data
    last = data.rfind(EOI, 2)
    if last >= 0:
        return data[:last + 2]
    return data + EOI
