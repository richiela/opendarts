"""Pick this platform's virtual-camera backend.

Another app cannot share a physical camera with this product on Windows
or Linux, so on both we own the real cameras and republish frames into
virtual ones that the other app reads instead. The mechanism differs -- a
DirectShow filter we wrote on Windows, the v4l2loopback kernel module on
Linux -- but the shape is identical, so the call sites in `run_product`
and `server` should not have to know which they are on.

macOS needs neither: AVFoundation shares cameras natively, and both
backends report `available() is False` there, which is the honest answer
rather than a silent no-op.

The two backends deliberately expose the same three names -- `available`,
`VirtualCameraSet` and (for register) `apply` -- so this module is a
selection, not an adapter. If they ever drift apart, the right fix is to
bring them back together rather than to translate here.
"""
from __future__ import annotations

import platform

from opendarts.live import v4l2_publish, v4l2_register, vcam_publish, vcam_register

if platform.system() == "Linux":
    publish = v4l2_publish
    register = v4l2_register
else:
    #: Windows really, and macOS by way of both backends answering False.
    publish = vcam_publish
    register = vcam_register


def backend_name() -> str:
    """Which mechanism this platform uses, for logs and diagnostics."""
    system = platform.system()
    if system == "Linux":
        return "v4l2loopback"
    if system == "Windows":
        return "DirectShow filter"
    return "none (AVFoundation shares cameras natively)"
