#!/bin/bash
# Cross-build the probe DLL from macOS/Linux. Needs mingw-w64:
#   brew install mingw-w64        (macOS)
#   apt install mingw-w64         (Debian/Ubuntu)
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-x86_64-w64-mingw32-g++}"
"$CXX" -shared -O2 -o vcam_probe.dll vcam_probe.cpp vcam_probe.def \
    -lstrmiids -lole32 -loleaut32 -luuid -lwindowscodecs \
    -static -static-libgcc -static-libstdc++ -Wl,--exclude-all-symbols
# -static matters: without it the DLL imports libwinpthread-1.dll, a MinGW
# runtime that will not exist on a target Windows machine, and regsvr32
# fails with a bare "module could not be found" that looks like a bad DLL
# rather than a missing dependency.
echo "built: $(ls -la vcam_probe.dll | awk '{print $5}') bytes"
