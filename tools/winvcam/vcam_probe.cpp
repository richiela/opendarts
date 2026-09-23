// vcam_probe -- three DirectShow video-input devices fed from the capture
// hub, so other software on the machine can read the rig's cameras.
//
// WHAT A DirectShow CLIENT NEEDS TO LIST A VIRTUAL CAMERA. Three things,
// each of which a typical client checks before it will show a device:
//
//   * a capture output pin that describes its formats (IAMStreamConfig).
//     A pinless filter enumerates, but can describe no format, and clients
//     that build a format list per device drop it as malformed.
//   * a DevicePath in the moniker's property bag. A filter registered
//     through IFilterMapper2 has none, and clients that parse vid_/pid_
//     out of DevicePath drop devices without one.
//   * a pin that answers AMPROPERTY_PIN_CATEGORY as a capture pin.
//
// STREAMING. The pin delivers real samples -- allocator negotiation, a
// push thread, state transitions. With no writer on the shared mapping it
// shows a synthetic moving test pattern, so the DirectShow streaming path
// can be checked independently of where frames come from.
//
// JPEG PASSTHROUGH (v5, 2026-09-17). The shared mapping can carry the
// camera's own JPEG, and MJPG is the preferred type. A client that asks
// for MJPG gets the JPEG bytes and Windows' own MJPEG decoder does the
// rest, exactly the graph it builds on a physical camera. Before this,
// MJPG was refused and such a client quietly connected RGB24 and used our
// decode instead of its own. The pin also honours SetFormat: once a
// client picks a type, only that type is offered, as a real device does
// -- otherwise the graph's direct-connect attempt finds RGB24 and never
// inserts the decoder. Conversions (WIC) cover a connection whose format
// differs from the payload: decode for RGB24, encode for MJPG.
//
// Build:   ./build.sh          (cross-compiles from macOS/Linux)
// Install: regsvr32 vcam_probe.dll      (elevation optional, see README)
// Remove:  regsvr32 /u vcam_probe.dll
//
// No signing required -- user-mode COM, not a kernel driver.

#include <initguid.h>   // must precede dshow.h; do NOT also include uuids.h
#include <windows.h>
#include <dshow.h>
#include <ks.h>
#include <ksmedia.h>
#include <olectl.h>
#include <wincodec.h>
#include <cstdio>
#include <cwchar>
#include <new>
#include <cstdarg>
#include "shared_frame.h"

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

DEFINE_GUID(CLSID_VCamProbe0,
    0x4d1f9a20, 0x8c33, 0x4b17, 0x9a, 0x64, 0x21, 0x70, 0xe5, 0x3b, 0x10, 0x00);
DEFINE_GUID(CLSID_VCamProbe1,
    0x4d1f9a20, 0x8c33, 0x4b17, 0x9a, 0x64, 0x21, 0x70, 0xe5, 0x3b, 0x10, 0x01);
DEFINE_GUID(CLSID_VCamProbe2,
    0x4d1f9a20, 0x8c33, 0x4b17, 0x9a, 0x64, 0x21, 0x70, 0xe5, 0x3b, 0x10, 0x02);

static const CLSID* const kClsids[3] = {
    &CLSID_VCamProbe0, &CLSID_VCamProbe1, &CLSID_VCamProbe2
};
static const wchar_t* const kNames[3] = {
    L"OpenDarts Probe Cam 0",
    L"OpenDarts Probe Cam 1",
    L"OpenDarts Probe Cam 2",
};

static HINSTANCE g_instance = nullptr;
static LONG g_lockCount = 0;

static void RegLog(const wchar_t* fmt, ...);   // defined with registration

// The formats offered. 1280x720@30 mirrors what the real board cameras
// report and what this product asks for, so a client that accepts these is
// accepting something it already knows how to handle.
static const int kWidth = 1280;
static const int kHeight = 720;
static const int kFps = 30;
static const REFERENCE_TIME kFrameTime = 10000000 / kFps;   // 100ns units
// How long the push thread waits for a new frame before re-pushing the
// last one, so a stalled writer does not look like a dead stream.
static const DWORD kRepeatMs = 100;

// ---------------------------------------------------------------------------
// Media-type helpers
//
// AM_MEDIA_TYPE owns a separately-allocated format block, so every copy and
// free has to handle both. Getting this wrong leaks on every enumeration,
// and enumeration happens repeatedly.
// ---------------------------------------------------------------------------

static void FreeMediaType(AM_MEDIA_TYPE& mt) {
    if (mt.cbFormat != 0) {
        CoTaskMemFree(mt.pbFormat);
        mt.cbFormat = 0;
        mt.pbFormat = nullptr;
    }
    if (mt.pUnk) { mt.pUnk->Release(); mt.pUnk = nullptr; }
}

static void DeleteMediaType(AM_MEDIA_TYPE* pmt) {
    if (!pmt) return;
    FreeMediaType(*pmt);
    CoTaskMemFree(pmt);
}

// Builds one uncompressed-or-MJPG video type. `bitCount`/`compression`
// differ per subtype; everything else is shared.
static AM_MEDIA_TYPE* CreateVideoMediaType(const GUID& subtype, WORD bitCount,
                                           DWORD compression) {
    AM_MEDIA_TYPE* mt = (AM_MEDIA_TYPE*)CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE));
    if (!mt) return nullptr;
    ZeroMemory(mt, sizeof(AM_MEDIA_TYPE));

    VIDEOINFOHEADER* vih = (VIDEOINFOHEADER*)CoTaskMemAlloc(sizeof(VIDEOINFOHEADER));
    if (!vih) { CoTaskMemFree(mt); return nullptr; }
    ZeroMemory(vih, sizeof(VIDEOINFOHEADER));

    vih->AvgTimePerFrame = kFrameTime;
    vih->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    vih->bmiHeader.biWidth = kWidth;
    vih->bmiHeader.biHeight = kHeight;
    vih->bmiHeader.biPlanes = 1;
    vih->bmiHeader.biBitCount = bitCount;
    vih->bmiHeader.biCompression = compression;
    // For MJPG this is the BUFFER bound, not a frame size: a JPEG of this
    // geometry is always smaller than its pixels.
    vih->bmiHeader.biSizeImage = (kWidth * kHeight * 24) / 8;
    vih->dwBitRate = vih->bmiHeader.biSizeImage * 8 * kFps;

    const bool fixed = IsEqualGUID(subtype, MEDIASUBTYPE_RGB24) != 0;
    mt->majortype = MEDIATYPE_Video;
    mt->subtype = subtype;
    mt->bFixedSizeSamples = fixed ? TRUE : FALSE;
    mt->bTemporalCompression = FALSE;
    mt->lSampleSize = fixed ? vih->bmiHeader.biSizeImage : 0;
    mt->formattype = FORMAT_VideoInfo;
    mt->cbFormat = sizeof(VIDEOINFOHEADER);
    mt->pbFormat = (BYTE*)vih;
    return mt;
}

// MJPG FIRST again as of v5, the order a camera lists them in. v4 put RGB24
// first because it could only produce raw pixels, and MJPG first would have
// labelled them JPEG. v5 delivers real JPEG on an MJPG connection, so the
// order goes back to matching the hardware.
static AM_MEDIA_TYPE* CreateTypeByIndex(int i) {
    switch (i) {
        case 0: return CreateVideoMediaType(MEDIASUBTYPE_MJPG, 24,
                                            MAKEFOURCC('M','J','P','G'));
        case 1: return CreateVideoMediaType(MEDIASUBTYPE_RGB24, 24, BI_RGB);
        default: return nullptr;
    }
}
static const int kTypeCount = 2;

// The type a client asked for with SetFormat, or GUID_NULL for "any".
// A real device offers only the selected type once one is set; offering
// both let DirectShow connect RGB24 straight to the client and skip the
// MJPEG decoder the client asked for.
//
// `pixelsOnly`: the writer is publishing pixels (a camera the capture hub
// could not read as JPEG). MJPG is then not offered at all, so the client
// connects RGB24 and gets those pixels untouched -- rather than the filter
// encoding them to JPEG only for the client to decode them straight back.
// A client that asked for MJPG gets RGB24 instead, the way the v4 filter
// always answered it.
static bool Offered(int i, const GUID& want, bool pixelsOnly) {
    if (pixelsOnly) return i == 1;
    if (IsEqualGUID(want, GUID_NULL)) return i >= 0 && i < kTypeCount;
    if (i == 0) return IsEqualGUID(want, MEDIASUBTYPE_MJPG) != 0;
    if (i == 1) return IsEqualGUID(want, MEDIASUBTYPE_RGB24) != 0;
    return false;
}

static AM_MEDIA_TYPE* CopyMediaType(const AM_MEDIA_TYPE* src) {
    if (!src) return nullptr;
    AM_MEDIA_TYPE* dst = (AM_MEDIA_TYPE*)CoTaskMemAlloc(sizeof(AM_MEDIA_TYPE));
    if (!dst) return nullptr;
    *dst = *src;
    if (src->cbFormat) {
        dst->pbFormat = (BYTE*)CoTaskMemAlloc(src->cbFormat);
        if (!dst->pbFormat) { CoTaskMemFree(dst); return nullptr; }
        CopyMemory(dst->pbFormat, src->pbFormat, src->cbFormat);
    }
    if (dst->pUnk) dst->pUnk->AddRef();
    return dst;
}

// ---------------------------------------------------------------------------
// IEnumMediaTypes
// ---------------------------------------------------------------------------

class EnumMediaTypesImpl : public IEnumMediaTypes {
public:
    explicit EnumMediaTypesImpl(GUID want = GUID_NULL, bool pixelsOnly = false, int pos = 0)
        : m_refs(1), m_pos(pos), m_want(want), m_pixelsOnly(pixelsOnly) {}

    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) || IsEqualIID(riid, IID_IEnumMediaTypes)) {
            *ppv = static_cast<IEnumMediaTypes*>(this); AddRef(); return S_OK;
        }
        *ppv = nullptr; return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_refs); }
    STDMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&m_refs);
        if (n == 0) delete this;
        return n;
    }

    STDMETHODIMP Next(ULONG cMediaTypes, AM_MEDIA_TYPE** ppMediaTypes,
                      ULONG* pcFetched) override {
        if (!ppMediaTypes) return E_POINTER;
        ULONG fetched = 0;
        while (fetched < cMediaTypes && m_pos < kTypeCount) {
            if (!Offered(m_pos, m_want, m_pixelsOnly)) { ++m_pos; continue; }
            AM_MEDIA_TYPE* mt = CreateTypeByIndex(m_pos);
            if (!mt) break;
            ppMediaTypes[fetched++] = mt;
            ++m_pos;
        }
        if (pcFetched) *pcFetched = fetched;
        return (fetched == cMediaTypes) ? S_OK : S_FALSE;
    }
    STDMETHODIMP Skip(ULONG cMediaTypes) override {
        m_pos += (int)cMediaTypes;
        return (m_pos <= kTypeCount) ? S_OK : S_FALSE;
    }
    STDMETHODIMP Reset() override { m_pos = 0; return S_OK; }
    STDMETHODIMP Clone(IEnumMediaTypes** ppEnum) override {
        if (!ppEnum) return E_POINTER;
        EnumMediaTypesImpl* e = new (std::nothrow) EnumMediaTypesImpl(m_want, m_pixelsOnly, m_pos);
        if (!e) return E_OUTOFMEMORY;
        *ppEnum = e;
        return S_OK;
    }

private:
    LONG m_refs;
    int m_pos;
    GUID m_want;
    bool m_pixelsOnly;
};

// ---------------------------------------------------------------------------
// JPEG <-> BGR through WIC, for a connection whose format differs from the
// payload. Windows' own codec: nothing to ship, and on an RGB24 connection
// the client still gets a Windows decode, as it would from the camera
// through the MJPEG decoder.
// ---------------------------------------------------------------------------

template <typename T> static void SafeRelease(T*& p) { if (p) { p->Release(); p = nullptr; } }

// Top-down BGR24 into `dst` (w*h*3). False unless the JPEG is exactly w x h.
static bool DecodeJpeg(IWICImagingFactory* f, const BYTE* src, UINT len,
                       BYTE* dst, int w, int h) {
    if (!f) return false;
    IWICStream* stream = nullptr;
    IWICBitmapDecoder* dec = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* conv = nullptr;
    bool ok = false;
    UINT fw = 0, fh = 0;
    WICRect rc = {0, 0, w, h};
    if (FAILED(f->CreateStream(&stream))) goto done;
    if (FAILED(stream->InitializeFromMemory(const_cast<BYTE*>(src), len))) goto done;
    if (FAILED(f->CreateDecoderFromStream(stream, nullptr,
                                          WICDecodeMetadataCacheOnDemand, &dec))) goto done;
    if (FAILED(dec->GetFrame(0, &frame))) goto done;
    if (FAILED(frame->GetSize(&fw, &fh)) || (int)fw != w || (int)fh != h) goto done;
    if (FAILED(f->CreateFormatConverter(&conv))) goto done;
    if (FAILED(conv->Initialize(frame, GUID_WICPixelFormat24bppBGR,
                                WICBitmapDitherTypeNone, nullptr, 0.0,
                                WICBitmapPaletteTypeCustom))) goto done;
    ok = SUCCEEDED(conv->CopyPixels(&rc, (UINT)w * 3, (UINT)w * 3 * h, dst));
done:
    SafeRelease(conv); SafeRelease(frame); SafeRelease(dec); SafeRelease(stream);
    return ok;
}

// Top-down BGR24 to JPEG in `dst` (capacity `cap`); *outLen gets its size.
static bool EncodeJpeg(IWICImagingFactory* f, const BYTE* bgr, int w, int h,
                       BYTE* dst, long cap, long* outLen) {
    if (!f) return false;
    IStream* mem = nullptr;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* props = nullptr;
    WICPixelFormatGUID fmt = GUID_WICPixelFormat24bppBGR;
    bool ok = false;
    HGLOBAL hg = nullptr;
    STATSTG st = {};
    if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &mem))) goto done;
    if (FAILED(f->CreateEncoder(GUID_ContainerFormatJpeg, nullptr, &enc))) goto done;
    if (FAILED(enc->Initialize(mem, WICBitmapEncoderNoCache))) goto done;
    if (FAILED(enc->CreateNewFrame(&frame, &props))) goto done;
    if (FAILED(frame->Initialize(props))) goto done;
    if (FAILED(frame->SetSize((UINT)w, (UINT)h))) goto done;
    if (FAILED(frame->SetPixelFormat(&fmt)) ||
        !IsEqualGUID(fmt, GUID_WICPixelFormat24bppBGR)) goto done;
    if (FAILED(frame->WritePixels((UINT)h, (UINT)w * 3, (UINT)w * 3 * h,
                                  const_cast<BYTE*>(bgr)))) goto done;
    if (FAILED(frame->Commit()) || FAILED(enc->Commit())) goto done;
    if (FAILED(mem->Stat(&st, STATFLAG_NONAME))) goto done;
    if (FAILED(GetHGlobalFromStream(mem, &hg))) goto done;
    if ((LONGLONG)st.cbSize.QuadPart > cap) goto done;
    {
        BYTE* p = (BYTE*)GlobalLock(hg);
        if (!p) goto done;
        memcpy(dst, p, (size_t)st.cbSize.QuadPart);
        GlobalUnlock(hg);
    }
    *outLen = (long)st.cbSize.QuadPart;
    ok = true;
done:
    SafeRelease(props); SafeRelease(frame); SafeRelease(enc); SafeRelease(mem);
    return ok;
}

static void FlipRows(const BYTE* src, BYTE* dst, int w, int h) {
    const long stride = (long)w * 3;
    for (int y = 0; y < h; ++y) {
        memcpy(dst + (long)y * stride, src + (long)(h - 1 - y) * stride, stride);
    }
}

// ---------------------------------------------------------------------------
// Shared-memory frame reader
//
// Opens lazily and re-opens on demand: the capture hub may start after
// a client, stop, or restart, and none of those should require touching
// the virtual camera. A reader with no writer simply reports no frame, and
// the pin falls back to the test pattern -- which makes "the hub is not
// running" visually obvious rather than a black screen that could mean
// anything.
// ---------------------------------------------------------------------------

// Three outcomes, not two. "No NEW frame" and "no writer at all" are
// completely different situations and collapsing them is what made the
// picture flash: the capture hub publishes at camera rate while the
// filter pushes on its own 30fps clock, so a large share of polls
// legitimately find nothing new -- and a real camera repeats its last
// frame rather than cutting to bars.
enum class FrameResult { Fresh, Repeated, NoSource };

class SharedFrameReader {
public:
    explicit SharedFrameReader(int slot) : m_slot(slot) {}
    ~SharedFrameReader() { Close(); delete[] m_data; delete[] m_tmp; }

    // Takes the newest complete frame, in whichever format the writer
    // published it. Fresh: Format()/Data()/Length() describe it. Repeated:
    // nothing new -- the caller re-serves what it made from the last one.
    FrameResult Poll(int width, int height) {
        const long cap = (long)width * height * 3;
        if (!m_view && !Open()) return Repeat();

        ODVCamHeader* h = (ODVCamHeader*)m_view;   // writable: counters live in its tail
        // Version 1 is the same layout with BGR24 only, so it reads fine.
        if (h->magic != ODVCAM_MAGIC ||
            (h->version != ODVCAM_VERSION && h->version != 1u)) {
            // Writer exited (it zeroes the magic on close) or is
            // rewriting the mapping. Drop the cached frame too: repeating
            // a picture from a hub that has stopped would show a frozen
            // board indefinitely, which is worse than visibly falling
            // back to the pattern.
            Close();
            m_have = false;
            return FrameResult::NoSource;
        }
        if ((int)h->width != width || (int)h->height != height) return Repeat();

        const uint32_t seq1 = h->sequence;
        if (seq1 & 1u) return Repeat();                // mid-write
        if (seq1 == m_lastSeq) return Repeat();        // nothing new yet

        const uint32_t fmt = h->format;
        long len;
        if (fmt == ODVCAM_FORMAT_BGR24) {
            if ((long)h->stride != (long)width * 3) return Repeat();
            len = cap;
        } else if (fmt == ODVCAM_FORMAT_MJPEG) {
            len = (long)h->stride;
            if (len < 4 || len > cap) return Repeat();
        } else {
            return Repeat();
        }

        if (!Reserve(cap)) return Repeat();
        memcpy(m_tmp, (const BYTE*)m_view + ODVCAM_HEADER_SIZE, len);

        // Re-read AFTER copying. A changed counter means the writer
        // overwrote the buffer mid-copy and this frame is torn.
        if (h->sequence != seq1) {
            h->reader_frames_torn++;
            return Repeat();
        }

        // Gaps in frame_index are the ONLY direct evidence that published
        // frames were never collected. Rates can only suggest it; this
        // counts it. Skipped on the first read, where there is no previous
        // index to compare against and every prior frame would look missed.
        const uint32_t idx = h->frame_index;
        if (m_haveIndex && idx > m_lastIndex + 1) {
            h->reader_frames_missed += (idx - m_lastIndex - 1);
        }
        m_lastIndex = idx;
        m_haveIndex = true;

        h->reader_frames_read++;
        h->reader_tick_ms = GetTickCount();
        m_lastSeq = seq1;

        BYTE* t = m_data; m_data = m_tmp; m_tmp = t;
        m_format = fmt;
        m_len = len;
        m_have = true;
        return FrameResult::Fresh;
    }

    // Set by the writer after each frame; null when unavailable.
    HANDLE FrameEvent() {
        if (!m_view) Open();
        return m_event;
    }

    // The format the writer is publishing right now, without taking a
    // frame; -1 when there is no live writer to ask.
    int PeekFormat() {
        if (!m_view && !Open()) return -1;
        const ODVCamHeader* h = (const ODVCamHeader*)m_view;
        if (h->magic != ODVCAM_MAGIC ||
            (h->version != ODVCAM_VERSION && h->version != 1u)) return -1;
        return (int)h->format;
    }

    uint32_t Format() const { return m_format; }
    const BYTE* Data() const { return m_data; }
    long Length() const { return m_len; }

private:
    // Nothing new. Without any previous frame there is genuinely nothing
    // to show, and the caller draws the pattern.
    FrameResult Repeat() const {
        return m_have ? FrameResult::Repeated : FrameResult::NoSource;
    }

    bool Reserve(long cap) {
        if (m_cap == cap && m_data && m_tmp) return true;
        delete[] m_data; delete[] m_tmp;
        m_data = new (std::nothrow) BYTE[cap];
        m_tmp = new (std::nothrow) BYTE[cap];
        m_cap = (m_data && m_tmp) ? cap : 0;
        m_have = false;
        return m_cap != 0;
    }

    bool Open() {
        // Rate-limited: OpenFileMapping on every frame with no writer
        // present is a syscall 30x a second producing nothing.
        const DWORD now = GetTickCount();
        if (m_lastTry && (now - m_lastTry) < 1000) return false;
        m_lastTry = now ? now : 1;

        wchar_t name[64];
        swprintf(name, 64, ODVCAM_NAME_FMT, m_slot);
        // READ|WRITE, not READ: the reader reports its own counters back
        // through the tail of the header, which is the only place the
        // "are frames being dropped" question can be answered from.
        m_map = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name);
        if (!m_map) return false;
        m_view = MapViewOfFile(m_map, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
        if (!m_view) { CloseHandle(m_map); m_map = nullptr; return false; }
        swprintf(name, 64, ODVCAM_EVENT_FMT, m_slot);
        m_event = OpenEventW(SYNCHRONIZE, FALSE, name);   // null is fine: poll
        return true;
    }
    void Close() {
        if (m_view) { UnmapViewOfFile(m_view); m_view = nullptr; }
        if (m_map) { CloseHandle(m_map); m_map = nullptr; }
        if (m_event) { CloseHandle(m_event); m_event = nullptr; }
        m_lastSeq = 0;
        m_haveIndex = false;
    }

    int m_slot;
    HANDLE m_map = nullptr;
    void* m_view = nullptr;
    HANDLE m_event = nullptr;
    uint32_t m_lastSeq = 0;
    BYTE* m_data = nullptr;       // the newest complete payload
    BYTE* m_tmp = nullptr;        // copied into first, swapped in if untorn
    long m_cap = 0;
    long m_len = 0;
    uint32_t m_format = ODVCAM_FORMAT_BGR24;
    bool m_have = false;
    uint32_t m_lastIndex = 0;
    bool m_haveIndex = false;
    DWORD m_lastTry = 0;
};

// ---------------------------------------------------------------------------
// The output pin
//
// Carries IAMStreamConfig because that is the conventional way a client
// reads a capture device's supported resolutions; a pinless filter could
// not answer it.
//
// IKsPropertySet is not optional decoration: DirectShow identifies a
// CAPTURE pin by asking it for AMPROPERTY_PIN_CATEGORY. A pin that cannot
// answer is not treated as a capture pin at all, however good its media
// types are.
// ---------------------------------------------------------------------------

class ProbeFilter;

class ProbePin : public IPin, public IAMStreamConfig, public IKsPropertySet,
                public IAMPushSource {
public:
    ProbePin(ProbeFilter* owner, int slot)
        : m_refs(1), m_owner(owner), m_slot(slot), m_reader(slot) {}
    ~ProbePin() { StopStreaming(); }

    HRESULT StartStreaming();
    void StopStreaming();

    // -- IUnknown
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) || IsEqualIID(riid, IID_IPin)) {
            *ppv = static_cast<IPin*>(this);
        } else if (IsEqualIID(riid, IID_IAMStreamConfig)) {
            *ppv = static_cast<IAMStreamConfig*>(this);
        } else if (IsEqualIID(riid, IID_IKsPropertySet)) {
            *ppv = static_cast<IKsPropertySet*>(this);
        } else if (IsEqualIID(riid, IID_IAMPushSource)) {
            // Live capture sources are push sources. Renderers ask for
            // this to decide clocking; refusing it makes some graphs
            // treat the stream as a file and mis-time it.
            *ppv = static_cast<IAMPushSource*>(this);
        } else {
            *ppv = nullptr; return E_NOINTERFACE;
        }
        AddRef();
        return S_OK;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_refs); }
    STDMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&m_refs);
        if (n == 0) delete this;
        return n;
    }

    // -- IPin. A real output-pin connection handshake: agree a media type,
    // get the downstream input pin's IMemInputPin, negotiate an allocator,
    // and remember all three for the push thread.
    STDMETHODIMP Connect(IPin* pReceivePin, const AM_MEDIA_TYPE* pmt) override {
        if (!pReceivePin) return E_POINTER;
        if (m_connected) return VFW_E_ALREADY_CONNECTED;

        // Our own full types, in preference order, narrowed by whatever the
        // caller specified (it may give a PARTIAL type -- just a subtype,
        // no format block -- which v4 copied as-is and then dereferenced).
        // The first one the receiver takes wins, which is how DirectShow's
        // own base classes agree a type.
        const bool pixelsOnly = PixelsOnly();
        HRESULT last = VFW_E_NO_ACCEPTABLE_TYPES;
        for (int i = 0; i < kTypeCount; ++i) {
            if (!Offered(i, m_want, pixelsOnly)) continue;
            AM_MEDIA_TYPE* use = CreateTypeByIndex(i);
            if (!use) return E_OUTOFMEMORY;
            const bool matches = !pmt ||
                ((IsEqualGUID(pmt->majortype, GUID_NULL) ||
                  IsEqualGUID(pmt->majortype, use->majortype)) &&
                 (IsEqualGUID(pmt->subtype, GUID_NULL) ||
                  IsEqualGUID(pmt->subtype, use->subtype)));
            if (!matches) { DeleteMediaType(use); continue; }
            const HRESULT hr = TryConnect(pReceivePin, use);   // owns `use`
            if (SUCCEEDED(hr)) {
                RegLog(L"slot %d connected as %ls%ls", m_slot,
                       IsEqualGUID(m_mt->subtype, MEDIASUBTYPE_MJPG) ? L"MJPG" : L"RGB24",
                       pixelsOnly ? L" (writer is publishing pixels)" : L"");
                return S_OK;
            }
            last = hr;
        }
        return last;
    }

    HRESULT TryConnect(IPin* pReceivePin, AM_MEDIA_TYPE* use) {
        HRESULT hr = pReceivePin->ReceiveConnection(static_cast<IPin*>(this), use);
        if (FAILED(hr)) { DeleteMediaType(use); return hr; }

        IMemInputPin* mem = nullptr;
        hr = pReceivePin->QueryInterface(IID_IMemInputPin, (void**)&mem);
        if (FAILED(hr)) { pReceivePin->Disconnect(); DeleteMediaType(use); return hr; }

        // Allocator: prefer the downstream pin's, fall back to asking it to
        // use one it supplies. A sample must come from the allocator the
        // receiver agreed to, or Receive() is entitled to reject it.
        IMemAllocator* alloc = nullptr;
        hr = mem->GetAllocator(&alloc);
        if (FAILED(hr) || !alloc) {
            hr = CoCreateInstance(CLSID_MemoryAllocator, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_IMemAllocator, (void**)&alloc);
        }
        if (FAILED(hr) || !alloc) { mem->Release(); pReceivePin->Disconnect();
                                    DeleteMediaType(use); return E_FAIL; }

        // Buffers are always pixel-sized: an MJPG sample is at most that.
        ALLOCATOR_PROPERTIES req = {}, actual = {};
        req.cBuffers = 4;                      // small ring; this is live video
        req.cbBuffer = (long)((VIDEOINFOHEADER*)use->pbFormat)->bmiHeader.biSizeImage;
        req.cbAlign = 1;
        mem->GetAllocatorRequirements(&req);   // honour downstream needs if stated
        if (req.cBuffers < 1) req.cBuffers = 4;
        if (req.cbAlign < 1) req.cbAlign = 1;
        req.cbBuffer = (long)((VIDEOINFOHEADER*)use->pbFormat)->bmiHeader.biSizeImage;

        hr = alloc->SetProperties(&req, &actual);
        if (SUCCEEDED(hr)) hr = mem->NotifyAllocator(alloc, FALSE);
        if (SUCCEEDED(hr)) hr = alloc->Commit();
        if (FAILED(hr)) {
            alloc->Release(); mem->Release(); pReceivePin->Disconnect();
            DeleteMediaType(use); return hr;
        }

        m_peer = pReceivePin; m_peer->AddRef();
        m_memInput = mem;
        m_allocator = alloc;
        m_mt = use;
        m_connected = true;
        return S_OK;
    }

    STDMETHODIMP ReceiveConnection(IPin*, const AM_MEDIA_TYPE*) override {
        // Output pin: a connection is never received, only made.
        return E_UNEXPECTED;
    }

    STDMETHODIMP Disconnect() override {
        StopStreaming();
        if (m_allocator) { m_allocator->Decommit(); m_allocator->Release(); m_allocator = nullptr; }
        if (m_memInput) { m_memInput->Release(); m_memInput = nullptr; }
        if (m_peer) { m_peer->Release(); m_peer = nullptr; }
        if (m_mt) { DeleteMediaType(m_mt); m_mt = nullptr; }
        m_connected = false;
        return S_OK;
    }

    STDMETHODIMP ConnectedTo(IPin** pPin) override {
        if (!pPin) return E_POINTER;
        *pPin = m_peer;
        if (m_peer) { m_peer->AddRef(); return S_OK; }
        return VFW_E_NOT_CONNECTED;
    }

    STDMETHODIMP ConnectionMediaType(AM_MEDIA_TYPE* pmt) override {
        if (!pmt) return E_POINTER;
        if (!m_connected || !m_mt) return VFW_E_NOT_CONNECTED;
        AM_MEDIA_TYPE* copy = CopyMediaType(m_mt);
        if (!copy) return E_OUTOFMEMORY;
        *pmt = *copy;
        CoTaskMemFree(copy);      // the block moved to the caller; free the shell
        return S_OK;
    }
    STDMETHODIMP QueryPinInfo(PIN_INFO* pInfo) override;
    STDMETHODIMP QueryDirection(PIN_DIRECTION* pPinDir) override {
        if (!pPinDir) return E_POINTER;
        *pPinDir = PINDIR_OUTPUT;
        return S_OK;
    }
    STDMETHODIMP QueryId(LPWSTR* Id) override {
        if (!Id) return E_POINTER;
        const wchar_t* id = L"Capture";
        size_t bytes = (wcslen(id) + 1) * sizeof(wchar_t);
        *Id = (LPWSTR)CoTaskMemAlloc(bytes);
        if (!*Id) return E_OUTOFMEMORY;
        CopyMemory(*Id, id, bytes);
        return S_OK;
    }
    STDMETHODIMP QueryAccept(const AM_MEDIA_TYPE* pmt) override {
        if (!pmt) return E_POINTER;
        if (!IsEqualGUID(pmt->majortype, MEDIATYPE_Video)) return S_FALSE;
        if (!IsEqualGUID(pmt->subtype, MEDIASUBTYPE_RGB24) &&
            !IsEqualGUID(pmt->subtype, MEDIASUBTYPE_MJPG)) return S_FALSE;
        // A format block, if given, must be our geometry: frames are never
        // scaled here.
        if (IsEqualGUID(pmt->formattype, FORMAT_VideoInfo) && pmt->pbFormat &&
            pmt->cbFormat >= sizeof(VIDEOINFOHEADER)) {
            const BITMAPINFOHEADER& bi = ((VIDEOINFOHEADER*)pmt->pbFormat)->bmiHeader;
            if (bi.biWidth != kWidth || abs(bi.biHeight) != kHeight) return S_FALSE;
        }
        return S_OK;
    }
    STDMETHODIMP EnumMediaTypes(IEnumMediaTypes** ppEnum) override {
        if (!ppEnum) return E_POINTER;
        EnumMediaTypesImpl* e = new (std::nothrow) EnumMediaTypesImpl(m_want, PixelsOnly());
        if (!e) return E_OUTOFMEMORY;
        *ppEnum = e;
        return S_OK;
    }
    STDMETHODIMP QueryInternalConnections(IPin**, ULONG* nPin) override {
        if (nPin) *nPin = 0;
        return E_NOTIMPL;
    }
    STDMETHODIMP EndOfStream() override { return S_OK; }
    STDMETHODIMP BeginFlush() override { return S_OK; }
    STDMETHODIMP EndFlush() override { return S_OK; }
    STDMETHODIMP NewSegment(REFERENCE_TIME, REFERENCE_TIME, double) override {
        return S_OK;
    }

    // -- IAMStreamConfig: the resolutions list a client reads.
    // SetFormat is how a client picks MJPG (OpenCV's DirectShow capture
    // calls it for CAP_PROP_FOURCC). v4 accepted and ignored it.
    STDMETHODIMP SetFormat(AM_MEDIA_TYPE* pmt) override {
        if (!pmt) { m_want = GUID_NULL; return S_OK; }
        if (QueryAccept(pmt) != S_OK) return VFW_E_INVALIDMEDIATYPE;
        if (IsEqualGUID(pmt->subtype, MEDIASUBTYPE_MJPG) && PixelsOnly()) {
            return VFW_E_INVALIDMEDIATYPE;     // see Offered()
        }
        if (m_connected && m_mt && !IsEqualGUID(m_mt->subtype, pmt->subtype)) {
            return VFW_E_WRONG_STATE;          // reconnect to change it
        }
        m_want = pmt->subtype;
        return S_OK;
    }
    STDMETHODIMP GetFormat(AM_MEDIA_TYPE** ppmt) override {
        if (!ppmt) return E_POINTER;
        if (m_connected && m_mt) { *ppmt = CopyMediaType(m_mt); }
        else { *ppmt = CreateTypeByIndex(Offered(0, m_want, PixelsOnly()) ? 0 : 1); }
        return *ppmt ? S_OK : E_OUTOFMEMORY;
    }
    STDMETHODIMP GetNumberOfCapabilities(int* piCount, int* piSize) override {
        if (!piCount || !piSize) return E_POINTER;
        *piCount = kTypeCount;
        *piSize = sizeof(VIDEO_STREAM_CONFIG_CAPS);
        return S_OK;
    }
    STDMETHODIMP GetStreamCaps(int iIndex, AM_MEDIA_TYPE** ppmt,
                               BYTE* pSCC) override {
        if (!ppmt || !pSCC) return E_POINTER;
        if (iIndex < 0 || iIndex >= kTypeCount) return S_FALSE;
        *ppmt = CreateTypeByIndex(iIndex);
        if (!*ppmt) return E_OUTOFMEMORY;

        VIDEO_STREAM_CONFIG_CAPS* caps = (VIDEO_STREAM_CONFIG_CAPS*)pSCC;
        ZeroMemory(caps, sizeof(VIDEO_STREAM_CONFIG_CAPS));
        caps->guid = FORMAT_VideoInfo;
        caps->VideoStandard = AnalogVideo_None;
        caps->InputSize.cx = kWidth;
        caps->InputSize.cy = kHeight;
        caps->MinCroppingSize.cx = kWidth;
        caps->MinCroppingSize.cy = kHeight;
        caps->MaxCroppingSize.cx = kWidth;
        caps->MaxCroppingSize.cy = kHeight;
        caps->CropGranularityX = 1;
        caps->CropGranularityY = 1;
        caps->MinOutputSize = caps->InputSize;
        caps->MaxOutputSize = caps->InputSize;
        caps->OutputGranularityX = 1;
        caps->OutputGranularityY = 1;
        caps->MinFrameInterval = kFrameTime;
        caps->MaxFrameInterval = kFrameTime;
        caps->MinBitsPerSecond = (LONG)(kWidth * kHeight * 24 * kFps);
        caps->MaxBitsPerSecond = caps->MinBitsPerSecond;
        return S_OK;
    }

    // -- IKsPropertySet: how DirectShow learns this is a CAPTURE pin.
    STDMETHODIMP Set(REFGUID, DWORD, LPVOID, DWORD, LPVOID, DWORD) override {
        return E_NOTIMPL;
    }
    STDMETHODIMP Get(REFGUID guidPropSet, DWORD dwPropID, LPVOID, DWORD,
                     LPVOID pPropData, DWORD cbPropData,
                     DWORD* pcbReturned) override {
        if (!IsEqualGUID(guidPropSet, AMPROPSETID_Pin)) return E_PROP_SET_UNSUPPORTED;
        if (dwPropID != AMPROPERTY_PIN_CATEGORY) return E_PROP_ID_UNSUPPORTED;
        if (pcbReturned) *pcbReturned = sizeof(GUID);
        if (!pPropData) return S_OK;                 // size query
        if (cbPropData < sizeof(GUID)) return E_UNEXPECTED;
        *(GUID*)pPropData = PIN_CATEGORY_CAPTURE;
        return S_OK;
    }
    STDMETHODIMP QuerySupported(REFGUID guidPropSet, DWORD dwPropID,
                                DWORD* pTypeSupport) override {
        if (!IsEqualGUID(guidPropSet, AMPROPSETID_Pin)) return E_PROP_SET_UNSUPPORTED;
        if (dwPropID != AMPROPERTY_PIN_CATEGORY) return E_PROP_ID_UNSUPPORTED;
        if (pTypeSupport) *pTypeSupport = KSPROPERTY_SUPPORT_GET;
        return S_OK;
    }

    // -- IAMPushSource: this is a live source, so the graph should not try
    // to seek it or reconstruct a file clock from it.
    STDMETHODIMP GetPushSourceFlags(ULONG* pFlags) override {
        if (!pFlags) return E_POINTER;
        *pFlags = 0;
        return S_OK;
    }
    STDMETHODIMP SetPushSourceFlags(ULONG) override { return E_NOTIMPL; }
    STDMETHODIMP SetStreamOffset(REFERENCE_TIME) override { return E_NOTIMPL; }
    STDMETHODIMP GetStreamOffset(REFERENCE_TIME* p) override {
        if (!p) return E_POINTER; *p = 0; return S_OK;
    }
    STDMETHODIMP GetMaxStreamOffset(REFERENCE_TIME* p) override {
        if (!p) return E_POINTER; *p = 0; return S_OK;
    }
    STDMETHODIMP SetMaxStreamOffset(REFERENCE_TIME) override { return E_NOTIMPL; }
    STDMETHODIMP GetLatency(REFERENCE_TIME* prtLatency) override {
        if (!prtLatency) return E_POINTER;
        *prtLatency = kFrameTime;
        return S_OK;
    }

private:
    static DWORD WINAPI ThreadProc(LPVOID self) {
        return ((ProbePin*)self)->Run();
    }
    DWORD Run();
    void FillTestPattern(BYTE* dst, long len, int frame);
    bool Convert(IWICImagingFactory* wic, bool mjpgOut, int w, int h,
                 BYTE* out, long cap, long* outLen, BYTE* scratch);

    LONG m_refs;
    ProbeFilter* m_owner;   // weak -- the filter outlives its pin
    int m_slot;
    bool PixelsOnly() { return m_reader.PeekFormat() == (int)ODVCAM_FORMAT_BGR24; }

    SharedFrameReader m_reader;
    bool m_sawWriter = false;
    GUID m_want = GUID_NULL;       // SetFormat's choice; GUID_NULL = any

    bool m_connected = false;
    IPin* m_peer = nullptr;
    IMemInputPin* m_memInput = nullptr;
    IMemAllocator* m_allocator = nullptr;
    AM_MEDIA_TYPE* m_mt = nullptr;

    HANDLE m_thread = nullptr;
    HANDLE m_stopEvent = nullptr;
    LONGLONG m_frameIndex = 0;
};

// ---------------------------------------------------------------------------
// IEnumPins
// ---------------------------------------------------------------------------

class EnumPinsImpl : public IEnumPins {
public:
    EnumPinsImpl(IPin* pin, int pos = 0) : m_refs(1), m_pin(pin), m_pos(pos) {
        if (m_pin) m_pin->AddRef();
    }
    ~EnumPinsImpl() { if (m_pin) m_pin->Release(); }

    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) || IsEqualIID(riid, IID_IEnumPins)) {
            *ppv = static_cast<IEnumPins*>(this); AddRef(); return S_OK;
        }
        *ppv = nullptr; return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_refs); }
    STDMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&m_refs);
        if (n == 0) delete this;
        return n;
    }

    STDMETHODIMP Next(ULONG cPins, IPin** ppPins, ULONG* pcFetched) override {
        if (!ppPins) return E_POINTER;
        ULONG fetched = 0;
        if (cPins > 0 && m_pos == 0 && m_pin) {
            m_pin->AddRef();
            ppPins[0] = m_pin;
            fetched = 1;
            m_pos = 1;
        }
        if (pcFetched) *pcFetched = fetched;
        return (fetched == cPins) ? S_OK : S_FALSE;
    }
    STDMETHODIMP Skip(ULONG cPins) override {
        m_pos += (int)cPins;
        return (m_pos <= 1) ? S_OK : S_FALSE;
    }
    STDMETHODIMP Reset() override { m_pos = 0; return S_OK; }
    STDMETHODIMP Clone(IEnumPins** ppEnum) override {
        if (!ppEnum) return E_POINTER;
        EnumPinsImpl* e = new (std::nothrow) EnumPinsImpl(m_pin, m_pos);
        if (!e) return E_OUTOFMEMORY;
        *ppEnum = e;
        return S_OK;
    }

private:
    LONG m_refs;
    IPin* m_pin;
    int m_pos;
};

// ---------------------------------------------------------------------------
// The filter
// ---------------------------------------------------------------------------

class ProbeFilter : public IBaseFilter {
public:
    ProbeFilter(const CLSID& clsid, int slot)
        : m_refs(1), m_clsid(clsid), m_pin(nullptr) {
        m_pin = new (std::nothrow) ProbePin(this, slot);
    }
    ~ProbeFilter() { if (m_pin) m_pin->Release(); }

    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) ||
            IsEqualIID(riid, IID_IPersist) ||
            IsEqualIID(riid, IID_IMediaFilter) ||
            IsEqualIID(riid, IID_IBaseFilter)) {
            *ppv = static_cast<IBaseFilter*>(this); AddRef(); return S_OK;
        }
        *ppv = nullptr; return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_refs); }
    STDMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&m_refs);
        if (n == 0) delete this;
        return n;
    }

    STDMETHODIMP GetClassID(CLSID* pClassID) override {
        if (!pClassID) return E_POINTER;
        *pClassID = m_clsid; return S_OK;
    }

    // State drives the pin's push thread. A live source starts pushing at
    // Pause, not Run: DirectShow expects a preroll sample to be available
    // so the graph can complete its transition, and a source that waits
    // for Run can leave the graph stuck in Paused.
    STDMETHODIMP Stop() override {
        m_state = State_Stopped;
        if (m_pin) m_pin->StopStreaming();
        return S_OK;
    }
    STDMETHODIMP Pause() override {
        m_state = State_Paused;
        if (m_pin) m_pin->StartStreaming();
        return S_OK;
    }
    STDMETHODIMP Run(REFERENCE_TIME) override {
        m_state = State_Running;
        if (m_pin) m_pin->StartStreaming();
        return S_OK;
    }
    STDMETHODIMP GetState(DWORD, FILTER_STATE* pState) override {
        if (!pState) return E_POINTER;
        *pState = m_state; return S_OK;
    }
    STDMETHODIMP SetSyncSource(IReferenceClock* pClock) override {
        if (m_clock) m_clock->Release();
        m_clock = pClock;
        if (m_clock) m_clock->AddRef();
        return S_OK;
    }
    STDMETHODIMP GetSyncSource(IReferenceClock** ppClock) override {
        if (!ppClock) return E_POINTER;
        *ppClock = m_clock;
        if (m_clock) m_clock->AddRef();
        return S_OK;
    }

    STDMETHODIMP EnumPins(IEnumPins** ppEnum) override {
        if (!ppEnum) return E_POINTER;
        EnumPinsImpl* e = new (std::nothrow) EnumPinsImpl(m_pin);
        if (!e) return E_OUTOFMEMORY;
        *ppEnum = e;
        return S_OK;
    }
    STDMETHODIMP FindPin(LPCWSTR Id, IPin** ppPin) override {
        if (!ppPin) return E_POINTER;
        *ppPin = nullptr;
        if (Id && wcscmp(Id, L"Capture") == 0 && m_pin) {
            m_pin->AddRef();
            *ppPin = m_pin;
            return S_OK;
        }
        return VFW_E_NOT_FOUND;
    }
    STDMETHODIMP QueryFilterInfo(FILTER_INFO* pInfo) override {
        if (!pInfo) return E_POINTER;
        wcsncpy(pInfo->achName, L"OpenDarts Probe Cam", MAX_FILTER_NAME - 1);
        pInfo->achName[MAX_FILTER_NAME - 1] = L'\0';
        pInfo->pGraph = m_graph;
        if (m_graph) m_graph->AddRef();
        return S_OK;
    }
    STDMETHODIMP JoinFilterGraph(IFilterGraph* pGraph, LPCWSTR) override {
        m_graph = pGraph;   // weak by COM convention -- the graph owns us
        return S_OK;
    }
    STDMETHODIMP QueryVendorInfo(LPWSTR*) override { return E_NOTIMPL; }

private:
    LONG m_refs;
    CLSID m_clsid;
    ProbePin* m_pin;
    FILTER_STATE m_state = State_Stopped;
    IReferenceClock* m_clock = nullptr;
    IFilterGraph* m_graph = nullptr;
};

STDMETHODIMP ProbePin::QueryPinInfo(PIN_INFO* pInfo) {
    if (!pInfo) return E_POINTER;
    ZeroMemory(pInfo, sizeof(PIN_INFO));
    pInfo->pFilter = (IBaseFilter*)m_owner;
    if (pInfo->pFilter) pInfo->pFilter->AddRef();
    pInfo->dir = PINDIR_OUTPUT;
    wcsncpy(pInfo->achName, L"Capture", MAX_PIN_NAME - 1);
    pInfo->achName[MAX_PIN_NAME - 1] = L'\0';
    return S_OK;
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

HRESULT ProbePin::StartStreaming() {
    if (!m_connected) return VFW_E_NOT_CONNECTED;
    if (m_thread) return S_OK;                  // already running; idempotent
    m_stopEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!m_stopEvent) return E_FAIL;
    m_frameIndex = 0;
    m_thread = CreateThread(nullptr, 0, &ProbePin::ThreadProc, this, 0, nullptr);
    if (!m_thread) { CloseHandle(m_stopEvent); m_stopEvent = nullptr; return E_FAIL; }
    return S_OK;
}

void ProbePin::StopStreaming() {
    if (m_stopEvent) SetEvent(m_stopEvent);
    if (m_thread) {
        // Bounded wait. A push thread blocked in Receive() on a wedged
        // downstream filter must not hang Stop() forever -- the graph is
        // entitled to expect Stop to return.
        if (WaitForSingleObject(m_thread, 2000) == WAIT_TIMEOUT) {
            // Deliberately not TerminateThread: killing a thread inside
            // Receive() would leak the sample and can corrupt the
            // allocator. Leaking the handle is the lesser harm.
        }
        CloseHandle(m_thread);
        m_thread = nullptr;
    }
    if (m_stopEvent) { CloseHandle(m_stopEvent); m_stopEvent = nullptr; }
}

// A moving pattern, not a static one: a still image cannot distinguish
// "streaming correctly" from "delivered one frame and stalled", which is
// exactly the failure worth seeing.
void ProbePin::FillTestPattern(BYTE* dst, long len, int frame) {
    if (!m_mt || !m_mt->pbFormat) { ZeroMemory(dst, len); return; }
    VIDEOINFOHEADER* vih = (VIDEOINFOHEADER*)m_mt->pbFormat;
    const int w = vih->bmiHeader.biWidth;
    const int h = abs(vih->bmiHeader.biHeight);
    const long stride = w * 3;
    const int shift = (frame * 4) % w;

    for (int y = 0; y < h; ++y) {
        BYTE* row = dst + (long)y * stride;
        if ((row - dst) + stride > len) break;
        for (int x = 0; x < w; ++x) {
            const int band = ((x + shift) * 8 / w) % 8;
            // BGR order, and the bottom-up bitmap convention means row 0
            // here is the BOTTOM of the picture.
            row[x*3 + 0] = (band & 1) ? 255 : 0;
            row[x*3 + 1] = (band & 2) ? 255 : 0;
            row[x*3 + 2] = (band & 4) ? 255 : 0;
        }
    }
}

// The newest payload, made into what this connection carries. `out` gets
// JPEG bytes on an MJPG connection, bottom-up BGR on RGB24.
bool ProbePin::Convert(IWICImagingFactory* wic, bool mjpgOut, int w, int h,
                       BYTE* out, long cap, long* outLen, BYTE* scratch) {
    const BYTE* src = m_reader.Data();
    const long len = m_reader.Length();
    const bool isJpeg = m_reader.Format() == ODVCAM_FORMAT_MJPEG;
    const long pixels = (long)w * h * 3;
    if (mjpgOut && isJpeg) {                 // the camera's bytes, untouched
        if (len > cap) return false;
        memcpy(out, src, len);
        *outLen = len;
        return true;
    }
    if (mjpgOut) {                           // pixels over MJPG: encode
        return EncodeJpeg(wic, src, w, h, out, cap, outLen);
    }
    if (pixels > cap) return false;
    if (isJpeg) {                            // JPEG over RGB24: decode
        if (!DecodeJpeg(wic, src, (UINT)len, scratch, w, h)) return false;
        src = scratch;
    }
    FlipRows(src, out, w, h);                // DirectShow RGB24 is bottom-up
    *outLen = pixels;
    return true;
}

DWORD ProbePin::Run() {
    // The thread owns its own COM apartment: it calls into the downstream
    // filter, which may marshal.
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    int w = kWidth, h = kHeight;
    if (m_mt && m_mt->pbFormat) {
        VIDEOINFOHEADER* vih = (VIDEOINFOHEADER*)m_mt->pbFormat;
        w = vih->bmiHeader.biWidth;
        h = abs(vih->bmiHeader.biHeight);
    }
    const bool mjpgOut = m_mt && IsEqualGUID(m_mt->subtype, MEDIASUBTYPE_MJPG);
    const long cap = (long)w * h * 3;

    // Needed only when the connection and the payload differ; a missing
    // factory just means those frames are not delivered.
    IWICImagingFactory* wic = nullptr;
    CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                     IID_IWICImagingFactory, (void**)&wic);

    // `out` holds what the last fresh frame became, so a poll with nothing
    // new re-serves it without converting again.
    BYTE* out = new (std::nothrow) BYTE[cap];
    BYTE* scratch = new (std::nothrow) BYTE[cap];
    long outLen = 0;
    bool haveOut = false;

    bool stopping = false;
    while (out && scratch && !stopping &&
           WaitForSingleObject(m_stopEvent, 0) != WAIT_OBJECT_0) {
        // PACED BY ARRIVAL. Wait for the writer's next frame, then push it.
        // A stalled writer still gets its last frame re-pushed every
        // kRepeatMs so the stream stays alive, as a camera's would.
        FrameResult fr = m_reader.Poll(w, h);
        const DWORD waitStart = GetTickCount();
        while (fr == FrameResult::Repeated) {
            const DWORD waited = GetTickCount() - waitStart;
            if (waited >= kRepeatMs) break;
            HANDLE ev = m_reader.FrameEvent();
            HANDLE handles[2] = { m_stopEvent, ev };
            const DWORD r = ev
                ? WaitForMultipleObjects(2, handles, FALSE, kRepeatMs - waited)
                : WaitForSingleObject(m_stopEvent, 2);
            if (r == WAIT_OBJECT_0) { stopping = true; break; }
            fr = m_reader.Poll(w, h);
        }
        if (stopping) break;

        if (fr == FrameResult::Fresh) {
            long n = 0;
            if (Convert(wic, mjpgOut, w, h, out, cap, &n, scratch)) {
                outLen = n;
                haveOut = true;
            }
            m_sawWriter = true;
        } else if (fr == FrameResult::NoSource) {
            haveOut = false;
        }

        IMediaSample* sample = nullptr;
        HRESULT hr = m_allocator->GetBuffer(&sample, nullptr, nullptr, 0);
        if (FAILED(hr) || !sample) break;      // allocator decommitted = shutting down

        bool deliver = false;
        BYTE* data = nullptr;
        if (SUCCEEDED(sample->GetPointer(&data)) && data) {
            const long room = sample->GetSize();
            if (haveOut && outLen <= room) {
                memcpy(data, out, outLen);
                sample->SetActualDataLength(outLen);
                deliver = true;
            } else if (!haveOut) {
                // Real frame if the hub has one, else the test pattern. The
                // fallback is not cosmetic: it makes "the capture hub is not
                // running" visually obvious, where a black frame would be
                // indistinguishable from a dead camera or a broken filter.
                if (mjpgOut) {
                    FillTestPattern(scratch, cap, (int)m_frameIndex);
                    long n = 0;
                    if (EncodeJpeg(wic, scratch, w, h, data, room, &n)) {
                        sample->SetActualDataLength(n);
                        deliver = true;
                    }
                } else if (cap <= room) {
                    FillTestPattern(data, cap, (int)m_frameIndex);
                    sample->SetActualDataLength(cap);
                    deliver = true;
                }
            }
        }

        if (deliver) {
            REFERENCE_TIME start = m_frameIndex * kFrameTime;
            REFERENCE_TIME end = start + kFrameTime;
            sample->SetTime(&start, &end);
            sample->SetSyncPoint(TRUE);            // every frame is a keyframe
            hr = m_memInput->Receive(sample);
        } else {
            hr = S_OK;
        }
        sample->Release();
        if (FAILED(hr)) break;
        ++m_frameIndex;

        // The pattern has no writer to pace it: one frame period.
        if (fr == FrameResult::NoSource &&
            WaitForSingleObject(m_stopEvent, 1000 / kFps) == WAIT_OBJECT_0) break;
    }

    delete[] out;
    delete[] scratch;
    SafeRelease(wic);
    CoUninitialize();
    return 0;
}

// ---------------------------------------------------------------------------
// Class factory
// ---------------------------------------------------------------------------

class ProbeFactory : public IClassFactory {
public:
    ProbeFactory(const CLSID& clsid, int slot)
        : m_refs(1), m_clsid(clsid), m_slot(slot) {}

    STDMETHODIMP QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) || IsEqualIID(riid, IID_IClassFactory)) {
            *ppv = static_cast<IClassFactory*>(this); AddRef(); return S_OK;
        }
        *ppv = nullptr; return E_NOINTERFACE;
    }
    STDMETHODIMP_(ULONG) AddRef() override { return InterlockedIncrement(&m_refs); }
    STDMETHODIMP_(ULONG) Release() override {
        LONG n = InterlockedDecrement(&m_refs);
        if (n == 0) delete this;
        return n;
    }
    STDMETHODIMP CreateInstance(IUnknown* pOuter, REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (pOuter) return CLASS_E_NOAGGREGATION;
        ProbeFilter* f = new (std::nothrow) ProbeFilter(m_clsid, m_slot);
        if (!f) return E_OUTOFMEMORY;
        HRESULT hr = f->QueryInterface(riid, ppv);
        f->Release();
        return hr;
    }
    STDMETHODIMP LockServer(BOOL fLock) override {
        if (fLock) InterlockedIncrement(&g_lockCount);
        else InterlockedDecrement(&g_lockCount);
        return S_OK;
    }

private:
    LONG m_refs;
    CLSID m_clsid;
    int m_slot;
};

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// regsvr32 reports a single HRESULT and nothing about which step produced
// it, so 0x80070005 from three different registry operations is one
// message with three meanings. This writes each step's real result to
// %TEMP%\\vcam_probe.log. Best-effort throughout: a logging failure must
// never change whether registration succeeds.
static void RegLog(const wchar_t* fmt, ...) {
    wchar_t path[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, path)) return;
    wcsncat(path, L"vcam_probe.log", MAX_PATH - wcslen(path) - 1);

    FILE* f = _wfopen(path, L"a, ccs=UTF-8");
    if (!f) return;

    SYSTEMTIME st;
    GetLocalTime(&st);
    fwprintf(f, L"[%02d:%02d:%02d] ", st.wHour, st.wMinute, st.wSecond);

    va_list args;
    va_start(args, fmt);
    vfwprintf(f, fmt, args);
    va_end(args);

    fwprintf(f, L"\n");
    fclose(f);
}

// ---------------------------------------------------------------------------
// WHICH REGISTRY HIVE -- i.e. why this ever needed admin, and why it now
// does not.
//
// HKEY_CLASSES_ROOT is a MERGED VIEW of HKLM\Software\Classes and
// HKCU\Software\Classes. READS see the union of both, with the HKCU half
// winning ties. WRITES go to the HKLM half, which needs elevation. That
// asymmetry is the entire content of the 0x80070005 (ERROR_ACCESS_DENIED)
// failure from a non-elevated regsvr32: DirectShow is not refusing an
// unprivileged filter, one registry hive is refusing a write.
//
// So when the HKLM half is not writable, register under
// HKCU\Software\Classes instead. The enumerator reads through the merged
// view and finds the filters there exactly as it would have found them in
// HKLM. What changes is scope -- current user rather than machine-wide --
// which for a diagnostic tool on somebody's own desktop is the more
// correct scope regardless.
static bool g_perUserReg = false;

static HKEY RegRoot(void) {
    return g_perUserReg ? HKEY_CURRENT_USER : HKEY_CLASSES_ROOT;
}

// Rewrite a path that is relative to the merged view ("CLSID\...") into
// the HKCU hive when that is where this registration is going.
static void ClassesPath(wchar_t* out, size_t cch, const wchar_t* suffix) {
    if (g_perUserReg) swprintf(out, cch, L"Software\\Classes\\%ls", suffix);
    else              swprintf(out, cch, L"%ls", suffix);
}

// Chosen by MEASUREMENT, not by asking whether the token is elevated: what
// decides this is whether the write actually succeeds, and it can fail for
// reasons that have nothing to do with UAC (group policy, a locked-down
// hive, a managed machine). Probing with a real create is the only check
// that cannot disagree with what the real writes are about to do.
//
// The probe key is VOLATILE, so even if the delete below is somehow
// skipped it does not survive a reboot -- this must not leave litter in
// the machine-wide hive of a machine it was only inspecting.
static void ChooseRegistryHive(void) {
    HKEY probe = nullptr;
    LONG rc = RegCreateKeyExW(HKEY_CLASSES_ROOT, L"CLSID\\ODVCamWriteProbe", 0,
                              nullptr, REG_OPTION_VOLATILE, KEY_WRITE, nullptr,
                              &probe, nullptr);
    if (rc == ERROR_SUCCESS) {
        RegCloseKey(probe);
        RegDeleteKeyW(HKEY_CLASSES_ROOT, L"CLSID\\ODVCamWriteProbe");
        g_perUserReg = false;
    } else {
        g_perUserReg = true;
    }
    RegLog(L"registry hive: %ls  (HKCR write probe rc=%ld)",
           g_perUserReg ? L"HKCU\\Software\\Classes -- per-user, no admin needed"
                        : L"HKCR -- machine-wide",
           rc);
}

static HRESULT RegisterServerKeys(const CLSID& clsid, const wchar_t* friendlyName) {
    wchar_t clsidStr[64];
    StringFromGUID2(clsid, clsidStr, 64);

    wchar_t modulePath[MAX_PATH];
    if (!GetModuleFileNameW(g_instance, modulePath, MAX_PATH)) return E_FAIL;

    wchar_t rel[256], keyPath[320];
    swprintf(rel, 256, L"CLSID\\%ls", clsidStr);
    ClassesPath(keyPath, 320, rel);

    HKEY key = nullptr;
    LONG rc = RegCreateKeyExW(RegRoot(), keyPath, 0, nullptr, 0,
                              KEY_WRITE, nullptr, &key, nullptr);
    if (rc != ERROR_SUCCESS) {
        // 5 here is ERROR_ACCESS_DENIED. It used to mean "not elevated",
        // but ChooseRegistryHive() has already fallen back to the per-user
        // hive in that case -- so reaching this now means a write failed
        // somewhere it was measured to succeed, which is a real anomaly
        // rather than the expected UAC story. Naming the key keeps the two
        // distinguishable.
        RegLog(L"RegCreateKeyEx(%ls\\%ls) FAILED rc=%ld  module=%ls",
               g_perUserReg ? L"HKCU" : L"HKCR", keyPath, rc, modulePath);
        return HRESULT_FROM_WIN32(rc);
    }
    RegLog(L"RegCreateKeyEx(%ls\\%ls) ok", g_perUserReg ? L"HKCU" : L"HKCR", keyPath);
    RegSetValueExW(key, nullptr, 0, REG_SZ, (const BYTE*)friendlyName,
                   (DWORD)((wcslen(friendlyName) + 1) * sizeof(wchar_t)));

    HKEY sub = nullptr;
    if (RegCreateKeyExW(key, L"InprocServer32", 0, nullptr, 0,
                        KEY_WRITE, nullptr, &sub, nullptr) == ERROR_SUCCESS) {
        RegSetValueExW(sub, nullptr, 0, REG_SZ, (const BYTE*)modulePath,
                       (DWORD)((wcslen(modulePath) + 1) * sizeof(wchar_t)));
        const wchar_t* model = L"Both";
        RegSetValueExW(sub, L"ThreadingModel", 0, REG_SZ, (const BYTE*)model,
                       (DWORD)((wcslen(model) + 1) * sizeof(wchar_t)));
        RegCloseKey(sub);
    }
    RegCloseKey(key);
    return S_OK;
}

// Some DirectShow clients read each device's DevicePath and parse a USB
// vendor/product id out of it (vid_XXXX / pid_XXXX), dropping devices
// where that fails. A filter registered through IFilterMapper2 has no
// DevicePath at all -- there is no hardware behind it -- so such a client
// drops it even though it advertises formats.
//
// The moniker's property bag is backed by this registry key, so a
// DevicePath written here is what the enumerator reads back. The value is
// shaped like a real USB path because that parse is all that inspects it;
// the VID/PID are deliberately not a real vendor's.
// `instance` must be the SAME string passed as RegisterFilter's szInstance
// -- that parameter names the registry key, and passing the friendly name
// there while looking the key up by CLSID is exactly the mismatch that
// made v3 fail to register at all.
// The fabricated USB path, defined once because two callers now write it:
// the machine-wide path (after IFilterMapper2 has made the Instance key)
// and the per-user path (which makes that key itself).
static void DevicePathFor(int index, wchar_t* out, size_t cch) {
    swprintf(out, cch,
             L"\\\\?\\usb#vid_0d00&pid_100%d#od_probe_%d#"
             L"{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global",
             index, index);
}

// The per-user stand-in for IFilterMapper2::RegisterFilter.
//
// The mapper can only write the machine-wide hive -- it goes through
// HKEY_CLASSES_ROOT like everything else, so under a non-elevated process
// it fails with the same ERROR_ACCESS_DENIED. What it writes, though, is
// just registry values, and the Instance key is a documented layout: the
// moniker's property bag IS this key. So per-user registration writes it
// directly.
//
// NO FilterData VALUE, deliberately. That blob is the serialized REGFILTER2
// the mapper builds, and it is consumed by graph building
// (IFilterMapper2::EnumMatchingFilters) -- not by device enumeration, which
// is the only thing that matters here. A typical client enumerating
// capture devices reads FriendlyName and DevicePath off the property bag,
// then binds the filter and asks the PIN for formats. None of that path touches
// FilterData. Written down because if these devices ever enumerate
// machine-wide but not per-user, this omission is the first suspect and the
// blob can be copied from a machine-wide registration.
static HRESULT RegisterInstanceKey(const CLSID& clsid, const wchar_t* name,
                                   int index) {
    wchar_t catStr[64], clsidStr[64];
    StringFromGUID2(CLSID_VideoInputDeviceCategory, catStr, 64);
    StringFromGUID2(clsid, clsidStr, 64);

    wchar_t rel[320], keyPath[400];
    swprintf(rel, 320, L"CLSID\\%ls\\Instance\\%ls", catStr, name);
    ClassesPath(keyPath, 400, rel);

    HKEY key = nullptr;
    LONG rc = RegCreateKeyExW(RegRoot(), keyPath, 0, nullptr, 0, KEY_WRITE,
                              nullptr, &key, nullptr);
    if (rc != ERROR_SUCCESS) {
        RegLog(L"RegCreateKeyEx(HKCU\\%ls) FAILED rc=%ld", keyPath, rc);
        return HRESULT_FROM_WIN32(rc);
    }

    // CLSID is what the enumerator binds through to reach the filter;
    // without it the key is an entry naming nothing.
    RegSetValueExW(key, L"CLSID", 0, REG_SZ, (const BYTE*)clsidStr,
                   (DWORD)((wcslen(clsidStr) + 1) * sizeof(wchar_t)));
    RegSetValueExW(key, L"FriendlyName", 0, REG_SZ, (const BYTE*)name,
                   (DWORD)((wcslen(name) + 1) * sizeof(wchar_t)));

    wchar_t devicePath[256];
    DevicePathFor(index, devicePath, 256);
    LONG drc = RegSetValueExW(key, L"DevicePath", 0, REG_SZ,
                              (const BYTE*)devicePath,
                              (DWORD)((wcslen(devicePath) + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
    RegLog(L"RegisterInstanceKey(%ls) at HKCU\\%ls  DevicePath rc=%ld",
           name, keyPath, drc);
    return (drc == ERROR_SUCCESS) ? S_OK : E_FAIL;
}

static HRESULT WriteDevicePath(const wchar_t* instance, const CLSID& clsid,
                               int index) {
    wchar_t catStr[64], clsidStr[64];
    StringFromGUID2(CLSID_VideoInputDeviceCategory, catStr, 64);
    StringFromGUID2(clsid, clsidStr, 64);

    // Try the instance name first, then the CLSID: szInstance defaults to
    // the CLSID when NULL, so both spellings exist in the wild and trying
    // both costs nothing.
    HKEY key = nullptr;
    wchar_t rel[320], keyPath[400];
    swprintf(rel, 320, L"CLSID\\%ls\\Instance\\%ls", catStr, instance);
    ClassesPath(keyPath, 400, rel);
    LONG orc = RegOpenKeyExW(RegRoot(), keyPath, 0, KEY_WRITE, &key);
    if (orc != ERROR_SUCCESS) {
        swprintf(rel, 320, L"CLSID\\%ls\\Instance\\%ls", catStr, clsidStr);
        ClassesPath(keyPath, 400, rel);
        orc = RegOpenKeyExW(RegRoot(), keyPath, 0, KEY_WRITE, &key);
    }
    if (orc != ERROR_SUCCESS) return HRESULT_FROM_WIN32(orc);

    wchar_t devicePath[256];
    DevicePathFor(index, devicePath, 256);

    LONG rc = RegSetValueExW(key, L"DevicePath", 0, REG_SZ,
                             (const BYTE*)devicePath,
                             (DWORD)((wcslen(devicePath) + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
    return (rc == ERROR_SUCCESS) ? S_OK : E_FAIL;
}

// Deletes from whichever hive g_perUserReg currently names. Deepest key
// first: RegDeleteKey refuses a key that still has subkeys, so removing
// the parent before InprocServer32 would silently leave the whole
// registration behind.
static void UnregisterServerKeys(const CLSID& clsid, const wchar_t* name) {
    wchar_t clsidStr[64], catStr[64];
    StringFromGUID2(clsid, clsidStr, 64);
    StringFromGUID2(CLSID_VideoInputDeviceCategory, catStr, 64);

    wchar_t rel[320], keyPath[400];

    swprintf(rel, 320, L"CLSID\\%ls\\InprocServer32", clsidStr);
    ClassesPath(keyPath, 400, rel);
    RegDeleteKeyW(RegRoot(), keyPath);

    swprintf(rel, 320, L"CLSID\\%ls", clsidStr);
    ClassesPath(keyPath, 400, rel);
    RegDeleteKeyW(RegRoot(), keyPath);

    // The Instance key: IFilterMapper2::UnregisterFilter removes this one
    // in the machine-wide case, but nothing does per-user, and a left-over
    // Instance key is the worst kind of litter -- the enumerator still
    // lists the device, binding fails, and the client reports a broken
    // camera that no longer exists.
    swprintf(rel, 320, L"CLSID\\%ls\\Instance\\%ls", catStr, name);
    ClassesPath(keyPath, 400, rel);
    RegDeleteKeyW(RegRoot(), keyPath);
}

STDAPI DllRegisterServer(void) {
    RegLog(L"---- DllRegisterServer ----");
    HRESULT hr = CoInitialize(nullptr);
    const bool didInit = SUCCEEDED(hr);

    // Decides machine-wide vs per-user for every write below.
    ChooseRegistryHive();

    // The mapper is only reachable in the machine-wide case -- it writes
    // through HKEY_CLASSES_ROOT, so unelevated it would fail with the very
    // error the per-user hive exists to avoid. RegisterInstanceKey() takes
    // its place there.
    IFilterMapper2* mapper = nullptr;
    if (!g_perUserReg) {
        hr = CoCreateInstance(CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
                              IID_IFilterMapper2, (void**)&mapper);
        if (FAILED(hr)) {
            RegLog(L"CoCreateInstance(FilterMapper2) FAILED hr=0x%08lx", (unsigned long)hr);
            if (didInit) CoUninitialize();
            return hr;
        }
        RegLog(L"FilterMapper2 ok");
    }

    // Declare the pin to the mapper as well as implementing it. v1 declared
    // cPins = 0; a mapper entry that admits no pins is a second way to look
    // malformed to anything reading the registry rather than instantiating.
    REGPINTYPES types[2] = {
        { &MEDIATYPE_Video, &MEDIASUBTYPE_MJPG },
        { &MEDIATYPE_Video, &MEDIASUBTYPE_RGB24 },
    };
    REGFILTERPINS pins[1] = {};
    pins[0].strName = const_cast<wchar_t*>(L"Capture");
    pins[0].bRendered = FALSE;
    pins[0].bOutput = TRUE;
    pins[0].bZero = FALSE;
    pins[0].bMany = FALSE;
    pins[0].clsConnectsToFilter = &CLSID_NULL;
    pins[0].strConnectsToPin = nullptr;
    pins[0].nMediaTypes = 2;
    pins[0].lpMediaType = types;

    int devicePathFailures = 0;

    REGFILTER2 rf = {};
    rf.dwVersion = 1;
    rf.dwMerit = MERIT_DO_NOT_USE;   // what real capture devices register with
    rf.cPins = 1;
    rf.rgPins = pins;

    for (int i = 0; i < 3; ++i) {
        HRESULT rhr = RegisterServerKeys(*kClsids[i], kNames[i]);
        if (FAILED(rhr)) { hr = rhr; break; }

        if (g_perUserReg) {
            // One key carries name, CLSID and DevicePath together, so
            // there is no second step to fail separately here.
            rhr = RegisterInstanceKey(*kClsids[i], kNames[i], i);
            if (FAILED(rhr)) { hr = rhr; devicePathFailures++; break; }
            continue;
        }

        IMoniker* moniker = nullptr;
        rhr = mapper->RegisterFilter(*kClsids[i], kNames[i], &moniker,
                                     &CLSID_VideoInputDeviceCategory,
                                     kNames[i], &rf);
        if (moniker) moniker->Release();
        RegLog(L"RegisterFilter(%ls) hr=0x%08lx", kNames[i], (unsigned long)rhr);
        if (FAILED(rhr)) { hr = rhr; break; }

        // After RegisterFilter, which is what creates the Instance key.
        // Deliberately NOT fatal: a probe that registers without a
        // DevicePath is still worth having -- it distinguishes "could not
        // write the path" from "wrote it and the client still refused". Aborting registration here turned a
        // partial failure into no probe at all.
        HRESULT dhr = WriteDevicePath(kNames[i], *kClsids[i], i);
        RegLog(L"WriteDevicePath(%ls) hr=0x%08lx", kNames[i], (unsigned long)dhr);
        if (FAILED(dhr)) devicePathFailures++;
    }

    if (mapper) mapper->Release();
    if (didInit) CoUninitialize();
    // Always S_OK on success, never S_FALSE. S_FALSE is a success code,
    // but regsvr32 is not consistent about it -- some builds test
    // `hr != S_OK` and report a perfectly good registration as a failure,
    // with a code that sends you looking for a permissions problem that
    // is not there. The partial-failure signal it carried is redundant:
    // the registry itself (and the log line below) shows whether
    // DevicePath actually landed.
    RegLog(L"DllRegisterServer result hr=0x%08lx devicePathFailures=%d",
           (unsigned long)hr, devicePathFailures);
    return SUCCEEDED(hr) ? S_OK : hr;
}

STDAPI DllUnregisterServer(void) {
    RegLog(L"---- DllUnregisterServer ----");
    HRESULT hr = CoInitialize(nullptr);
    const bool didInit = SUCCEEDED(hr);

    // BOTH hives, unconditionally, rather than whichever one this process
    // could write today. A machine that was registered elevated and is now
    // being cleaned up unelevated (or the reverse) is the ordinary case
    // once per-user registration exists, and deleting a key that was never
    // there costs one failed call. Asking ChooseRegistryHive() instead
    // would clean exactly one hive and silently leave the other
    // registered -- devices that still enumerate and can no longer bind.
    IFilterMapper2* mapper = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FilterMapper2, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_IFilterMapper2, (void**)&mapper)) && mapper) {
        for (int i = 0; i < 3; ++i) {
            mapper->UnregisterFilter(&CLSID_VideoInputDeviceCategory,
                                     kNames[i], *kClsids[i]);
        }
        mapper->Release();
    }
    for (int pass = 0; pass < 2; ++pass) {
        g_perUserReg = (pass == 1);
        for (int i = 0; i < 3; ++i) UnregisterServerKeys(*kClsids[i], kNames[i]);
    }
    g_perUserReg = false;

    if (didInit) CoUninitialize();
    return hr;
}

// ---------------------------------------------------------------------------
// COM entry points
// ---------------------------------------------------------------------------

STDAPI DllGetClassObject(REFCLSID rclsid, REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    for (int i = 0; i < 3; ++i) {
        if (IsEqualCLSID(rclsid, *kClsids[i])) {
            ProbeFactory* f = new (std::nothrow) ProbeFactory(*kClsids[i], i);
            if (!f) return E_OUTOFMEMORY;
            HRESULT hr = f->QueryInterface(riid, ppv);
            f->Release();
            return hr;
        }
    }
    return CLASS_E_CLASSNOTAVAILABLE;
}

STDAPI DllCanUnloadNow(void) {
    return (g_lockCount == 0) ? S_OK : S_FALSE;
}

extern "C" BOOL WINAPI DllMain(HINSTANCE inst, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_instance = inst;
        DisableThreadLibraryCalls(inst);
    }
    return TRUE;
}
