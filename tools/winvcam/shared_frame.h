// Shared-memory frame contract between the capture hub (Python, writer)
// and the DirectShow virtual cameras (this DLL, reader).
//
// One named mapping per camera slot: Local\ODVCam0, Local\ODVCam1, ...
// "Local\" rather than "Global\" deliberately -- both processes run as the
// same desktop user, and Global\ needs SeCreateGlobalPrivilege, which would
// mean running the capture hub elevated for no benefit.
//
// TEARING. The writer publishes whole frames into a buffer the reader can
// sample at any moment, so a naive copy can catch a half-written frame.
// This uses a seqlock, which suits the access pattern exactly: one writer,
// rare readers, and no need for the writer to ever block.
//
//   writer:  seq -> odd      (write in progress)
//            copy pixels
//            seq -> even     (complete, and a NEW value)
//
//   reader:  read seq; if odd, the frame is mid-write -- skip it
//            copy pixels
//            read seq again; if it changed, the copy was torn -- discard
//
// The reader never blocks the writer and a torn read costs one dropped
// frame rather than a corrupted one. At 30fps a dropped frame is invisible;
// a torn one is a visible glitch that would look like a capture bug.
//
// PAYLOAD, one of two formats, named per frame by `format`:
//
//   BGR24  pixels, top-down -- what OpenCV hands back, so the writer copies
//          without converting. `stride` is bytes per row (width*3).
//   MJPEG  the camera's own JPEG, exactly as the camera sent it (repaired
//          to end at its EOI marker). `stride` is its length in bytes.
//
// MJPEG exists so a consumer gets what the real camera would give it: a
// client that asks the virtual camera for MJPG receives the JPEG and does
// its own decode, as it would plugged into the hardware (2026-09-17). The
// filter converts whenever the connection and the payload differ -- decode
// for an RGB24 connection, encode for MJPG over a BGR24 payload.
//
// DirectShow's RGB24 is bottom-up by convention, so the READER flips rows.
// Doing it on the reader side keeps the cost off the capture loop, which is
// the latency-sensitive side.
//
// The mapping is sized for BGR24 pixels, so any JPEG of the same geometry
// fits in it.

#ifndef OPENDARTS_SHARED_FRAME_H
#define OPENDARTS_SHARED_FRAME_H

#include <stdint.h>

#define ODVCAM_MAGIC    0x4344564FU   /* 'ODVC' little-endian */
// 2 since the payload can be MJPEG. A version-1 reader refuses the mapping
// and shows its test pattern, rather than repeating its last frame forever
// because every new one is in a format it skips.
#define ODVCAM_VERSION  2U

#define ODVCAM_FORMAT_BGR24 0U
#define ODVCAM_FORMAT_MJPEG 1U

// Header is 64 bytes: one cache line, so the seqlock counter and the
// geometry never straddle a line boundary.
typedef struct ODVCamHeader {
    uint32_t magic;        // ODVCAM_MAGIC -- distinguishes a live mapping
                           // from one a crashed writer left zeroed
    uint32_t version;      // ODVCAM_VERSION; readers refuse anything else
    uint32_t width;
    uint32_t height;
    uint32_t stride;       // BGR24: bytes per row (width*3)
                           // MJPEG: payload length in bytes
    uint32_t format;       // ODVCAM_FORMAT_*, per frame
    uint32_t sequence;     // seqlock: odd = write in progress
    uint32_t frame_index;  // monotonic; lets a reader spot a stalled writer
    uint64_t timestamp_ns; // capture time, writer's monotonic clock

    // --- written by the READER, read by the writer ----------------------
    //
    // The reverse direction exists to answer one question the writer
    // cannot answer alone: is the consumer keeping up? A writer can only
    // see that it published; whether anything collected the frame, and
    // whether any were missed in between, is only visible from the
    // reading end.
    //
    // `frames_missed` is the number that matters. The reader compares
    // frame_index against the last one it saw; a gap greater than one
    // means the writer published frames the reader never collected. That
    // is the only direct evidence of dropped frames on this path --
    // everything else is inference from rates.
    //
    // These are plain non-atomic counters, deliberately. They are
    // diagnostics: a torn read of a statistic costs an inaccurate number
    // once, and paying for atomics on the frame path to protect a counter
    // would be the wrong trade.
    uint32_t reader_frames_read;
    uint32_t reader_frames_missed;  // frame_index gaps -- REAL dropped frames
    uint32_t reader_frames_torn;    // seqlock caught a mid-write copy
    uint32_t reader_tick_ms;        // GetTickCount at the last successful read
    uint8_t  reserved[8];
} ODVCamHeader;

// The writer must only touch the first 40 bytes -- rewriting the whole
// header would clobber the reader's counters every frame.
#define ODVCAM_WRITER_BYTES 40

#define ODVCAM_HEADER_SIZE 64

// Total mapping size for a given geometry.
#define ODVCAM_MAPPING_SIZE(w, h) (ODVCAM_HEADER_SIZE + (size_t)(w) * (h) * 3)

// Name template. %d is the camera slot.
#define ODVCAM_NAME_FMT L"Local\\ODVCam%d"

// Auto-reset event the writer sets after each complete frame, so the
// reader pushes when a frame ARRIVES. It used to sleep a fixed frame period
// after each push, which with the push's own work ran slower than the
// camera and dropped ~18% of frames (measured on a Windows rig, 2026-09-17). A reader that cannot
// open it falls back to polling.
#define ODVCAM_EVENT_FMT L"Local\\ODVCamEvt%d"

#endif  // OPENDARTS_SHARED_FRAME_H
