# JASH-005 — Browser Microphone Ingestion Design

Status: approved architecture; implementation not started

## 1. Objective and scope

JASH-005 adds ephemeral browser microphone ingestion to the existing VoxSentinel live-ML path:

~~~text
browser microphone
    ↓
getUserMedia
    ↓
AudioContext + AudioWorklet
    ↓
versioned binary PCM audio WebSocket
    ↓
call-scoped StreamingAudioCanonicalizer
    ↓
continuous mono 16 kHz float32 PCM
    ↓
existing AASISTWindowScheduler
    ↓
existing bounded AsyncCallSession
    ↓
existing AASIST / MLRiskProvider
    ↓
existing risk WebSocket
    ↓
existing VoxSentinel UI
~~~

This task is audio ingestion only. It does not add speaker verification, speaker enrollment, final risk fusion, telephony, persistence, authentication, or UI redesign.

The existing complete-container POST /api/v1/calls/{call_id}/audio WAV/FLAC feeder remains unchanged.

## 2. Existing boundaries to preserve

- MockRiskProvider remains deterministic and independent of microphone and ML runtime code.
- MLRiskProvider continues consuming the existing shared AsyncCallSession.
- The existing risk WebSocket remains:
  ~~~text
  WS /api/v1/calls/{call_id}/risk-stream
  ~~~
- The new audio transport is separate:
  ~~~text
  WS /api/v1/calls/{call_id}/audio-stream
  ~~~
- Backend canonicalization remains the authority for conversion to mono 16 kHz float32.
- Existing AASIST window size, hop, aggregation, thresholds, and policy remain unchanged.
- ECAPA speaker evidence remains NOT_EVALUATED.

## 3. Browser capture model

Microphone access begins only after an explicit user gesture. No permission request occurs on page load, component mount, scenario selection, or mock-mode initialization.

The browser uses getUserMedia, AudioContext, and a dedicated AudioWorklet. AudioContext.sampleRate means the browser/Web Audio processing rate delivered to the AudioWorklet. It is not necessarily the physical microphone hardware’s native rate; the browser or operating system may already resample device input before the worklet.

The browser reports the actual AudioContext.sampleRate in audio_start. It does not resample to 16 kHz and does not create AASIST windows.

## 4. Microphone state machine

The frontend exposes:

~~~text
IDLE
REQUESTING_PERMISSION
CONNECTING
STREAMING
STOPPING
ERROR
~~~

STREAMING is forbidden until microphone permission succeeded, the AudioContext and AudioWorklet are ready, the backend call is live, the risk WebSocket is connected, and the audio WebSocket accepted ownership and returned audio_ready.

## 5. AudioWorklet transport accumulation

AudioWorklet process callbacks commonly contain small render quanta. The microphone transport layer owns a bounded accumulator.

~~~text
TARGET_TRANSPORT_SAMPLES = 1024 samples per channel
~~~

Approximate duration is 21.33 ms at 48 kHz and 23.22 ms at 44.1 kHz.

The accumulator receives samples, preserves the first Web Audio frame index represented by buffered samples, emits approximately 1024-sample transport frames, and retains only a small residual until the next frame. It must not resample, create AASIST windows, calculate risk, persist audio, or maintain an unbounded retry queue.

On normal stop, a final non-empty residual may be sent if frame-aligned and permitted by the protocol. It must not be zero-padded merely to reach 1024 samples.

If a produced frame is dropped because of browser backpressure, its sequence still advances and the next transmitted frame exposes the gap.

## 6. Versioned audio WebSocket protocol

### 6.1 Start message

The first client message must be:

~~~json
{
  "type": "audio_start",
  "protocol_version": 1,
  "sample_rate": 48000,
  "channels": 1,
  "sample_format": "float32le"
}
~~~

sample_rate is the actual AudioContext/Web Audio processing rate. The backend validates the message, claims the call’s single audio-producer ownership, creates the call-scoped canonicalizer, and responds:

~~~json
{
  "type": "audio_ready",
  "protocol_version": 1,
  "call_id": "..."
}
~~~

No binary frame may be sent before audio_ready.

### 6.2 Binary frame v1

Each binary message contains a fixed 32-byte header followed by PCM payload.

| Field | Encoding | Size |
|---|---|---:|
| magic | ASCII VXAF | 4 bytes |
| protocol version | uint8 | 1 byte |
| header length | uint8 | 1 byte |
| flags | uint16 little-endian | 2 bytes |
| frame sequence | uint64 little-endian | 8 bytes |
| first sample frame | uint64 little-endian | 8 bytes |
| sample count per channel | uint32 little-endian | 4 bytes |
| payload byte length | uint32 little-endian | 4 bytes |

The total header size is exactly 32 bytes.

Payload is little-endian IEEE-754 Float32 PCM interleaved by channel. Payload length must equal:

~~~text
sample_count_per_channel * channels * 4
~~~

first_sample_frame is the AudioContext/Web Audio frame index of the first sample in the transport frame. The final source position is first_sample_frame + sample_count_per_channel - 1.

### 6.3 Validation

The backend rejects wrong magic, unsupported version, invalid header length, unsupported flags, non-increasing frame sequence, regressing first_sample_frame, malformed payload length, unsupported sample format/rate/channels, NaN or infinite PCM, data before valid audio_start and ownership, data after audio_stop, and data for cancelled or closed sessions.

Frame sequence gaps greater than one are accepted and recorded as transport telemetry. They are not authenticity evidence and must not generate risk reasons.

Normal stop messages are:

~~~json
{ "type": "audio_stop" }
~~~

The server acknowledges with:

~~~json
{ "type": "audio_stopped" }
~~~

## 7. Stateful canonicalization

Exactly one StreamingAudioCanonicalizer exists per active audio producer.

It validates source metadata, downmixes supported interleaved channels to mono, resamples the continuous mono stream to 16 kHz, preserves source timing and sequence ranges, emits continuous finite Float32 samples to the existing scheduler, and disposes deterministically.

### 7.1 Resampler

The preferred implementation is soxr.ResampleStream.

The backend dependency must be pinned to an exact version only after a real compatibility installation and test in the documented backend environment. The compatibility gate verifies import, stateful chunked conversion, output shape, and teardown. The exact version then belongs in backend/requirements.txt.

If soxr cannot be installed and reproduced in the documented backend environment, implementation stops and reports the blocker. It must not silently substitute per-frame scipy resampling.

The resampler retains filter and fractional-phase state across every binary frame, never independently resamples WebSocket frames, downmixes before resampling, emits continuous mono 16 kHz float32, flushes only samples derived from received audio, never pads or fabricates audio, and releases all state on stop, reset, disconnect, or failed startup.

The existing complete WAV/FLAC conversion implementation remains unchanged.

### 7.2 Window metadata

The existing AASIST scheduler remains responsible for 64,600-sample windows and 8,000-sample hops. It receives continuous canonical output from the persistent canonicalizer.

Each model window carries metadata equivalent to:

~~~text
audio_source_frame_start
audio_source_frame_end
audio_source_transport_sequence_start
audio_source_transport_sequence_end
audio_source_gap_count
audio_window_sequence
~~~

This metadata is attached to the existing AudioWindow/session path and may be exposed optionally on LiveRiskEvent without changing risk semantics.

The audio WebSocket uses the exact AsyncCallSession already owned by JASH-004. No second queue, ML session, scheduler, or call lifecycle is permitted.

## 8. Audio timing and correlation

The AudioWorklet’s authoritative source clock is the Web Audio processing timeline: currentFrame, currentTime, and sampleRate. The worklet derives first_sample_frame from currentFrame and accumulator offsets. performance.now() is not used inside the AudioWorklet as the authoritative per-frame capture timestamp.

At startup:

~~~text
context_origin_perf_ms =
    performance.now() - audioContext.currentTime * 1000
~~~

For a model window:

~~~text
capture_end_perf_ms =
    context_origin_perf_ms
    + audio_source_frame_end / audioContext.sampleRate * 1000
~~~

When its risk event arrives:

~~~text
steady_state_latency_ms =
    risk_event_arrival_performance_now - capture_end_perf_ms
~~~

This is browser-observed Web Audio processing and transport latency, not a physical microphone ADC timestamp.

Warm-up latency is measured separately:

~~~text
audio_ready / STREAMING → first real risk event
~~~

The two measurements must not be mixed.

## 9. Session ownership and lifecycle

### 9.1 Ownership

One active microphone producer is allowed per call. A second producer is explicitly rejected and only the rejected WebSocket is closed. Streams are never merged.

### 9.2 Transactional startup

Startup is provisional until audio_ready.

~~~text
explicit user gesture
→ generation token
→ getUserMedia
→ AudioContext
→ AudioWorklet
→ backend call create
→ backend call start
→ risk WebSocket
→ audio WebSocket
→ audio_start
→ ownership + canonicalizer creation
→ audio_ready
→ enable transport delivery
→ STREAMING
~~~

For every partial-start failure:

1. invalidate the startup generation token;
2. stop AudioWorklet delivery;
3. send audio_stop if the audio socket is ready;
4. close the audio WebSocket;
5. close the risk WebSocket;
6. release audio ownership;
7. dispose the streaming canonicalizer;
8. cancel and close the ML runtime session if it exists;
9. stop the backend call or transition an already-created failed start to FAILED;
10. disconnect AudioWorklet nodes;
11. close the AudioContext;
12. stop every acquired MediaStream track;
13. clear the accumulator and local telemetry;
14. enter ERROR unless the user explicitly reset to IDLE.

If getUserMedia resolves after cancellation, returned tracks are stopped immediately and stale callbacks cannot modify the current session.

### 9.3 Normal stop

~~~text
stop producing frames
→ send final permitted residual
→ audio_stop
→ backend finalizes received resampler input
→ audio_stopped
→ release ownership
→ dispose canonicalizer
→ close audio WebSocket
→ close risk WebSocket
→ stop tracks
→ disconnect nodes
→ close AudioContext
→ existing backend stop lifecycle
~~~

No zero-padding is performed solely to create another model window.

### 9.4 Reset

Reset is idempotent. It invalidates the generation, cancels startup or streaming work, closes both WebSockets, disposes the canonicalizer and runtime session, stops tracks, disconnects nodes, closes the AudioContext, clears local state, and returns to IDLE.

### 9.5 Unexpected disconnect

An unexpected audio WebSocket disconnect releases ownership, disposes the canonicalizer, cancels/closes the JASH-004 runtime session, prevents MLRiskProvider from waiting indefinitely, stops or fails the live call, tears down frontend risk and microphone resources, reports ERROR or DISCONNECTED, and does not automatically reconnect.

An unexpected risk WebSocket disconnect while microphone streaming is active triggers the same coordinated teardown. Normal audio_stop is handled separately.

## 10. Backpressure

### 10.1 Browser

Use named high and low WebSocket.bufferedAmount thresholds. Above HIGH, newly produced transport frames are dropped rather than added to an application retry queue. Sending resumes below LOW.

Track:

~~~text
browser_frames_produced
browser_frames_sent
browser_frames_dropped
sequence_gap_count
~~~

### 10.2 Backend

Preserve the JASH-004 bounded queue:

~~~text
queue full
→ drop oldest pending model window
→ retain newest complete model window
~~~

The queue never exceeds capacity and existing dropped-window telemetry remains authoritative.

## 11. Privacy

Raw microphone audio is ephemeral. JASH-005 adds no LocalStorage, IndexedDB, downloads, backend files, database persistence, raw PCM logging, or repository recordings.

The UI may show:

~~~text
LIVE AUDIO ANALYSIS
Raw audio is not retained
~~~

## 12. Policy and ML scope

JASH-004 policy is unchanged.

Real microphone AASIST mode may emit only:

~~~text
NORMAL
20 / LOW / MONITOR
~~~

or:

~~~text
ELEVATED_AUTHENTICITY_REVIEW
70 / HIGH / REQUIRE_CALLBACK
~~~

It must never emit CRITICAL, REQUIRE_SUPERVISOR, or BLOCK_ACTION. AASIST thresholds, persistence, aggregation, and score semantics do not change. ECAPA remains NOT_EVALUATED.

## 13. Mock mode

With VOXSENTINEL_RISK_PROVIDER=mock, the deterministic path remains:

~~~text
18 → 27 → 43 → 61 → 79 → 92
CRITICAL
BLOCK_ACTION
~~~

Mock mode never requests microphone permission, creates AudioContext, loads AudioWorklet, opens the audio WebSocket, or requires ML/AASIST.

## 14. Baseline gate

Before production implementation begins, recreate or use the documented backend environment from backend/requirements.txt and run the complete backend suite.

The earlier missing-soundfile shell failure is not valid baseline evidence. If the recreated backend baseline is not green, production JASH-005 changes must not begin.

## 15. Testing requirements

### Frontend

Cover explicit permission gesture, microphone states, mock no-microphone behavior, AudioContext sample-rate metadata, AudioWorklet accumulation, 1024-sample framing, residual final frame, exact binary header bytes, sequence generation, first_sample_frame tracking, bounded bufferedAmount behavior, dropped sequence visibility, every partial-start rollback, stop/reset/disconnect cleanup, stale generation suppression, and no audio persistence.

### Backend

Cover LIVE-call validation, audio_start validation, exact 32-byte parser, sequence and first_sample_frame monotonicity, accepted gaps, finite PCM, payload alignment, duplicate producer rejection, stateful soxr continuity across arbitrary network-frame boundaries, comparison against a continuous reference within an explicitly documented numerical tolerance, continuous mono 16 kHz behavior, source metadata propagation, reuse of the existing scheduler/session, bounded queue behavior, normal-stop flush, disconnect/reset cleanup, and no malformed-input risk events.

### Integration

Eventually exercise:

~~~text
AudioWorklet
→ binary audio WebSocket
→ StreamingAudioCanonicalizer
→ AASIST windows
→ real AASIST
→ MLRiskProvider
→ risk WebSocket
→ frontend
~~~

Physical/browser acceptance records browser/version, AudioContext.sampleRate, channels, duration, produced/sent/dropped frames, sequence gaps, canonical samples, generated and dropped AASIST windows, raw scores, aggregate/policy states, warm-up latency, steady-state capture-end-to-risk-arrival latency, and ML latency.

## 16. Acceptance criteria

Acceptance passes only when:

1. a real browser obtains permission after explicit user action;
2. actual AudioContext rate is recorded;
3. audio_ready precedes STREAMING;
4. binary frames have valid headers and metadata;
5. frame drops are observable through sequence gaps;
6. one stateful canonicalizer serves the session;
7. continuous output reaches the existing scheduler;
8. real AASIST inference occurs;
9. events return through the existing risk WebSocket;
10. the UI updates without fabricated events;
11. normal stop closes all resources;
12. unexpected disconnect terminates the live session without an orphan task;
13. mock mode remains independent and deterministic.

Human speech is not expected to always score LOW. Thresholds must not be tuned from one microphone session.

## 17. Out of scope

No ECAPA integration, speaker enrollment or identification, speaker fusion, Twilio/PSTN/SIP, OTP or callback implementation, database, authentication, microphone recording storage, AASIST retraining, threshold tuning, calibration, automatic reconnect, UI redesign, or JASH-006.

## 18. Self-review

- Canonicalization is stateful and call-scoped; no per-frame resampling is permitted.
- AudioContext rate is distinguished from hardware rate.
- Audio timing uses Web Audio frame indices, not AudioWorklet performance.now().
- Transport accumulation targets 1024 samples and remains bounded.
- Browser and backend backpressure are explicit.
- Startup is provisional until audio_ready and rolls back every partial resource allocation.
- Normal stop and unexpected disconnect are distinct.
- The existing WAV/FLAC feeder remains unchanged.
- Mock mode, JASH-004 policy, and ECAPA NOT_EVALUATED status are preserved.
- No implementation branch or production code is included.

Open decisions: NONE.
