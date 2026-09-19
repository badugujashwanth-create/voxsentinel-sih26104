export const AUDIO_PROTOCOL_VERSION = 1;
export const AUDIO_HEADER_LENGTH = 32;
export const AUDIO_FLAGS = 0;
export const TARGET_TRANSPORT_SAMPLES = 1024;

export interface AudioStartMetadata {
  protocol_version: number;
  sample_rate: number;
  channels: 1 | 2;
  sample_format: "float32le";
}

export interface TransportFrame {
  sequence: bigint;
  firstSampleFrame: bigint;
  channels: 1 | 2;
  samples: Float32Array;
}

/** Encodes the fixed VXAF v1 header and interleaved PCM payload. */
export function encodeAudioFrame(frame: TransportFrame): ArrayBuffer {
  const sampleCount = frame.samples.length / frame.channels;
  if (!Number.isInteger(sampleCount) || sampleCount < 1 || sampleCount > TARGET_TRANSPORT_SAMPLES) throw new Error("Invalid transport sample count");
  const payloadBytes = frame.samples.byteLength;
  const output = new ArrayBuffer(AUDIO_HEADER_LENGTH + payloadBytes);
  const bytes = new Uint8Array(output);
  bytes.set(new TextEncoder().encode("VXAF"), 0);
  const view = new DataView(output);
  view.setUint8(4, AUDIO_PROTOCOL_VERSION);
  view.setUint8(5, AUDIO_HEADER_LENGTH);
  view.setUint16(6, AUDIO_FLAGS, true);
  view.setBigUint64(8, frame.sequence, true);
  view.setBigUint64(16, frame.firstSampleFrame, true);
  view.setUint32(24, sampleCount, true);
  view.setUint32(28, payloadBytes, true);
  new Uint8Array(output, AUDIO_HEADER_LENGTH).set(new Uint8Array(frame.samples.buffer, frame.samples.byteOffset, payloadBytes));
  return output;
}

/** Encodes the JSON control message that negotiates browser PCM transport. */
export function encodeAudioStart(metadata: AudioStartMetadata): string {
  return JSON.stringify({ type: "audio_start", ...metadata });
}
