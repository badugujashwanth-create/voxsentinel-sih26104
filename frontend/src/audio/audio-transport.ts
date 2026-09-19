import { encodeAudioFrame, encodeAudioStart, type AudioStartMetadata, type TransportFrame } from "./protocol";

export const AUDIO_WS_HIGH_WATERMARK_BYTES = 262_144;
export const AUDIO_WS_LOW_WATERMARK_BYTES = 65_536;

export interface AudioSocketLike {
  readonly bufferedAmount: number;
  readonly readyState: number;
  send(data: string | ArrayBuffer): void;
  close?(): void;
}

export interface AudioTransportTelemetry {
  browser_frames_produced: number;
  browser_frames_sent: number;
  browser_frames_dropped: number;
  sequence_gap_count: number;
}

/** Adds sequence numbers and bounded WebSocket backpressure to transport frames. */
export class AudioTransport {
  private readonly socket: AudioSocketLike;
  private nextSequence = 1n;
  private congested = false;
  public readonly telemetry: AudioTransportTelemetry = { browser_frames_produced: 0, browser_frames_sent: 0, browser_frames_dropped: 0, sequence_gap_count: 0 };

  public constructor(socket: AudioSocketLike) {
    this.socket = socket;
  }

  /** Sends audio_start before frame delivery. */
  public sendStart(metadata: AudioStartMetadata): void {
    this.socket.send(encodeAudioStart(metadata));
  }

  /** Sends or deterministically drops one produced frame. */
  public sendFrame(frame: Omit<TransportFrame, "sequence">): boolean {
    const sequence = this.nextSequence;
    this.nextSequence += 1n;
    this.telemetry.browser_frames_produced += 1;
    if (this.socket.readyState !== 1 || this.shouldDrop()) {
      this.telemetry.browser_frames_dropped += 1;
      this.telemetry.sequence_gap_count += 1;
      return false;
    }
    this.socket.send(encodeAudioFrame({ ...frame, sequence }));
    this.telemetry.browser_frames_sent += 1;
    return true;
  }

  /** Sends the normal-stop control message and closes the socket. */
  public stop(): void {
    if (this.socket.readyState === 1) this.socket.send(JSON.stringify({ type: "audio_stop" }));
    this.socket.close?.();
  }

  private shouldDrop(): boolean {
    if (this.congested && this.socket.bufferedAmount < AUDIO_WS_LOW_WATERMARK_BYTES) this.congested = false;
    if (!this.congested && this.socket.bufferedAmount > AUDIO_WS_HIGH_WATERMARK_BYTES) this.congested = true;
    return this.congested;
  }
}
