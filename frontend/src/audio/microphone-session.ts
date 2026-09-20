import { AudioTransport, type AudioSocketLike } from "./audio-transport";
import { AudioWorkletBridge, type AudioFrameHandler } from "./audio-worklet-bridge";
import { encodeAudioStart } from "./protocol";
import type { AudioTransportTelemetry } from "./audio-transport";
import type { AudioTimingAnchor } from "./timing";
import { createAudioTimingAnchor } from "./timing";

export type MicrophoneState = "IDLE" | "REQUESTING_PERMISSION" | "CONNECTING" | "STREAMING" | "STOPPING" | "ERROR";

interface MicrophoneSocket extends AudioSocketLike {
  onopen: (() => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: (() => void) | null;
  onclose: (() => void) | null;
}

export interface MicrophoneSessionDependencies {
  getUserMedia?: (constraints: MediaStreamConstraints) => Promise<MediaStream>;
  createAudioContext?: () => AudioContext;
  createSocket?: (url: string) => MicrophoneSocket;
  createBridge?: (context: AudioContext, channels: 1 | 2, onFrame: AudioFrameHandler, source: AudioNode) => AudioWorkletBridge;
  createMediaStreamSource?: (stream: MediaStream) => MediaStreamAudioSourceNode;
  baseUrl?: string;
}

/** Coordinates permission, audio_ready gating, transport, and microphone cleanup. */
export class MicrophoneSession {
  private readonly dependencies: Required<Pick<MicrophoneSessionDependencies, "getUserMedia" | "createAudioContext" | "createSocket" | "createBridge">> & Pick<MicrophoneSessionDependencies, "baseUrl" | "createMediaStreamSource">;
  private readonly onStateChange?: (state: MicrophoneState) => void;
  private generation = 0;
  private stream: MediaStream | undefined;
  private context: AudioContext | undefined;
  private socket: MicrophoneSocket | undefined;
  private bridge: AudioWorkletBridge | undefined;
  private sourceNode: MediaStreamAudioSourceNode | undefined;
  private transport: AudioTransport | undefined;
  private callId: string | undefined;
  private audioReadyAt: number | undefined;
  private timingAnchor: AudioTimingAnchor | undefined;
  private lastTransportTelemetry: AudioTransportTelemetry | undefined;
  private lastSampleRate: number | undefined;
  public state: MicrophoneState = "IDLE";

  public constructor(dependencies: MicrophoneSessionDependencies = {}, onStateChange?: (state: MicrophoneState) => void) {
    this.dependencies = {
      getUserMedia: dependencies.getUserMedia ?? ((constraints) => navigator.mediaDevices.getUserMedia(constraints)),
      createAudioContext: dependencies.createAudioContext ?? (() => new AudioContext()),
      createSocket: dependencies.createSocket ?? ((url) => new WebSocket(url) as unknown as MicrophoneSocket),
      createBridge: dependencies.createBridge ?? ((context, channels, onFrame, source) => new AudioWorkletBridge(context, channels, onFrame, source)),
      createMediaStreamSource: dependencies.createMediaStreamSource,
      baseUrl: dependencies.baseUrl,
    };
    this.onStateChange = onStateChange;
  }

  /** Requests permission only when called by an explicit operator action. */
  public async start(callId: string): Promise<void> {
    this.callId = callId;
    this.lastTransportTelemetry = undefined;
    this.lastSampleRate = undefined;
    this.audioReadyAt = undefined;
    this.timingAnchor = undefined;
    const token = ++this.generation;
    this.setState("REQUESTING_PERMISSION");
    try {
      this.stream = await this.dependencies.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      if (token !== this.generation) return this.cleanupResources();
      this.context = this.dependencies.createAudioContext();
      if (this.context.state === "suspended") await this.context.resume();
      this.sourceNode = this.dependencies.createMediaStreamSource ? this.dependencies.createMediaStreamSource(this.stream) : this.context.createMediaStreamSource(this.stream);
      this.setState("CONNECTING");
      this.socket = this.dependencies.createSocket(buildAudioWebSocketUrl(this.dependencies.baseUrl, callId));
      await waitForSocketOpen(this.socket);
      this.socket.send(encodeAudioStart({ protocol_version: 1, sample_rate: this.context.sampleRate, channels: 1, sample_format: "float32le" }));
      await waitForAudioReady(this.socket);
      this.lastSampleRate = this.context.sampleRate;
      this.audioReadyAt = Number.isFinite(performance.now()) ? performance.now() : Date.now();
      const contextCurrentTime = Number.isFinite(this.context.currentTime) ? this.context.currentTime : 0;
      this.timingAnchor = createAudioTimingAnchor(contextCurrentTime, this.audioReadyAt, this.context.sampleRate);
      if (token !== this.generation) return this.cleanupResources();
      this.transport = new AudioTransport(this.socket);
      this.bridge = this.dependencies.createBridge(this.context, 1, (frame) => this.transport?.sendFrame(frame), this.sourceNode);
      await this.bridge.start();
      this.socket.onclose = () => { void this.handleUnexpectedDisconnect(); };
      this.setState("STREAMING");
    } catch (error) {
      await this.cleanupResources();
      this.setState("ERROR");
      throw error instanceof Error ? error : new Error("Microphone startup failed");
    }
  }

  /** Stops frames, closes transport, and releases every browser resource. */
  public async stop(): Promise<void> {
    if (this.state === "IDLE" || this.state === "ERROR") return this.cleanupResources();
    this.setState("STOPPING");
    ++this.generation;
    await this.bridge?.flush();
    this.transport?.stop();
    await this.cleanupResources();
    this.setState("IDLE");
  }

  /** Invalidates stale callbacks and performs idempotent teardown. */
  public reset(): void {
    ++this.generation;
    void this.cleanupResources();
    this.setState("IDLE");
  }

  /** Releases resources without initiating another lifecycle transition. */
  public dispose(): void {
    ++this.generation;
    void this.cleanupResources();
  }

  /** Returns ephemeral microphone metadata without exposing audio samples. */
  public getAcceptanceSnapshot(): { call_id?: string; microphone_state: MicrophoneState; audio_context_sample_rate?: number; channels?: number; audio_ready_at?: number; timing_anchor?: AudioTimingAnchor; transport?: AudioTransportTelemetry; cleanup: { tracks_active: boolean; audio_context_active: boolean; audio_socket_active: boolean } } {
    return { call_id: this.callId, microphone_state: this.state, audio_context_sample_rate: this.context?.sampleRate ?? this.lastSampleRate, channels: this.context ? 1 : (this.lastTransportTelemetry ? 1 : undefined), audio_ready_at: this.audioReadyAt, timing_anchor: this.timingAnchor, transport: this.transport?.getTelemetry() ?? this.lastTransportTelemetry, cleanup: { tracks_active: Boolean(this.stream), audio_context_active: Boolean(this.context), audio_socket_active: Boolean(this.socket) } };
  }

  private async handleUnexpectedDisconnect(): Promise<void> {
    if (this.state !== "STREAMING") return;
    ++this.generation;
    this.setState("ERROR");
    await this.cleanupResources();
  }

  private async cleanupResources(): Promise<void> {
    if (this.transport) this.lastTransportTelemetry = this.transport.getTelemetry();
    this.bridge?.dispose();
    this.bridge = undefined;
    if (this.socket && this.socket.readyState === 1) this.socket.close?.();
    this.socket = undefined;
    this.transport = undefined;
    this.sourceNode?.disconnect();
    this.sourceNode = undefined;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = undefined;
    if (this.context) await this.context.close();
    this.context = undefined;
  }

  private setState(state: MicrophoneState): void {
    this.state = state;
    this.onStateChange?.(state);
  }

}

function buildAudioWebSocketUrl(baseUrl: string | undefined, callId: string): string {
  const origin = baseUrl ?? window.location.origin;
  const url = new URL(origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/api/v1/calls/${encodeURIComponent(callId)}/audio-stream`;
  return url.toString();
}

function waitForSocketOpen(socket: MicrophoneSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    if (socket.readyState === 1) {
      resolve();
      return;
    }
    socket.onopen = () => resolve();
    socket.onerror = () => reject(new Error("Audio WebSocket connection failed"));
    socket.onclose = () => reject(new Error("Audio WebSocket closed before connection"));
  });
}

function waitForAudioReady(socket: MicrophoneSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(String(event.data)) as { type?: string };
        if (message.type === "audio_ready") resolve();
        else reject(new Error("Backend did not acknowledge audio_start"));
      } catch (error) {
        reject(error instanceof Error ? error : new Error("Malformed audio_ready response"));
      }
    };
    socket.onerror = () => reject(new Error("Audio WebSocket audio_ready failed"));
    socket.onclose = () => reject(new Error("Audio WebSocket closed before audio_ready"));
  });
}
