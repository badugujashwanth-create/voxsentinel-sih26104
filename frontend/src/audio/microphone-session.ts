import { AudioTransport, type AudioSocketLike } from "./audio-transport";
import { AudioWorkletBridge, type AudioFrameHandler } from "./audio-worklet-bridge";
import { encodeAudioStart } from "./protocol";

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
    this.bridge?.flush();
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

  private async handleUnexpectedDisconnect(): Promise<void> {
    if (this.state !== "STREAMING") return;
    ++this.generation;
    this.setState("ERROR");
    await this.cleanupResources();
  }

  private async cleanupResources(): Promise<void> {
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
  });
}

function waitForAudioReady(socket: MicrophoneSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.onmessage = (event) => {
      const message = JSON.parse(String(event.data)) as { type?: string };
      if (message.type === "audio_ready") resolve();
      else reject(new Error("Backend did not acknowledge audio_start"));
    };
    socket.onerror = () => reject(new Error("Audio WebSocket audio_ready failed"));
  });
}
