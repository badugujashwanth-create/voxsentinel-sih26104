import { validateLiveRiskEvent, type LiveRiskEvent } from "../../domain/risk";
import type { RiskStreamHandlers, RiskStreamSource } from "./RiskStreamSource";

const RISK_STREAM_PATH = "/api/v1/calls";
export const WS_UNKNOWN_CALL = 4404;
export const WS_CALL_NOT_LIVE = 4409;
export const WS_ML_FAILURE = 1011;

/** Connects the console to the future backend risk stream contract. */
export class WebSocketRiskStreamSource implements RiskStreamSource {
  private readonly callId: string;
  private readonly baseUrl: string;
  private readonly wsBaseUrl: string;
  private socket: WebSocket | undefined;
  private handlers: RiskStreamHandlers | undefined;
  private lastSequence = 0;
  private paused = false;

  /** Creates an adapter for one call and backend origin. */
  public constructor(callId: string, baseUrl?: string, wsBaseUrl?: string) {
    this.callId = callId;
    this.baseUrl = baseUrl?.trim() || getConfiguredBaseUrl();
    this.wsBaseUrl = wsBaseUrl?.trim() || (baseUrl ? this.baseUrl : getConfiguredWsBaseUrl());
  }

  /** Opens the backend stream and attaches validated message handlers. */
  public async start(handlers: RiskStreamHandlers): Promise<void> {
    this.lastSequence = 0;
    this.paused = false;
    this.handlers = handlers;
    this.emitStatus("CONNECTING");
    this.socket = new WebSocket(buildStreamUrl(this.wsBaseUrl, this.callId));
    this.socket.onopen = () => this.emitStatus("LIVE");
    this.socket.onmessage = (message) => this.handleMessage(message.data);
    this.socket.onerror = () => this.reportError(new Error("Risk stream connection failed"));
    this.socket.onclose = (event) => {
      if (event?.code === WS_UNKNOWN_CALL) this.reportError(new Error("Backend call was not found (4404)"));
      if (event?.code === WS_CALL_NOT_LIVE) this.reportError(new Error("Backend call is not live (4409)"));
      if (event?.code === WS_ML_FAILURE) this.reportError(new Error(event.reason || "ML risk stream failed"));
      this.emitStatus("DISCONNECTED");
    };
  }

  /** Closes the backend stream and reports disconnection. */
  public async stop(): Promise<void> {
    this.socket?.close();
    this.socket = undefined;
    this.lastSequence = 0;
    this.paused = false;
    this.emitStatus("DISCONNECTED");
  }

  /** Marks the live adapter paused without closing the transport. */
  public pause(): void {
    this.paused = true;
    this.emitStatus("PAUSED");
  }

  /** Marks the live adapter active again. */
  public resume(): void {
    this.paused = false;
    this.emitStatus("LIVE");
  }

  /** Clears event ordering and returns the adapter to idle state. */
  public reset(): void {
    this.lastSequence = 0;
    this.paused = false;
    this.emitStatus("IDLE");
  }

  /** Closes the socket and releases all consumer references. */
  public dispose(): void {
    this.socket?.close();
    this.socket = undefined;
    this.handlers = undefined;
    this.lastSequence = 0;
  }

  /** Parses, validates, and orders one untrusted socket message. */
  private handleMessage(data: unknown): void {
    if (this.paused) return;
    try {
      const event = validateLiveRiskEvent(parseMessage(data));
      if (event.sequence <= this.lastSequence) throw new Error("Risk event sequence is out of order");
      this.lastSequence = event.sequence;
      this.handlers?.onEvent(event);
    } catch (error) {
      this.reportError(error instanceof Error ? error : new Error("Malformed risk event"));
    }
  }

  /** Reports an adapter error through both the status and error channels. */
  private reportError(error: Error): void {
    this.emitStatus("ERROR");
    this.handlers?.onError(error);
  }

  /** Publishes a status while the consumer is attached. */
  private emitStatus(status: Parameters<RiskStreamHandlers["onStatusChange"]>[0]): void {
    this.handlers?.onStatusChange(status);
  }
}

/** Parses text or object socket data into an unknown payload for validation. */
function parseMessage(data: unknown): unknown {
  if (typeof data === "string") return JSON.parse(data) as unknown;
  return data;
}

/** Builds the backend WebSocket endpoint without hardcoded hostnames. */
function buildStreamUrl(baseUrl: string, callId: string): string {
  const parsed = new URL(baseUrl || window.location.origin);
  parsed.protocol = parsed.protocol === "https:" || parsed.protocol === "wss:" ? "wss:" : "ws:";
  parsed.pathname = `${trimTrailingSlash(parsed.pathname)}${RISK_STREAM_PATH}/${encodeURIComponent(callId)}/risk-stream`;
  parsed.search = "";
  return parsed.toString();
}

/** Reads the optional Vite backend origin from environment configuration. */
function getConfiguredBaseUrl(): string {
  return import.meta.env.VITE_API_BASE_URL?.trim() || window.location.origin;
}

/** Reads the optional backend WebSocket origin from environment configuration. */
function getConfiguredWsBaseUrl(): string {
  return import.meta.env.VITE_API_WS_URL?.trim() || getConfiguredBaseUrl();
}

/** Removes trailing slashes before appending the stream endpoint. */
function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export type { LiveRiskEvent };
