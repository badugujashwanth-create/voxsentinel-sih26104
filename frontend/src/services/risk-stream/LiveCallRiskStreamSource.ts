import type { ScenarioId } from "../../scenarios/scenarios";
import { CallSessionClient, type CallSessionClientOptions, type CallSessionOperations, type CreateCallRequest, type CreateCallResponse } from "../call-session/CallSessionClient";
import type { RiskStreamHandlers, RiskStreamSource, RiskStreamStatus } from "./RiskStreamSource";
import { WebSocketRiskStreamSource } from "./WebSocketRiskStreamSource";
import type { MicrophoneSession } from "../../audio/microphone-session";

export interface LiveCallRequest {
  claimed_identity: string;
  scenario: ScenarioId;
  transaction_value?: number;
  currency?: string;
}

export interface LiveCallRiskStreamSourceOptions extends CallSessionClientOptions {
  request: LiveCallRequest;
  microphone?: Pick<MicrophoneSession, "start" | "stop" | "reset" | "dispose">;
}

/** Orchestrates backend call creation, start, streaming, and stop lifecycle. */
export class LiveCallRiskStreamSource implements RiskStreamSource {
  private readonly request: CreateCallRequest;
  private readonly client: CallSessionOperations;
  private readonly baseUrl: string | undefined;
  private stream: WebSocketRiskStreamSource | undefined;
  private session: CreateCallResponse | undefined;
  private started = false;
  private handlers: RiskStreamHandlers | undefined;
  private readonly microphone?: Pick<MicrophoneSession, "start" | "stop" | "reset" | "dispose">;

  /** Creates a live source that owns one backend call session. */
  public constructor(options: LiveCallRiskStreamSourceOptions, client: CallSessionOperations = new CallSessionClient(options)) {
    this.request = options.request;
    this.client = client;
    this.baseUrl = options.baseUrl;
    this.microphone = options.microphone;
  }

  /** Creates and starts the backend call before opening its risk stream. */
  public async start(handlers: RiskStreamHandlers): Promise<void> {
    this.handlers = handlers;
    this.emitStatus("CONNECTING");
    try {
      this.session = await this.client.createCall(this.request);
      await this.client.startCall(this.session.call_id);
      this.started = true;
      this.stream = new WebSocketRiskStreamSource(this.session.call_id, this.baseUrl);
      await this.stream.start(this.createStreamHandlers(handlers));
      await this.microphone?.start(this.session.call_id);
    } catch (error) {
      this.reportError(error instanceof Error ? error : new Error("Live call setup failed"));
    }
  }

  /** Closes the stream and completes a started backend call. */
  public async stop(): Promise<void> {
    const stream = this.stream;
    this.stream = undefined;
    await stream?.stop();
    await this.microphone?.stop();
    await this.stopBackendSession();
    this.emitStatus("DISCONNECTED");
  }

  /** Pauses event delivery while leaving the live transport open. */
  public pause(): void {
    this.stream?.pause();
  }

  /** Resumes event delivery on the existing live transport. */
  public resume(): void {
    this.stream?.resume();
  }

  /** Disposes the stream and asynchronously completes the backend session. */
  public reset(): void {
    this.stream?.dispose();
    this.microphone?.reset();
    this.stream = undefined;
    this.started = false;
    void this.stopBackendSession();
    this.emitStatus("IDLE");
  }

  /** Releases handlers and closes any live resources without retaining state. */
  public dispose(): void {
    this.stream?.dispose();
    this.microphone?.dispose();
    this.stream = undefined;
    this.started = false;
    this.handlers = undefined;
    void this.stopBackendSession();
  }

  /** Wraps stream status so a normal socket close completes the backend call. */
  private createStreamHandlers(handlers: RiskStreamHandlers): RiskStreamHandlers {
    return {
      onEvent: handlers.onEvent,
      onError: handlers.onError,
      onStatusChange: (status) => {
        handlers.onStatusChange(status);
        if (status === "DISCONNECTED") void this.stopBackendSession();
      },
    };
  }

  /** Stops a started backend session once and releases its identity. */
  private async stopBackendSession(): Promise<void> {
    if (!this.session || !this.started) {
      this.session = undefined;
      return;
    }
    const callId = this.session.call_id;
    this.session = undefined;
    this.started = false;
    try {
      await this.client.stopCall(callId);
    } catch (error) {
      this.reportError(error instanceof Error ? error : new Error("Live call stop failed"));
    }
  }

  /** Publishes a lifecycle status to the active consumer. */
  private emitStatus(status: RiskStreamStatus): void {
    this.handlers?.onStatusChange(status);
  }

  /** Reports a setup or cleanup failure through the stream error contract. */
  private reportError(error: Error): void {
    this.handlers?.onStatusChange("ERROR");
    this.handlers?.onError(error);
  }
}
