import type { ScenarioId } from "../../scenarios/scenarios";
import { CallSessionClient, type CallSessionClientOptions, type CallSessionOperations, type CreateCallRequest, type CreateCallResponse } from "../call-session/CallSessionClient";
import type { RiskStreamHandlers, RiskStreamSource, RiskStreamStatus } from "./RiskStreamSource";
import { WebSocketRiskStreamSource } from "./WebSocketRiskStreamSource";
import type { MicrophoneSession } from "../../audio/microphone-session";
import { createAcceptanceSnapshot, publishAcceptanceSnapshot, type AcceptanceInference } from "../../audio/acceptance-telemetry";
import type { LiveRiskEvent } from "../../domain/risk";
import { captureEndPerformanceMs, steadyStateLatencyMs } from "../../audio/timing";

export interface LiveCallRequest {
  claimed_identity: string;
  scenario: ScenarioId;
  transaction_value?: number;
  currency?: string;
  speaker_profile_id?: string;
}

type MicrophoneLike = Pick<MicrophoneSession, "start" | "stop" | "reset" | "dispose"> & { getAcceptanceSnapshot?: MicrophoneSession["getAcceptanceSnapshot"] };

export interface LiveCallRiskStreamSourceOptions extends CallSessionClientOptions {
  request: LiveCallRequest;
  microphone?: MicrophoneLike;
  wsBaseUrl?: string;
}

/** Orchestrates backend call creation, start, streaming, and stop lifecycle. */
export class LiveCallRiskStreamSource implements RiskStreamSource {
  private readonly request: CreateCallRequest;
  private readonly client: CallSessionOperations;
  private readonly baseUrl: string | undefined;
  private readonly wsBaseUrl: string | undefined;
  private stream: WebSocketRiskStreamSource | undefined;
  private session: CreateCallResponse | undefined;
  private started = false;
  private stopping = false;
  private handlers: RiskStreamHandlers | undefined;
  private readonly microphone?: MicrophoneLike;
  private lifecycleGeneration = 0;
  private readonly acceptanceInferences: AcceptanceInference[] = [];
  private static readonly maxAcceptanceInferences = 50;

  /** Creates a live source that owns one backend call session. */
  public constructor(options: LiveCallRiskStreamSourceOptions, client: CallSessionOperations = new CallSessionClient(options)) {
    this.request = options.request;
    this.client = client;
    this.baseUrl = options.baseUrl;
    this.wsBaseUrl = options.wsBaseUrl;
    this.microphone = options.microphone;
  }

  /** Creates and starts the backend call before opening its risk stream. */
  public async start(handlers: RiskStreamHandlers): Promise<void> {
    const generation = ++this.lifecycleGeneration;
    this.handlers = handlers;
    this.acceptanceInferences.length = 0;
    this.emitStatus("CONNECTING");
    try {
      this.session = await this.client.createCall(this.request);
      await this.client.startCall(this.session.call_id);
      this.started = true;
      if (generation !== this.lifecycleGeneration) {
        await this.stopBackendSession();
        return;
      }
      this.stream = new WebSocketRiskStreamSource(this.session.call_id, this.baseUrl, this.wsBaseUrl);
      await this.stream.start(this.createStreamHandlers(handlers));
      await this.microphone?.start(this.session.call_id);
    } catch (error) {
      await this.rollbackStart();
      this.reportError(error instanceof Error ? error : new Error("Live call setup failed"));
    }
  }

  /** Closes the stream and completes a started backend call. */
  public async stop(): Promise<void> {
    this.lifecycleGeneration += 1;
    this.stopping = true;
    const stream = this.stream;
    this.stream = undefined;
    await stream?.stop();
    await this.microphone?.stop();
    this.publishAcceptanceState();
    await this.stopBackendSession();
    this.stopping = false;
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
    this.lifecycleGeneration += 1;
    this.stopping = true;
    void this.finishReset();
  }

  private async finishReset(): Promise<void> {
    this.stream?.dispose();
    await this.microphone?.reset();
    this.stream = undefined;
    this.publishAcceptanceState();
    await this.stopBackendSession();
    this.stopping = false;
    this.emitStatus("IDLE");
  }

  /** Releases handlers and closes any live resources without retaining state. */
  public dispose(): void {
    this.lifecycleGeneration += 1;
    this.stopping = true;
    void this.finishDispose();
  }

  private async finishDispose(): Promise<void> {
    this.stream?.dispose();
    await this.microphone?.dispose();
    this.stream = undefined;
    this.publishAcceptanceState();
    this.handlers = undefined;
    await this.stopBackendSession();
    this.stopping = false;
  }

  /** Wraps stream status so a normal socket close completes the backend call. */
  private createStreamHandlers(handlers: RiskStreamHandlers): RiskStreamHandlers {
    return {
      onEvent: (event) => { handlers.onEvent(event); this.recordAcceptanceEvent(event); },
      onError: handlers.onError,
      onStatusChange: (status) => {
        handlers.onStatusChange(status);
        if (status === "DISCONNECTED" && !this.stopping) void this.handleUnexpectedRiskDisconnect();
      },
    };
  }

  /** Records bounded risk metadata and publishes the dev-only browser snapshot. */
  private recordAcceptanceEvent(event: LiveRiskEvent): void {
    const mic = this.microphone?.getAcceptanceSnapshot?.();
    const steadyLatency = event.audio_source_frame_end !== undefined && mic?.timing_anchor ? steadyStateLatencyMs(performance.now(), captureEndPerformanceMs(BigInt(event.audio_source_frame_end), mic.timing_anchor)) : null;
    this.appendAcceptanceInference({ audio_window_sequence: event.audio_window_sequence, raw_spoof_score: event.synthetic_probability, aggregate_score: event.aggregate_spoof_evidence ?? event.synthetic_probability, policy_state: event.overall_risk_score === 70 ? "ELEVATED_AUTHENTICITY_REVIEW" : "NORMAL", overall_risk_score: event.overall_risk_score, risk_level: event.risk_level, recommended_action: event.recommended_action, ml_inference_latency_ms: event.inference_latency_ms, steady_state_latency_ms: steadyLatency, score_semantics: "uncalibrated", speaker_similarity: event.speaker_similarity ?? null, speaker_state: event.speaker_state, fusion_state: event.fusion_state });
    this.publishAcceptanceState(event.call_id);
  }

  /** Publishes the current ephemeral acceptance snapshot. */
  private appendAcceptanceInference(inference: AcceptanceInference): void {
    if (this.acceptanceInferences.length === LiveCallRiskStreamSource.maxAcceptanceInferences) this.acceptanceInferences.shift();
    this.acceptanceInferences.push(inference);
  }

  private publishAcceptanceState(callId?: string): void {
    const mic = this.microphone?.getAcceptanceSnapshot?.();
    const resolvedCallId = callId ?? this.session?.call_id ?? mic?.call_id;
    if (!resolvedCallId) return;
    publishAcceptanceSnapshot(createAcceptanceSnapshot({ call_id: resolvedCallId, microphone_state: mic?.microphone_state, audio_context_sample_rate: mic?.audio_context_sample_rate, channels: mic?.channels, browser_frames_produced: mic?.transport?.browser_frames_produced, browser_frames_sent: mic?.transport?.browser_frames_sent, browser_frames_dropped: mic?.transport?.browser_frames_dropped, transport_sequence_gap_count: mic?.transport?.sequence_gap_count, cleanup: mic?.cleanup, inferences: this.acceptanceInferences }));
  }

  /** Rolls back every resource acquired by a failed live startup. */
  private async rollbackStart(): Promise<void> {
    this.stopping = true;
    const stream = this.stream;
    this.stream = undefined;
    await stream?.stop();
    await this.microphone?.stop();
    this.publishAcceptanceState();
    await this.stopBackendSession();
    this.stopping = false;
  }

  /** Terminates microphone capture when the risk transport disappears unexpectedly. */
  private async handleUnexpectedRiskDisconnect(): Promise<void> {
    await this.microphone?.stop();
    this.publishAcceptanceState();
    await this.stopBackendSession();
  }

  /** Stops a started backend session once and releases its identity. */
  private async stopBackendSession(): Promise<void> {
    if (!this.session || !this.started) return;
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
