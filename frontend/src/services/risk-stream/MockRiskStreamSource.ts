import { validateLiveRiskEvent } from "../../domain/risk";
import { SCENARIOS, type ScenarioId } from "../../scenarios/scenarios";
import type { RiskStreamHandlers, RiskStreamSource, RiskStreamStatus } from "./RiskStreamSource";

const DEFAULT_INTERVAL_MS = 700;

/** Emits a repeatable scenario event sequence without backend dependencies. */
export class MockRiskStreamSource implements RiskStreamSource {
  private readonly scenarioId: ScenarioId;
  private readonly intervalMs: number;
  private cursor = 0;
  private timer: ReturnType<typeof setInterval> | undefined;
  private handlers: RiskStreamHandlers | undefined;

  /** Creates a mock source for one deterministic scenario. */
  public constructor(scenarioId: ScenarioId, intervalMs = DEFAULT_INTERVAL_MS) {
    this.scenarioId = scenarioId;
    this.intervalMs = intervalMs;
  }

  /** Starts emitting events on the configured interval. */
  public async start(handlers: RiskStreamHandlers): Promise<void> {
    this.handlers = handlers;
    this.emitStatus("CONNECTING");
    this.emitStatus("LIVE");
    this.schedule();
  }

  /** Stops future events and marks the source disconnected. */
  public async stop(): Promise<void> {
    this.clearTimer();
    this.emitStatus("DISCONNECTED");
  }

  /** Pauses event delivery without losing the current cursor. */
  public pause(): void {
    this.clearTimer();
    this.emitStatus("PAUSED");
  }

  /** Resumes event delivery from the current cursor. */
  public resume(): void {
    if (this.cursor >= SCENARIOS[this.scenarioId].events.length) return;
    this.emitStatus("LIVE");
    this.schedule();
  }

  /** Resets the source cursor and status for a fresh scenario run. */
  public reset(): void {
    this.clearTimer();
    this.cursor = 0;
    this.emitStatus("IDLE");
  }

  /** Clears timers and listeners so the source cannot retain the UI. */
  public dispose(): void {
    this.clearTimer();
    this.handlers = undefined;
    this.cursor = 0;
  }

  /** Schedules the next deterministic event while a run is active. */
  private schedule(): void {
    this.clearTimer();
    this.timer = setInterval(() => this.emitNextEvent(), this.intervalMs);
  }

  /** Emits the next validated event and stops at the scenario boundary. */
  private emitNextEvent(): void {
    const event = SCENARIOS[this.scenarioId].events[this.cursor];
    if (!event) {
      this.clearTimer();
      this.emitStatus("DISCONNECTED");
      return;
    }

    try {
      this.handlers?.onEvent(validateLiveRiskEvent(event));
      this.cursor += 1;
    } catch (error) {
      this.clearTimer();
      this.emitError(error instanceof Error ? error : new Error("Mock risk event failed"));
    }
  }

  /** Publishes a stream status when a consumer is attached. */
  private emitStatus(status: RiskStreamStatus): void {
    this.handlers?.onStatusChange(status);
  }

  /** Publishes a stream error and records an error status. */
  private emitError(error: Error): void {
    this.emitStatus("ERROR");
    this.handlers?.onError(error);
  }

  /** Clears the current interval handle. */
  private clearTimer(): void {
    if (this.timer !== undefined) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }
}
