import type { LiveRiskEvent } from "../../domain/risk";

export type RiskStreamStatus = "IDLE" | "CONNECTING" | "LIVE" | "PAUSED" | "DISCONNECTED" | "ERROR";

export interface RiskStreamHandlers {
  onEvent(event: LiveRiskEvent): void;
  onStatusChange(status: RiskStreamStatus): void;
  onError(error: Error): void;
}

export interface RiskStreamSource {
  start(handlers: RiskStreamHandlers): Promise<void>;
  stop(): Promise<void>;
  pause(): void;
  resume(): void;
  reset(): void;
  dispose(): void;
}
