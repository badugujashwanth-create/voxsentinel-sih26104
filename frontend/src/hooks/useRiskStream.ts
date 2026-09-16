import { useCallback, useEffect, useState } from "react";
import type { LiveRiskEvent } from "../domain/risk";
import type { RiskStreamSource, RiskStreamStatus } from "../services/risk-stream/RiskStreamSource";

export interface RiskStreamState {
  events: LiveRiskEvent[];
  status: RiskStreamStatus;
  error: Error | null;
}

/** Binds one risk source to React state without exposing timers to components. */
export function useRiskStream(source: RiskStreamSource | null): RiskStreamState & {
  start(): Promise<void>;
  stop(): Promise<void>;
  pause(): void;
  resume(): void;
  reset(): void;
} {
  const [events, setEvents] = useState<LiveRiskEvent[]>([]);
  const [status, setStatus] = useState<RiskStreamStatus>("IDLE");
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    setEvents([]);
    setStatus("IDLE");
    setError(null);
    return () => source?.dispose();
  }, [source]);

  const start = useCallback(async () => {
    if (!source) return;
    await source.start({
      onEvent: (event) => setEvents((current) => [...current, event]),
      onStatusChange: setStatus,
      onError: setError,
    });
  }, [source]);

  const stop = useCallback(async () => source?.stop(), [source]);
  const pause = useCallback(() => source?.pause(), [source]);
  const resume = useCallback(() => source?.resume(), [source]);
  const reset = useCallback(() => {
    source?.reset();
    setEvents([]);
    setStatus("IDLE");
    setError(null);
  }, [source]);

  return { events, status, error, start, stop, pause, resume, reset };
}
