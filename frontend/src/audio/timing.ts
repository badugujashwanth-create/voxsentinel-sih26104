export interface AudioTimingAnchor {
  contextOriginPerfMs: number;
  sampleRate: number;
}

/** Anchors the AudioContext processing timeline to browser performance time. */
export function createAudioTimingAnchor(contextCurrentTime: number, performanceNow: number, sampleRate: number): AudioTimingAnchor {
  if (!Number.isFinite(contextCurrentTime) || !Number.isFinite(performanceNow) || !Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error("Audio timing anchor inputs are invalid");
  return { contextOriginPerfMs: performanceNow - contextCurrentTime * 1_000, sampleRate };
}

/** Converts the final contributing source frame into a browser performance timestamp. */
export function captureEndPerformanceMs(sourceFrameEnd: bigint, anchor: AudioTimingAnchor): number {
  if (sourceFrameEnd < 0n) throw new Error("Source frame end must be non-negative");
  return anchor.contextOriginPerfMs + (Number(sourceFrameEnd) / anchor.sampleRate) * 1_000;
}

/** Calculates browser-observed processing/transport latency, not ADC latency. */
export function steadyStateLatencyMs(eventArrivalPerformanceNow: number, captureEndPerformance: number): number {
  const latency = eventArrivalPerformanceNow - captureEndPerformance;
  return Math.max(0, latency);
}
