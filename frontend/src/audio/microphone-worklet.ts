import { AudioFrameAccumulator } from "./audio-frame-accumulator";
import { selectTransportChannels } from "./worklet-input";

declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
}

declare const currentFrame: number;
declare function registerProcessor(name: string, processor: unknown): void;

class VoxSentinelMicrophoneProcessor extends AudioWorkletProcessor {
  private readonly accumulator: AudioFrameAccumulator;

  public constructor(options?: AudioWorkletNodeOptions) {
    super();
    const channels = options?.processorOptions?.channels === 2 ? 2 : 1;
    this.accumulator = new AudioFrameAccumulator(1024, channels);
    this.port.onmessage = (event: MessageEvent) => {
      if (event.data?.type === "flush") {
        this.emitResidual();
        this.port.postMessage({ type: "flush_complete" });
      }
      if (event.data?.type === "reset") this.accumulator.reset();
    };
  }

  public process(inputs: Float32Array[][]): boolean {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) return true;
    const channels = selectTransportChannels(input, this.accumulator.channelCount);
    for (const frame of this.accumulator.append(channels, BigInt(currentFrame))) {
      this.port.postMessage({ type: "frame", firstSampleFrame: frame.firstSampleFrame, channels: frame.channels, samples: frame.samples }, [frame.samples.buffer]);
    }
    return true;
  }

  private emitResidual(): void {
    const frame = this.accumulator.flush();
    if (frame) this.port.postMessage({ type: "frame", firstSampleFrame: frame.firstSampleFrame, channels: frame.channels, samples: frame.samples }, [frame.samples.buffer]);
  }
}

registerProcessor("voxsentinel-microphone", VoxSentinelMicrophoneProcessor);
