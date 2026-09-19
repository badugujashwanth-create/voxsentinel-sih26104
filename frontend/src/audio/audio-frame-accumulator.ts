import { TARGET_TRANSPORT_SAMPLES, type TransportFrame } from "./protocol";

/** Accumulates one runtime stream of Web Audio quanta into transport frames. */
export class AudioFrameAccumulator {
  private readonly targetSamples: number;
  private readonly channels: 1 | 2;
  private buffers: Float32Array[];
  private bufferStart: bigint | undefined;

  public constructor(targetSamples = TARGET_TRANSPORT_SAMPLES, channels: 1 | 2 = 1) {
    this.targetSamples = targetSamples;
    this.channels = channels;
    this.buffers = Array.from({ length: channels }, () => new Float32Array(0));
  }

  /** Appends a render quantum and emits all complete transport frames. */
  public append(channelSamples: Float32Array[], firstSampleFrame: bigint): TransportFrame[] {
    this.validateQuantum(channelSamples);
    if (this.bufferStart === undefined) this.bufferStart = firstSampleFrame;
    const expected = this.bufferStart + BigInt(this.buffers[0].length);
    if (firstSampleFrame > expected) {
      this.reset();
      this.bufferStart = firstSampleFrame;
    }
    this.buffers = this.buffers.map((buffer, index) => concat(buffer, channelSamples[index]));
    const frames: TransportFrame[] = [];
    while (this.buffers[0].length >= this.targetSamples) {
      const samples = interleave(this.buffers.map((buffer) => buffer.slice(0, this.targetSamples)));
      frames.push({ sequence: 0n, firstSampleFrame: this.bufferStart!, channels: this.channels, samples });
      this.buffers = this.buffers.map((buffer) => buffer.slice(this.targetSamples));
      this.bufferStart = this.bufferStart! + BigInt(this.targetSamples);
    }
    return frames;
  }

  /** Emits the final aligned residual without adding fabricated samples. */
  public flush(): TransportFrame | null {
    if (this.buffers[0].length === 0 || this.bufferStart === undefined) return null;
    const frame = { sequence: 0n, firstSampleFrame: this.bufferStart, channels: this.channels, samples: interleave(this.buffers) } satisfies TransportFrame;
    this.buffers = Array.from({ length: this.channels }, () => new Float32Array(0));
    this.bufferStart = undefined;
    return frame;
  }

  /** Discards residual samples on reset or error. */
  public reset(): void {
    this.buffers = Array.from({ length: this.channels }, () => new Float32Array(0));
    this.bufferStart = undefined;
  }

  public get residualSamples(): number {
    return this.buffers[0].length;
  }

  private validateQuantum(channelSamples: Float32Array[]): void {
    if (channelSamples.length !== this.channels || channelSamples.some((samples) => samples.length !== channelSamples[0].length)) throw new Error("Audio quantum channel shape is invalid");
  }
}

function concat(left: Float32Array, right: Float32Array): Float32Array {
  const result = new Float32Array(left.length + right.length);
  result.set(left);
  result.set(right, left.length);
  return result;
}

function interleave(channels: Float32Array[]): Float32Array {
  const result = new Float32Array(channels[0].length * channels.length);
  for (let sample = 0; sample < channels[0].length; sample += 1) {
    for (let channel = 0; channel < channels.length; channel += 1) result[sample * channels.length + channel] = channels[channel][sample];
  }
  return result;
}
