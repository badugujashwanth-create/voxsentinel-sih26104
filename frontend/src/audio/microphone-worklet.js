/* global AudioWorkletProcessor, currentFrame, registerProcessor */

const TARGET_TRANSPORT_SAMPLES = 1024;

class AudioFrameAccumulator {
  constructor(targetSamples = TARGET_TRANSPORT_SAMPLES, channels = 1) {
    this.targetSamples = targetSamples;
    this.channels = channels;
    this.buffers = Array.from({ length: channels }, () => new Float32Array(0));
    this.bufferStart = undefined;
  }

  append(channelSamples, firstSampleFrame) {
    if (channelSamples.length !== this.channels || channelSamples.some((samples) => samples.length !== channelSamples[0].length)) throw new Error("Audio quantum channel shape is invalid");
    if (this.bufferStart === undefined) this.bufferStart = firstSampleFrame;
    const expected = this.bufferStart + BigInt(this.buffers[0].length);
    if (firstSampleFrame > expected) {
      this.reset();
      this.bufferStart = firstSampleFrame;
    }
    this.buffers = this.buffers.map((buffer, index) => concat(buffer, channelSamples[index]));
    const frames = [];
    while (this.buffers[0].length >= this.targetSamples) {
      const samples = interleave(this.buffers.map((buffer) => buffer.slice(0, this.targetSamples)));
      frames.push({ firstSampleFrame: this.bufferStart, channels: this.channels, samples });
      this.buffers = this.buffers.map((buffer) => buffer.slice(this.targetSamples));
      this.bufferStart += BigInt(this.targetSamples);
    }
    return frames;
  }

  flush() {
    if (this.buffers[0].length === 0 || this.bufferStart === undefined) return null;
    const frame = { firstSampleFrame: this.bufferStart, channels: this.channels, samples: interleave(this.buffers) };
    this.buffers = Array.from({ length: this.channels }, () => new Float32Array(0));
    this.bufferStart = undefined;
    return frame;
  }

  reset() {
    this.buffers = Array.from({ length: this.channels }, () => new Float32Array(0));
    this.bufferStart = undefined;
  }
}

function concat(left, right) {
  const result = new Float32Array(left.length + right.length);
  result.set(left);
  result.set(right, left.length);
  return result;
}

function interleave(channels) {
  const result = new Float32Array(channels[0].length * channels.length);
  for (let sample = 0; sample < channels[0].length; sample += 1) {
    for (let channel = 0; channel < channels.length; channel += 1) result[sample * channels.length + channel] = channels[channel][sample];
  }
  return result;
}

function selectTransportChannels(input, channels) {
  if (channels === 1) {
    if (input.length < 1) throw new Error("Audio quantum has no input channel");
    return [input[0]];
  }
  if (input.length < 2) throw new Error("Audio quantum is missing a stereo channel");
  return [input[0], input[1]];
}

class VoxSentinelMicrophoneProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const channels = options?.processorOptions?.channels === 2 ? 2 : 1;
    this.accumulator = new AudioFrameAccumulator(TARGET_TRANSPORT_SAMPLES, channels);
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") {
        this.emitResidual();
        this.port.postMessage({ type: "flush_complete" });
      }
      if (event.data?.type === "reset") this.accumulator.reset();
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) return true;
    const channels = selectTransportChannels(input, this.accumulator.channels);
    for (const frame of this.accumulator.append(channels, BigInt(currentFrame))) this.port.postMessage({ type: "frame", firstSampleFrame: frame.firstSampleFrame, channels: frame.channels, samples: frame.samples }, [frame.samples.buffer]);
    return true;
  }

  emitResidual() {
    const frame = this.accumulator.flush();
    if (frame) this.port.postMessage({ type: "frame", firstSampleFrame: frame.firstSampleFrame, channels: frame.channels, samples: frame.samples }, [frame.samples.buffer]);
  }
}

registerProcessor("voxsentinel-microphone", VoxSentinelMicrophoneProcessor);
