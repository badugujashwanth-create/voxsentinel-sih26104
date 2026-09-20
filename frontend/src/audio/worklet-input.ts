/** Selects the configured transport channels from one Web Audio render quantum. */
export function selectTransportChannels(input: Float32Array[], channels: 1 | 2): Float32Array[] {
  if (channels === 1) {
    if (input.length < 1) throw new Error("Audio quantum has no input channel");
    return [input[0]];
  }
  if (input.length < 2) throw new Error("Audio quantum is missing a stereo channel");
  return [input[0], input[1]];
}
