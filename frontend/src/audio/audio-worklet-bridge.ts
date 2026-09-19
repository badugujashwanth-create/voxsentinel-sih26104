import type { TransportFrame } from "./protocol";

export type AudioFrameHandler = (frame: Omit<TransportFrame, "sequence">) => void;

/** Owns the browser AudioWorklet node and its lifecycle, not audio policy. */
export class AudioWorkletBridge {
  private readonly context: AudioContext;
  private readonly channels: 1 | 2;
  private readonly onFrame: AudioFrameHandler;
  private readonly source?: AudioNode;
  private node: AudioWorkletNode | undefined;

  public constructor(context: AudioContext, channels: 1 | 2, onFrame: AudioFrameHandler, source?: AudioNode) {
    this.context = context;
    this.channels = channels;
    this.onFrame = onFrame;
    this.source = source;
  }

  /** Loads the dedicated worklet, attaches the frame callback, and connects it. */
  public async start(): Promise<void> {
    await this.context.audioWorklet.addModule(new URL("./microphone-worklet.ts", import.meta.url));
    this.node = new AudioWorkletNode(this.context, "voxsentinel-microphone", { processorOptions: { channels: this.channels } });
    this.node.port.onmessage = (event: MessageEvent) => {
      if (event.data?.type === "frame") this.onFrame(event.data);
    };
    this.source?.connect(this.node);
    this.node.connect(this.context.destination);
  }

  /** Requests a final residual frame without padding. */
  public flush(): void {
    this.node?.port.postMessage({ type: "flush" });
  }

  /** Disposes the node and prevents future callbacks. */
  public dispose(): void {
    this.node?.port.postMessage({ type: "reset" });
    this.node?.disconnect();
    this.source?.disconnect();
    this.node = undefined;
  }
}
