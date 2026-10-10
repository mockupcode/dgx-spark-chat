// Runs on the audio thread: downsample the microphone to 16 kHz mono int16
// and hand out 512-sample frames (32 ms), the unit the server's VAD works in.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;            // fractional read position into the input stream
    this.prev = 0;           // last input sample of the previous block
    this.frame = new Int16Array(512);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    const n = ch.length;
    // Linear interpolation; `pos` is relative to the start of this block,
    // with index -1 meaning the last sample of the previous block.
    while (this.pos < n - 1) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = i < 0 ? this.prev : ch[i];
      const b = ch[i + 1];
      let s = a + (b - a) * frac;
      s = Math.max(-1, Math.min(1, s));
      this.frame[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.filled === this.frame.length) {
        this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
        this.frame = new Int16Array(512);
        this.filled = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= n;
    this.prev = ch[n - 1];
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
