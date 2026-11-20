// Collects audio from the graph and posts ~100 ms Float32 chunks (mono mix) to the page.
class Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = (options.processorOptions && options.processorOptions.chunk) || 1600;
    this.buf = new Float32Array(this.size);
    this.n = 0;
  }
  process(inputs) {
    const chans = inputs[0];
    if (chans && chans.length) {
      const len = chans[0].length;
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let c = 0; c < chans.length; c++) s += chans[c][i];
        this.buf[this.n++] = s / chans.length;
        if (this.n === this.size) {
          this.port.postMessage(this.buf.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("capture", Capture);
