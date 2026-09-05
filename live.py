"""
Real-time personal voice gate. Stage 2: 48 kHz path + denoiser + identity gate.

PIPELINE
--------
    mic 48 kHz -> DeepFilterNet3 -> gain -> virtual output -> Discord
                        |
                        +-> ring buffer -> decimate to 16 kHz -> ECAPA -> gain target

WHAT THE DENOISER DOES AND DOESN'T DO
-------------------------------------
DeepFilterNet is a NOISE suppressor. It kills fans, keyboards, HVAC, and
reverb tail. It is trained to PRESERVE speech, so it will not remove
others voices. It treats their speech as signal. Concurrent speech is a
separation problem, which is stage 3.

WHAT THIS GATE CANNOT DO
------------------------
When someone talks at the same time as you, the gate is open, because you are
talking. Both voices pass. This is not a tuning problem and no threshold fixes
it: a gate decides when to pass audio, not which parts of it to pass. Overlapped
speech needs a mask over the mixture, which is what probe.py is investigating.
Record a session with `--record-debug` and run probe.py on it.

THREE THREADS, DELIBERATELY
---------------------------
The audio callback is hard-realtime and must never block: it denoises one
512-sample frame (~1.5 ms of work per 10.7 ms of audio) and applies the current
gain with a ramp so changes are inaudible. It does no file I/O and takes no
locks. Debug frames go onto a queue and a writer thread does the writing.

The analysis thread is soft-realtime: every ~200 ms it grabs the last 1.5 s from
the ring, decimates it, embeds it, and updates the gain target. Speaker identity
is simply not decidable from 10 ms of audio, you need about a second of
context. The gain the callback applies is therefore a couple hundred ms behind
the decision, which the hangover parameter papers over.

Usage:
    python live.py --list-devices
    python live.py --in 2 --out 7 --monitor
    python live.py --in 2 --out 7 --threshold 0.42 --record-debug session.wav
"""

import argparse
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

import dsp

ANALYZE_EVERY = 0.20  # seconds between identity decisions
RAMP_MS = 15          # gain fade, prevents clicks
HANGOVER = 0.40       # hold the gate open this long after you stop
EMA = 0.6             # smoothing on the similarity score
FLOOR = 0.02          # closed-gate gain. Not zero: dead silence reads as a
                      # broken connection; -34 dB reads as "quiet room."
VAD_RMS = 0.004       # below this, don't bother embedding

# Debug frames waiting to be written. At 512 samples per frame this is about
# 2.7 seconds of slack, far more than a disk write ever needs. If it ever
# fills, the callback drops the frame rather than blocking. A dropped debug
# frame costs you 10 ms of a recording; a blocked callback costs a dropout in
# the actual call.
DEBUG_QUEUE_FRAMES = 256


class Ring:
    """Circular buffer written by the audio callback, read by the analysis
    thread. A torn read costs at most one frame of stale audio inside a 1.5 s
    window, which is harmless, so no lock.

    Writes are two slice assignments rather than fancy indexing, because the
    callback runs 94 times a second and building an index array each time is
    an allocation the realtime thread does not need to make.
    """

    def __init__(self, n):
        self.buf = np.zeros(n, dtype=np.float32)
        self.n = n
        self.w = 0

    def push(self, x):
        m = len(x)
        if m >= self.n:              # a write larger than the ring: keep the tail
            self.buf[:] = x[-self.n:]
            self.w = 0
            return
        end = self.w + m
        if end <= self.n:
            self.buf[self.w:end] = x
        else:
            k = self.n - self.w
            self.buf[self.w:] = x[:k]
            self.buf[:end - self.n] = x[k:]
        self.w = end % self.n

    def latest(self):
        """Oldest sample first. self.w points at the next write, so everything
        from w to the end is older than everything before w."""
        return np.concatenate([self.buf[self.w:], self.buf[:self.w]])


class Pipeline:
    def __init__(self, names, centroids, threshold, monitor=False, debug=None):
        self.names = names
        self.centroids = centroids
        self.threshold = threshold
        self.monitor = monitor
        self.encoder = dsp.load_encoder()
        self.denoiser = dsp.Denoiser()
        self.ring = Ring(int(dsp.WINDOW_SEC * dsp.SR))
        self.gain = FLOOR
        self.target = FLOOR
        self.score = 0.0
        self.who = ""
        self.last_open = 0.0
        self.running = True
        self.error = None
        self.short_blocks = 0
        self.debug_drops = 0

        # Precomputed so the callback does no arithmetic it can hoist. The
        # ramp traverses the full 0-to-1 gain range in RAMP_MS.
        self._step = 1.0 / (RAMP_MS * dsp.SR / 1000.0)
        self._ramp_idx = np.arange(1, dsp.FRAME + 1, dtype=np.float32)

        self.debug_path = debug
        self._debug_q = queue.Queue(maxsize=DEBUG_QUEUE_FRAMES) if debug else None
        self._debug_thread = None
        if debug:
            self._debug_thread = threading.Thread(target=self._write_debug,
                                                  daemon=True)
            self._debug_thread.start()

    # ---------------- audio thread ----------------
    def callback(self, indata, outdata, frames, t, status):
        if status:
            print(status, file=sys.stderr, flush=True)
        raw = indata[:, 0].copy()

        # process_frame demands exactly dsp.FRAME samples and raises otherwise,
        # and an exception in here tears down the stream. PortAudio should
        # honour the blocksize we asked for, but a short block at teardown or
        # from a device that ignores blocksize would otherwise end the call.
        # Pass the audio through undenoised instead: 10 ms of fan noise is a
        # better outcome than a dead microphone.
        if frames != dsp.FRAME:
            self.short_blocks += 1
            outdata[:, 0] = raw * self.gain
            return

        clean = self.denoiser.process_frame(raw)
        self.ring.push(clean)

        # Snapshot the target once. The analysis thread can change it at any
        # moment, and reading it twice could pair a ramp direction with clip
        # bounds from the other side of the change.
        target = self.target
        gain = self.gain
        ramp = np.clip(
            gain + np.sign(target - gain) * self._step * self._ramp_idx,
            min(gain, target), max(gain, target),
        )
        self.gain = float(ramp[-1])
        out = clean * ramp
        outdata[:, 0] = out

        if self._debug_q is not None:
            try:
                self._debug_q.put_nowait(np.stack([raw, out], axis=1))
            except queue.Full:
                self.debug_drops += 1

    # ---------------- debug writer thread ----------------
    def _write_debug(self):
        """File I/O lives here and nowhere near the callback (CLAUDE.md
        invariant 2). L = raw mic, R = gated output, which is the layout
        probe.py expects."""
        try:
            with sf.SoundFile(self.debug_path, "w", samplerate=dsp.SR,
                              channels=2, subtype="PCM_16") as f:
                while True:
                    block = self._debug_q.get()
                    if block is None:
                        return
                    f.write(np.clip(block, -1.0, 1.0))
        except Exception as e:
            # Losing the debug recording is annoying. Taking the call down with
            # it would be worse, so this thread dies quietly and the callback
            # goes on dropping frames into a queue nobody drains.
            print(f"\ndebug writer stopped: {e}", file=sys.stderr, flush=True)

    # ---------------- analysis thread ----------------
    def analyze(self):
        while self.running:
            time.sleep(ANALYZE_EVERY)
            try:
                self._analyze_once()
            except Exception as e:
                # This thread is a daemon. Left unhandled, one exception kills
                # it, the gain target freezes at whatever it last was, and the
                # gate appears to work but never changes again. Fail open: a
                # filter that stopped filtering still lets you be heard, while
                # one stuck closed silently ends your side of the call.
                self.error = repr(e)
                self.target = 1.0
                print(f"\nanalysis thread error, gate failing open: {e}",
                      file=sys.stderr, flush=True)
                self.running = False
                return

    def _analyze_once(self):
        win = self.ring.latest()

        if float(np.sqrt(np.mean(win**2))) < VAD_RMS:
            self._maybe_close()
            return

        e = dsp.embed(self.encoder, dsp.to_analysis(win))
        sim, i = dsp.best_score(self.centroids, e)
        self.score = EMA * self.score + (1 - EMA) * sim
        self.who = self.names[i]

        if self.score > self.threshold:
            self.target = 1.0
            self.last_open = time.time()
        else:
            self._maybe_close()

        if self.monitor:
            bar = "#" * int(max(0.0, self.score) * 40)
            print(f"\r{self.who:>10} sim={self.score:+.3f} "
                  f"gain={self.gain:.2f} {bar:<40}", end="", flush=True)

    def _maybe_close(self):
        if time.time() - self.last_open > HANGOVER:
            self.target = FLOOR

    def close(self):
        """Call this AFTER the stream is closed, so the callback has stopped
        pushing and the writer can drain the last of the queue."""
        self.running = False
        if self._debug_q is not None:
            try:
                # Bounded: if the writer thread already died, say on a full
                # disk, a blocking put here would hang shutdown forever.
                self._debug_q.put(None, timeout=2.0)
            except queue.Full:
                pass
            self._debug_thread.join(timeout=5.0)


def open_stream(inp, out, callback):
    """Open the duplex stream, with a readable failure for the common macOS case.

    A duplex sd.Stream spans one input device and one output device. When those
    are two different physical devices they run on two different clocks, and
    CoreAudio will not bridge them for us. The fix is an Aggregate Device, which
    makes the OS do the bridging before PortAudio ever sees it.
    """
    try:
        return sd.Stream(device=(inp, out), samplerate=dsp.SR,
                         blocksize=dsp.FRAME, channels=1, dtype="float32",
                         latency="low", callback=callback)
    except Exception as e:
        raise SystemExit(
            f"could not open a duplex stream on devices ({inp}, {out}): {e}\n\n"
            f"On macOS this usually means the mic and the virtual cable are "
            f"separate devices on separate clocks. Open Audio MIDI Setup, "
            f"create an Aggregate Device containing both, and pass its index "
            f"for --in and --out. `python live.py --list-devices` will show it "
            f"once it exists."
        ) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--in", dest="inp", type=int, help="input device (AT2020)")
    ap.add_argument("--out", dest="out", type=int, help="output device (virtual cable)")
    ap.add_argument("--speakers", default="speakers.npz")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--monitor", action="store_true",
                    help="print live similarity, use this to tune the threshold")
    ap.add_argument("--record-debug", default=None, metavar="WAV",
                    help="write a stereo wav: L=raw mic, R=gated output. "
                         "This is your evaluation data, and probe.py's input. "
                         "Listen for false rejects (your speech cut) and leaks "
                         "(their speech through).")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    dsp.limit_torch_threads()

    d = np.load(args.speakers, allow_pickle=False)
    names = [str(n) for n in d["names"]]
    centroids = d["centroids"]
    threshold = args.threshold if args.threshold is not None else float(d["threshold"])
    print(f"enrolled: {names}  threshold={threshold:.3f}")

    p = Pipeline(names, centroids, threshold,
                 monitor=args.monitor, debug=args.record_debug)
    threading.Thread(target=p.analyze, daemon=True).start()

    with open_stream(args.inp, args.out, p.callback):
        print("running - Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    # Outside the `with`: the stream is closed and the callback has stopped, so
    # the debug writer can drain what is left without racing new frames in.
    p.close()

    if p.short_blocks:
        print(f"\n{p.short_blocks} blocks arrived at the wrong size and bypassed "
              f"the denoiser")
    if p.debug_drops:
        print(f"{p.debug_drops} debug frames dropped; the writer could not keep up")
    if args.record_debug:
        print(f"\nwrote {args.record_debug}")
        print("Find a stretch where two people talk at once, then:")
        print(f"  python probe.py --debug-wav {args.record_debug} "
              f"--speaker {names[0]}")


if __name__ == "__main__":
    main()
