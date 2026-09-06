"""
The realtime loop's state and its two threads.

TWO THREADS, DELIBERATELY
-------------------------
The audio callback is hard-realtime and must never block. It denoises one
512-sample frame, about 0.30 ms of work per 10.67 ms of audio, and applies the
current gain with a ramp so changes are inaudible. It does no file I/O, takes
no locks, and allocates as little as it can. Debug frames go onto a queue and a
writer thread does the writing. This is CLAUDE.md invariant 2.

The analysis thread is soft-realtime. Every ANALYZE_EVERY seconds it grabs the
last dsp.WINDOW_SEC from the ring, decimates it, embeds it, and updates the
gain target. Speaker identity is not decidable from 10 ms of audio; ECAPA needs
about a second of context. The gain the callback applies is therefore a couple
of hundred milliseconds behind the decision, which HANGOVER papers over.
"""

import queue
import sys
import threading
import time

import numpy as np
import soundfile as sf

from .. import dsp
from .ring import Ring

ANALYZE_EVERY = 0.20  # seconds between identity decisions
RAMP_MS = 15          # gain fade. CLAUDE.md invariant 3: stepping gain clicks.
HANGOVER = 0.40       # hold the gate open this long after you stop
EMA = 0.6             # smoothing on the similarity score

# Closed-gate gain. CLAUDE.md invariant 4: not zero. Dead silence reads as a
# dropped connection, while -34 dB reads as a quiet room.
FLOOR = 0.02

VAD_RMS = 0.004       # below this the room is quiet; nobody is talking

# Debug frames waiting to be written. At 512 samples per frame this is about
# 2.7 seconds of slack, far more than a disk write ever needs. If it fills, the
# callback drops the frame rather than blocking: a dropped debug frame costs
# 10 ms of a recording, a blocked callback costs a dropout in the actual call.
DEBUG_QUEUE_FRAMES = 256


class Pipeline:
    def __init__(self, names, centroids, threshold, monitor=False, debug=None,
                 in_channel=0, out_channels=(0,)):
        self.in_channel = in_channel
        self.out_channels = tuple(out_channels)
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
        self.skipped_windows = 0

        # Precomputed so the callback does no arithmetic it can hoist. The ramp
        # traverses the full 0-to-1 gain range in RAMP_MS.
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
        raw = indata[:, self.in_channel].copy()

        # process_frame demands exactly dsp.FRAME samples and raises otherwise,
        # and an exception here tears down the stream. PortAudio should honour
        # the blocksize we asked for, but a short block at teardown or from a
        # device that ignores blocksize would otherwise end the call. Pass the
        # audio through undenoised instead: 10 ms of fan noise is a better
        # outcome than a dead microphone.
        if frames != dsp.FRAME:
            self.short_blocks += 1
            self._emit(outdata, raw * self.gain)
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
        self._emit(outdata, out)

        if self._debug_q is not None:
            try:
                self._debug_q.put_nowait(np.stack([raw, out], axis=1))
            except queue.Full:
                self.debug_drops += 1

    def _emit(self, outdata, signal):
        """Write the mono result to the chosen output channels, silence the rest.

        Everything not written has to be zeroed explicitly: PortAudio hands
        back an uninitialised buffer, so an untouched channel plays whatever
        was in that memory, which is the previous block or noise.
        """
        outdata[:] = 0.0
        for c in self.out_channels:
            outdata[:, c] = signal

    # ---------------- debug writer thread ----------------
    def _write_debug(self):
        """File I/O lives here and nowhere near the callback.

        L = raw mic, R = gated output, which is the layout pvi.probe expects.
        """
        try:
            with sf.SoundFile(self.debug_path, "w", samplerate=dsp.SR,
                              channels=2, subtype="PCM_16") as f:
                while True:
                    block = self._debug_q.get()
                    if block is None:
                        return
                    f.write(np.clip(block, -1.0, 1.0))
        except Exception as e:  # noqa: BLE001
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
            except Exception as e:  # noqa: BLE001
                # This thread is a daemon. Left unhandled, one exception kills
                # it, the gain target freezes at whatever it last was, and the
                # gate appears to work but never changes again.
                #
                # Fail OPEN: a filter that stopped filtering still lets you be
                # heard, while one stuck closed silently ends your side of the
                # call and looks identical from your end.
                self.error = repr(e)
                self.target = 1.0
                print(f"\nanalysis thread error, gate failing open: {e}",
                      file=sys.stderr, flush=True)
                self.running = False
                return

    def _analyze_once(self):
        win = self.ring.latest()

        if dsp.rms(win) < VAD_RMS:
            self._maybe_close()
            return

        # A window can clear the loudness check and still be mostly silence:
        # the one straddling the moment you start talking is perhaps 30%
        # speech, and ECAPA returns something closer to a fingerprint of the
        # room than of you. Embedding it produces a low score that the EMA then
        # carries for several windows, which is what clips your first word.
        #
        # Skip the decision rather than make a bad one. Do NOT close the gate
        # here: this is a transition, not silence, and silence is handled
        # above. Whatever the gate was doing it keeps doing until a window
        # arrives that is worth judging, about 750 ms after speech onset at the
        # default window and hop.
        if dsp.speech_coverage(win) < dsp.MIN_COVERAGE:
            self.skipped_windows += 1
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

    def start_analysis(self):
        threading.Thread(target=self.analyze, daemon=True).start()

    def close(self):
        """Call AFTER the stream is closed, so the callback has stopped pushing
        and the writer can drain the last of the queue."""
        self.running = False
        if self._debug_q is not None:
            try:
                # Bounded: if the writer thread already died, say on a full
                # disk, a blocking put here would hang shutdown forever.
                self._debug_q.put(None, timeout=2.0)
            except queue.Full:
                pass
            self._debug_thread.join(timeout=5.0)
