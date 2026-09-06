"""
Timing for the realtime path. Nothing here had ever been measured.

WHY THIS EXISTS
---------------
The audio callback runs every dsp.FRAME samples, which at dsp.SR is 10.67 ms.
Everything that runs inside it has to finish in less than that, every single
time, or the stream drops out. Not on average: the worst case is what you hear.

CLAUDE.md's invariant 2 says any model that goes in the callback must be
benchmarked against that budget first. This is that benchmark. Until it has
been run on the machine in question, every statement about what fits in the
callback is a guess.

WHAT THE NUMBERS MEAN
---------------------
p50 is the typical cost. p99 and max are the ones that matter: a component that
is fast 99 times out of 100 and blows the budget on the hundredth produces an
audible click every couple of seconds.

RTF is the real-time factor: seconds of compute per second of audio. Below 1.0
means the component can keep up with the audio at all. That is necessary but
NOT sufficient for realtime, because it says nothing about latency: a model
that needs a 4-second chunk to produce output cannot be used in a call however
fast it runs, since you would arrive 4 seconds late to your own conversation.

Usage:
    python bench.py
    python bench.py --iters 5000        # tighter tails
    python bench.py --skip tse          # denoiser and encoder only
"""

import argparse
import time

import numpy as np
import torch

import dsp

# Enough iterations that the p99 means something. At 2000 frames we are
# measuring 21 seconds of audio, so the tail includes a few garbage collections
# and at least one scheduler hiccup, which is the point.
DEFAULT_ITERS = 2000

# The budget, in milliseconds. One callback's worth of audio.
BUDGET_MS = 1000.0 * dsp.FRAME / dsp.SR

# Chunk lengths to time the extraction model at, in seconds. The short ones
# matter most: a streaming version would process something in this range, and
# per-chunk overhead dominates there.
TSE_CHUNKS = [0.25, 0.5, 1.0, 2.0, 4.0]


def timeit(fn, iters, warmup=20):
    """Run fn iters times, return milliseconds per call as an array.

    Warms up first because the first calls pay for lazy allocation, kernel
    selection and page faults, and including those would report a worst case
    that only ever happens once.
    """
    for _ in range(warmup):
        fn()
    out = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t = time.perf_counter()
        fn()
        out[i] = (time.perf_counter() - t) * 1000.0
    return out


def report(name, ms, budget=None):
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    line = (f"  {name:<34}{p50:8.3f}{p95:8.3f}{p99:8.3f}{ms.max():9.3f}")
    if budget:
        worst = ms.max() / budget
        line += f"{worst * 100:8.1f}%"
    print(line)
    return p50, p99, ms.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["denoiser", "ecapa", "tse"])
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads. 1 is what live.py uses; raise it to "
                         "see what an offline or larger-budget path could do.")
    a = ap.parse_args()

    dsp.limit_torch_threads(a.threads)
    print(f"callback budget: {BUDGET_MS:.2f} ms per {dsp.FRAME} samples "
          f"at {dsp.SR} Hz")
    print(f"torch threads: {torch.get_num_threads()}\n")
    print(f"  {'component':<34}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>9}"
          f"{'of budget':>9}")
    print(f"  {'-' * 34}{'-' * 42}")

    results = {}

    if "denoiser" not in a.skip:
        den = dsp.Denoiser()
        frame = (np.random.randn(dsp.FRAME) * 0.05).astype(np.float32)
        ms = timeit(lambda: den.process_frame(frame), a.iters)
        results["denoiser"] = report("DeepFilterNet, one frame", ms, BUDGET_MS)

    if "ecapa" not in a.skip:
        # Runs on the analysis thread, not the callback, so it is not against
        # the budget. It still has to finish inside ANALYZE_EVERY or decisions
        # queue up behind each other.
        enc = dsp.load_encoder()
        win = (np.random.randn(int(dsp.WINDOW_SEC * dsp.ANALYSIS_SR)) * 0.05
               ).astype(np.float32)
        ms = timeit(lambda: dsp.embed(enc, win), max(50, a.iters // 40))
        report("ECAPA, one 1.5 s window", ms)
        print(f"  {'':>34}(analysis thread, budget is 200 ms between decisions)")

    if "tse" not in a.skip:
        import tse
        print()
        model = tse.load("espnet")
        enroll = torch.randn(1, tse.SR * 3) * 0.03
        print(f"\n  {'extraction chunk':<34}{'p50':>8}{'p95':>8}{'p99':>8}"
              f"{'max':>9}{'RTF':>9}")
        print(f"  {'-' * 34}{'-' * 42}")
        for secs in TSE_CHUNKS:
            mix = torch.randn(1, int(tse.SR * secs)) * 0.03
            def run():
                with torch.no_grad():
                    model(mix, enroll)
            n = max(5, int(20 / max(secs, 0.25)))
            ms = timeit(run, n, warmup=3)
            p50 = float(np.percentile(ms, 50))
            rtf = (p50 / 1000.0) / secs
            p95, p99 = np.percentile(ms, [95, 99])
            print(f"  {secs:>4.2f} s of audio at {tse.SR} Hz{'':<12}"
                  f"{p50:8.1f}{p95:8.1f}{p99:8.1f}{ms.max():9.1f}{rtf:9.2f}")
        results["tse_rtf"] = rtf

    print("\nRTF below 1.0 means it can keep up with the audio. It says nothing")
    print("about latency: a model needing a long chunk to produce output cannot")
    print("be used in a call however fast it runs.")


if __name__ == "__main__":
    main()
