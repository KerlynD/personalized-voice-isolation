"""
Timing for the realtime path.

WHY THIS EXISTS
---------------
The audio callback runs every dsp.FRAME samples, which at dsp.SR is 10.67 ms.
Everything inside it has to finish in less than that EVERY time, not on
average: the worst case is what you hear. CLAUDE.md invariant 2 says any model
going in the callback must be benchmarked against that budget first, and this
is that benchmark.

WHAT THE NUMBERS MEAN
---------------------
p50 is the typical cost. p99 and max are the ones that matter: a component that
is fast 99 times out of 100 and blows the budget on the hundredth produces an
audible click every couple of seconds.

RTF is the real-time factor, seconds of compute per second of audio. Below 1.0
means a component can keep up with the audio at all. That is necessary but NOT
sufficient for realtime, because it says nothing about latency: a model needing
a 4-second chunk to produce output cannot be used in a call however fast it
runs, since you would arrive four seconds late to your own conversation.
"""

import argparse
import time

import numpy as np
import torch

from .. import dsp

# Enough iterations that p99 means something. At 2000 frames this measures 21
# seconds of audio, so the tail includes a few garbage collections and at least
# one scheduler hiccup, which is the point.
DEFAULT_ITERS = 2000

BUDGET_MS = 1000.0 * dsp.FRAME / dsp.SR

# Chunk lengths to time the extraction model at. The short ones matter most: a
# streaming version would process something in this range, and per-call
# overhead dominates there.
TSE_CHUNKS = [0.25, 0.5, 1.0, 2.0, 4.0]


def timeit(fn, iters, warmup=20):
    """Run fn iters times, return milliseconds per call.

    Warms up first because the first calls pay for lazy allocation, kernel
    selection and page faults, and including those reports a worst case that
    only ever happens once.
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
    line = f"  {name:<34}{p50:8.3f}{p95:8.3f}{p99:8.3f}{ms.max():9.3f}"
    if budget:
        line += f"{ms.max() / budget * 100:8.1f}%"
    print(line)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="bench.py",
        description="Time each component against the audio callback budget.")
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["denoiser", "ecapa", "tse"])
    ap.add_argument("--threads", type=int, default=1,
                    help="torch threads. 1 is what live.py uses; raise it to "
                         "see what an offline path could do.")
    ap.add_argument("--backend", default=None,
                    help="extraction backend to time; defaults to tse.DEFAULT")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    dsp.limit_torch_threads(args.threads)

    print(f"callback budget: {BUDGET_MS:.2f} ms per {dsp.FRAME} samples "
          f"at {dsp.SR} Hz")
    print(f"torch threads: {torch.get_num_threads()}\n")
    print(f"  {'component':<34}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>9}"
          f"{'of budget':>9}")
    print(f"  {'-' * 34}{'-' * 42}")

    if "denoiser" not in args.skip:
        den = dsp.Denoiser()
        frame = (np.random.randn(dsp.FRAME) * 0.05).astype(np.float32)
        report("DeepFilterNet, one frame",
               timeit(lambda: den.process_frame(frame), args.iters), BUDGET_MS)

    if "ecapa" not in args.skip:
        # Runs on the analysis thread, not the callback, so it is not against
        # the budget. It still has to finish inside ANALYZE_EVERY or decisions
        # queue up behind each other.
        encoder = dsp.load_encoder()
        win = (np.random.randn(int(dsp.WINDOW_SEC * dsp.ANALYSIS_SR)) * 0.05
               ).astype(np.float32)
        report("ECAPA, one 1.5 s window",
               timeit(lambda: dsp.embed(encoder, win), max(50, args.iters // 40)))
        print(f"  {'':>34}(analysis thread, budget is 200 ms between decisions)")

    if "tse" not in args.skip:
        from .. import tse
        print()
        model = tse.load(args.backend or tse.DEFAULT)
        enroll = torch.randn(1, tse.SR * 3) * 0.03
        print(f"\n  {'extraction chunk':<34}{'p50':>8}{'p95':>8}{'p99':>8}"
              f"{'max':>9}{'RTF':>9}")
        print(f"  {'-' * 34}{'-' * 42}")
        for secs in TSE_CHUNKS:
            mix = torch.randn(1, int(tse.SR * secs)) * 0.03

            def run():
                with torch.no_grad():
                    model(mix, enroll)

            ms = timeit(run, max(5, int(20 / max(secs, 0.25))), warmup=3)
            p50 = float(np.percentile(ms, 50))
            p95, p99 = np.percentile(ms, [95, 99])
            print(f"  {secs:>4.2f} s of audio at {tse.SR} Hz{'':<12}"
                  f"{p50:8.1f}{p95:8.1f}{p99:8.1f}{ms.max():9.1f}"
                  f"{(p50 / 1000.0) / secs:9.2f}")

    print("\nRTF below 1.0 means it can keep up with the audio. It says nothing")
    print("about latency: a model needing a long chunk to produce output cannot")
    print("be used in a call however fast it runs.")
    return 0
