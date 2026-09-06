"""Command line for the realtime filter."""

import argparse
import time

import sounddevice as sd

from .. import dsp
from ..enroll import store
from . import guide
from .devices import open_stream, parse_channels
from .pipeline import Pipeline


def build_parser():
    ap = argparse.ArgumentParser(
        prog="live.py",
        description="Realtime personal voice gate: mic -> denoise -> gate -> "
                    "virtual output.")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--in", dest="inp", type=int, help="input device")
    ap.add_argument("--out", dest="out", type=int,
                    help="output device (virtual cable)")
    ap.add_argument("--in-channel", type=int, default=0, metavar="N",
                    help="which input channel carries the mic, counting from 0. "
                         "On an Aggregate Device the members stack end to end, "
                         "so this is 0 only if the mic is the first member.")
    ap.add_argument("--out-channel", default="0", metavar="N[,N]",
                    help="which output channel(s) to send to, counting from 0. "
                         "Give both channels of a stereo virtual cable, e.g. "
                         "2,3, or the far end hears you in one ear.")
    ap.add_argument("--speakers", default="speakers.npz")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the enrolled cutoff; tune with --monitor")
    ap.add_argument("--guide", action="store_true",
                    help="run the scripted probe session: prints a clock, tells "
                         "you who should be talking, and stops on its own. Use "
                         "with --record-debug.")
    ap.add_argument("--monitor", action="store_true",
                    help="print live similarity, use this to tune the threshold")
    ap.add_argument("--record-debug", default=None, metavar="WAV",
                    help="write a stereo wav: L=raw mic, R=gated output. This "
                         "is your evaluation data and probe.py's input. Listen "
                         "for false rejects (your speech cut) and leaks (their "
                         "speech through).")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_devices:
        print(sd.query_devices())
        return 0

    dsp.limit_torch_threads()

    try:
        out_channels = parse_channels(args.out_channel)
    except ValueError:
        ap.error(f"--out-channel must be numbers separated by commas, "
                 f"got {args.out_channel!r}")

    names, centroids, enrolled_threshold = store.load(args.speakers)
    threshold = args.threshold if args.threshold is not None else enrolled_threshold
    print(f"enrolled: {names}  threshold={threshold:.3f}")
    print(f"input:  device {args.inp} channel {args.in_channel}")
    print(f"output: device {args.out} channel(s) "
          f"{','.join(str(c) for c in out_channels)}")

    p = Pipeline(names, centroids, threshold,
                 monitor=args.monitor and not args.guide,
                 debug=args.record_debug,
                 in_channel=args.in_channel, out_channels=out_channels)
    p.start_analysis()

    lead_in = 0.0
    with open_stream(args.inp, args.out, args.in_channel, out_channels,
                     p.callback):
        if args.guide:
            lead_in = guide.run(p)
        else:
            print("running - Ctrl+C to stop")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    # Outside the `with`: the stream is closed and the callback has stopped, so
    # the debug writer can drain what is left without racing new frames in.
    p.close()

    if p.skipped_windows:
        print(f"\n{p.skipped_windows} windows skipped as mostly silence")
    if p.short_blocks:
        print(f"\n{p.short_blocks} blocks arrived at the wrong size and "
              f"bypassed the denoiser")
    if p.debug_drops:
        print(f"{p.debug_drops} debug frames dropped; the writer could not "
              f"keep up")

    if args.record_debug:
        print(f"\nwrote {args.record_debug}")
        if args.guide:
            print("\nWhere each phase landed in the file:")
            both_at = None
            for a, b, label in guide.phases(guide.PROBE_SCRIPT):
                print(f"    {guide.mmss(a + lead_in)} - "
                      f"{guide.mmss(b + lead_in)}   {label}")
                if both_at is None and label.startswith("BOTH"):
                    both_at = a + lead_in
            # Skip the first few seconds of the overlap: one person is usually
            # still finishing a sentence while the other starts.
            print("\nRun the probe on the overlap:")
            print(f"  python probe.py --debug-wav {args.record_debug} "
                  f"--start {both_at + 5:.0f} --dur 30")
        else:
            print("Find a stretch where two people talk at once, then:")
            print(f"  python probe.py --debug-wav {args.record_debug} "
                  f"--start <seconds> --dur 30")
    return 0
