"""Command line for enrollment. Orchestration only; the logic lives beside it."""

import argparse
import pathlib

import numpy as np
import sounddevice as sd

from .. import dsp
from . import analysis, record as rec, store, threshold as thr

INTRO = """
================================================================
ENROLLMENT: teaching the filter what your voice looks like
================================================================

You will record two kinds of clip.

  {clips} clips of YOU     These build your voiceprint. Only your voice
                       can be on them. If anyone else talks during one,
                       their voice gets averaged into your voiceprint and
                       the filter partly learns to accept them too.

  {impostors} clips of THEM   These are the counter-example. They tell the
                       filter how close someone else gets to scoring like
                       you, which is what sets the cutoff. You must be
                       silent on these.

Each {seconds:.0f}s clip is cut into {window}s windows, the same length the
live filter looks at, and each window becomes one 192-number voiceprint.
Windows that are less than {coverage:.0%} speech are thrown away, so talk
continuously. Long pauses are wasted recording time.

Vary your delivery across your clips: normal, quiet, loud, laughing, leaning
in, leaning back. A voiceprint built from eight identical clips only
recognises you when you sound exactly like that.
================================================================"""


def record_usable(encoder, denoiser, seconds, header, instructions,
                  device, channel, min_windows):
    """Record until the clip is usable, or the operator skips or stops.

    Losing eight good recordings because the ninth was too quiet is not an
    acceptable failure, and nothing here is expensive to redo, so ask.

    Returns (wav, embeddings), or (None, None) if the clip was skipped.
    """
    while True:
        wav = rec.record(seconds, header, instructions,
                         device=device, channel=channel)
        embs, covs = analysis.clip_embeddings(encoder, denoiser, wav)
        skipped = len(covs) - len(embs)
        if len(embs) >= min_windows:
            note = f", {skipped} mostly silent" if skipped else ""
            print(f"  {len(embs)} usable windows{note}")
            return wav, embs

        best = max(covs) if covs else 0.0
        print(f"  NOT USABLE: only {len(embs)} of {len(covs)} windows were at "
              f"least {dsp.MIN_COVERAGE:.0%} speech (need {min_windows}).")
        print(f"  The best window was {best:.0%} speech.")
        if best < 0.2:
            print("  That is close to silence. Check that the right microphone "
                  "is selected and that whoever is talking is actually audible.")
        else:
            print("  Talk continuously for the whole clip, without long pauses, "
                  "and a little louder or closer to the mic.")
        ans = input("  [Enter] to re-record, 's' to skip this clip, "
                    "'a' to abort > ").strip().lower()
        if ans.startswith("s"):
            return None, None
        if ans.startswith("a"):
            raise SystemExit("aborted; clips already recorded are still on disk")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="enroll.py",
        description="Record reference clips and build a speaker voiceprint.")
    # Not required=True: argparse enforces that before --list-devices can be
    # handled, so listing devices would error out asking for a name you cannot
    # know yet.
    ap.add_argument("--name", help="label for this speaker")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--impostors", type=int, default=0,
                    help="clips of people who should NOT pass the gate")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index; use --list-devices to find it")
    ap.add_argument("--channel", type=int, default=0, metavar="N",
                    help="which input channel carries the mic, counting from 0. "
                         "Must match live.py --in-channel, or you enroll on one "
                         "microphone and run on another.")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--from-clips", action="store_true",
                    help="reuse the clips a previous run already saved instead "
                         "of recording new ones. Use this to resume after a "
                         "failed enrollment without re-recording everything.")
    ap.add_argument("--add", action="store_true",
                    help="append to an existing enrollment instead of replacing")
    ap.add_argument("--out", default="speakers.npz")
    ap.add_argument("--clip-dir", default="clips",
                    help="where to save reference recordings for probe.py")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.list_devices:
        print(sd.query_devices())
        return 0
    if not args.name:
        ap.error("--name is required (or use --list-devices)")

    if args.device is None and not args.from_clips:
        # Only when about to record. --from-clips reads audio off disk, so
        # there is no device to get wrong.
        #
        # Enrolling on one microphone and running on another is the easiest way
        # to get a gate that never opens, and nothing about it looks like an
        # error. Say which device is about to be used.
        default_in = sd.query_devices(kind="input")
        print(sd.query_devices())
        print(f"\nNo --device given. Recording from the system default: "
              f"{default_in['name']!r}")
        print("If that is not the microphone you will run live.py on, stop now "
              "and pass --device. Enrolling on the wrong mic produces a gate "
              "that silently never opens.")
        input("[Enter] to continue with the default > ")

    if not args.add and not args.from_clips:
        stale = rec.stale_clips(args.clip_dir, args.name)
        if stale:
            print(f"\n{len(stale)} clip(s) from a previous enrollment will be "
                  f"deleted:")
            for f in stale[:4]:
                print(f"    {f}")
            if len(stale) > 4:
                print(f"    ... and {len(stale) - 4} more")
            print("Move them elsewhere now if you want to keep them.")
            input("[Enter] to delete and continue, Ctrl+C to abort > ")
            for f in stale:
                f.unlink()

    encoder = dsp.load_encoder()
    denoiser = dsp.Denoiser()

    print(INTRO.format(clips=args.clips, impostors=args.impostors,
                       seconds=args.seconds, window=dsp.WINDOW_SEC,
                       coverage=dsp.MIN_COVERAGE))

    if args.from_clips:
        print(f"\nReusing clips already recorded in {args.clip_dir}/{args.name}/")
        per_clip, clip_paths = analysis.load_saved_clips(
            encoder, denoiser, args.clip_dir, args.name)
    else:
        per_clip, clip_paths = [], []
        for i in range(args.clips):
            wav, embs = record_usable(
                encoder, denoiser, args.seconds,
                f"[YOU]  clip {i+1} of {args.clips}  --  {args.name} alone",
                rec.TARGET_INSTRUCTIONS
                + [f"Delivery for this one: {rec.DELIVERY[i % len(rec.DELIVERY)]}"],
                args.device, args.channel, analysis.MIN_WINDOWS_PER_CLIP)
            if wav is None:
                continue
            path = rec.save_clip(args.clip_dir, args.name, i + 1, wav)
            print(f"  saved {path}")
            per_clip.append(embs)
            clip_paths.append(path)

    if len(per_clip) < 2:
        raise SystemExit("need at least 2 usable clips to enroll")

    mine = np.concatenate(per_clip)
    centroid = dsp.centroid(mine)
    self_sims = analysis.leave_one_clip_out_sims(per_clip)

    print(f"\n{len(mine)} usable windows from {len(per_clip)} clips")
    print(f"Self-similarity (held out): p{thr.SELF_PERCENTILE}="
          f"{np.percentile(self_sims, thr.SELF_PERCENTILE):.3f} "
          f"min={self_sims.min():.3f} mean={self_sims.mean():.3f}")

    # Reported per clip because a single bad clip is actionable, delete it,
    # while a uniformly low spread is not: that means the microphone or the
    # room is the problem, not any one recording.
    print(f"    {'clip':<26}{'coherence':>10}{'vs others':>11}")
    suspect = []
    for i, (internal, external) in enumerate(analysis.clip_quality(per_clip)):
        flag, bad = analysis.verdict(internal, external)
        if bad:
            suspect.append(clip_paths[i])
        label = clip_paths[i].name if i < len(clip_paths) else f"clip {i+1}"
        print(f"    {label:<26}{internal:>10.3f}{external:>11.3f}{flag}")

    if suspect:
        print("\n  Those clips have windows that disagree with each other, which "
              "usually\n  means somebody else was audible. To drop them and "
              "rebuild without\n  re-recording anything else:\n")
        for f in suspect:
            print(f"      rm {f}")
        print(f"      python enroll.py --name {args.name} --from-clips "
              f"--device {args.device} --channel {args.channel} "
              f"--impostors {args.impostors}\n")

    threshold = thr.DEFAULT_THRESHOLD
    if args.impostors:
        others = []
        for i in range(args.impostors):
            wav, embs = record_usable(
                encoder, denoiser, args.seconds,
                f"[THEM] impostor clip {i+1} of {args.impostors}",
                rec.IMPOSTOR_INSTRUCTIONS,
                args.device, args.channel, analysis.MIN_WINDOWS_PER_CLIP)
            if wav is None:
                continue
            rec.save_clip(args.clip_dir, "_impostors", i + 1, wav)
            others.append(embs)

        if not others:
            print(f"\nNo usable impostor clips, so no cutoff could be measured. "
                  f"Falling back to the placeholder {thr.DEFAULT_THRESHOLD}; "
                  f"tune it with live.py --monitor.")
        else:
            other_sims = np.concatenate(others) @ centroid
            threshold, margin, floor, ceiling = thr.compute(self_sims, other_sims)
            print(f"Impostor similarity: p{thr.IMPOSTOR_PERCENTILE}={ceiling:.3f} "
                  f"max={other_sims.max():.3f} mean={other_sims.mean():.3f}")
            print(f"Separation margin: {margin:+.3f}  "
                  f"(your p{thr.SELF_PERCENTILE}={floor:.3f} vs their "
                  f"p{thr.IMPOSTOR_PERCENTILE}={ceiling:.3f})")
            print(f"Threshold placed {thr.THRESHOLD_BIAS:.0%} of the way from "
                  f"the impostor ceiling toward your floor")
            if margin <= 0:
                print("  Distributions overlap, the gate will make errors. "
                      "Usually this means the impostor clips were recorded much "
                      "quieter than yours, or your own clips vary more than the "
                      "two voices differ. More and longer clips of both usually "
                      "fixes it.")

    names, centroids, threshold = store.merge(
        args.out, args.name, centroid, threshold, args.add)
    store.save(args.out, names, centroids, threshold)

    print(f"\nSaved {args.out}: {list(names)}  threshold={threshold:.3f}")
    print(f"Reference clips in {args.clip_dir}/{args.name}/")
    print("\nNext:")
    print("  python live.py --list-devices")
    print("  python live.py --in <mic> --out <cable> --monitor  "
          "# check the threshold before trusting it")
    return 0
