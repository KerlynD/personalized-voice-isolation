"""Command line for the offline extraction probe."""

import argparse
import pathlib

import numpy as np
import soundfile as sf

from .. import dsp, probe as pkg, tse
from ..enroll import store
from . import audio, extract, report

# Peak level of the written files. Below 1.0 so that an unbounded relu mask
# overshooting the mixture does not clip on the way to disk.
OUT_PEAK = 0.89


def build_parser():
    p = argparse.ArgumentParser(
        prog="probe.py",
        description="Offline speaker-conditioned extraction on a real capture. "
                    "Non-causal, 8 kHz, far slower than real time.",
        epilog="Run with --explain for how to read the result.")
    p.add_argument("--debug-wav", help="stereo capture from live.py "
                                       "--record-debug (L=raw, R=gated)")
    p.add_argument("--explain", action="store_true",
                   help="print the four outcomes and what each one implies")
    p.add_argument("--speakers", default="speakers.npz")
    p.add_argument("--speaker", default=None,
                   help="which enrolled speaker to extract; defaults to the first")
    p.add_argument("--clip-dir", default="clips",
                   help="where enroll.py saved the reference recordings")
    p.add_argument("--enroll-wav", default=None,
                   help="override the reference clips with one specific file")
    p.add_argument("--out-dir", default="probe_out")
    p.add_argument("--start", type=float, default=0.0,
                   help="seconds into the capture")
    p.add_argument("--dur", type=float, default=0.0,
                   help="seconds to process, 0 for all")
    p.add_argument("--chunk-sec", type=float, default=extract.CHUNK_SEC,
                   help="0 processes the whole file at once; needs more RAM")
    p.add_argument("--backend", default=tse.DEFAULT, choices=sorted(tse.BACKENDS),
                   help="which extraction model. 'espnet' is Apache-2.0 code "
                        "with CC-BY-4.0 weights and can be shipped; "
                        "'speakerbeam' is the original BUT release and is "
                        "licensed for evaluation only.")
    p.add_argument("--control", default=None, metavar="DIR_OR_WAV",
                   help="run a second pass conditioned on somebody else and "
                        "report how much the two differ. Defaults to "
                        "clips/_impostors when it exists. This separates 'the "
                        "model ignored the enrollment' from 'the model "
                        "correctly passed a target that already dominated'.")
    p.add_argument("--no-control", action="store_true",
                   help="skip the control pass; roughly halves the runtime")
    p.add_argument("--device", default="cpu", help="cpu, or cuda if you have it")
    return p


def _resolve_control(args):
    """Which clips to condition the control pass on, or None."""
    if args.no_control:
        return None
    control = args.control
    if control is None:
        d = pathlib.Path(args.clip_dir) / "_impostors"
        control = str(d) if any(d.glob("*.wav")) else None
    if not control:
        return None
    cp = pathlib.Path(control)
    return sorted(cp.glob("*.wav")) if cp.is_dir() else [cp]


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.explain:
        print(pkg.__doc__)
        return 0
    if not args.debug_wav:
        ap.error("--debug-wav is required (or use --explain)")

    # Deliberately NOT limiting torch threads. That exists to keep ECAPA off
    # the cores the audio callback needs, which only matters while a realtime
    # stream is open. This is an offline batch job with no callback to starve.

    names = cents = None
    if pathlib.Path(args.speakers).exists():
        names, cents, _ = store.load(args.speakers)
    speaker = args.speaker or (names[0] if names else None)
    if speaker is None and args.enroll_wav is None:
        raise SystemExit(
            f"no {args.speakers} and no --enroll-wav. Run enroll.py first, or "
            f"pass --enroll-wav pointing at a clean solo recording.")
    if names and speaker not in names:
        raise SystemExit(f"{speaker!r} is not enrolled in {args.speakers}: {names}")

    raw48, gated48, sr = audio.read_debug_wav(args.debug_wav)
    if sr != dsp.SR:
        print(f"note: capture is {sr} Hz, not {dsp.SR}; resampling anyway")

    # Trim before resampling so the requested window is exact in source samples.
    i0 = int(args.start * sr)
    i1 = len(raw48) if args.dur <= 0 else min(len(raw48), i0 + int(args.dur * sr))
    if i0 >= len(raw48):
        raise SystemExit(f"--start {args.start}s is past the end of a "
                         f"{len(raw48) / sr:.1f}s file")
    raw48, gated48 = raw48[i0:i1], None if gated48 is None else gated48[i0:i1]

    mix = dsp.resample_to(raw48, sr, tse.SR)
    clips = ([pathlib.Path(args.enroll_wav)] if args.enroll_wav
             else audio.find_enrollment(args.clip_dir, speaker))
    enroll, n_used = audio.read_enrollment(clips)
    print(f"target     {speaker or '(from --enroll-wav)'}")
    print(f"mixture    {len(mix) / tse.SR:.1f}s at {tse.SR} Hz")
    print(f"enrollment {len(enroll) / tse.SR:.1f}s from {n_used} of "
          f"{len(clips)} clip(s)")

    mix_n, mix_gain = audio.rms_normalize(mix)
    enroll_n, _ = audio.rms_normalize(enroll)

    model = tse.load(args.backend, device=args.device)
    print("extracting")
    est = audio.match_scale(
        extract.run(model, mix_n, enroll_n, chunk_sec=args.chunk_sec),
        mix_n) / mix_gain
    removed = mix - est

    # One shared gain across every written file. Per-file peak normalization
    # would hide exactly what the probe looks for: whether the interferer got
    # quieter relative to the target.
    peak = max(float(np.abs(mix).max()), float(np.abs(est).max()), 1e-9)
    g = OUT_PEAK / peak
    if g < 1.0:
        print(f"note: scaling all outputs by {g:.2f} to fit; relative levels "
              f"preserved")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pathlib.Path(args.debug_wav).stem}_{args.backend}"

    def write(tag, sig, gain=None):
        f = out_dir / f"{stem}_{tag}.wav"
        sf.write(f, (sig * (g if gain is None else gain)).astype(np.float32),
                 tse.SR)
        return f

    written = [write("mix", mix), write("extracted", est)]
    if gated48 is not None:
        written.append(write("gate", dsp.resample_to(gated48, sr, tse.SR)))
    # Amplified to its own peak: it sits far below the mixture and the question
    # is what it IS, not how loud. If this sounds like the interferer, the
    # model targets the right voice even when little was removed.
    written.append(write("removed", removed,
                         OUT_PEAK / max(float(np.abs(removed).max()), 1e-9)))

    print("\nwrote:")
    for f in written:
        print(f"  {f}")
    print(f"\nextracted vs mixture: correlation "
          f"{np.corrcoef(est, mix)[0, 1]:+.4f}, removed energy "
          f"{dsp.db(dsp.rms(removed), dsp.rms(mix)):+.1f} dB")

    encoder = None

    def enc():
        nonlocal encoder
        if encoder is None:
            encoder = dsp.load_encoder(device=args.device)
        return encoder

    if cents is not None:
        print("\nECAPA similarity (relative only, see pvi.probe.report):")
        s_mix = report.score(enc(), cents, mix)
        s_est = report.score(enc(), cents, est)
        print(f"  {'speaker':<20} {'mixture':>8} {'extracted':>10} {'delta':>8}")
        for i, n in enumerate(names):
            print(f"  {n:<20} {s_mix[i]:8.3f} {s_est[i]:10.3f} "
                  f"{s_est[i] - s_mix[i]:+8.3f}")

    cpaths = _resolve_control(args)
    if cpaths:
        print(f"\ncontrol pass, conditioned on {cpaths[0].parent} instead")
        c_enroll, _ = audio.read_enrollment(cpaths)
        c_n, _ = audio.rms_normalize(c_enroll)
        c_est = audio.match_scale(
            extract.run(model, mix_n, c_n, chunk_sec=args.chunk_sec),
            mix_n) / mix_gain
        print(f"  wrote {write('control', c_est)}")
        report.print_conditioning(float(np.corrcoef(est, c_est)[0, 1]))

        if cents is not None:
            other = report.centroid_from_clips(enc(), cpaths)
            if other is not None:
                rows = report.attribution(
                    enc(), cents[names.index(speaker)], other,
                    {"mix": mix, "extracted": est,
                     "removed": removed, "control": c_est})
                report.print_attribution(rows)
                report.print_verdict(rows)

    print("\nListen to _extracted against _mix, not against the 48 kHz original.")
    print("Find a stretch where two people talk at once; that is the experiment.")
    print("  python probe.py --explain     # the four outcomes")
    return 0
