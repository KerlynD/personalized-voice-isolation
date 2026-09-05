"""
Enrollment: build speaker embedding centroids and save reference clips.

WHAT AN EMBEDDING IS
--------------------
ECAPA-TDNN (trained on VoxCeleb) maps speech to a 192-dim vector, trained so
that clips from the same person land close together and clips from different
people land far apart. It does not know who you are - it just produces
coordinates in "speaker space."

Enrollment = record several clips, embed each, average, normalize. That average
is your centroid. At runtime we embed incoming audio and take the cosine
similarity against it.

WHY THIS EMBEDS WINDOWS AND NOT CLIPS
-------------------------------------
live.py never sees a 6-second clip. It sees a rolling dsp.WINDOW_SEC window and
embeds that. ECAPA embeddings are duration-sensitive, so a centroid built from
6-second clips sits at a systematically different distance from 1.5-second
window embeddings than from other 6-second ones. Same code path, different
answer: precisely the silent drift dsp.py's docstring warns about, and it does
not raise anything, it just makes your threshold wrong.

So each clip is sliced into dsp.WINDOW_SEC windows and every window is embedded.
The centroid is the mean over all windows from all clips. Recording longer clips
still helps, because it buys more windows and more variety within a window.

WHAT GETS WRITTEN
-----------------
  speakers.npz          names, centroids, threshold. Biometric data. Gitignored.
  clips/<name>/*.wav    the raw recordings, before denoising

The clips are not a debug artifact. probe.py needs a waveform of you speaking
alone, because TD-SpeakerBeam learned its own speaker representation and cannot
consume an ECAPA vector. Saving them here is what makes the probe runnable
without a second recording session. They are saved raw, pre-denoise, so that
whatever consumes them later can make its own decision about preprocessing.

Run this on the microphone and in the room you'll actually use.

Usage:
    python enroll.py --name angel --device 2 --clips 8 --impostors 4
    python enroll.py --name cohost --device 2 --clips 8 --add
"""

import argparse
import os
import pathlib

import numpy as np
import sounddevice as sd
import soundfile as sf

import dsp

# Where the threshold sits between the impostor ceiling and your own floor.
# 0.0 puts it right at the loudest impostor score, which leaks constantly.
# 1.0 puts it at your own worst window, which cuts you off constantly.
# Biased below the midpoint on purpose: a false reject chops your own sentence
# in half mid-call, which listeners notice immediately and cannot recover from,
# while a leak is a second of someone else's voice that the far end can parse
# around. Raise it if you care more about privacy than about being interrupted.
THRESHOLD_BIAS = 0.35

# Used when no impostor clips were recorded. This is a placeholder, not a
# calibration. Cosine similarity ranges roughly 0.2 to 0.8 in practice and the
# useful value depends on mic and room, so tune it with `live.py --monitor`.
DEFAULT_THRESHOLD = 0.35

# A clip must be at least this many windows long to be worth keeping. Below
# this you are averaging almost the same window with itself.
MIN_WINDOWS_PER_CLIP = 3


def record(seconds, prompt, device=None):
    input(f"\n{prompt}\n  [Enter] to start, then talk for {seconds:.0f}s > ")
    buf = sd.rec(int(seconds * dsp.SR), samplerate=dsp.SR, channels=1,
                 dtype="float32", device=device)
    sd.wait()
    wav = np.ascontiguousarray(buf[:, 0])
    rms = float(np.sqrt(np.mean(wav**2)))
    peak = float(np.max(np.abs(wav)))
    print(f"  rms={rms:.4f} peak={peak:.3f}")
    if rms < 0.005:
        print("  WARNING: very quiet. Check input device and gain.")
    if peak > 0.99:
        print("  WARNING: clipping. Back off the gain or move off-axis.")
    return wav


def clip_embeddings(encoder, denoiser, wav48):
    """The one true preprocessing path, must match live.py exactly.

    Returns one embedding per dsp.WINDOW_SEC window, shape (n_windows, EMB_DIM).
    Denoising happens on the whole clip before windowing, which is what live.py
    effectively does too: it denoises continuously and windows the result.
    """
    clean48 = denoiser.process(wav48)
    embs = [dsp.embed(encoder, dsp.to_analysis(w))
            for w in dsp.analysis_windows(clean48)]
    if not embs:
        raise ValueError(
            f"clip is shorter than one {dsp.WINDOW_SEC}s window after denoising; "
            f"record with --seconds at least {dsp.WINDOW_SEC + 1:.0f}"
        )
    return np.stack(embs)


def save_clip(clip_dir, name, index, wav48):
    """Write the raw recording for probe.py to use as an enrollment reference."""
    d = pathlib.Path(clip_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{index:02d}.wav"
    sf.write(path, wav48, dsp.SR)
    return path


def leave_one_clip_out_sims(per_clip):
    """Honest self-similarity: score each clip against a centroid built without it.

    Scoring a clip against a centroid it helped build is circular, and with a
    handful of clips the inflation is large enough to matter. It pushes
    self_sims.min() up, which pushes the threshold up, which produces a gate
    that cuts you off more than enrollment predicted. Holding each clip out in
    turn removes that. Held out by CLIP and not by window, because windows from
    the same clip share a delivery and a position in front of the mic, so
    leaving out one window of ten leaks nearly everything.
    """
    out = []
    for i, held in enumerate(per_clip):
        rest = np.concatenate(per_clip[:i] + per_clip[i + 1:])
        c = rest.mean(0)
        c /= np.linalg.norm(c) + 1e-9
        out.append(held @ c)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="label for this speaker")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--impostors", type=int, default=0,
                    help="clips of people who should NOT pass the gate")
    ap.add_argument("--device", type=int, default=None,
                    help="input device index; use --list-devices to find it")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--add", action="store_true",
                    help="append to an existing enrollment instead of replacing")
    ap.add_argument("--out", default="speakers.npz")
    ap.add_argument("--clip-dir", default="clips",
                    help="where to save reference recordings for probe.py")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.device is None:
        # Enrolling on one microphone and running on another is the single
        # easiest way to get a gate that never opens, and nothing about it
        # looks like an error. Say which device is about to be used.
        default_in = sd.query_devices(kind="input")
        print(sd.query_devices())
        print(f"\nNo --device given. Recording from the system default: "
              f"{default_in['name']!r}")
        print("If that is not the microphone you will run live.py on, stop now "
              "and pass --device. Enrolling on the wrong mic produces a gate "
              "that silently never opens.")
        input("[Enter] to continue with the default > ")

    dsp.limit_torch_threads()
    encoder = dsp.load_encoder()
    denoiser = dsp.Denoiser()

    print(f"\nEach {args.seconds:.0f}s clip becomes about "
          f"{int((args.seconds - dsp.WINDOW_SEC) / dsp.ENROLL_HOP_SEC) + 1} "
          f"embeddings of {dsp.WINDOW_SEC}s each, the same length live.py uses.")
    print("Vary your delivery across clips: normal, quiet, loud, laughing, "
          "leaning in, leaning back. Uniform enrollment gives a brittle gate.")

    per_clip = []
    for i in range(args.clips):
        wav = record(args.seconds, f"Clip {i+1}/{args.clips} - {args.name}",
                     device=args.device)
        embs = clip_embeddings(encoder, denoiser, wav)
        if len(embs) < MIN_WINDOWS_PER_CLIP:
            print(f"  skipping: only {len(embs)} windows, need "
                  f"{MIN_WINDOWS_PER_CLIP}")
            continue
        path = save_clip(args.clip_dir, args.name, i + 1, wav)
        print(f"  {len(embs)} windows, saved {path}")
        per_clip.append(embs)

    if len(per_clip) < 2:
        raise SystemExit("need at least 2 usable clips to enroll")

    mine = np.concatenate(per_clip)
    centroid = mine.mean(0)
    centroid /= np.linalg.norm(centroid)

    self_sims = leave_one_clip_out_sims(per_clip)
    print(f"\n{len(mine)} windows from {len(per_clip)} clips")
    print(f"Self-similarity (held out): min={self_sims.min():.3f} "
          f"mean={self_sims.mean():.3f}")
    if self_sims.min() < 0.55:
        print("  One clip is an outlier, probably clipped, silent, or noisy. "
              "Consider re-recording.")

    threshold = DEFAULT_THRESHOLD
    if args.impostors:
        others = []
        for i in range(args.impostors):
            wav = record(args.seconds,
                         f"Impostor {i+1}/{args.impostors} - someone else, "
                         f"sitting where they normally sit",
                         device=args.device)
            others.append(clip_embeddings(encoder, denoiser, wav))
        other_sims = np.concatenate(others) @ centroid
        print(f"Impostor similarity: max={other_sims.max():.3f} "
              f"mean={other_sims.mean():.3f}")

        lo, hi = float(other_sims.max()), float(self_sims.min())
        threshold = lo + THRESHOLD_BIAS * (hi - lo)
        margin = hi - lo
        print(f"Separation margin: {margin:+.3f}")
        print(f"Threshold placed {THRESHOLD_BIAS:.0%} of the way from the "
              f"impostor ceiling toward your floor")
        if margin <= 0:
            print("  Distributions overlap, the gate will make errors. More "
                  "and longer enrollment clips usually fixes this.")

    names, centroids = [], []
    if args.add and os.path.exists(args.out):
        prev = np.load(args.out, allow_pickle=False)
        names = [str(n) for n in prev["names"]]
        centroids = list(prev["centroids"])
        threshold = min(threshold, float(prev["threshold"]))
    if args.name in names:
        centroids[names.index(args.name)] = centroid
    else:
        names.append(args.name)
        centroids.append(centroid)

    np.savez(args.out, names=np.array(names), centroids=np.stack(centroids),
             threshold=threshold)
    print(f"\nSaved {args.out}: {list(names)}  threshold={threshold:.3f}")
    print(f"Reference clips in {args.clip_dir}/{args.name}/")
    print("\nNext:")
    print("  python live.py --list-devices")
    print(f"  python live.py --in <mic> --out <cable> --monitor  "
          f"# check the threshold before trusting it")


if __name__ == "__main__":
    main()
