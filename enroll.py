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

# Which percentile of each distribution counts as its edge. NOT the min and max:
# taking the single worst of ~80 enrollment windows lets one bad window set your
# operating point, and on a real enrollment that one window (the opening of the
# first clip, mostly room tone) dragged the floor from 0.33 down to 0.10 and
# turned a workable +0.12 margin into an unusable -0.11. Percentiles say "ignore
# the worst 5% of my own windows and the best 5% of theirs", which is the right
# question, because the gate does not have to be right on every single window.
# The score live.py compares is smoothed by EMA across several windows anyway.
SELF_PERCENTILE = 5
IMPOSTOR_PERCENTILE = 95

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

# Cycled through the target's clips so the voiceprint covers a range of
# deliveries rather than eight takes of the same sentence at the same volume.
DELIVERY = [
    "normal speaking voice, how you sound on a call",
    "a bit quieter, like it is late and someone is asleep",
    "louder and more animated, like you are excited about something",
    "normal again, but lean back away from the mic",
    "normal, but lean in close to the mic",
    "faster, like you are telling a story you are into",
    "slower and lower, like you are explaining something carefully",
    "normal speaking voice again",
]


def record(seconds, header, instructions, device=None, channel=0):
    """Prompt, then capture `seconds` of one channel of `device`.

    sd.rec's `mapping` counts channels from 1 the way macOS does, while the
    rest of this project counts from 0 the way the arrays do, hence the +1.
    """
    print(f"\n{header}")
    for line in instructions:
        print(f"    {line}")
    input(f"  [Enter] to start recording {seconds:.0f}s > ")
    buf = sd.rec(int(seconds * dsp.SR), samplerate=dsp.SR,
                 dtype="float32", device=device, mapping=[channel + 1])
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

    Returns one embedding per usable dsp.WINDOW_SEC window, shape
    (n_windows, EMB_DIM), plus how many windows were skipped. Denoising happens
    on the whole clip before windowing, which is what live.py effectively does
    too: it denoises continuously and windows the result.

    Windows that are mostly silence are dropped rather than embedded. They
    happen at the start of a clip, before you begin talking, and in the gaps
    between sentences. Their embeddings are not "you speaking quietly", they
    are noise, and averaging them into the centroid moves it toward the room
    rather than toward your voice. live.py applies the same test at runtime.
    """
    clean48 = denoiser.process(wav48)
    embs, skipped = [], 0
    for w in dsp.analysis_windows(clean48):
        if dsp.speech_coverage(w) < dsp.MIN_COVERAGE:
            skipped += 1
            continue
        embs.append(dsp.embed(encoder, dsp.to_analysis(w)))
    if not embs:
        raise ValueError(
            f"no window in this clip was at least {dsp.MIN_COVERAGE:.0%} speech. "
            f"Either the clip is shorter than {dsp.WINDOW_SEC}s, or you were not "
            f"talking for most of it. Talk continuously through the whole clip."
        )
    return np.stack(embs), skipped


def save_clip(clip_dir, name, index, wav48):
    """Write the raw recording to clips/<name>/.

    For the enrolled speaker these are probe.py's reference clips. Impostor
    clips go to clips/_impostors/ and are saved for a different reason: without
    them you cannot re-derive a threshold, re-check a margin, or diagnose a bad
    enrollment without dragging the other person back to the microphone.
    """
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
    # Not required=True: argparse enforces that before we get a chance to
    # handle --list-devices, so `enroll.py --list-devices` would error out
    # asking for a name you cannot know yet.
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
    ap.add_argument("--add", action="store_true",
                    help="append to an existing enrollment instead of replacing")
    ap.add_argument("--out", default="speakers.npz")
    ap.add_argument("--clip-dir", default="clips",
                    help="where to save reference recordings for probe.py")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.name:
        ap.error("--name is required (or use --list-devices)")

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

    # A fresh enrollment replaces the voiceprint, so it has to replace the
    # clips too. probe.py reads every wav in clips/<name>/, so a leftover clip
    # from a previous run would silently become part of the probe's reference
    # audio, which is exactly the kind of stale-data bug that looks like a bad
    # model result.
    if not args.add:
        stale = sorted((pathlib.Path(args.clip_dir) / args.name).glob("*.wav"))
        stale += sorted((pathlib.Path(args.clip_dir) / "_impostors").glob("*.wav"))
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

    print(f"""
================================================================
ENROLLMENT: teaching the filter what your voice looks like
================================================================

You will record two kinds of clip.

  {args.clips} clips of YOU     These build your voiceprint. Only your voice
                       can be on them. If anyone else talks during one,
                       their voice gets averaged into your voiceprint and
                       the filter partly learns to accept them too.

  {args.impostors} clips of THEM   These are the counter-example. They tell the
                       filter how close someone else gets to scoring like
                       you, which is what sets the cutoff. You must be
                       silent on these.

Each {args.seconds:.0f}s clip is cut into {dsp.WINDOW_SEC}s windows, the same length the
live filter looks at, and each window becomes one 192-number voiceprint.
Windows that are less than {dsp.MIN_COVERAGE:.0%} speech are thrown away, so talk
continuously. Long pauses are wasted recording time.

Vary your delivery across your clips: normal, quiet, loud, laughing, leaning
in, leaning back. A voiceprint built from eight identical clips only
recognises you when you sound exactly like that.
================================================================""")

    per_clip = []
    for i in range(args.clips):
        wav = record(
            args.seconds,
            f"[YOU]  clip {i+1} of {args.clips}  --  {args.name} alone",
            ["Only you. Nobody else in the room may speak, at any volume.",
             "Talk continuously for the whole time: read something out loud,",
             "describe your day, count. Do not pause for more than a second.",
             f"Delivery for this one: {DELIVERY[i % len(DELIVERY)]}"],
            device=args.device, channel=args.channel)
        embs, skipped = clip_embeddings(encoder, denoiser, wav)
        if len(embs) < MIN_WINDOWS_PER_CLIP:
            print(f"  skipping clip: only {len(embs)} usable windows, need "
                  f"{MIN_WINDOWS_PER_CLIP}. Talk for the whole {args.seconds:.0f}s.")
            continue
        path = save_clip(args.clip_dir, args.name, i + 1, wav)
        note = f", {skipped} mostly silent" if skipped else ""
        print(f"  {len(embs)} windows{note}, saved {path}")
        per_clip.append(embs)

    if len(per_clip) < 2:
        raise SystemExit("need at least 2 usable clips to enroll")

    mine = np.concatenate(per_clip)
    centroid = mine.mean(0)
    centroid /= np.linalg.norm(centroid)

    self_sims = leave_one_clip_out_sims(per_clip)
    print(f"\n{len(mine)} usable windows from {len(per_clip)} clips")
    print(f"Self-similarity (held out): p{SELF_PERCENTILE}="
          f"{np.percentile(self_sims, SELF_PERCENTILE):.3f} "
          f"min={self_sims.min():.3f} mean={self_sims.mean():.3f}")
    # Reported per clip because a single low clip is actionable (re-record it)
    # while a uniformly low spread is not (it means the mic or room is the
    # problem, or that ECAPA cannot pin your voice down in this environment).
    for i, embs in enumerate(per_clip):
        rest = np.concatenate(per_clip[:i] + per_clip[i + 1:])
        c = rest.mean(0); c /= np.linalg.norm(c) + 1e-9
        m = float((embs @ c).mean())
        flag = "   <-- unlike your other clips" if m < 0.45 else ""
        print(f"    clip {i+1}: mean={m:+.3f}{flag}")

    threshold = DEFAULT_THRESHOLD
    if args.impostors:
        others = []
        for i in range(args.impostors):
            wav = record(
                args.seconds,
                f"[THEM] impostor clip {i+1} of {args.impostors}",
                ["The OTHER person talks. You stay completely silent.",
                 "They sit where they normally sit, at their normal distance.",
                 "Have them talk at a NORMAL to LOUD level, not quietly. The",
                 "cutoff is calibrated against how loud they actually get, so",
                 "a whispered impostor clip sets it too permissive."],
                device=args.device, channel=args.channel)
            embs, _ = clip_embeddings(encoder, denoiser, wav)
            save_clip(args.clip_dir, "_impostors", i + 1, wav)
            others.append(embs)
        other_sims = np.concatenate(others) @ centroid
        hi = float(np.percentile(self_sims, SELF_PERCENTILE))
        lo = float(np.percentile(other_sims, IMPOSTOR_PERCENTILE))
        print(f"Impostor similarity: p{IMPOSTOR_PERCENTILE}={lo:.3f} "
              f"max={other_sims.max():.3f} mean={other_sims.mean():.3f}")

        threshold = lo + THRESHOLD_BIAS * (hi - lo)
        margin = hi - lo
        print(f"Separation margin: {margin:+.3f}  "
              f"(your p{SELF_PERCENTILE}={hi:.3f} vs their "
              f"p{IMPOSTOR_PERCENTILE}={lo:.3f})")
        print(f"Threshold placed {THRESHOLD_BIAS:.0%} of the way from the "
              f"impostor ceiling toward your floor")
        if margin <= 0:
            print("  Distributions overlap, the gate will make errors. Usually "
                  "this means the impostor clips were recorded much quieter "
                  "than yours, or your own clips vary more than the two voices "
                  "differ. More and longer clips of both usually fixes it.")

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
