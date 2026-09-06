"""
Scoring probe outputs, and saying what the numbers mean.

READ THIS BEFORE TRUSTING ANY NUMBER HERE
-----------------------------------------
These similarities are comparable to each other and to nothing else. They are
NOT comparable to pvi.live's threshold and must never be used to calibrate it.

CLAUDE.md invariant 1 says enrollment and inference share one preprocessing
path: Denoiser, then to_analysis, then embed. This path differs on both counts.
The audio has been decimated to 8 kHz and upsampled back to 16 kHz, so
everything above 4 kHz is gone and ECAPA sees a spectrum it was not enrolled
on; and the probe deliberately skips the denoiser, because inserting
DeepFilterNet ahead of the extractor would confound the thing being measured.

What the numbers are good for is comparison WITHIN one run: this file against
that file, this speaker against that speaker.
"""

import numpy as np
import soundfile as sf

from .. import dsp, tse


def centroid_from_clips(encoder, paths):
    """Build an ECAPA centroid from raw clips, for the control speaker.

    speakers.npz only holds people who were enrolled. The interferer
    deliberately was not, but scoring against them is what turns "did anything
    change?" into "did THEIR voice get quieter?", so their centroid is derived
    here. Skips the denoiser to match everything else in this module.
    """
    win = int(dsp.WINDOW_SEC * dsp.ANALYSIS_SR)
    hop = int(dsp.ENROLL_HOP_SEC * dsp.ANALYSIS_SR)
    embs = []
    for path in paths:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = dsp.resample_to(x.mean(axis=1), sr, dsp.ANALYSIS_SR)
        for i in range(0, max(1, len(x) - win + 1), hop):
            w = x[i:i + win]
            if len(w) < win:
                break
            embs.append(dsp.embed(encoder, w))
    return dsp.centroid(embs) if embs else None


def score(encoder, centroids, wav, sr=tse.SR):
    """Cosine similarity of a clip against a set of centroids."""
    return centroids @ dsp.embed(encoder, dsp.resample_to(wav, sr, dsp.ANALYSIS_SR))


def attribution(encoder, target_centroid, other_centroid, signals, sr=tse.SR):
    """Score every named signal against both speakers. Returns {tag: (you, them)}."""
    rows = {}
    for tag, sig in signals.items():
        e = dsp.embed(encoder, dsp.resample_to(sig, sr, dsp.ANALYSIS_SR))
        rows[tag] = (float(target_centroid @ e), float(other_centroid @ e))
    return rows


def print_attribution(rows):
    print(f"\n  {'file':<12}{'target':>9}{'other':>9}  leans")
    for tag, (a, b) in rows.items():
        print(f"  {tag:<12}{a:>9.3f}{b:>9.3f}  "
              f"{'target' if a > b else 'OTHER'}")


def print_verdict(rows):
    """The headline question, and it is not "did the interferer get quieter".

    Extraction shifts every speaker's score a little, because cleaning the
    audio up lifts them all. A model that merely attenuates everything lowers
    both. What matters is whether it lowered THEM more than YOU, which is the
    difference between separating two voices and turning both down.
    """
    d_them = rows["mix"][1] - rows["extracted"][1]
    d_you = rows["mix"][0] - rows["extracted"][0]
    print(f"\n  their score fell by {d_them:+.3f}, yours by {d_you:+.3f}")

    if d_them > 0.01 and d_them > 2 * d_you:
        print(f"  Theirs fell {d_them / max(d_you, 1e-6):.1f}x further than "
              f"yours. That is separation: the model is\n  removing them and "
              f"keeping you.")
    elif d_them > 0.01 and d_them > d_you:
        print("  Theirs fell further than yours, so it is separating, but it "
              "is taking\n  some of you with them. Listen to _removed for how "
              "much.")
    elif d_them > 0.01:
        print("  Yours fell as far or further. It is attenuating both voices "
              "rather than\n  separating them, which is not what this is for.")
    else:
        print("  Their voice did not move away from the output. If the "
              "correlation above\n  shows the conditioning working, this "
              "recording did not ask the model to do\n  anything: record one "
              "where both voices are equally loud.")

    removed = rows["removed"]
    if removed[1] > removed[0]:
        print(f"  What it removed leans THEM ({removed[1]:.3f} vs "
              f"{removed[0]:.3f}), the right voice.")
    else:
        print(f"  What it removed leans YOU ({removed[0]:.3f} vs "
              f"{removed[1]:.3f}), the wrong voice.")


def print_conditioning(same):
    """A speaker-conditioned model handed a different speaker must give a
    different answer. If it does not, the conditioning is inert and any
    apparent success is the model passing whoever was loudest."""
    print(f"  target-conditioned vs control-conditioned: correlation {same:+.4f}")
    if same > 0.9:
        print("  The two are nearly identical, so the enrollment is not "
              "changing the answer.\n  Whatever came out is the model passing "
              "the loudest voice, not extracting yours.")
    else:
        print("  The two differ, so the conditioning is doing real work.")
