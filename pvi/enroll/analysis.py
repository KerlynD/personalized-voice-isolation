"""
Turning clips into embeddings, and judging whether a clip is any good.

WHY THIS EMBEDS WINDOWS AND NOT CLIPS
-------------------------------------
pvi.live never sees a 6-second clip. It sees a rolling dsp.WINDOW_SEC window
and embeds that. ECAPA embeddings are duration-sensitive, so a centroid built
from 6-second clips sits at a systematically different distance from
1.5-second window embeddings than from other 6-second ones. Same code path,
different answer: the silent drift pvi.dsp exists to prevent. It raises
nothing, it just makes the threshold wrong.

So each clip is sliced into dsp.WINDOW_SEC windows and every window is embedded.
The centroid is the mean over all windows from all clips. Recording longer
clips still helps, because it buys more windows and more variety within one.
"""

import pathlib

import numpy as np
import soundfile as sf

from .. import dsp

# A clip must be at least this many usable windows long to be worth keeping.
# Below this you are averaging almost the same window with itself.
MIN_WINDOWS_PER_CLIP = 3

# How much a clip's own windows must agree with each other. This separates the
# two ways a clip can be unusual, which need opposite responses:
#
#   coherent but far from the others  you talked differently on that clip.
#                                     Keep it. It widens the voiceprint.
#   incoherent                        the windows disagree among themselves,
#                                     because some caught a second voice and
#                                     some did not. Delete it. It drags the
#                                     voiceprint toward whoever else was there.
#
# Measured on real enrollments: clean clips sit at 0.78 to 0.81 regardless of
# how loud or quiet the delivery was, and a clip with someone audible in the
# background came in at 0.71 while scoring 0.19 against the rest. Dropping that
# one clip moved the self-similarity floor from 0.187 to 0.485.
#
# Coherence alone does not condemn a clip. It only explains one that is ALREADY
# far from the others, which is what EXTERNAL_MIN decides. A clip whose windows
# scatter while still sitting on top of the voiceprint is just within-take
# variation, and flagging it cost floor on real data.
COHERENCE_MIN = 0.75
EXTERNAL_MIN = 0.45


def clip_embeddings(encoder, denoiser, wav48):
    """The one true preprocessing path. Must match pvi.live exactly.

    Returns (embeddings, coverages): one embedding per usable window, shape
    (n, EMB_DIM), plus the coverage of every window whether kept or not, so the
    caller can explain a rejection.

    Denoising happens on the whole clip before windowing, which is what
    pvi.live effectively does too: it denoises continuously and windows the
    result.

    Windows that are mostly silence are dropped rather than embedded. They
    happen at the start of a clip, before you begin talking, and in the gaps
    between sentences. Their embeddings are not "you speaking quietly", they
    are the room, and averaging them into the centroid moves it away from your
    voice.

    Returns an empty array rather than raising when nothing is usable. A clip
    that yields nothing is a recording problem, not a program error, and the
    caller is better placed to offer a retry than an exception is to end the
    session with eight good clips already in hand.
    """
    clean48 = denoiser.process(wav48)
    embs, covs = [], []
    for w in dsp.analysis_windows(clean48):
        c = dsp.speech_coverage(w)
        covs.append(c)
        if c < dsp.MIN_COVERAGE:
            continue
        embs.append(dsp.embed(encoder, dsp.to_analysis(w)))
    out = np.stack(embs) if embs else np.zeros((0, dsp.EMB_DIM), np.float32)
    return out, covs


def load_saved_clips(encoder, denoiser, clip_dir, name,
                     min_windows=MIN_WINDOWS_PER_CLIP):
    """Rebuild embeddings from clips a previous run already recorded.

    The clips are raw audio, so re-deriving embeddings from them is exactly
    equivalent to having just recorded them. This is what makes a failed
    enrollment resumable instead of a total loss.
    """
    paths = sorted((pathlib.Path(clip_dir) / name).glob("*.wav"))
    if not paths:
        raise SystemExit(f"no saved clips in {clip_dir}/{name}/ to reuse")
    per_clip, kept = [], []
    for path in paths:
        wav, sr = sf.read(path, dtype="float32")
        if sr != dsp.SR:
            raise SystemExit(f"{path} is {sr} Hz, expected {dsp.SR}")
        embs, covs = clip_embeddings(encoder, denoiser,
                                     np.ascontiguousarray(wav))
        if len(embs) < min_windows:
            print(f"  {path.name}: only {len(embs)} usable windows, skipping")
            continue
        skipped = len(covs) - len(embs)
        note = f", {skipped} mostly silent" if skipped else ""
        print(f"  {path.name}: {len(embs)} usable windows{note}")
        per_clip.append(embs)
        kept.append(path)
    return per_clip, kept


def leave_one_clip_out_sims(per_clip):
    """Honest self-similarity: score each clip against a centroid built without it.

    Scoring a clip against a centroid it helped build is circular, and with a
    handful of clips the inflation is large enough to matter: it raises the
    floor, which raises the threshold, which produces a gate that cuts you off
    more than enrollment predicted.

    Held out by CLIP and not by window, because windows from the same clip
    share a delivery and a position in front of the microphone, so leaving out
    one window of ten leaks nearly everything.
    """
    out = []
    for i, held in enumerate(per_clip):
        rest = np.concatenate(per_clip[:i] + per_clip[i + 1:])
        out.append(held @ dsp.centroid(rest))
    return np.concatenate(out)


def clip_quality(per_clip):
    """Per clip, (internal coherence, similarity to the other clips).

    Two numbers because one cannot tell the two failure modes apart. See
    COHERENCE_MIN.
    """
    rows = []
    for i, embs in enumerate(per_clip):
        internal = float((embs @ dsp.centroid(embs)).mean())
        rest = np.concatenate(per_clip[:i] + per_clip[i + 1:])
        external = float((embs @ dsp.centroid(rest)).mean())
        rows.append((internal, external))
    return rows


def verdict(internal, external):
    """What to tell the operator about one clip. Returns (flag, is_suspect)."""
    if external >= EXTERNAL_MIN:
        return "", False                    # agrees with the rest, nothing to say
    if internal < COHERENCE_MIN:
        return "  <-- another voice or noise on this clip", True
    return "  <-- unusual delivery, but consistent. Keeping it is fine.", False
