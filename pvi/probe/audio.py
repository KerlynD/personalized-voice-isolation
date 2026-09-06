"""Reading captures and enrollment, and getting levels onto a comparable scale."""

import pathlib

import numpy as np
import soundfile as sf

from .. import dsp, tse

# The extraction models trained on LibriMix, which is loudness-normalized. A
# close-mic capture is far hotter than that. Normalizing input RMS to roughly
# what the model saw removes one confound, so that a negative result means
# "wrong acoustic condition" rather than "wrong level".
TARGET_RMS = 0.03

# The auxiliary network averages over the whole enrollment, so longer is
# generally better, but it was trained on 3-second segments. Below the floor
# the speaker embedding is too noisy for the result to mean anything; above the
# ceiling you spend time and memory for no additional speaker information, and
# encoding it costs about 37 ms per 3 seconds.
MIN_ENROLL_SEC = 2.0
MAX_ENROLL_SEC = 20.0


def read_debug_wav(path):
    """Read a pvi.live --record-debug capture. Returns (raw, gated, sr).

    Stereo by contract: L is the raw mic before any gain, R is what actually
    went to the virtual device. `gated` is None for a mono file, which is
    accepted so an ordinary recording can be probed too.
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        print(f"note: {path} is mono; treating it as the raw channel, "
              f"no gate to compare")
        return x[:, 0], None, sr
    if x.shape[1] > 2:
        print(f"note: {path} has {x.shape[1]} channels; using 0 as raw and "
              f"1 as gated")
    return x[:, 0], x[:, 1], sr


def find_enrollment(clip_dir, speaker):
    """Collect the reference clips pvi.enroll saved for this speaker."""
    d = pathlib.Path(clip_dir) / speaker
    clips = sorted(d.glob("*.wav"))
    if not clips:
        raise SystemExit(
            f"no enrollment clips in {d}.\n"
            f"The probe needs a WAVEFORM of {speaker} speaking alone, because "
            f"the extraction model learned its own speaker representation and "
            f"cannot use an ECAPA vector. Either re-run pvi.enroll, which saves "
            f"clips there, or pass --enroll-wav pointing at any clean solo "
            f"recording."
        )
    return clips


def read_enrollment(paths, sr_out=tse.SR):
    """Read reference clips, downmix, concatenate. Returns (audio, n_used).

    Concatenated rather than picked one at a time because the auxiliary network
    averages over time: more material means a steadier speaker embedding, and
    pvi.enroll deliberately varies delivery across clips, so using several
    covers more of how the speaker actually sounds.
    """
    parts, total = [], 0.0
    for p in paths:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = dsp.resample_to(x.mean(axis=1), sr, sr_out)
        parts.append(x)
        total += len(x) / sr_out
        if total >= MAX_ENROLL_SEC:
            break
    x = np.concatenate(parts)[: int(MAX_ENROLL_SEC * sr_out)]

    secs = len(x) / sr_out
    if secs < MIN_ENROLL_SEC:
        print(f"warning: enrollment is {secs:.1f}s, under the "
              f"{MIN_ENROLL_SEC}s floor. The speaker embedding will be noisy "
              f"and a poor result may say more about the enrollment than about "
              f"the model.")
    return x, len(parts)


def rms_normalize(x, target=TARGET_RMS):
    """Scale to a target RMS. Returns (scaled, gain) so it can be undone."""
    g = target / (dsp.rms(x) + 1e-12)
    return (np.asarray(x, dtype=np.float32) * g).astype(np.float32), g


def match_scale(est, ref):
    """Put a scale-invariant estimate back on the reference's scale.

    THIS IS NOT COSMETIC. Ignoring it produced a silent listening test and cost
    this project a wasted round of measurement.

    ConvTasNet-family models trained with SI-SDR loss (scale-invariant
    signal-to-distortion ratio) are free to output any gain: an estimate 1000x
    too loud scores exactly the same as a perfect one, so nothing in training
    ever pushes the output toward the input's scale. The BUT checkpoint settled
    on roughly 350,000x, and its own bundled example does the same, so it is a
    property of the model and not of the audio. The upstream demo notebook
    divides by max() for this reason. The ESPnet backend trains with plain SNR
    and does not need this, but applying it there is harmless.

    The scalar below is the least-squares fit: the value of a minimizing
    ||a*est - ref||. It puts the estimate at the level where it best explains
    the mixture, which is the level the target actually had in that mixture.
    That is what makes "is the interferer quieter than before?" answerable by
    ear.

    Peak normalization would also make the file audible but not comparable: it
    would take the level from whatever single sample happened to be loudest, so
    one click would rescale the whole comparison.
    """
    denom = float(np.dot(est, est))
    if denom <= 1e-20:
        return np.zeros_like(est)
    a = float(np.dot(est, ref)) / denom
    return (a * est).astype(np.float32)
