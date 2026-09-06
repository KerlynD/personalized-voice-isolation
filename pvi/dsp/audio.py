"""
Resampling and window selection. No models, no state, no I/O.

Everything here works on whole arrays rather than streaming frames. That is
deliberate: a polyphase resampler carries filter state across calls, and doing
it per-frame means either threading that state through or accepting an edge
artifact at every block boundary. Whole windows have neither problem and cost
about a millisecond.
"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from .constants import (ANALYSIS_SR, COVERAGE_ABS_RMS, COVERAGE_FLOOR_DB,
                        COVERAGE_FRAME, ENROLL_HOP_SEC, SR, WINDOW_SEC)


def to_analysis(wav48):
    """48 kHz -> 16 kHz, the rate ECAPA-TDNN expects.

    resample_poly applies a proper anti-alias filter. Without one, everything
    above 8 kHz folds back down into the speech band as inharmonic garbage and
    the embedding shifts for reasons that have nothing to do with the speaker.
    """
    return resample_poly(wav48, 1, 3).astype(np.float32)


def resample_to(x, sr_in, sr_out):
    """Rational resample between arbitrary rates.

    48000 -> 8000 reduces to exactly 1/6 and 48000 -> 16000 to 1/3, so no
    approximation is involved on any path this project actually takes. The
    denominator limit only matters for odd device rates.
    """
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    r = Fraction(int(sr_out), int(sr_in)).limit_denominator(1000)
    return resample_poly(x, r.numerator, r.denominator).astype(np.float32)


def analysis_windows(wav48, window_sec=WINDOW_SEC, hop_sec=ENROLL_HOP_SEC):
    """Slice a 48 kHz clip into the same length windows pvi.live embeds.

    Yields views, not copies. A clip shorter than one window yields nothing:
    padding it out to length would put silence into the embedding, and a
    centroid built partly from silence sits between "you" and "nobody", which
    drags the gate open on room tone.
    """
    win = int(window_sec * SR)
    hop = int(hop_sec * SR)
    for start in range(0, max(0, len(wav48) - win + 1), hop):
        yield wav48[start:start + win]


def speech_coverage(wav48):
    """Fraction of a window that is actually speech, 0.0 to 1.0.

    Cheap: frame energies and two comparisons, no model. Call it before paying
    for an embedding. See MIN_COVERAGE for why loudness alone is not enough.

    Used identically by pvi.enroll, to reject windows before they reach the
    centroid, and by pvi.live, to skip a decision rather than make a bad one.
    That shared use is the point: a window enrollment would have thrown away
    must not be one the runtime happily embeds, or the two ends of the pipeline
    disagree about what counts as speech.
    """
    n = len(wav48) // COVERAGE_FRAME
    if n == 0:
        return 0.0
    frames = wav48[:n * COVERAGE_FRAME].reshape(n, COVERAGE_FRAME)
    e = np.sqrt((frames ** 2).mean(axis=1))
    peak = e.max()
    if peak <= 0.0:
        return 0.0
    rel = peak * (10.0 ** (-COVERAGE_FLOOR_DB / 20.0))
    return float(((e > rel) & (e > COVERAGE_ABS_RMS)).mean())


def rms(x):
    """Root mean square. Enough places wanted this to be worth naming."""
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def db(x, ref=1.0):
    """Level in dB relative to `ref`, with a floor so silence does not divide."""
    return 20.0 * np.log10((x + 1e-12) / (ref + 1e-12))
