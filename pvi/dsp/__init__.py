"""
Shared signal path. Every other package imports from here and none of them
reimplements any part of it.

Why this package exists: the audio that produces your enrolled embedding must
travel through the SAME processing chain as the audio at runtime. If you enroll
on raw mic capture but run inference on denoised-and-decimated audio, the
embeddings live in slightly different corners of the space and your similarity
scores are quietly wrong. One code path, no drift. This is CLAUDE.md invariant
1, and it fails silently every time.

Signal chain, identical in enrollment and inference:

    mic @ 48 kHz mono
      -> DeepFilterNet3 denoise (48 kHz, causal, ~32 ms latency)
      -> [audio path]     gain -> output device -> the call
      -> [analysis path]  decimate 48k -> 16k -> ECAPA-TDNN -> 192-dim vector

"Same code path" is necessary but not sufficient. ECAPA embeddings also depend
on how much audio you hand it: a 6-second clip and a 1.5-second window of the
same voice do not land at the same distance from a centroid. That is why
WINDOW_SEC lives here and why pvi.enroll embeds sub-windows of exactly that
length rather than whole clips. Sharing the constant is part of sharing the
path, and getting this wrong once cost a real enrollment its entire margin.

Flat re-exports below, so `from pvi import dsp; dsp.SR` keeps working and no
caller needs to know which file a name lives in.
"""

from .audio import (analysis_windows, db, resample_to, rms, speech_coverage,
                    to_analysis)
from .constants import (ANALYSIS_SR, COVERAGE_ABS_RMS, COVERAGE_FLOOR_DB,
                        COVERAGE_FRAME, EMB_DIM, ENROLL_HOP_SEC, FRAME,
                        MIN_COVERAGE, MODEL_DIR, ORT_THREADS, SR, WINDOW_SEC)
from .denoise import Denoiser
from .embedding import (best_score, centroid, embed, limit_torch_threads,
                        load_encoder, normalize)

__all__ = [
    "SR", "ANALYSIS_SR", "FRAME", "EMB_DIM", "WINDOW_SEC", "ENROLL_HOP_SEC",
    "MIN_COVERAGE", "COVERAGE_FRAME", "COVERAGE_FLOOR_DB", "COVERAGE_ABS_RMS",
    "ORT_THREADS", "MODEL_DIR",
    "to_analysis", "resample_to", "analysis_windows", "speech_coverage",
    "rms", "db",
    "load_encoder", "limit_torch_threads", "embed", "normalize", "centroid",
    "best_score",
    "Denoiser",
]
