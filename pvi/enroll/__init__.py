"""
Enrollment: build a speaker voiceprint and the threshold that uses it.

WHAT ENROLLMENT PRODUCES
------------------------
  speakers.npz          names, centroids, threshold. Biometric data.
  clips/<name>/*.wav    the raw recordings, before denoising
  clips/_impostors/     the counter-examples, kept for re-deriving a threshold

The clips are not a debug artifact. pvi.probe needs a WAVEFORM of the target
speaking alone, because the extraction models learned their own speaker
representation and cannot consume an ECAPA vector. Saving them here is what
makes a probe run possible without a second recording session.
"""

from .analysis import (COHERENCE_MIN, EXTERNAL_MIN, MIN_WINDOWS_PER_CLIP,
                       clip_embeddings, clip_quality, leave_one_clip_out_sims,
                       load_saved_clips, verdict)
from .record import (DELIVERY, IMPOSTOR_INSTRUCTIONS, TARGET_INSTRUCTIONS,
                     record, save_clip, stale_clips)
from .threshold import (DEFAULT_THRESHOLD, IMPOSTOR_PERCENTILE,
                        SELF_PERCENTILE, THRESHOLD_BIAS, compute)

__all__ = [
    "record", "save_clip", "stale_clips", "DELIVERY",
    "TARGET_INSTRUCTIONS", "IMPOSTOR_INSTRUCTIONS",
    "clip_embeddings", "load_saved_clips", "leave_one_clip_out_sims",
    "clip_quality", "verdict",
    "MIN_WINDOWS_PER_CLIP", "COHERENCE_MIN", "EXTERNAL_MIN",
    "compute", "SELF_PERCENTILE", "IMPOSTOR_PERCENTILE", "THRESHOLD_BIAS",
    "DEFAULT_THRESHOLD",
]
