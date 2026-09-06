"""
Numbers shared by every stage. Changing one here changes it everywhere, which
is the point.

CLAUDE.md invariant 5: the sample rates are load-bearing and must not be
unified. The audio path is 48 kHz because that is DeepFilterNet's native rate;
ECAPA-TDNN requires 16 kHz because that is what it was trained at; the
extraction models run at 8 kHz for the same reason. Each conversion is a
deliberate, filtered step, not an accident to be tidied away.
"""

import pathlib

SR = 48000           # audio path, full band
ANALYSIS_SR = 16000  # ECAPA-TDNN was trained at 16 kHz
FRAME = 512          # 10.67 ms at 48 kHz; DeepFilterNet's native frame size
EMB_DIM = 192        # ECAPA-TDNN output dimensionality

# How much audio goes into one identity decision. Speaker identity is not
# decidable from 10 ms; ECAPA needs roughly a second of context to be stable.
# Longer is more accurate but lags harder behind the moment someone starts
# talking, and the gate's hangover has to cover that lag. Shared by pvi.live
# (the window it embeds) and pvi.enroll (the window it enrolls on). Changing it
# in one place only would reintroduce exactly the drift this module prevents.
WINDOW_SEC = 1.5

# Hop between enrollment sub-windows. Smaller gives more embeddings per clip
# and a smoother centroid, at the cost of enrollment time and of neighbouring
# windows being highly correlated, so they add less information than their
# count suggests.
ENROLL_HOP_SEC = 0.5

# A window has to be mostly speech before it is worth embedding. Measured on
# real enrollment audio: overall window loudness barely predicts embedding
# quality (r = +0.05), but the fraction of the window that is actually speech
# predicts it far better (r = +0.36). A window that is 70% room tone still
# clears a plain loudness check, and ECAPA returns a fingerprint of nothing in
# particular for it. Those windows were the three worst in a real enrollment.
#
# Lower this and transitional windows (silence, then you start talking) leak in
# and pollute both the centroid and the runtime score. Raise it and you throw
# away usable audio and react more slowly to speech onset.
MIN_COVERAGE = 0.5

# A 20 ms frame counts as active if it is audible in absolute terms AND within
# 20 dB of the loudest frame in its window. The relative half is what makes
# this work across mic gains: a quiet talker's speech is still 20 dB above
# their own room tone.
COVERAGE_FRAME = 960          # 20 ms at 48 kHz
COVERAGE_FLOOR_DB = 20.0
COVERAGE_ABS_RMS = 0.004      # same absolute floor pvi.live uses as its VAD

# ONNX Runtime thread pool size for the denoiser. One, not the default of "all
# cores": at 512 samples per frame the pool's synchronization costs more than
# the inference it parallelizes, and the extra threads compete with the
# CoreAudio realtime thread. Measured in pvi.bench. See CLAUDE.md invariant 2.
ORT_THREADS = 1

# Downloaded weights land next to the package rather than in the working
# directory, so running any script from another directory does not silently
# re-download 80 MB into a second location.
MODEL_DIR = pathlib.Path(__file__).resolve().parents[2] / "models"
