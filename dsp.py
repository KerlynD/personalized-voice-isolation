"""
Shared signal path. enroll.py, live.py and probe.py all import from here.

Why this module exists: the audio that produces your enrolled embedding must
travel through the SAME processing chain as the audio at runtime. If you enroll
on raw 16 kHz mic capture but run inference on denoised-and-decimated audio,
the embeddings live in slightly different corners of the space and your
similarity scores are quietly wrong. One code path, no drift.

Signal chain (identical in enrollment and inference):

    mic @ 48 kHz mono
      -> DeepFilterNet3 denoise (48 kHz, causal, ~32 ms latency)
      -> [audio path]     gain -> output device -> Discord
      -> [analysis path]  decimate 48k -> 16k -> ECAPA-TDNN -> 192-dim vector

"Same code path" is necessary but not sufficient. ECAPA embeddings also depend
on how much audio you hand it: a 6-second clip and a 1.5-second window of the
same voice do not land at the same distance from a centroid. That is why
WINDOW_SEC lives here and why enroll.py embeds sub-windows of exactly that
length rather than whole clips. Sharing the constant is part of sharing the path.
"""

import pathlib

import numpy as np
import torch
from scipy.signal import resample_poly

SR = 48000           # audio path, full band
ANALYSIS_SR = 16000  # ECAPA-TDNN was trained at 16 kHz
FRAME = 512          # 10.67 ms at 48 kHz; DeepFilterNet's native frame size
EMB_DIM = 192

# How much audio goes into one identity decision. Speaker identity is not
# decidable from 10 ms; ECAPA needs roughly a second of context to be stable.
# Longer is more accurate but lags harder behind the moment someone starts
# talking, and the gate's hangover has to cover that lag. This value is shared
# by live.py (window it embeds) and enroll.py (window it enrolls on). Changing
# it in one place only would reintroduce exactly the drift this module exists
# to prevent.
WINDOW_SEC = 1.5

# Hop between enrollment sub-windows. Smaller gives more embeddings per clip
# and a smoother centroid, at the cost of enrollment time and of neighbouring
# windows being highly correlated (so they add less information than their
# count suggests).
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
COVERAGE_ABS_RMS = 0.004      # same absolute floor live.py uses as its VAD

# ONNX Runtime thread pool size for the denoiser. One, not the default of
# "all cores": at 512 samples per frame the pool's synchronization costs more
# than the inference it parallelizes, and the extra threads compete with the
# CoreAudio realtime thread. See invariant 2 in CLAUDE.md.
ORT_THREADS = 1

# Downloaded weights land next to this file rather than in the working
# directory, so running any script from another directory does not silently
# re-download 80 MB into a second location.
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "models"


def load_encoder(device="cpu"):
    """ECAPA-TDNN speaker embedder. ~80 MB, downloaded on first use."""
    from speechbrain.inference import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_DIR / "ecapa"),
        run_opts={"device": device},
    )


def limit_torch_threads(n=1):
    """Keep torch off the audio thread's cores. For live.py only.

    ECAPA runs every ANALYZE_EVERY seconds in a background thread. Left to its
    defaults torch will spread that across every core, which is exactly the
    kind of CPU contention that turns into dropouts on the realtime callback.
    One thread is plenty for a 1.5 s window.

    enroll.py and probe.py do not call this. Neither has an audio callback
    running while it computes (enroll.py blocks on sd.wait() before embedding),
    so throttling them would only make them slower.
    """
    torch.set_num_threads(n)


def to_analysis(wav48):
    """48 kHz -> 16 kHz. resample_poly applies a proper anti-alias filter.

    We do this on whole ~1.5s windows rather than per-frame, so there are no
    filter edge artifacts to carry state for. Costs about a millisecond.
    """
    return resample_poly(wav48, 1, 3).astype(np.float32)


def embed(encoder, wav16):
    """16 kHz mono float32 -> unit-length speaker embedding."""
    t = torch.from_numpy(np.ascontiguousarray(wav16)).float().unsqueeze(0)
    with torch.no_grad():
        e = encoder.encode_batch(t).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def analysis_windows(wav48, window_sec=WINDOW_SEC, hop_sec=ENROLL_HOP_SEC):
    """Slice a 48 kHz clip into the same length windows live.py embeds.

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

    Used identically by enroll.py (to reject windows before they reach the
    centroid) and live.py (to skip a decision rather than make a bad one). That
    shared use is the point: a window that enrollment would have thrown away
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


class Denoiser:
    """Thin wrapper over deepfilter-stream so the rest of the code doesn't
    care about the backend. Swap this class out to try DPDFNet or anything
    else - the interface is process_frame / process / reset.

    One instance owns one stream, and a stream owns recurrent state. Do not
    share an instance between the audio callback and anything else: the
    underlying library guards its block API with a non-blocking lock, but
    process_frame is deliberately unguarded for speed, so interleaved calls
    corrupt state silently rather than raising.
    """

    def __init__(self, atten_lim_db=None, ort_threads=ORT_THREADS):
        from deepfilter_stream import DeepFilterModel
        self._model = DeepFilterModel(
            intra_op_num_threads=ort_threads,
            inter_op_num_threads=ort_threads,
        )
        self._stream = self._model.new_stream(atten_lim_db=atten_lim_db)
        self.frame_size = self._stream.frame_size  # 512 @ 48 kHz

        # frame_size is read out of the ONNX graph, while live.py hardcodes
        # dsp.FRAME as the PortAudio blocksize. If a future model ships a
        # different frame, process_frame would raise on every single callback.
        # Fail here, once, with an explanation instead.
        if self.frame_size != FRAME:
            raise RuntimeError(
                f"denoiser wants {self.frame_size}-sample frames but dsp.FRAME "
                f"is {FRAME}. FRAME is the PortAudio blocksize as well, so the "
                f"two have to agree. Update FRAME and re-check the latency "
                f"budget in CLAUDE.md invariant 6."
            )

    def process_frame(self, frame48):
        """Lowest-latency path. Exactly frame_size samples in and out.
        Carries recurrent state between calls - do not interleave streams."""
        return self._stream.process_frame(frame48)

    def process(self, wav48):
        """Whole-clip path, for enrollment. Resets state afterward.

        Sample-for-sample identical to feeding the same audio through
        process_frame in 512-sample blocks, because that is what it does
        internally. The one difference is that this starts from a cold
        recurrent state each call while live.py's stream is always warm, so
        the first frame or two of an enrollment clip is denoised slightly
        differently than it would be mid-call. At 6 seconds a clip that is
        under 1% of the audio and well inside the noise floor of the
        similarity score.
        """
        out = self._stream.process(wav48, sr=SR)
        tail = self._stream.flush()
        self._stream.reset()
        return np.concatenate([out, tail]).astype(np.float32)

    def reset(self):
        self._stream.reset()


def best_score(centroids, e):
    """Cosine similarity against every enrolled speaker; return the best match.

    With one enrolled speaker this is just a dot product. With several, it's an
    allowlist: the gate opens for anyone enrolled. That's the multi-person
    case - you and a co-host, say - as opposed to multiple people each running
    their own copy, which needs no code change at all.
    """
    sims = centroids @ e
    i = int(np.argmax(sims))
    return float(sims[i]), i
