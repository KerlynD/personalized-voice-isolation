"""
Noise suppression: a thin wrapper over deepfilter-stream.

WHAT THIS DOES AND DOES NOT DO
------------------------------
DeepFilterNet is a NOISE suppressor. It removes fans, keyboards, HVAC and
reverb tail. It is trained to PRESERVE speech, so it will not remove another
person's voice: it treats their speech as signal, because it is. Never describe
it as separation. Concurrent speech is what pvi.tse is for.

Measured cost, from pvi.bench: 0.30 ms per frame at the median and 0.74 ms at
the worst, against a 10.67 ms callback budget. It is 7% of the budget and is
the only model cheap enough to sit in the callback today.
"""

import numpy as np

from .constants import FRAME, ORT_THREADS, SR


class Denoiser:
    """One instance owns one stream, and a stream owns recurrent state.

    Do not share an instance between the audio callback and anything else. The
    underlying library guards its block API with a non-blocking lock, but
    process_frame is deliberately unguarded for speed, so interleaved calls
    corrupt state silently rather than raising.

    Swap this class out to try a different suppressor; the interface the rest
    of the project relies on is process_frame / process / reset.
    """

    def __init__(self, atten_lim_db=None, ort_threads=ORT_THREADS):
        from deepfilter_stream import DeepFilterModel
        self._model = DeepFilterModel(
            intra_op_num_threads=ort_threads,
            inter_op_num_threads=ort_threads,
        )
        self._stream = self._model.new_stream(atten_lim_db=atten_lim_db)
        self.frame_size = self._stream.frame_size  # 512 @ 48 kHz

        # frame_size is read out of the ONNX graph, while pvi.live uses
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

        Carries recurrent state between calls, so do not interleave streams.
        """
        return self._stream.process_frame(frame48)

    def process(self, wav48):
        """Whole-clip path, for enrollment. Resets state afterward.

        Sample-for-sample identical to feeding the same audio through
        process_frame in 512-sample blocks, because that is what it does
        internally. The one difference is that this starts from a cold
        recurrent state each call while pvi.live's stream is always warm, so
        the first frame or two of an enrollment clip is denoised slightly
        differently than it would be mid-call. At six seconds a clip that is
        under 1% of the audio, well inside the noise floor of the similarity
        score.
        """
        out = self._stream.process(wav48, sr=SR)
        tail = self._stream.flush()
        self._stream.reset()
        return np.concatenate([out, tail]).astype(np.float32)

    def reset(self):
        self._stream.reset()
