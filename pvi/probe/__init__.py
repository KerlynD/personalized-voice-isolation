"""
Offline test of a speaker-conditioned extraction model on real recordings.

WHY THIS EXISTS
---------------
The gate in pvi.live mutes other people's turns. It cannot touch a voice that
overlaps yours, because a gate decides WHEN to pass audio and not WHICH PARTS
of it. Fixing that means a speaker-conditioned mask estimator, and there were
two very different ways to get one:

    streaming engineering  take a model that already works on this audio and
                           make it causal, higher rate, and callback-sized
    fine-tuning            take a model that does not work on this audio and
                           retrain it on data that matches the room

This probe decided which. The answer, measured on two real recordings, was
streaming engineering: the released models transfer zero-shot. See
docs/PLAN.md.

    python -m pvi.enroll   ->  speakers.npz + clips/<name>/*.wav
    python -m pvi.live     ->  session.wav  (--record-debug, L=raw, R=gated)
    python -m pvi.probe    ->  probe_out/*.wav, to listen to

It remains useful for any new backend, any new room, and any new microphone.
It is offline, non-causal, 8 kHz and far slower than real time, and none of
that should leak into pvi.live.

WHAT IT WRITES
--------------
    *_mix.wav          the raw mic channel, band-limited to the model's rate
    *_extracted.wav    the model's output, conditioned on you
    *_gate.wav         what the v0.1 gate sent, band-limited the same way
    *_removed.wav      mixture minus extracted: what the model took OUT
    *_control.wav      the same mixture conditioned on somebody ELSE

All at one shared gain so their relative loudness is honest, except _removed,
which is amplified to its own peak because the question is what it IS rather
than how loud.

READING THE RESULT
------------------
Compare _extracted against _mix, never against the original 48 kHz recording.
Both went through the same decimation, so anything you hear between them is the
model. Against the 48 kHz original you will hear everything above 4 kHz missing
and read it as extraction damage. It is not; it is the checkpoint's sample rate.

Find a stretch where both people talk at once. That stretch is the whole
experiment. Every stretch where one person talks tells you nothing you did not
already know from the gate.

Then judge one of four outcomes:

  Interferer ducked, target intact.
      The other voice drops to a murmur under yours and your speech comes
      through unchanged. "Murmur" is the ceiling: single-channel extraction
      makes a competing voice quiet and unintelligible, it does not delete it.
      The next question becomes latency, not modelling.

  Interferer ducked, target damaged.
      The other voice drops but yours acquires warble, lisping or dropouts.
      The conditioning works and the model is fighting the level mismatch.
      Points at fine-tuning on matched data, not a different architecture.

  Nothing ducked.
      Output sounds like the input. Check the control pass before concluding
      anything: if conditioning on somebody else gives a very different answer,
      the model is working and your recording simply did not ask it for
      anything, because the target already dominated. That is what happened at
      +11.6 dB on the first real session. Re-record with both voices at equal
      volume before deciding.

  Wrong speaker extracted.
      The interferer comes through and you are suppressed. Check the enrollment
      first: a clip that is too short, or one with two voices on it, does this
      before the model is at fault.

Whatever you conclude, it is one recording in one room. Record a second session
on a different day before treating the answer as settled.
"""

from .audio import (MAX_ENROLL_SEC, MIN_ENROLL_SEC, TARGET_RMS,
                    find_enrollment, match_scale, read_debug_wav,
                    read_enrollment, rms_normalize)
from .extract import CHUNK_OVERLAP_SEC, CHUNK_SEC, run
from .report import (attribution, centroid_from_clips, print_attribution,
                     print_conditioning, print_verdict, score)

__all__ = [
    "read_debug_wav", "find_enrollment", "read_enrollment", "rms_normalize",
    "match_scale", "TARGET_RMS", "MIN_ENROLL_SEC", "MAX_ENROLL_SEC",
    "run", "CHUNK_SEC", "CHUNK_OVERLAP_SEC",
    "centroid_from_clips", "score", "attribution", "print_attribution",
    "print_verdict", "print_conditioning",
]
