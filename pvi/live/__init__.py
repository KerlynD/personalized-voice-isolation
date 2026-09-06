"""
The realtime loop. This is the product; everything else supports it.

    mic 48 kHz -> DeepFilterNet3 -> gain -> virtual output -> the call
                        |
                        +-> ring -> decimate to 16 kHz -> ECAPA -> gain target

WHAT THIS GATE CANNOT DO
------------------------
When somebody talks at the same time as you, the gate is open, because you are
talking. Both voices pass. This is not a tuning problem and no threshold fixes
it: a gate decides WHEN to pass audio, not WHICH PARTS of it to pass.
Overlapped speech needs a mask over the mixture, which is pvi.tse.

WHY THE GATE SURVIVES ANYWAY
----------------------------
Measured on a real recording: on the interferer-only phase, the extraction
model attenuates by 1.1 dB where this gate attenuates by 34. TD-SpeakerBeam
trained on Libri2Mix, where the target is present in every mixture, so it was
never taught what to do when the target is silent. The mask handles overlap and
the gate handles absence. They are different problems and the product needs
both. See docs/PLAN.md phase 4.
"""

from .devices import open_stream, parse_channels
from .guide import PROBE_SCRIPT, mmss, phases
from .pipeline import (ANALYZE_EVERY, DEBUG_QUEUE_FRAMES, EMA, FLOOR,
                       HANGOVER, RAMP_MS, VAD_RMS, Pipeline)
from .ring import Ring

__all__ = ["Pipeline", "Ring", "open_stream", "parse_channels",
           "PROBE_SCRIPT", "mmss", "phases", "ANALYZE_EVERY", "RAMP_MS",
           "HANGOVER", "EMA", "FLOOR", "VAD_RMS", "DEBUG_QUEUE_FRAMES"]
