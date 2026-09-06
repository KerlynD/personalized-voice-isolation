"""
Target-speech-extraction backends.

Two implementations of the same idea, TD-SpeakerBeam, from two different groups
with very different licences. That difference is why this is a package rather
than a single loader: one of these can ship and one cannot, and the distinction
has to be impossible to lose track of.

    espnet        Apache-2.0 code, CC-BY-4.0 weights. Shippable. The default,
                  and measurably the better separator of the two.
    speakerbeam   BUT/NTT evaluation-only. The reference the project's first
                  result was measured against, kept for comparison only.

Both expose the same interface, so callers do not care which is loaded:

    model = tse.load("espnet")      # a torch Module
    y = model(mix, enroll)          # (B, T) in, (B, T) out, both at tse.SR

ARCHITECTURE, side by side, read out of the two checkpoints:

                        speakerbeam       espnet
    filterbank          512 x 16 / 8      256 x 32 / 16
    TCN blocks          8 x 3             8 x 4
    bottleneck          128               256
    speaker embedding   128               256
    adaptation          multiply at 7     multiply at 7
    sample rate         8000              8000
    causal              no                no
    training loss       SI-SDR            SNR

Neither is vendored. Both are fetched at runtime into gitignored models/.
"""

from . import espnet, speakerbeam

# Both models are 8 kHz. It is a property of the checkpoints, not of the
# method: their filterbanks are learned at that rate, so feeding one 16 kHz
# audio would halve the effective analysis window and put it well outside
# anything it has seen. Band-limiting to 4 kHz is the cost of using released
# weights, and lifting it is phase 2 of docs/PLAN.md.
SR = 8000

BACKENDS = {
    "espnet": (espnet.load, espnet.LICENCE),
    "speakerbeam": (speakerbeam.load, speakerbeam.LICENCE),
}

DEFAULT = "espnet"


def load(name=DEFAULT, device="cpu"):
    """Load a backend by name. Returns a torch Module taking (mix, enroll)."""
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; pick from {sorted(BACKENDS)}")
    fn, licence = BACKENDS[name]
    print(f"backend: {name}  [{licence}]")
    return fn(device=device)
