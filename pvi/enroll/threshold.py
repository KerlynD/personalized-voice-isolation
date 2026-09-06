"""
Turning two distributions of similarity scores into one number.

A voiceprint alone does not tell you how close is close enough. The impostor
clips answer that: they measure where somebody else's voice lands relative to
your centroid. The gap between your floor and their ceiling is the margin, and
the threshold goes inside it.

    your p5     0.485   anything below this and you cut yourself off
                  ^ the margin the threshold must sit in
    their p95   0.222   anything above this and they get through
"""

import numpy as np

# Which percentile of each distribution counts as its edge. NOT the min and
# max: taking the single worst of ~80 enrollment windows lets one bad window
# set your operating point. On a real enrollment one window, the opening of the
# first clip and mostly room tone, dragged the floor from 0.33 to 0.10 and
# turned a workable +0.12 margin into an unusable -0.11.
#
# Percentiles ask the right question, because the gate does not have to be
# correct on every single window: the score pvi.live compares is smoothed by an
# exponential moving average across several of them.
SELF_PERCENTILE = 5
IMPOSTOR_PERCENTILE = 95

# Where the threshold sits between the impostor ceiling and your own floor.
# 0.0 puts it right at their loudest score, which leaks constantly. 1.0 puts it
# at your own worst window, which cuts you off constantly.
#
# Biased below the midpoint on purpose: a false reject chops your own sentence
# in half mid-call, which listeners notice immediately and cannot recover from,
# while a leak is a second of someone else's voice the far end can parse
# around. Raise it if you care more about privacy than about being interrupted.
THRESHOLD_BIAS = 0.35

# Used when no impostor clips were recorded. A placeholder, not a calibration.
# Cosine similarity runs roughly 0.2 to 0.8 in practice and the useful value
# depends on microphone and room, so tune it with `python -m pvi.live --monitor`.
DEFAULT_THRESHOLD = 0.35


def compute(self_sims, other_sims):
    """Threshold and margin from the two score distributions.

    Returns (threshold, margin, floor, ceiling). A margin at or below zero
    means the distributions overlap and no threshold can separate them.
    """
    floor = float(np.percentile(self_sims, SELF_PERCENTILE))
    ceiling = float(np.percentile(other_sims, IMPOSTOR_PERCENTILE))
    return (ceiling + THRESHOLD_BIAS * (floor - ceiling),
            floor - ceiling, floor, ceiling)
