"""Running an extraction model over a long recording."""

import numpy as np
import torch

from .. import tse
from .audio import match_scale

# Whole-file inference on a ConvTasNet-shaped network allocates activations
# proportional to clip length, so a three-minute capture will exhaust memory on
# a laptop. Chunking is a memory concession, not part of the method.
#
# Chunks are crossfaded rather than butt-joined because global layer norm
# normalizes over the whole sequence: two chunks processed independently land
# at slightly different output levels, and a hard splice steps the gain
# audibly, for the same reason RAMP_MS exists in pvi.live.
CHUNK_SEC = 20.0
CHUNK_OVERLAP_SEC = 1.0


def run(model, mix, enroll, chunk_sec=CHUNK_SEC, overlap_sec=CHUNK_OVERLAP_SEC,
        sr=tse.SR, verbose=True):
    """Extract the enrolled speaker from `mix`. Both arrays at `sr`.

    The enrollment is passed to every chunk rather than embedded once. That
    costs about 37 ms per 3 seconds of reference audio per chunk, which is
    real: it is most of why a probe run is slow. It is kept because the two
    backends expose different internals, and phase 1 of docs/PLAN.md replaces
    this path with a cached embedding anyway.
    """
    enroll_t = torch.from_numpy(np.ascontiguousarray(enroll)).float().unsqueeze(0)

    def one(seg):
        seg_t = torch.from_numpy(np.ascontiguousarray(seg)).float().unsqueeze(0)
        with torch.no_grad():
            y = model(seg_t, enroll_t)
        y = y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        # Per chunk, against that chunk's own mixture. Each chunk is a separate
        # forward pass, and a scale-invariant model returns them at unrelated
        # levels. Crossfading those would splice a step into the output;
        # matching here means the crossfade joins signals already on one scale.
        return match_scale(y, np.asarray(seg, dtype=np.float32))

    n = len(mix)
    if chunk_sec <= 0 or n <= int(chunk_sec * sr):
        return one(mix)[:n]

    chunk = int(chunk_sec * sr)
    overlap = max(1, int(overlap_sec * sr))
    hop = chunk - overlap
    if hop <= 0:
        raise ValueError("chunk_sec must be larger than overlap_sec")

    # Build the starts up front so the tail folds into the last chunk. A stub
    # final chunk of a few hundred samples is shorter than the network's
    # receptive field and produces garbage.
    starts = list(range(0, max(1, n - overlap), hop))
    while len(starts) > 1 and n - starts[-1] < overlap * 2:
        starts.pop()

    out = np.zeros(n, dtype=np.float32)
    fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    for i, s in enumerate(starts):
        e = n if i == len(starts) - 1 else min(n, s + chunk)
        y = one(mix[s:e])[: e - s]
        if verbose:
            print(f"  chunk {i + 1}/{len(starts)}  "
                  f"{s / sr:6.1f}s - {e / sr:6.1f}s")
        if i == 0:
            out[s:e] = y
            continue
        k = min(overlap, len(y))
        out[s:s + k] = out[s:s + k] * (1.0 - fade[:k]) + y[:k] * fade[:k]
        out[s + k:e] = y[k:]
    return out
