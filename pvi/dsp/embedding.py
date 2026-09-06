"""
Speaker embeddings: ECAPA-TDNN, and scoring against enrolled centroids.

WHAT AN EMBEDDING IS
--------------------
ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation
Time-Delay Neural Network), trained on VoxCeleb, maps speech to a 192-number
vector, trained so that clips from the same person land close together and
clips from different people land far apart. It does not know who anyone is. It
produces coordinates in a space where distance means "how differently these two
people sound".

Similarity is cosine similarity, which ranges -1 to +1 and sits roughly 0.2 to
0.8 in practice. It is not a percentage and not a probability. Both vectors are
unit length, so the cosine is a plain dot product.
"""

import numpy as np
import torch

from .constants import MODEL_DIR


def load_encoder(device="cpu"):
    """ECAPA-TDNN speaker embedder. About 80 MB, downloaded on first use."""
    from speechbrain.inference import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(MODEL_DIR / "ecapa"),
        run_opts={"device": device},
    )


def limit_torch_threads(n=1):
    """Keep torch off the audio thread's cores. For pvi.live only.

    ECAPA runs every ANALYZE_EVERY seconds on a background thread. Left to its
    defaults torch spreads that across every core, which is exactly the CPU
    contention that turns into dropouts on the realtime callback. One thread is
    plenty for a 1.5 s window: pvi.bench measures it at 13.5 ms against a
    200 ms budget.

    pvi.enroll and pvi.probe do not call this. Neither has an audio callback
    running while it computes, so throttling them only makes them slower.
    """
    torch.set_num_threads(n)


def embed(encoder, wav16):
    """16 kHz mono float32 -> unit-length speaker embedding."""
    t = torch.from_numpy(np.ascontiguousarray(wav16)).float().unsqueeze(0)
    with torch.no_grad():
        e = encoder.encode_batch(t).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def normalize(v):
    """Unit-length, safe on an all-zero vector."""
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def centroid(embeddings):
    """Mean of a stack of embeddings, renormalized to unit length.

    The mean of unit vectors is not itself unit length, and every comparison
    downstream assumes it is, so this is not optional.
    """
    return normalize(np.asarray(embeddings, dtype=np.float32).mean(axis=0))


def best_score(centroids, e):
    """Cosine similarity against every enrolled speaker; return the best match.

    With one enrolled speaker this is a dot product. With several it is an
    allowlist: the gate opens for anyone enrolled. That is the multi-person
    case, you and a co-host say, as opposed to several people each running
    their own copy, which needs no code change at all.
    """
    sims = centroids @ e
    i = int(np.argmax(sims))
    return float(sims[i]), i
