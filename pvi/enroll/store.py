"""
Reading and writing speakers.npz.

BIOMETRIC DATA. An ECAPA centroid is not reversible into audio, but it is a
stable identifier that matches its owner across recordings, so it is treated
like one: gitignored, never committed, never uploaded.

Format, deliberately plain so it loads with allow_pickle=False:

    names       (n,)         unicode, the labels you enrolled under
    centroids   (n, 192)     unit-length ECAPA centroids
    threshold   scalar       cosine similarity, shared by every enrolled name
"""

import os

import numpy as np

from .. import dsp


def load(path):
    """Returns (names, centroids, threshold). Centroids are renormalized."""
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    mat = np.asarray(z["centroids"], dtype=np.float32)
    if mat.ndim != 2 or mat.shape[1] != dsp.EMB_DIM:
        raise ValueError(
            f"{path} centroids have shape {mat.shape}, "
            f"expected (n, {dsp.EMB_DIM})"
        )
    return names, dsp.normalize(mat), float(z["threshold"])


def save(path, names, centroids, threshold):
    np.savez(path, names=np.array(names),
             centroids=np.stack(centroids), threshold=threshold)


def merge(path, name, centroid, threshold, add):
    """Combine one speaker into an existing enrollment, or start fresh.

    With --add the threshold is the minimum across everyone enrolled, because
    the gate is an allowlist: it must open for whoever is hardest to recognise,
    or that person gets cut off.
    """
    names, centroids = [], []
    if add and os.path.exists(path):
        prev_names, prev_cents, prev_thr = load(path)
        names = list(prev_names)
        centroids = list(prev_cents)
        threshold = min(threshold, prev_thr)
    if name in names:
        centroids[names.index(name)] = centroid
    else:
        names.append(name)
        centroids.append(centroid)
    return names, centroids, threshold
