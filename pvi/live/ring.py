"""The buffer between the audio callback and the analysis thread."""

import numpy as np


class Ring:
    """Circular buffer written by the audio callback, read by the analysis
    thread. A torn read costs at most one frame of stale audio inside a 1.5 s
    window, which is harmless, so no lock.

    Writes are two slice assignments rather than fancy indexing, because the
    callback runs 94 times a second and building an index array each time is an
    allocation the realtime thread does not need to make.
    """

    def __init__(self, n):
        self.buf = np.zeros(n, dtype=np.float32)
        self.n = n
        self.w = 0

    def push(self, x):
        m = len(x)
        if m >= self.n:              # a write larger than the ring: keep the tail
            self.buf[:] = x[-self.n:]
            self.w = 0
            return
        end = self.w + m
        if end <= self.n:
            self.buf[self.w:end] = x
        else:
            k = self.n - self.w
            self.buf[self.w:] = x[:k]
            self.buf[:end - self.n] = x[k:]
        self.w = end % self.n

    def latest(self):
        """Oldest sample first. self.w points at the next write, so everything
        from w to the end is older than everything before w."""
        return np.concatenate([self.buf[self.w:], self.buf[:self.w]])
