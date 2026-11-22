"""Readers for the standard ANNS dataset formats (fvecs/ivecs, fbin/ibin)."""

import numpy as np


def read_fvecs(path, dtype=np.float32):
    data = np.fromfile(path, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view(dtype).copy()


def read_ivecs(path):
    return read_fvecs(path, dtype=np.int32)


def read_fbin(path, dtype=np.float32, count=None):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        if count is not None:
            n = min(n, count)
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d).copy()


def read_ibin(path, count=None):
    return read_fbin(path, dtype=np.int32, count=count)
