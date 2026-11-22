# src/ann/load_texmex.py
import os
import numpy as np


def read_fvecs(filename: str) -> np.ndarray:
    """
    Read TexMex .fvecs file.

    Format:
    [int32 dimension][float32 values...]
    """
    data = np.fromfile(filename, dtype=np.int32)

    if data.size == 0:
        raise ValueError(f"Empty file: {filename}")

    d = data[0]
    if d <= 0:
        raise ValueError(f"Invalid dimension {d} in {filename}")

    data = data.reshape(-1, d + 1)
    vectors = data[:, 1:].view(np.float32)

    return vectors.copy()


def read_ivecs(filename: str) -> np.ndarray:
    """
    Read TexMex .ivecs file.

    Format:
    [int32 dimension][int32 values...]
    """
    data = np.fromfile(filename, dtype=np.int32)

    if data.size == 0:
        raise ValueError(f"Empty file: {filename}")

    d = data[0]
    if d <= 0:
        raise ValueError(f"Invalid dimension {d} in {filename}")

    data = data.reshape(-1, d + 1)
    vectors = data[:, 1:]

    return vectors.copy()


def read_bvecs(filename: str, count: int = -1) -> np.ndarray:
    """
    Read Texmex .bvecs file.
    Each vector is: [int32 dimension][uint8 values...]
    Total bytes per vector: 4 + dimension
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "rb") as f:
        # Read the first 4 bytes to find the dimension d
        d_bytes = f.read(4)
        if len(d_bytes) < 4:
            return np.empty((0, 0), dtype=np.uint8)
        d = int(np.frombuffer(d_bytes, dtype=np.int32)[0])

        record_size = 4 + d

        if count > 0:
            # Check file size to avoid reading out of bounds
            f.seek(0, 2)
            file_size = f.tell()
            max_possible = file_size // record_size
            actual_count = min(count, max_possible)

            f.seek(0)
            raw_data = np.fromfile(f, dtype=np.uint8, count=actual_count * record_size)
        else:
            f.seek(0)
            raw_data = np.fromfile(f, dtype=np.uint8)
            actual_count = raw_data.size // record_size
            raw_data = raw_data[:actual_count * record_size]

    data = raw_data.reshape(actual_count, record_size)
    # slice off the first 4 bytes of each vector (which is the dimension int32)
    # and copy to make the memory contiguous
    return data[:, 4:].copy()


def read_fbin(filename: str, start: int = 0, count: int = -1) -> np.ndarray:
    """
    Read a big-ann-benchmarks .fbin file (used by Deep1B).

    Format:
        [uint32 N][uint32 d]  then  N*d float32 (row-major, no per-vector header)

    Unlike .fvecs the dimension is NOT repeated per vector. Reads a [start, start+count)
    slice without loading the whole file; count<0 reads to the end.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        if header.size < 2:
            raise ValueError(f"Truncated header in {filename}")
        n, d = int(header[0]), int(header[1])

        if start >= n:
            return np.empty((0, d), dtype=np.float32)

        if count < 0:
            count = n - start
        count = min(count, n - start)

        # data begins after the 8-byte header
        f.seek(8 + start * d * 4)
        data = np.fromfile(f, dtype=np.float32, count=count * d)

    return data.reshape(-1, d)


def read_fbin_mmap(filename: str) -> np.ndarray:
    """
    Memory-map a .fbin file as a (N, d) float32 array without reading it into RAM.
    Ideal for the 384GB Deep1B base: index/slice lazily, OS pages in on demand.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    n, d = np.fromfile(filename, dtype=np.uint32, count=2)
    n, d = int(n), int(d)
    return np.memmap(filename, dtype=np.float32, mode="r", offset=8, shape=(n, d))


def read_groundtruth_bin(filename: str):
    """
    Read a big-ann-benchmarks groundtruth .bin file.

    Format:
        [uint32 N][uint32 K]  then  N*K uint32 ids  then  N*K float32 distances

    Returns (ids, dists), each shaped (N, K). dists may be empty if absent.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "rb") as f:
        n, k = np.fromfile(f, dtype=np.uint32, count=2)
        n, k = int(n), int(k)
        ids = np.fromfile(f, dtype=np.uint32, count=n * k).reshape(n, k)
        dists = np.fromfile(f, dtype=np.float32, count=n * k)
        dists = dists.reshape(n, k) if dists.size == n * k else dists

    return ids, dists


def read_bvecs_slice(filename: str, start: int, count: int) -> np.ndarray:
    """
    Read a slice of vectors from a Texmex .bvecs file without loading the whole file.
    Each vector is: [int32 dimension][uint8 values...]
    Total bytes per vector: 4 + dimension
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "rb") as f:
        # Read the first 4 bytes to find the dimension d
        d_bytes = f.read(4)
        if len(d_bytes) < 4:
            return np.empty((0, 0), dtype=np.uint8)
        d = int(np.frombuffer(d_bytes, dtype=np.int32)[0])

        record_size = 4 + d
        offset = start * record_size

        # Check size to avoid reading out of bounds
        f.seek(0, 2)
        file_size = f.tell()
        max_possible = file_size // record_size

        if start >= max_possible:
            return np.empty((0, d), dtype=np.uint8)

        actual_count = min(count, max_possible - start)

        f.seek(offset)
        raw_data = np.fromfile(f, dtype=np.uint8, count=actual_count * record_size)

    data = raw_data.reshape(actual_count, record_size)
    return data[:, 4:].copy()
