"""Power-of-Two (PoT) encoder/decoder — golden model for S-ANNS.

Follows the CAL representation: each float32 dimension is approximated by S
signed power-of-two terms, v_i ~= sum_j (-1)^{s_ij} 2^{k_ij}, where each term is
the nearest power of two to the running residual. Exponent indices live in a
contiguous 15-position window [kmin, kmax] (4-bit index, one spare code = null
term), plus one sign bit -> 5 bits/term (signed) or 4 bits/term (unsigned).

Terms are stored compactly as int8 codes: 0 = null, otherwise
sign * (k - kmin + 1) with sign in {+1,-1}.
"""

import numpy as np

NULL = 0


def exponent_window(X, positions=15):
    """Pick the 15-position exponent window [kmin, kmax] from the data.

    kmax covers the largest magnitude coordinate (nearest-PoT rounding can pick
    the exponent above floor(log2|x|), so allow round-up).
    """
    m = np.max(np.abs(X))
    if m == 0:
        return -positions + 1, 0
    kmax = int(np.ceil(np.log2(m * 2.0 / 3.0)))  # nearest-PoT of m never exceeds 2^kmax
    if 2.0 ** kmax < m * 2.0 / 3.0:
        kmax += 1
    kmin = kmax - positions + 1
    return kmin, kmax


def _nearest_pot(r, kmin, kmax):
    """Nearest signed power of two to residual r (elementwise).

    Returns (code, value): code = sign*(k-kmin+1) int8, value = (-1)^s 2^k.
    Residuals closer to 0 than to +-2^kmin become null terms.
    """
    sign = np.where(r < 0, -1, 1).astype(np.int8)
    a = np.abs(r)
    # nearest exponent: k = round(log2 a) with linear-distance tie at 1.5*2^k
    with np.errstate(divide="ignore"):
        kf = np.floor(np.log2(np.maximum(a, np.finfo(np.float32).tiny)))
    up = a > 1.5 * np.exp2(kf)
    k = kf + up
    k = np.clip(k, kmin, kmax)
    val = sign * np.exp2(k).astype(np.float32)
    # null if 0 is at least as close as the nearest representable PoT
    null = a <= np.exp2(float(kmin)) * 0.5
    code = np.where(null, NULL, sign * (k - kmin + 1)).astype(np.int8)
    val = np.where(null, np.float32(0), val).astype(np.float32)
    return code, val


def encode(X, S, kmin, kmax, signed=True):
    """Greedy residual PoT encoding. X: (n, d) float32 -> codes (n, d, S) int8."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    codes = np.empty((n, d, S), dtype=np.int8)
    r = X.copy()
    for j in range(S):
        code, val = _nearest_pot(r, kmin, kmax)
        if not signed:
            # unsigned corpora: negative residual terms are not representable
            neg = code < 0
            code = np.where(neg, NULL, code).astype(np.int8)
            val = np.where(neg, np.float32(0), val)
        codes[:, :, j] = code
        r -= val
    return codes


def decode(codes, kmin, j=None):
    """Reconstruct using the first j terms (j=None -> all)."""
    if j is None:
        j = codes.shape[2]
    c = codes[:, :, :j].astype(np.int32)
    sign = np.sign(c).astype(np.float32)
    k = np.abs(c).astype(np.float32) + (kmin - 1)
    val = sign * np.exp2(k)
    val[c == NULL] = 0.0
    return val.sum(axis=2, dtype=np.float32)
