"""Minimal example of the greedy reference PoT encoder in pot.py.

Run: python3 example.py
"""
import numpy as np

import pot


def main():
    # The Fig. 3 value 0.6875 is exact with S = 3 signed terms. Fig. 3 shows
    # 2^-1 + 2^-2 - 2^-4; the greedy encoder finds the equivalent
    # 2^-1 + 2^-3 + 2^-4 (both have zero error).
    x = np.array([[0.6875]], dtype=np.float32)
    kmin, kmax = -14, 0                       # 15-position exponent window
    codes = pot.encode(x, S=3, kmin=kmin, kmax=kmax)
    terms = [(int(np.sign(c)), int(abs(c)) + kmin - 1) for c in codes[0, 0] if c != pot.NULL]
    print("0.6875 ->", " ".join(f"{'+' if s > 0 else '-'}2^{k}" for s, k in terms),
          "=", float(pot.decode(codes, kmin)[0, 0]))

    # relative approximation error and inner-product error vs. S on random data
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 96)).astype(np.float32) * 0.1
    q = rng.standard_normal(96).astype(np.float32)
    kmin, kmax = pot.exponent_window(X)
    print(f"window [kmin, kmax] = [{kmin}, {kmax}]")
    for S in (2, 3, 4, 5):
        codes = pot.encode(X, S, kmin, kmax)
        Xh = pot.decode(codes, kmin)
        rel = np.linalg.norm(X - Xh) / np.linalg.norm(X)
        ip = np.abs(X @ q - Xh @ q).mean() / np.abs(X @ q).mean()
        print(f"S={S}: {5 * S:2d} bits/dim  rel. L2 error {rel:.4f}  mean |dIP|/|IP| {ip:.4f}")


if __name__ == "__main__":
    main()
