# src/ann/02_3_gist1M_mitchell_recall.py
"""
GIST1M: what does Mitchell's logarithmic multiplier cost in RECALL?

Motivation (CAL revision): in the FP32 PE design space, Mitchell's approximation
is the one baseline cheaper than the S-ANN shift PE -- its MAC collapses to a
fraction adder because the FP32 exponent already stores the characteristic. But
every product is underestimated by 0..11.1% (log2(1+f) ~= f drops the f_a*f_b
cross term), the error is data-dependent, and there is no accuracy knob. This
script measures what that does to recall@k.

Methodology (matches 02_2 shared-coarse):
  * ONE coarse quantizer (same cache file as 02_2), so routing is identical for
    every method; candidates per query are the SAME probed lists.
  * Ranking over those candidates:
      exact    : score = ||v||^2 - 2 q.v with exact float products (upper bound;
                 equals true L2 ranking)
      mitchell : same expansion, but every cross product q_i*v_i goes through
                 the Mitchell model; ||v||^2 stays exact (precomputed offline in
                 a PIM, as stored metadata -- this FAVORS Mitchell)
      pot_S    : exact arithmetic on CSD/PoT-quantized base vectors built with
                 the same fp16-LUT as 02_2 (--S/--mode/--bmin/--bmax), i.e. the
                 S-ANN scan, for a same-candidates reference
  * Recall differences between the three columns are purely scan arithmetic /
    storage quantization; IVF effects cancel.

The Mitchell path is elementwise numpy (frexp/ldexp), not BLAS, so it is the
runtime bottleneck: ~1-2 s per query at nlist=4096, nprobe=128. Use
--nprobe-list around the recall-0.9 operating point rather than a dense sweep.

Example:
  python 02_3_gist1M_mitchell_recall.py --data-dir ../../data/datasets/gist1M \
      --nlist 4096 --nprobe-list 8 16 32 64 128 --S 3 --bmin -14 --bmax 0
"""

import argparse
import gc
import os
import time

import faiss
import numpy as np
import pandas as pd

from load_texmex import read_fvecs, read_ivecs
from utils import recall_at_k


# ---------------- CSD candidates + fp16 LUT (same as 02_2) ----------------
def make_pow2_csd_candidates(S, signed, bmin, bmax, term_budget="at_most"):
    if S == 0:
        return np.array([0.0], dtype=np.float32)
    powers = [float(2 ** b) for b in range(bmin, bmax + 1)]
    terms = powers + [-p for p in powers] if signed else powers
    sums_exact_prev = {0.0}
    sums_at_most = {0.0}
    for _ in range(S):
        nxt = set()
        for v in sums_exact_prev:
            for t in terms:
                nxt.add(v + t)
        sums_exact_prev = nxt
        sums_at_most.update(nxt)
    cand = sums_exact_prev if term_budget == "exact" else sums_at_most
    return np.array(sorted(cand), dtype=np.float32)


def _nearest_candidate(vals, cand):
    r = np.clip(np.searchsorted(cand, vals, side="left"), 0, cand.size - 1)
    l = np.clip(r - 1, 0, cand.size - 1)
    cl, cr = cand[l], cand[r]
    return np.where(np.abs(vals - cr) < np.abs(vals - cl), cr, cl).astype(np.float32)


def build_fp16_lut(S, signed, bmin, bmax, term_budget="at_most"):
    cand = make_pow2_csd_candidates(S, signed, bmin, bmax, term_budget)
    u16 = np.arange(65536, dtype=np.uint16)
    f16 = u16.view(np.float16).astype(np.float32)
    finite = np.isfinite(f16)
    lut = np.zeros(65536, dtype=np.float32)
    lut[finite] = _nearest_candidate(f16[finite], cand)
    return lut, cand.size


def fp16_lut_quantize(x, lut):
    u16 = x.astype(np.float16).view(np.uint16)
    return np.ascontiguousarray(lut[u16], dtype=np.float32)


# ---------------- shared coarse (same cache convention as 02_2) ----------------
def build_fp16_ivf(d, nlist):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    index.by_residual = False
    index.cp.niter = 10
    return index


def get_shared_coarse(coarse_path, d, nlist, xt_float):
    if os.path.exists(coarse_path):
        print(f"Loading cached coarse quantizer from {coarse_path}")
        return faiss.read_index(coarse_path)
    tmpl = build_fp16_ivf(d, nlist)
    tmpl.train(np.ascontiguousarray(xt_float, dtype=np.float32))
    faiss.write_index(tmpl.quantizer, coarse_path)
    return faiss.read_index(coarse_path)


# ---------------- Mitchell product model ----------------
def mitchell_prod(a, b):
    """Elementwise Mitchell product: log2(1+f) ~= f in both directions.
    Exact when either significand is a power of two; otherwise underestimates
    by up to 11.1% (worst at f_a = f_b = 0.5)."""
    s = np.sign(a) * np.sign(b)
    ma, ea = np.frexp(np.abs(a))          # |a| = ma * 2^ea, ma in [0.5, 1)
    mb, eb = np.frexp(np.abs(b))
    fa = 2.0 * ma - 1.0                    # fraction of the [1,2) significand
    fb = 2.0 * mb - 1.0
    e = ea + eb - 2
    fsum = fa + fb
    carry = fsum >= 1.0
    mant = np.where(carry, fsum, 1.0 + fsum)
    out = s * np.ldexp(mant.astype(np.float32), e + carry)
    return np.where((a == 0) | (b == 0), np.float32(0.0), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="gist1M")
    ap.add_argument("--data-dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "data", "datasets", "gist1M"))
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe-list", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--S", type=int, default=3, help="PoT reference term count")
    ap.add_argument("--mode", type=str, default="signed", choices=["signed", "positive"])
    ap.add_argument("--bmin", type=int, default=-14)
    ap.add_argument("--bmax", type=int, default=0)
    ap.add_argument("--term-budget", type=str, default="at_most")
    ap.add_argument("--cache-dir", type=str, default="../../data/cache")
    ap.add_argument("--train-size", type=int, default=500_000)
    ap.add_argument("--max-query", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads or os.cpu_count() or 8)

    t0 = time.time()
    xb = np.ascontiguousarray(read_fvecs(os.path.join(args.data_dir, "gist_base.fvecs")),
                              dtype=np.float32)
    xq = np.ascontiguousarray(read_fvecs(os.path.join(args.data_dir, "gist_query.fvecs")),
                              dtype=np.float32)
    xt = read_fvecs(os.path.join(args.data_dir, "gist_learn.fvecs"))[: args.train_size]
    I_gt = read_ivecs(os.path.join(args.data_dir, "gist_groundtruth.ivecs")).astype(np.int64)
    if args.max_query is not None:
        xq, I_gt = xq[: args.max_query], I_gt[: args.max_query]
    nb, d = xb.shape
    nq = xq.shape[0]
    print(f"[{time.time()-t0:6.1f}s] base {xb.shape}, query {xq.shape}")

    # shared coarse (same cache file 02_2 uses -> identical routing)
    os.makedirs(args.cache_dir, exist_ok=True)
    coarse_path = os.path.join(
        args.cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_flat_full.index")
    coarse = get_shared_coarse(coarse_path, d, args.nlist, xt)
    print(f"[{time.time()-t0:6.1f}s] coarse ready (nlist={args.nlist})")

    # bucket base ids by their coarse assignment
    _, assign = coarse.search(xb, 1)
    assign = assign[:, 0]
    order = np.argsort(assign, kind="stable")
    sorted_ids = order.astype(np.int64)
    counts = np.bincount(assign, minlength=args.nlist)
    offsets = np.zeros(args.nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"[{time.time()-t0:6.1f}s] base assigned to lists")

    # PoT reference base (same fp16-LUT as 02_2)
    lut, cand_count = build_fp16_lut(args.S, args.mode == "signed",
                                     args.bmin, args.bmax, args.term_budget)
    xb_pot = fp16_lut_quantize(xb, lut)
    print(f"[{time.time()-t0:6.1f}s] PoT S={args.S}/{args.mode} base built "
          f"(candidates={cand_count})")

    vnorm = (xb.astype(np.float64) ** 2).sum(axis=1)
    vnorm_pot = (xb_pot.astype(np.float64) ** 2).sum(axis=1)

    maxp = max(args.nprobe_list)
    _, probe = coarse.search(xq, maxp)

    out_dir = os.path.join("results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    np_tag = f"{min(args.nprobe_list)}-{max(args.nprobe_list)}"
    out_path = os.path.join(
        out_dir, f"mitchell_{args.mode}_S{args.S}_bmin{args.bmin}_bmax{args.bmax}_"
                 f"nlist{args.nlist}_nprobe{np_tag}.csv")

    results = []
    for npb in args.nprobe_list:
        I_ex = np.full((nq, args.k), -1, dtype=np.int64)
        I_mi = np.full((nq, args.k), -1, dtype=np.int64)
        I_po = np.full((nq, args.k), -1, dtype=np.int64)
        ncand = 0
        t_np = time.time()
        for qi in range(nq):
            lists = probe[qi, :npb]
            cand_ids = np.concatenate(
                [sorted_ids[offsets[l]:offsets[l + 1]] for l in lists])
            ncand += len(cand_ids)
            q = xq[qi]
            cb = xb[cand_ids]

            # exact: ||v||^2 - 2 q.v  (== true L2 ranking)
            s_ex = vnorm[cand_ids] - 2.0 * (cb @ q)
            # mitchell: cross products through the Mitchell model, ||v||^2 exact
            s_mi = vnorm[cand_ids] - 2.0 * mitchell_prod(cb, q[None, :]).sum(axis=1,
                                                                             dtype=np.float64)
            # pot reference: exact arithmetic on S-term quantized vectors
            s_po = vnorm_pot[cand_ids] - 2.0 * (xb_pot[cand_ids] @ q)

            kk = min(args.k, len(cand_ids))
            I_ex[qi, :kk] = cand_ids[np.argsort(s_ex)[:kk]]
            I_mi[qi, :kk] = cand_ids[np.argsort(s_mi)[:kk]]
            I_po[qi, :kk] = cand_ids[np.argsort(s_po)[:kk]]
        dt = time.time() - t_np
        r_ex = recall_at_k(I_ex, I_gt, args.k)
        r_mi = recall_at_k(I_mi, I_gt, args.k)
        r_po = recall_at_k(I_po, I_gt, args.k)
        print(f"nprobe={npb:>4} | exact={r_ex:.5f} | mitchell={r_mi:.5f} | "
              f"pot_S{args.S}={r_po:.5f} | cands/q={ncand // nq} | {dt:.1f}s", flush=True)
        for method, rec in [("exact", r_ex), ("mitchell", r_mi),
                            (f"pot_S{args.S}_{args.mode}", r_po)]:
            results.append({
                "dataset": args.dataset, "nb": nb, "method": method,
                "mode": args.mode, "S": args.S, "nlist": args.nlist, "nprobe": npb,
                "bmin": args.bmin, "bmax": args.bmax,
                f"recall@{args.k}": rec, "cands_per_query": ncand // nq,
                "scan_time_sec": dt,
            })
        pd.DataFrame(results).to_csv(out_path, index=False)
        gc.collect()

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
