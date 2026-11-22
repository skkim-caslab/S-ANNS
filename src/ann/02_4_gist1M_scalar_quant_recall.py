# src/ann/02_4_gist1M_scalar_quant_recall.py
"""
GIST1M: recall of scalar-quantized storage (bf16 / fp16 / SQ8) vs PoT and exact.

Compares PoT against simply scalar-quantizing the float32 corpus to int8 or
bf16. Methodology matches 02_3 (shared coarse, identical probed
candidates per query, ranking isolated to the storage representation):

  exact  : float32 base, exact L2 (upper bound)
  bf16   : base rounded to bfloat16 (round-to-nearest-even), 16 bits/dim
  fp16   : base rounded to IEEE fp16, 16 bits/dim
  sq8    : per-dimension affine 256-level uniform quantization
           (faiss QT_8bit style, min/max trained on the learn set), 8 bits/dim
  pot_S  : the S-ANN encoding (same fp16-LUT as 02_2), 5S bits/dim

All methods rank with exact arithmetic on the dequantized vectors, so recall
differences come purely from the stored representation.

Example:
  python 02_4_gist1M_scalar_quant_recall.py --data-dir ../../data/datasets/gist1M \
      --nlist 16384 --nprobe-list 32 64 128 256 512 --S 3
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


# ---------------- CSD candidates + fp16 LUT (same as 02_2/02_3) ----------------
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
    return lut


def fp16_lut_quantize(x, lut):
    u16 = x.astype(np.float16).view(np.uint16)
    return np.ascontiguousarray(lut[u16], dtype=np.float32)


# ---------------- scalar quantize-dequantize models ----------------
def bf16_qdq(x):
    """Round-to-nearest-even to bfloat16, back to float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) \
        & np.uint32(0xFFFF0000)
    return r.view(np.float32).copy()


def fp16_qdq(x):
    return x.astype(np.float16).astype(np.float32)


def sq8_train(xt):
    """Per-dimension min/max over the train set (faiss QT_8bit rangestat)."""
    return xt.min(axis=0).astype(np.float32), xt.max(axis=0).astype(np.float32)


def sq8_qdq(x, vmin, vmax):
    """256-level per-dimension uniform quantization, bin-center reconstruction."""
    scale = (vmax - vmin) / 256.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    code = np.clip(np.floor((x - vmin) / scale), 0, 255)
    return (vmin + (code + 0.5) * scale).astype(np.float32)


def sq8_sym_pervec(x):
    """Symmetric int8, per-vector max-abs scale (applied to DB and queries)."""
    s = np.abs(x).max(axis=1, keepdims=True) / 127.0
    s = np.where(s > 0, s, 1.0).astype(np.float32)
    return (np.clip(np.rint(x / s), -127, 127) * s).astype(np.float32)


def sq8_sym_global(x, s):
    """Symmetric int8, one dataset-wide scale (99.9th-pct clip) shared with
    the query side -- the calibration a bank-level integer MAC PIM needs."""
    return (np.clip(np.rint(x / s), -127, 127) * np.float32(s)).astype(np.float32)


# ---------------- shared coarse (same cache convention as 02_2/02_3) ----------------
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="gist1M")
    ap.add_argument("--data-dir", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "data", "datasets", "gist1M"))
    ap.add_argument("--nlist", type=int, default=16384)
    ap.add_argument("--nprobe-list", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--S-list", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--windows", type=int, nargs="+", default=[15, 7],
                    help="exponent-window positions; 15 -> 4-bit index (5-bit "
                         "signed term), 7 -> 3-bit index (4-bit signed term). "
                         "bmax is fixed at 0, bmin = -(positions-1).")
    ap.add_argument("--mode", type=str, default="signed", choices=["signed", "positive"])
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
    print(f"[{time.time()-t0:6.1f}s] base {xb.shape}, query {xq.shape}", flush=True)

    # storage variants; PoT spans the (S, window) grid -- two orthogonal
    # precision knobs: term count and per-term index width
    vmin, vmax = sq8_train(xt)
    s_gl = float(np.percentile(np.abs(xt), 99.9) / 127.0)
    variants = {
        "exact": xb,
        "bf16": bf16_qdq(xb),
        "fp16": fp16_qdq(xb),
        "sq8": sq8_qdq(xb, vmin, vmax),
        "sq8sym_pv": sq8_sym_pervec(xb),
        "sq8sym_gl": sq8_sym_global(xb, s_gl),
    }
    # symmetric schemes quantize the query too; everything else keeps fp32 queries
    variants["bf16s"] = variants["bf16"]              # bf16 store + bf16 query
    queries = {m: xq for m in variants}
    xq_bf16 = bf16_qdq(xq)
    queries["bf16s"] = xq_bf16
    queries["sq8sym_pv"] = sq8_sym_pervec(xq)
    queries["sq8sym_gl"] = sq8_sym_global(xq, s_gl)
    sign_bit = 1 if args.mode == "signed" else 0
    for S in args.S_list:
        for pos in args.windows:
            idx_bits = max(1, (pos + 1 - 1).bit_length())   # positions + null code
            bits = S * (sign_bit + idx_bits)
            lut = build_fp16_lut(S, args.mode == "signed", -(pos - 1), 0)
            name = f"pot_S{S}w{pos}_{args.mode}_{bits}b"
            variants[name] = fp16_lut_quantize(xb, lut)
            queries[name] = xq
            variants[name + "_qb"] = variants[name]   # same rows, bf16 query
            queries[name + "_qb"] = xq_bf16
    del xt
    gc.collect()
    norms = {m: (v.astype(np.float64) ** 2).sum(axis=1) for m, v in variants.items()}
    for m, v in variants.items():
        err = float(np.abs(v - xb).mean())
        print(f"  variant {m:14s} mean|err|={err:.3e}", flush=True)

    # shared coarse -> identical candidates for every variant
    os.makedirs(args.cache_dir, exist_ok=True)
    coarse_path = os.path.join(
        args.cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_flat_full.index")
    coarse = get_shared_coarse(coarse_path, d, args.nlist, xb[: args.train_size])
    _, assign = coarse.search(xb, 1)
    assign = assign[:, 0]
    order = np.argsort(assign, kind="stable")
    sorted_ids = order.astype(np.int64)
    counts = np.bincount(assign, minlength=args.nlist)
    offsets = np.zeros(args.nlist + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"[{time.time()-t0:6.1f}s] coarse ready, base assigned", flush=True)

    maxp = max(args.nprobe_list)
    _, probe = coarse.search(xq, maxp)

    out_dir = os.path.join("results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    np_tag = f"{min(args.nprobe_list)}-{max(args.nprobe_list)}"
    s_tag = "-".join(map(str, args.S_list))
    w_tag = "-".join(map(str, args.windows))
    out_path = os.path.join(
        out_dir, f"scalarq_grid_{args.mode}_S{s_tag}_w{w_tag}_"
                 f"nlist{args.nlist}_nprobe{np_tag}.csv")

    results = []
    for npb in args.nprobe_list:
        I_pred = {m: np.full((nq, args.k), -1, dtype=np.int64) for m in variants}
        ncand = 0
        t_np = time.time()
        for qi in range(nq):
            cand_ids = np.concatenate(
                [sorted_ids[offsets[l]:offsets[l + 1]] for l in probe[qi, :npb]])
            ncand += len(cand_ids)
            kk = min(args.k, len(cand_ids))
            for m, v in variants.items():
                q = queries[m][qi]
                s = norms[m][cand_ids] - 2.0 * (v[cand_ids] @ q).astype(np.float64)
                I_pred[m][qi, :kk] = cand_ids[np.argsort(s)[:kk]]
        dt = time.time() - t_np
        recs = {m: recall_at_k(I_pred[m], I_gt, args.k) for m in variants}
        print(f"nprobe={npb:>4} | " +
              " | ".join(f"{m}={r:.5f}" for m, r in recs.items()) +
              f" | cands/q={ncand // nq} | {dt:.1f}s", flush=True)
        for m, r in recs.items():
            results.append({
                "dataset": args.dataset, "nb": nb, "method": m, "mode": args.mode,
                "nlist": args.nlist, "nprobe": npb,
                f"recall@{args.k}": r, "cands_per_query": ncand // nq,
                "scan_time_sec": dt,
            })
        pd.DataFrame(results).to_csv(out_path, index=False)
        gc.collect()

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
