# src/ann/02_2_gist1M_quantization_recall_bruteforce.py
"""
GIST1M = TexMex GIST (float32, 960-dim, L2/Euclidean) IVF recall with power-of-two/
CSD brute-force quantization.

GIST is only 1M x 960 (base ~3.6 GB float32, fp16 index ~1.9 GB), so -- unlike the
1B datasets -- everything fits in RAM: NO streaming, NO on-disk assignment cache,
NO shared HNSW coarse. This is the small-dataset analogue of the DEEP/TEXT float
pipeline.

Why fp16-LUT (like DEEP/TEXT), not the 256-entry LUT (like SIFT/SPACE):
  GIST scalars are CONTINUOUS floats, so there is no small integer alphabet. We round
  each base scalar to fp16 first (<=65536 distinct values) and gather its nearest CSD
  candidate through a 65536-entry LUT indexed by the fp16 bit pattern. The stored
  index is fp16 anyway, so this adds no extra error beyond fp16 rounding.

Metric: L2. Verified on the real base (20k sample): vectors are NOT unit-normalized
(L2 norm 2.49 +/- 0.72) and every scalar is NON-NEGATIVE (0% negative, range
[0, ~1.1], mean|x| ~= 0.069) -> keep METRIC_L2.
  * Because values are non-negative, the "positive" mode (+2^b terms only) is the
    natural fit and halves the candidate count vs "signed"; consider --modes positive
    (or both, to compare). Default below is "signed" for parity with the other scripts.
  * DO NOT reuse DEEP's bmin/bmax (-16/-2): GIST's magnitude (~0.069, max ~1.1)
    differs. Defaults -8/0 (2^-8=0.004 .. 2^0=1.0) roughly bracket the range but are
    NOT tuned -- the startup value-distribution diagnostic prints |x| percentiles;
    use it to pick/sweep --bmin/--bmax.

Files (TexMex .fvecs / .ivecs, dimension prefixed per vector):
  gist_base.fvecs        1,000,000 x 960
  gist_query.fvecs           1,000 x 960
  gist_learn.fvecs         500,000 x 960   (used to train the IVF coarse quantizer)
  gist_groundtruth.ivecs     1,000 x 100
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import faiss

from load_texmex import read_fvecs, read_ivecs
from utils import recall_at_k


# ------------------------------------------------------------
# CSD candidate generation + fp16-LUT quantization (same as DEEP/TEXT)
# ------------------------------------------------------------
def make_pow2_csd_candidates(S, signed, bmin, bmax, term_budget="at_most"):
    if S < 0:
        raise ValueError("S must be non-negative.")
    if bmin > bmax:
        raise ValueError("bmin must be <= bmax.")
    if term_budget not in ["at_most", "exact"]:
        raise ValueError("term_budget must be 'at_most' or 'exact'.")
    if S == 0:
        return np.array([0.0], dtype=np.float32)

    powers = [float(2 ** b) for b in range(bmin, bmax + 1)]
    terms = powers + [-p for p in powers] if signed else powers

    sums_exact_prev = {0.0}
    sums_at_most = {0.0}
    for _ in range(S):
        sums_exact_next = set()
        for v in sums_exact_prev:
            for t in terms:
                sums_exact_next.add(v + t)
        sums_exact_prev = sums_exact_next
        sums_at_most.update(sums_exact_prev)

    candidates = sums_exact_prev if term_budget == "exact" else sums_at_most
    return np.array(sorted(candidates), dtype=np.float32)


def _nearest_candidate(vals, cand):
    """Map each scalar in vals to its nearest candidate (vectorized)."""
    r = np.clip(np.searchsorted(cand, vals, side="left"), 0, cand.size - 1)
    l = np.clip(r - 1, 0, cand.size - 1)
    cl, cr = cand[l], cand[r]
    return np.where(np.abs(vals - cr) < np.abs(vals - cl), cr, cl).astype(np.float32)


def build_fp16_lut(S, signed, bmin, bmax, term_budget):
    """65536-entry LUT: quantize every finite fp16 value to its nearest CSD
    candidate, indexed by the fp16 bit pattern (uint16). Returns (lut, cand_count, sec)."""
    t0 = time.time()
    cand = make_pow2_csd_candidates(S, signed, bmin, bmax, term_budget)
    u16 = np.arange(65536, dtype=np.uint16)
    f16 = u16.view(np.float16).astype(np.float32)
    finite = np.isfinite(f16)
    lut = np.zeros(65536, dtype=np.float32)
    lut[finite] = _nearest_candidate(f16[finite], cand)
    return lut, cand.size, time.time() - t0


def fp16_lut_quantize(x, lut):
    """Round to fp16, gather nearest CSD candidate via the LUT. lut=None -> fp16 only."""
    if lut is None:
        return np.ascontiguousarray(x, dtype=np.float32)
    u16 = x.astype(np.float16).view(np.uint16)
    return np.ascontiguousarray(lut[u16], dtype=np.float32)


# ------------------------------------------------------------
# fp16 IVF index with a SHARED coarse quantizer (flat coarse -- 1M is small,
# no HNSW needed). The coarse is trained ONCE on FLOAT training vectors and reused
# by the flat baseline and every CSD config, so flat-vs-CSD recall differences come
# ONLY from base-vector quantization distortion (identical routing) -- this matches
# the 1B shared-coarse methodology (02_2_deep1B_...).
# ------------------------------------------------------------
def build_fp16_ivf(d, nlist):
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    index.by_residual = False
    index.cp.niter = 10
    return index


def get_shared_coarse(coarse_path, d, nlist, xt_float):
    """Train the IVF coarse quantizer ONCE on FLOAT xt and cache it to disk, then
    return a fresh read_index copy (Python-owned -> no lifetime pitfalls). Reused if
    already cached. Writing/reading (not returning tmpl.quantizer directly) avoids the
    dangling-pointer segfault when the training template is garbage-collected."""
    if os.path.exists(coarse_path):
        print(f"Loading cached coarse quantizer from {coarse_path}")
        return faiss.read_index(coarse_path)
    tmpl = build_fp16_ivf(d, nlist)
    tmpl.train(np.ascontiguousarray(xt_float, dtype=np.float32))
    faiss.write_index(tmpl.quantizer, coarse_path)
    return faiss.read_index(coarse_path)


def make_shared_index(coarse, d, nlist, xt_float):
    """Fresh fp16 IVF that REUSES the shared (float-trained) coarse quantizer.
    We still call train() so the fp16 ScalarQuantizer gets its trained state, but
    because `coarse` is already trained (ntotal==nlist) faiss's train_q1 SKIPS
    re-clustering the coarse -> routing (centroids) stays IDENTICAL across all configs.
    own_fields=False so deleting a config's index does not free the shared coarse
    (owned by the caller's `coarse` variable)."""
    index = build_fp16_ivf(d, nlist)
    index.quantizer = coarse
    index.own_fields = False
    index.train(np.ascontiguousarray(xt_float, dtype=np.float32))
    return index


def error_stats(orig, quant):
    err = quant - orig
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    nmse = mse / (float(np.mean(orig ** 2)) + 1e-12)
    return mse, mae, max_abs, nmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gist1M")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "..", "data", "datasets", "gist1M"))
    parser.add_argument("--base-file", type=str, default="gist_base.fvecs")
    parser.add_argument("--query-file", type=str, default="gist_query.fvecs")
    parser.add_argument("--learn-file", type=str, default="gist_learn.fvecs")
    parser.add_argument("--gt-file", type=str, default="gist_groundtruth.ivecs")

    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--nprobe-list", type=int, nargs="+", default=None,
                        help="Sweep these nprobe on ONE built index per config.")
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--S-list", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--modes", type=str, nargs="+", default=["signed"],
                        choices=["positive", "signed"])
    parser.add_argument("--term-budget", type=str, default="at_most",
                        choices=["at_most", "exact"])
    # NOTE: GIST values ~[0, 1.5]; the paper uses --bmin -14 --bmax 0 (see run_gist1m.sh).
    parser.add_argument("--bmin", type=int, default=-8)
    parser.add_argument("--bmax", type=int, default=0)

    parser.add_argument("--cache-dir", type=str, default="../../data/cache",
                        help="Where to cache the shared coarse quantizer.")
    parser.add_argument("--train-size", type=int, default=500_000)
    parser.add_argument("--max-base", type=int, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads or os.cpu_count() or 8)
    nprobe_list = args.nprobe_list if args.nprobe_list else [args.nprobe]

    base_path = os.path.join(args.data_dir, args.base_file)
    query_path = os.path.join(args.data_dir, args.query_file)
    learn_path = os.path.join(args.data_dir, args.learn_file)
    gt_path = os.path.join(args.data_dir, args.gt_file)
    for p in [base_path, query_path, learn_path, gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    print("Loading base / query / learn / gt (all in RAM)...")
    xb = read_fvecs(base_path).astype(np.float32)
    xq = read_fvecs(query_path).astype(np.float32)
    xt = read_fvecs(learn_path).astype(np.float32)[:args.train_size]
    I_gt = read_ivecs(gt_path).astype(np.int64)

    if args.max_base is not None:
        xb = xb[:args.max_base]
    if args.max_query is not None:
        xq = xq[:args.max_query]
        I_gt = I_gt[:args.max_query]
    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)

    nb, d_val = xb.shape

    # --- value-distribution diagnostic to guide bmin/bmax choice ---
    s = xb[: min(nb, 50_000)].reshape(-1)
    pct = np.percentile(np.abs(s[s != 0]), [50, 90, 99, 99.9])
    print("\n[value diagnostic] |x| percentiles over base sample:")
    print(f"  min={s.min():.5f}  max={s.max():.5f}  mean|x|={np.mean(np.abs(s)):.5f}")
    print(f"  |x| p50={pct[0]:.5f}  p90={pct[1]:.5f}  p99={pct[2]:.5f}  p99.9={pct[3]:.5f}")
    print(f"  -> pick bmin/bmax so 2^bmin <= smallest meaningful magnitude and "
          f"2^bmax covers ~p99.9. (current: bmin={args.bmin} -> {2.0**args.bmin:.5f}, "
          f"bmax={args.bmax} -> {2.0**args.bmax:.5f})\n")

    print(f"dataset           = {args.dataset}")
    print(f"xb shape          = {xb.shape}   xq={xq.shape}   xt={xt.shape}   gt={I_gt.shape}")
    print(f"nlist / nprobe(s) = {args.nlist} / {nprobe_list}")
    print(f"modes / S         = {args.modes} / {args.S_list}")
    print(f"bmin / bmax       = {args.bmin} / {args.bmax}   term_budget={args.term_budget}\n")

    out_dir = os.path.join("results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    m_tag = "-".join(args.modes)
    s_tag = "-".join(map(str, args.S_list))
    np_tag = (f"nprobe{args.nprobe}" if not args.nprobe_list
              else f"nprobesweep{min(nprobe_list)}-{max(nprobe_list)}")
    out_path = os.path.join(
        out_dir,
        f"quant_brute_force_{m_tag}_S{s_tag}_term{args.term_budget}_"
        f"bmin{args.bmin}_bmax{args.bmax}_nlist{args.nlist}_{np_tag}.csv")

    # --- shared coarse: trained ONCE on FLOAT xt, reused by flat + all CSD configs ---
    os.makedirs(args.cache_dir, exist_ok=True)
    nb_tag = "full" if args.max_base is None else str(nb)
    coarse_path = os.path.join(
        args.cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_flat_{nb_tag}.index")
    print("Preparing shared coarse quantizer (once, on float xt)...")
    t_c0 = time.time()
    coarse = get_shared_coarse(coarse_path, d_val, args.nlist, xt)
    coarse_train_t = time.time() - t_c0
    print(f"  coarse ready in {coarse_train_t:.1f}s (nlist={args.nlist}, shared by all configs)\n")

    results = []

    def search_sweep(index, label):
        out = []
        for p in nprobe_list:
            index.nprobe = int(p)
            t0 = time.time()
            _, I = index.search(xq, args.k)
            st = time.time() - t0
            rec = recall_at_k(I, I_gt, args.k)
            qps = float(xq.shape[0]) / max(st, 1e-12)
            print(f"  {label} | nprobe={p:>4} | recall@{args.k}={rec:.5f} | "
                  f"QPS={qps:.1f} | search={st:.3f}s", flush=True)
            out.append((int(p), rec, st, qps))
        return out

    def run_config(lut, quant_label, mode, S, cand_count, q_t):
        # Only the BASE vectors are quantized; the coarse (routing) is the shared
        # float-trained one, so flat vs CSD differ only in stored-vector distortion.
        xb_q = fp16_lut_quantize(xb, lut)
        if lut is None:
            mse = mae = maxae = nmse = 0.0
        else:
            mse, mae, maxae, nmse = error_stats(xb, xb_q)

        index = make_shared_index(coarse, d_val, args.nlist, xt)
        train_t = 0.0  # coarse is shared/pretrained once (see coarse_train_t); only SQ retrained
        t0 = time.time()
        index.add(xb_q)
        add_t = time.time() - t0

        sweep = search_sweep(index, quant_label)
        for (p, rec, st, qps) in sweep:
            results.append({
                "dataset": args.dataset, "nb": nb,
                "method": "original" if lut is None else "pow2_brute_force",
                "quantizer": "original" if lut is None else "brute_force",
                "mode": mode, "S": S, "nlist": args.nlist, "nprobe": p,
                "bmin": None if lut is None else args.bmin,
                "bmax": None if lut is None else args.bmax,
                "term_budget": None if lut is None else args.term_budget,
                "candidate_count": cand_count,
                f"recall@{args.k}": rec, "search_time_sec": st, "qps": qps,
                "train_time_sec": train_t, "add_time_sec": add_t, "quant_time_sec": q_t,
                "mse": mse, "mae": mae, "max_abs_err": maxae, "nmse": nmse,
            })
        pd.DataFrame(results).to_csv(out_path, index=False)
        del index
        gc.collect()

    if not args.skip_baseline:
        print("=== Baseline (original float -> fp16) ===")
        run_config(None, "original", "float", 0, None, 0.0)

    for mode in args.modes:
        signed = mode == "signed"
        for S in args.S_list:
            print(f"\n=== brute_force | {mode} | S={S} ===")
            lut, cand_count, q_t = build_fp16_lut(S, signed, args.bmin, args.bmax, args.term_budget)
            print(f"  candidate_count={cand_count} | fp16-LUT build={q_t:.3f}s")
            run_config(lut, f"brute_force/{mode}/S{S}", mode, S, cand_count, q_t)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
