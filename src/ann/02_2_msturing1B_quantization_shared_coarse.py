# src/ann/02_2_msturing1B_quantization_shared_coarse.py
"""
MSTURING1B = Microsoft Turing-ANNS-1B (float32, 100-dim, L2/Euclidean) IVF recall
with a SHARED coarse structure, adapted from 02_2_deep1B_quantization_shared_coarse.py.

Data: big-ann-benchmarks fbin distribution (NeurIPS'21 competition).
  base1b.fbin      header [uint32 N][uint32 d], then N*d float32   (N=1e9, d=100)
  query100K.fbin   same layout                                     (N=1e5, d=100)
  query_gt100.bin  header [uint32 N][uint32 k], then N*k uint32 ids, then N*k float32
                   dists.  read_ibin() below reads ONLY the leading ids block (the
                   trailing dists are ignored) -> id-recall@k works unchanged.

KEY POINTS vs DEEP1B (which this is copied from):
  (a) MSTuring base vectors are UNIT-NORMALIZED (verified on the real data: L2 norm
      = 1.0000, std 0.0000). Like DEEP's unit vectors, L2 and IP therefore rank
      identically for a fixed query, so METRIC_L2 (used below) is correct and matches
      the flat-vs-CSD comparison. Scalar magnitude mean|x| ~= 0.079 -- essentially
      identical to DEEP (~0.08).
  (b) d=100 (vs DEEP's 96). fp16 index ~= 1e9 * 2 * 100 = 200 GB in RAM -> fits the
      436GB machine; run per-config with --single so the OS reclaims RAM between
      configs.
  (c) Because the value distribution matches DEEP (~0.08 scalar, unit norm), DEEP's
      bmin/bmax default of -16/-2 transfers directly and is a sound starting point
      (observed base range ~[-0.54, 0.50]). A quick b-sweep on --max-base 10_000_000
      is still worth running to confirm before the full-1B numbers.

Everything else (fp16-LUT quantization, HNSW shared coarse, cached assignment,
add_core into precomputed lists) is identical to the DEEP1B pipeline.

Original DEEP1B notes retained below for reference:
--------------------------------------------------------------------------------
1. Data is DEEP1B .fbin float32 (header: uint32 N, uint32 d; then N*d float32).
   Values are ~unit-normalized (observed |x| ~0.08 per scalar). Because values are
   CONTINUOUS floats, there is no small integer alphabet -> the 256-entry LUT trick
   used for SIFT/SPACE does not apply directly.
2. fp16-LUT quantization: the index stores fp16 anyway, so we round each base
   scalar to fp16 FIRST (<=65536 distinct values) and gather through a 65536-entry
   LUT indexed by the fp16 bit pattern (uint16). This turns the per-scalar
   brute-force search (searchsorted over thousands of candidates for 1e11 scalars,
   ~2.5 h/config) into a one-time 65536-entry LUT build + a cheap gather
   (~5-10 min/config, ~20x faster). Measured extra error vs exact float32
   quantization: +0% at S=2, +0.23% at S=3, +4% at S=4 (where recall is already
   saturated) -> negligible for the recall targets.
3. Coarse quantizer is HNSW (like SPACE): flat coarse would make the one-time 1B
   assignment take days; HNSW makes it ~minutes and is shared by every config so
   its (tiny) approximation cancels in the flat-vs-CSD comparison.
4. DEEP is float and effectively tie-free -> strict id-recall@10 only (no
   tie-aware). DEEP unit vectors: L2 and IP rank identically, index uses METRIC_L2.

fp16 storage is slightly LOSSY for DEEP (unlike integer SIFT/SPACE), but the fp16
floor MSE (~4.5e-10) is far below the CSD quantization MSE, so it does not distort
the flat-vs-CSD comparison. In-RAM ~192 GB for 1B x 96 dim fp16.

Pipeline (one-time coarse, then per-config):
  1. (once) Train HNSW centroids on float base.
  2. (once) Assign every float base vector to its centroid -> cached int32 array.
  3. (per S) Read base, fp16-LUT quantize, add_core into precomputed lists (no
     coarse search), sweep nprobe with float queries.
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import faiss

from utils import recall_at_k


# ------------------------------------------------------------
# .fbin / .ibin readers (header = uint32 N, uint32 d/k)
# ------------------------------------------------------------
def read_bin_header(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
    return int(n), int(d)


def read_fbin_slice(path, start, count):
    """Float32 slice, shape (count, d). count=None -> to end."""
    n, d = read_bin_header(path)
    if count is None:
        count = n - start
    count = min(count, n - start)
    with open(path, "rb") as f:
        f.seek(8 + start * d * 4)
        a = np.fromfile(f, dtype=np.float32, count=count * d)
    return a.reshape(count, d)


def read_ibin(path, count=None):
    """Int32 matrix (N, k) from .ibin (ground-truth ids)."""
    n, k = read_bin_header(path)
    if count is None:
        count = n
    count = min(count, n)
    with open(path, "rb") as f:
        f.seek(8)
        a = np.fromfile(f, dtype=np.int32, count=count * k)
    return a.reshape(count, k).astype(np.int64)


# ------------------------------------------------------------
# Brute-force CSD candidate generation (same logic as SIFT/SPACE)
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
    candidate, indexed by the fp16 bit pattern (uint16). Returns (lut, build_sec)."""
    t0 = time.time()
    cand = make_pow2_csd_candidates(S, signed, bmin, bmax, term_budget)
    u16 = np.arange(65536, dtype=np.uint16)
    f16 = u16.view(np.float16).astype(np.float32)
    finite = np.isfinite(f16)
    lut = np.zeros(65536, dtype=np.float32)
    lut[finite] = _nearest_candidate(f16[finite], cand)
    return lut, cand.size, time.time() - t0


# ------------------------------------------------------------
# Shared coarse structure (built once, cached on disk) -- HNSW coarse
# ------------------------------------------------------------
def build_fp16_ivf(d, nlist, hnsw_m, hnsw_efc):
    quantizer = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_L2)
    quantizer.hnsw.efConstruction = hnsw_efc
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    index.by_residual = False
    index.cp.niter = 10
    return index


def get_coarse_quantizer(coarse_path, xt, d, nlist, hnsw_m, hnsw_efc):
    if os.path.exists(coarse_path):
        print(f"Loading cached coarse quantizer from {coarse_path}")
        return faiss.read_index(coarse_path)
    print(f"Training HNSW coarse quantizer (nlist={nlist}, M={hnsw_m}) on float xt...")
    t0 = time.time()
    tmpl = build_fp16_ivf(d, nlist, hnsw_m, hnsw_efc)
    tmpl.train(xt)
    print(f"  coarse train_time={time.time()-t0:.1f} sec")
    faiss.write_index(tmpl.quantizer, coarse_path)
    return faiss.read_index(coarse_path)


def get_base_assignment(assign_path, coarse, base_path, nb, chunk_size, hnsw_efsearch):
    if os.path.exists(assign_path):
        print(f"Loading cached base assignment from {assign_path}")
        assign = np.load(assign_path)
        if assign.shape[0] != nb:
            raise ValueError(f"Cached assignment has {assign.shape[0]} rows, expected {nb}. "
                             f"Delete {assign_path} to recompute.")
        return assign
    try:
        faiss.downcast_index(coarse).hnsw.efSearch = hnsw_efsearch
    except Exception:
        pass
    print(f"Computing base->cluster assignment for {nb:,} vectors (one-time)...")
    assign = np.empty(nb, dtype=np.int32)
    t0 = time.time()
    done = 0
    while done < nb:
        cur = min(chunk_size, nb - done)
        xb = read_fbin_slice(base_path, done, cur)
        _, idx = coarse.search(xb, 1)
        assign[done:done + cur] = idx.ravel().astype(np.int32)
        done += cur
        el = time.time() - t0
        eta = el / done * (nb - done)
        print(f"  assigned {done:,}/{nb:,} ({100.0*done/nb:5.1f}%) | "
              f"elapsed={el:7.1f}s | eta={eta:7.1f}s", flush=True)
    np.save(assign_path, assign)
    print(f"  saved assignment to {assign_path} ({assign.nbytes/1e9:.1f} GB)")
    return assign


def add_quantized_base(index, base_path, nb, assign, chunk_size, lut):
    """Read float base, fp16-LUT quantize (lut[fp16_bits(x)]), insert fp16 codes
    into precomputed lists via add_core. lut=None -> store original (fp16 baseline)."""
    t_add0 = time.time()
    done = 0
    while done < nb:
        cur = min(chunk_size, nb - done)
        raw = read_fbin_slice(base_path, done, cur)  # float32 (cur, d)
        if lut is None:
            xb_use = np.ascontiguousarray(raw, dtype=np.float32)
        else:
            u16 = raw.astype(np.float16).view(np.uint16)
            xb_use = np.ascontiguousarray(lut[u16], dtype=np.float32)
        ids = np.arange(done, done + cur, dtype=np.int64)
        pre = assign[done:done + cur].astype(np.int64)
        index.add_core(cur, faiss.swig_ptr(xb_use),
                       faiss.swig_ptr(ids), faiss.swig_ptr(pre))
        done += cur
        el = time.time() - t_add0
        eta = el / done * (nb - done)
        print(f"    added {done:,}/{nb:,} ({100.0*done/nb:5.1f}%) | "
              f"elapsed={el:7.1f}s | eta={eta:7.1f}s", flush=True)
    return time.time() - t_add0


def make_shared_index(coarse_path, d, nlist, hnsw_m, hnsw_efc, hnsw_efsearch):
    index = build_fp16_ivf(d, nlist, hnsw_m, hnsw_efc)
    coarse = faiss.read_index(coarse_path)
    try:
        faiss.downcast_index(coarse).hnsw.efSearch = hnsw_efsearch
    except Exception:
        pass
    index.quantizer = coarse
    index.own_fields = True
    index.is_trained = True
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="msturing1b")
    parser.add_argument("--data-dir", type=str, default="../../data/datasets/msturing1b")
    parser.add_argument("--base-file", type=str, default="base1b.fbin")
    parser.add_argument("--query-file", type=str, default="query100K.fbin")
    parser.add_argument("--gt-file", type=str, default="query_gt100.bin")

    parser.add_argument("--nlist", type=int, default=32768)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--nprobe-list", type=int, nargs="+", default=None,
                        help="Sweep these nprobe on ONE built index per config "
                             "(index built once, re-searched per nprobe). Overrides --nprobe.")
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--S-list", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--modes", type=str, nargs="+", default=["signed"],
                        choices=["positive", "signed"])
    parser.add_argument("--term-budget", type=str, default="at_most",
                        choices=["at_most", "exact"])
    parser.add_argument("--bmin", type=int, default=-16)
    parser.add_argument("--bmax", type=int, default=-2)

    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=40)
    parser.add_argument("--hnsw-ef-search", type=int, default=256)

    parser.add_argument("--max-base", type=int, default=None)
    parser.add_argument("--max-query", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=2_000_000)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--assign-only", action="store_true",
                        help="Compute & cache the coarse quantizer + base assignment, then exit.")
    parser.add_argument("--single", action="store_true",
                        help="Process exactly ONE pending config then exit (code 42 if none).")
    parser.add_argument("--cache-dir", type=str, default="../../data/cache",
                        help="Where to cache coarse + assignment (needs tens of GB "
                             "for the 1B corpora).")
    parser.add_argument("--threads", type=int, default=None,
                        help="Cap OpenMP/faiss threads (shared machine etiquette). "
                             "Default: all cores.")
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads or os.cpu_count() or 31)
    nprobe_list = args.nprobe_list if args.nprobe_list else [args.nprobe]

    base_path = os.path.join(args.data_dir, args.base_file)
    query_path = os.path.join(args.data_dir, args.query_file)
    gt_path = os.path.join(args.data_dir, args.gt_file)
    for p in [base_path, query_path, gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    total_nb, d_val = read_bin_header(base_path)
    nb = min(args.max_base, total_nb) if args.max_base is not None else total_nb

    print("Loading query vectors...")
    xq = read_fbin_slice(query_path, 0, None)
    print("Loading ground truth (ids)...")
    I_gt = read_ibin(gt_path)
    if args.max_query is not None:
        xq = xq[:args.max_query]
        I_gt = I_gt[:args.max_query]
    xq = np.ascontiguousarray(xq, dtype=np.float32)

    cache_dir = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    nb_tag = "full" if args.max_base is None else str(nb)
    coarse_path = os.path.join(cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_hnsw_{nb_tag}.index")
    assign_path = os.path.join(cache_dir, f"assign_{args.dataset}_nlist{args.nlist}_hnsw_{nb_tag}.npy")

    print(f"d                 = {d_val}")
    print(f"base to index     = {nb:,}")
    print(f"nlist / nprobe(s) = {args.nlist} / {nprobe_list}")
    print(f"modes / S         = {args.modes} / {args.S_list}")
    print(f"bmin / bmax       = {args.bmin} / {args.bmax}   term_budget={args.term_budget}")
    print(f"est. RAM per index= {nb * 2 * d_val / 1e9:.1f} GB (fp16)")
    print(f"cache_dir         = {cache_dir}\n")

    # ---- one-time shared coarse structure ----
    print("Loading training vectors (from base)...")
    xt = read_fbin_slice(base_path, 0, max(args.train_size, 40 * args.nlist))
    coarse = get_coarse_quantizer(coarse_path, xt, d_val, args.nlist,
                                  args.hnsw_m, args.hnsw_ef_construction)
    del xt
    gc.collect()
    assign = get_base_assignment(assign_path, coarse, base_path, nb, args.chunk_size,
                                 args.hnsw_ef_search)
    del coarse
    gc.collect()

    if args.assign_only:
        print(f"\n--assign-only: coarse + assignment cached in {cache_dir}. Exiting.")
        return

    out_dir = os.path.join("results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    m_tag = "-".join(args.modes)
    s_tag = "-".join(map(str, args.S_list))
    np_tag = (f"nprobe{args.nprobe}" if not args.nprobe_list
              else f"nprobesweep{min(nprobe_list)}-{max(nprobe_list)}")
    out_path = os.path.join(
        out_dir,
        f"quant_sharedcoarse_brute_force_{m_tag}_S{s_tag}_term{args.term_budget}_"
        f"bmin{args.bmin}_bmax{args.bmax}_nlist{args.nlist}_{np_tag}_{nb_tag}.csv")

    results = []
    done_keys = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path)
        results = prev.to_dict("records")
        done_keys = {(r["quantizer"], r["mode"], int(r["S"])) for r in results}
        print(f"Resuming: {len(results)} existing rows loaded from {out_path}")

    def save():
        pd.DataFrame(results).to_csv(out_path, index=False)

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
                  f"QPS={qps:.2f} | search={st:.2f}s", flush=True)
            out.append((int(p), rec, st, qps))
        return out

    units = []
    if not args.skip_baseline:
        units.append(("original", "float", 0))
    for mode in args.modes:
        for S in args.S_list:
            units.append(("brute_force", mode, S))

    pending = [u for u in units if u not in done_keys]
    print(f"\n{len(done_keys & set(units))}/{len(units)} units done, {len(pending)} pending.")
    if not pending:
        print("All units complete.")
        if args.single:
            sys.exit(42)
        return

    nan_stats = {"mse": float("nan"), "mae": float("nan"),
                 "max_abs_err": float("nan"), "nmse": float("nan")}

    def process_unit(unit):
        qname, mode, S = unit
        signed = mode == "signed"
        if qname == "original":
            print("\n=== Baseline (original float -> fp16, shared coarse) ===")
            index = make_shared_index(coarse_path, d_val, args.nlist,
                                      args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search)
            add_t = add_quantized_base(index, base_path, nb, assign, args.chunk_size, None)
            sweep = search_sweep(index, "original")
            for (p, rec, st, qps) in sweep:
                results.append({
                    "dataset": args.dataset, "nb": nb, "method": "original",
                    "quantizer": "original", "mode": "float", "S": 0,
                    "nlist": args.nlist, "nprobe": p, "bmin": None, "bmax": None,
                    "term_budget": None, "candidate_count": None,
                    f"recall@{args.k}": rec, "search_time_sec": st, "qps": qps,
                    "add_time_sec": add_t, "quant_time_sec": 0.0,
                    "mse": 0.0, "mae": 0.0, "max_abs_err": 0.0, "nmse": 0.0,
                })
        else:
            print(f"\n=== brute_force | {mode} | S={S} ===")
            lut, cand_count, q_t = build_fp16_lut(S, signed, args.bmin, args.bmax, args.term_budget)
            print(f"  candidate_count={cand_count} | fp16-LUT build={q_t:.3f}s")
            index = make_shared_index(coarse_path, d_val, args.nlist,
                                      args.hnsw_m, args.hnsw_ef_construction, args.hnsw_ef_search)
            add_t = add_quantized_base(index, base_path, nb, assign, args.chunk_size, lut)
            sweep = search_sweep(index, f"brute_force/{mode}/S{S}")
            for (p, rec, st, qps) in sweep:
                results.append({
                    "dataset": args.dataset, "nb": nb, "method": "pow2_brute_force",
                    "quantizer": "brute_force", "mode": mode, "S": S,
                    "nlist": args.nlist, "nprobe": p, "bmin": args.bmin, "bmax": args.bmax,
                    "term_budget": args.term_budget, "candidate_count": cand_count,
                    f"recall@{args.k}": rec, "search_time_sec": st, "qps": qps,
                    "add_time_sec": add_t, "quant_time_sec": q_t,
                    **nan_stats,
                })
        save()
        del index
        gc.collect()

    if args.single:
        process_unit(pending[0])
        print(f"\nProcessed 1 unit ({pending[0]}); {len(pending)-1} pending. Saved to {out_path}")
        return

    for unit in pending:
        process_unit(unit)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
