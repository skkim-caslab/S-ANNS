# src/ann/02_3_1B_mitchell_recall.py
"""
DEEP1B / MSTuring1B: recall cost of Mitchell's logarithmic multiplier at 1B scale.

Companion to 02_3_gist1M_mitchell_recall.py (same question, same product model).
At 1B scale the probed candidate pool is millions of vectors per query, far too
large for the elementwise-numpy Mitchell path, so this script RERANKS:

  1. Reuse the SAME shared coarse quantizer + base assignment caches the 02_2
     paper runs created (coarse_{ds}_nlist{n}_hnsw_full.index / assign_*.npy).
  2. Build the fp16 baseline IVF index once (lut=None, identical to the 02_2
     fp16 baseline) and search top-C (default 2000) per query per nprobe.
  3. Reconstruct the C candidates from the in-RAM fp16 codes (direct map; no
     disk I/O) and rank them two ways: exact cross products, and Mitchell cross
     products with exact ||v||^2 (same favors-Mitchell formulation as GIST).
  4. Truncation audit: Mitchell recall is also computed from the top-C/2
     candidates; if it matches top-C, C is large enough (Mitchell shrinks a
     product's magnitude by at most 11.1%, so far-away candidates cannot enter
     the top-10 -- the audit verifies this empirically).

Output CSV matches the GIST Mitchell CSV schema so the paper figure picks the
curve up automatically (method in {exact, mitchell, mitchell_halfC}).

Examples (paper operating nlists; caches must exist from the 02_2 runs):
  python 02_3_1B_mitchell_recall.py --dataset deep1b     --nlist 131072 \
      --nprobe-list 8 16 32 64 128 --max-query 1000
  python 02_3_1B_mitchell_recall.py --dataset msturing1b --nlist 262144 \
      --nprobe-list 8 16 32 64 128 512 --max-query 1000
"""

import argparse
import gc
import os
import time

import faiss
import numpy as np
import pandas as pd

from utils import recall_at_k

PRESET = {
    "deep1b": dict(data_dir="../../data/datasets/deep1b", base="base.1B.fbin",
                   query="query.public.10K.fbin", gt="groundtruth.public.10K.ibin"),
    "msturing1b": dict(data_dir="../../data/datasets/msturing1b", base="base1b.fbin",
                       query="query100K.fbin", gt="query_gt100.bin"),
}


# ---------------- .fbin / .ibin readers (same as 02_2) ----------------
def read_bin_header(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
    return int(n), int(d)


def read_fbin_slice(path, start, count):
    n, d = read_bin_header(path)
    if count is None:
        count = n - start
    count = min(count, n - start)
    with open(path, "rb") as f:
        f.seek(8 + start * d * 4)
        a = np.fromfile(f, dtype=np.float32, count=count * d)
    return a.reshape(count, d)


def read_ibin(path, count=None):
    n, k = read_bin_header(path)
    if count is None:
        count = n
    count = min(count, n)
    with open(path, "rb") as f:
        f.seek(8)
        a = np.fromfile(f, dtype=np.int32, count=count * k)
    return a.reshape(count, k).astype(np.int64)


# ---------------- shared coarse plumbing (same as 02_2) ----------------
def build_fp16_ivf(d, nlist, hnsw_m, hnsw_efc):
    quantizer = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_L2)
    quantizer.hnsw.efConstruction = hnsw_efc
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
    index.by_residual = False
    index.cp.niter = 10
    return index


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


def add_base(index, base_path, nb, assign, chunk_size):
    t0 = time.time()
    done = 0
    while done < nb:
        cur = min(chunk_size, nb - done)
        xb = np.ascontiguousarray(read_fbin_slice(base_path, done, cur),
                                  dtype=np.float32)
        ids = np.arange(done, done + cur, dtype=np.int64)
        pre = assign[done:done + cur].astype(np.int64)
        index.add_core(cur, faiss.swig_ptr(xb),
                       faiss.swig_ptr(ids), faiss.swig_ptr(pre))
        done += cur
        el = time.time() - t0
        print(f"    added {done:,}/{nb:,} ({100.0*done/nb:5.1f}%) | "
              f"elapsed={el:7.1f}s | eta={el/done*(nb-done):7.1f}s", flush=True)


# ---------------- Mitchell product model (identical to GIST script) ----------------
def mitchell_prod(a, b):
    s = np.sign(a) * np.sign(b)
    ma, ea = np.frexp(np.abs(a))
    mb, eb = np.frexp(np.abs(b))
    fa = 2.0 * ma - 1.0
    fb = 2.0 * mb - 1.0
    e = ea + eb - 2
    fsum = fa + fb
    carry = fsum >= 1.0
    mant = np.where(carry, fsum, 1.0 + fsum)
    out = s * np.ldexp(mant.astype(np.float32), e + carry)
    return np.where((a == 0) | (b == 0), np.float32(0.0), out)


def fetch_rows(index, ids):
    try:
        return index.reconstruct_batch(ids)
    except AttributeError:                      # older faiss: per-id reconstruct
        out = np.empty((len(ids), index.d), dtype=np.float32)
        for j, i_ in enumerate(ids):
            out[j] = index.reconstruct(int(i_))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=list(PRESET))
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--nlist", type=int, required=True)
    ap.add_argument("--nprobe-list", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank-C", type=int, default=2000)
    ap.add_argument("--max-query", type=int, default=1000)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef-construction", type=int, default=40)
    ap.add_argument("--hnsw-ef-search", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    ap.add_argument("--cache-dir", type=str, default="../../data/cache")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads or os.cpu_count() or 31)
    p = PRESET[args.dataset]
    data_dir = args.data_dir or p["data_dir"]
    base_path = os.path.join(data_dir, p["base"])
    nb, d = read_bin_header(base_path)

    xq = np.ascontiguousarray(read_fbin_slice(os.path.join(data_dir, p["query"]),
                                              0, args.max_query), dtype=np.float32)
    I_gt = read_ibin(os.path.join(data_dir, p["gt"]))[: args.max_query]
    nq = xq.shape[0]
    print(f"base {nb:,} x {d}, queries {nq}", flush=True)

    coarse_path = os.path.join(
        args.cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_hnsw_full.index")
    assign_path = os.path.join(
        args.cache_dir, f"assign_{args.dataset}_nlist{args.nlist}_hnsw_full.npy")
    for path in (coarse_path, assign_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run the 02_2 script (or --assign-only) for "
                f"nlist={args.nlist} first; this script reuses the paper caches.")
    assign = np.load(assign_path)

    t0 = time.time()
    index = make_shared_index(coarse_path, d, args.nlist, args.hnsw_m,
                              args.hnsw_ef_construction, args.hnsw_ef_search)
    print("Adding fp16 baseline codes (one-time for this run)...", flush=True)
    add_base(index, base_path, nb, assign, args.chunk_size)
    del assign
    gc.collect()
    index.make_direct_map()          # id -> (list, offset), enables reconstruct
    print(f"[{time.time()-t0:8.1f}s] index ready", flush=True)

    out_dir = os.path.join("results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    np_tag = f"{min(args.nprobe_list)}-{max(args.nprobe_list)}"
    out_path = os.path.join(
        out_dir, f"mitchell_rerank{args.rerank_C}_nlist{args.nlist}_nprobe{np_tag}.csv")

    C = args.rerank_C
    results = []
    for npb in args.nprobe_list:
        index.nprobe = npb
        t_np = time.time()
        D_f, I_f = index.search(xq, C)
        t_search = time.time() - t_np

        I_mi = np.full((nq, args.k), -1, dtype=np.int64)
        I_mh = np.full((nq, args.k), -1, dtype=np.int64)
        I_re = np.full((nq, args.k), -1, dtype=np.int64)
        for qi in range(nq):
            ids = I_f[qi]
            ids = ids[ids >= 0]
            if len(ids) == 0:
                continue
            # decode candidates from the in-RAM fp16 index: no disk I/O.
            # fp16 rounding error (~1e-3 relative) is negligible against the
            # Mitchell effect being measured (up to 11% per product).
            rows = fetch_rows(index, ids)
            ids_s = ids
            q = xq[qi]
            vnorm = (rows.astype(np.float64) ** 2).sum(axis=1)
            s_ex = vnorm - 2.0 * (rows @ q).astype(np.float64)
            s_mi = vnorm - 2.0 * mitchell_prod(rows, q[None, :]).sum(
                axis=1, dtype=np.float64)
            kk = min(args.k, len(ids_s))
            I_re[qi, :kk] = ids_s[np.argsort(s_ex)[:kk]]
            I_mi[qi, :kk] = ids_s[np.argsort(s_mi)[:kk]]
            # truncation audit: redo Mitchell on the top-C/2 faiss candidates
            half = np.isin(ids_s, I_f[qi, : C // 2])
            s_h = s_mi[half]
            ih = ids_s[half]
            kk2 = min(args.k, len(ih))
            I_mh[qi, :kk2] = ih[np.argsort(s_h)[:kk2]]
        dt = time.time() - t_np
        r_fa = recall_at_k(I_f[:, : args.k], I_gt, args.k)
        r_re = recall_at_k(I_re, I_gt, args.k)
        r_mi = recall_at_k(I_mi, I_gt, args.k)
        r_mh = recall_at_k(I_mh, I_gt, args.k)
        print(f"nprobe={npb:>4} | faiss={r_fa:.5f} | exact={r_re:.5f} | "
              f"mitchell={r_mi:.5f} | mitchell@C/2={r_mh:.5f} | "
              f"search={t_search:.1f}s total={dt:.1f}s", flush=True)
        for method, rec in [("exact", r_re), ("mitchell", r_mi),
                            ("mitchell_halfC", r_mh), ("faiss_fp16", r_fa)]:
            results.append({
                "dataset": args.dataset, "nb": nb, "method": method,
                "mode": "signed", "S": 0, "nlist": args.nlist, "nprobe": npb,
                "rerank_C": C, f"recall@{args.k}": rec,
                "search_time_sec": t_search, "total_time_sec": dt,
            })
        pd.DataFrame(results).to_csv(out_path, index=False)
        gc.collect()

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
