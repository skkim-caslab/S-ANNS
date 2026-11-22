# src/ann/02_3_text100m_mitchell_recall.py
"""
T2I-100M: recall cost of Mitchell's logarithmic multiplier under MIPS.

Companion to 02_3_1B_mitchell_recall.py with the two text1B differences:
  - metric is INNER PRODUCT (gt is IP; query normalized, base NOT), so the
    score is the cross product itself. Unlike the L2 scripts there is no
    exact ||v||^2 term to shelter Mitchell -- every product in the score is
    Mitchell-approximated, which is the exact MIPS formulation.
  - reuses the metric-tagged coarse/assign caches the 02_5 T2I run created
    (coarse_text2image100m_nlist{n}_hnsw_ip_full.index / assign_..._ip_full.npy).

Same rerank protocol as every other same-candidates run: fp16 baseline index,
top-C candidates per query per nprobe, C/2 truncation audit. Output CSV name
matches the mitchell_* pattern so the Fig 4 T2I panel picks the curve up
automatically (method in {exact, mitchell, mitchell_halfC, faiss_fp16}).

Example:
  python 02_3_text100m_mitchell_recall.py --nlist 32768 \
      --nprobe-list 4 8 16 32 64 128 256 512 1024 --max-query 1000 --threads 28
"""

import argparse
import gc
import os
import time

import faiss
import numpy as np
import pandas as pd

from utils import recall_at_k


# ---------------- readers (drop_cache variant, as 02_5_text1B) ----------------
def read_bin_header(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
    return int(n), int(d)


def read_fbin_slice(path, start, count, drop_cache=False):
    n, d = read_bin_header(path)
    if count is None:
        count = n - start
    count = min(count, n - start)
    off = 8 + start * d * 4
    nbytes = count * d * 4
    with open(path, "rb") as f:
        f.seek(off)
        a = np.fromfile(f, dtype=np.float32, count=count * d)
        if drop_cache:
            try:
                f.flush()
                os.posix_fadvise(f.fileno(), off, nbytes, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
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


# ---------------- shared coarse plumbing (IP metric) ----------------
def build_fp16_ivf(d, nlist, hnsw_m, hnsw_efc):
    quantizer = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    quantizer.hnsw.efConstruction = hnsw_efc
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, d, nlist, faiss.ScalarQuantizer.QT_fp16,
        faiss.METRIC_INNER_PRODUCT)
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
        xb = np.ascontiguousarray(
            read_fbin_slice(base_path, done, cur, drop_cache=True),
            dtype=np.float32)
        ids = np.arange(done, done + cur, dtype=np.int64)
        pre = assign[done:done + cur].astype(np.int64)
        index.add_core(cur, faiss.swig_ptr(xb),
                       faiss.swig_ptr(ids), faiss.swig_ptr(pre))
        done += cur
        el = time.time() - t0
        print(f"    added {done:,}/{nb:,} ({100.0*done/nb:5.1f}%) | "
              f"elapsed={el:7.1f}s | eta={el/done*(nb-done):7.1f}s", flush=True)


# ---------------- Mitchell product model (identical to the other scripts) ----
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
    except AttributeError:
        out = np.empty((len(ids), index.d), dtype=np.float32)
        for j, i_ in enumerate(ids):
            out[j] = index.reconstruct(int(i_))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="text2image100m")
    ap.add_argument("--data-dir", type=str, default="../../data/datasets/text2image100m")
    ap.add_argument("--base-file", type=str, default="base.100M.fbin")
    ap.add_argument("--query-file", type=str, default="query.public.100K.fbin")
    ap.add_argument("--gt-file", type=str, default="groundtruth.public.100K.ibin")
    ap.add_argument("--nlist", type=int, required=True)
    ap.add_argument("--nprobe-list", type=int, nargs="+",
                    default=[4, 8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank-C", type=int, default=2000)
    ap.add_argument("--max-query", type=int, default=1000)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef-construction", type=int, default=40)
    ap.add_argument("--hnsw-ef-search", type=int, default=256)
    ap.add_argument("--chunk-size", type=int, default=1_000_000)
    ap.add_argument("--cache-dir", type=str, default="../../data/cache")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    faiss.omp_set_num_threads(args.threads or os.cpu_count() or 31)
    base_path = os.path.join(args.data_dir, args.base_file)
    nb, d = read_bin_header(base_path)

    xq = np.ascontiguousarray(
        read_fbin_slice(os.path.join(args.data_dir, args.query_file),
                        0, args.max_query), dtype=np.float32)
    I_gt = read_ibin(os.path.join(args.data_dir, args.gt_file))[: args.max_query]
    nq = xq.shape[0]
    print(f"base {nb:,} x {d}, queries {nq}, metric=ip", flush=True)

    coarse_path = os.path.join(
        args.cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_hnsw_ip_full.index")
    assign_path = os.path.join(
        args.cache_dir, f"assign_{args.dataset}_nlist{args.nlist}_hnsw_ip_full.npy")
    for path in (coarse_path, assign_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run the 02_5 T2I pipeline (run_t2i100m.sh "
                f"step 1) for nlist={args.nlist} first; this script reuses its caches.")
    assign = np.load(assign_path, mmap_mode="r")

    t0 = time.time()
    index = make_shared_index(coarse_path, d, args.nlist, args.hnsw_m,
                              args.hnsw_ef_construction, args.hnsw_ef_search)
    print("Adding fp16 baseline codes (one-time for this run)...", flush=True)
    add_base(index, base_path, nb, assign, args.chunk_size)
    del assign
    gc.collect()
    index.make_direct_map()
    print(f"[{time.time()-t0:8.1f}s] index ready", flush=True)

    out_dir = os.path.join("results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    np_tag = f"{min(args.nprobe_list)}-{max(args.nprobe_list)}"
    out_path = os.path.join(
        out_dir, f"mitchell_rerank{args.rerank_C}_ip_nlist{args.nlist}_nprobe{np_tag}.csv")

    C = args.rerank_C
    results = []
    for npb in args.nprobe_list:
        index.nprobe = npb
        t_np = time.time()
        _, I_f = index.search(xq, C)
        t_search = time.time() - t_np

        I_mi = np.full((nq, args.k), -1, dtype=np.int64)
        I_mh = np.full((nq, args.k), -1, dtype=np.int64)
        I_re = np.full((nq, args.k), -1, dtype=np.int64)
        for qi in range(nq):
            ids = I_f[qi]
            ids = ids[ids >= 0]
            if len(ids) == 0:
                continue
            rows = fetch_rows(index, ids)
            q = xq[qi]
            # MIPS: rank by descending inner product (ascending negated score);
            # the whole score is cross products, so Mitchell touches all of it
            s_ex = -(rows.astype(np.float64) @ q.astype(np.float64))
            s_mi = -mitchell_prod(rows, q[None, :]).sum(axis=1, dtype=np.float64)
            kk = min(args.k, len(ids))
            I_re[qi, :kk] = ids[np.argsort(s_ex)[:kk]]
            I_mi[qi, :kk] = ids[np.argsort(s_mi)[:kk]]
            half = np.isin(ids, I_f[qi, : C // 2])
            s_h = s_mi[half]
            ih = ids[half]
            kk2 = min(args.k, len(ih))
            I_mh[qi, :kk2] = ih[np.argsort(s_h)[:kk2]]
        dt = time.time() - t_np
        r_fa = recall_at_k(I_f[:, : args.k], I_gt, args.k)
        r_re = recall_at_k(I_re, I_gt, args.k)
        r_mi = recall_at_k(I_mi, I_gt, args.k)
        r_mh = recall_at_k(I_mh, I_gt, args.k)
        print(f"nprobe={npb:>5} | faiss={r_fa:.5f} | exact={r_re:.5f} | "
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
