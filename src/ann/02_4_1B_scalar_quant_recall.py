# src/ann/02_4_1B_scalar_quant_recall.py
"""
DEEP1B / MSTuring1B: recall of scalar-quantized storage (bf16 / fp16-ref / SQ8)
via the same top-C rerank used by 02_3_1B_mitchell_recall.py.

Same shared coarse caches, same fp16 baseline index, same candidate sets; the
C candidates are reconstructed and re-ranked after quantize-dequantize with
each storage scheme, so recall differences are purely the stored precision.
(The reconstructed rows are already fp16-rounded, which is why fp16 itself is
the 'exact' reference here; bf16 and SQ8 are applied on top, a negligible
composition error next to the 8-bit step being measured.)

  exact : fp16-reconstructed rows as-is (the paper's baseline representation)
  bf16  : rows rounded to bfloat16
  sq8   : per-dimension affine 256-level uniform quantization, min/max trained
          on a 2M-vector sample of the base (faiss QT_8bit style)

Includes the same C/2 truncation audit as 02_3 (reported as <method>_halfC).

Examples (run on the large-memory machine; caches must exist from 02_2 runs):
  python 02_4_1B_scalar_quant_recall.py --dataset deep1b     --nlist 131072 \
      --nprobe-list 8 16 32 64 128 256 512 1024 --max-query 1000
  python 02_4_1B_scalar_quant_recall.py --dataset msturing1b --nlist 262144 \
      --nprobe-list 2 4 8 16 32 64 128 256 512 --max-query 1000
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


def fetch_rows(index, ids):
    try:
        return index.reconstruct_batch(ids)
    except AttributeError:
        out = np.empty((len(ids), index.d), dtype=np.float32)
        for j, i_ in enumerate(ids):
            out[j] = index.reconstruct(int(i_))
        return out


def bf16_qdq(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) \
        & np.uint32(0xFFFF0000)
    return r.view(np.float32).copy()


def sq8_train(xt):
    return xt.min(axis=0).astype(np.float32), xt.max(axis=0).astype(np.float32)


def sq8_qdq(x, vmin, vmax):
    scale = (vmax - vmin) / 256.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    code = np.clip(np.floor((x - vmin) / scale), 0, 255)
    return (vmin + (code + 0.5) * scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=list(PRESET))
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--nlist", type=int, required=True)
    ap.add_argument("--nprobe-list", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank-C", type=int, default=2000)
    ap.add_argument("--max-query", type=int, default=1000)
    ap.add_argument("--sq8-train", type=int, default=2_000_000)
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

    vmin, vmax = sq8_train(read_fbin_slice(base_path, 0, args.sq8_train))
    print("sq8 stats trained", flush=True)

    coarse_path = os.path.join(
        args.cache_dir, f"coarse_{args.dataset}_nlist{args.nlist}_hnsw_full.index")
    assign_path = os.path.join(
        args.cache_dir, f"assign_{args.dataset}_nlist{args.nlist}_hnsw_full.npy")
    for path in (coarse_path, assign_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run the 02_2 script "
                                    f"(or --assign-only) for nlist={args.nlist} first.")
    assign = np.load(assign_path)

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
        out_dir, f"scalarq_rerank{args.rerank_C}_nlist{args.nlist}_nprobe{np_tag}.csv")

    C = args.rerank_C
    schemes = ["exact", "bf16", "sq8"]
    results = []
    for npb in args.nprobe_list:
        index.nprobe = npb
        t_np = time.time()
        _, I_f = index.search(xq, C)
        t_search = time.time() - t_np

        I_pred = {m: np.full((nq, args.k), -1, dtype=np.int64) for m in schemes}
        I_half = {m: np.full((nq, args.k), -1, dtype=np.int64) for m in schemes}
        for qi in range(nq):
            ids = I_f[qi]
            ids = ids[ids >= 0]
            if len(ids) == 0:
                continue
            rows = fetch_rows(index, ids)
            q = xq[qi]
            kk = min(args.k, len(ids))
            half = np.isin(ids, I_f[qi, : C // 2])
            for m in schemes:
                if m == "exact":
                    v = rows
                elif m == "bf16":
                    v = bf16_qdq(rows)
                else:
                    v = sq8_qdq(rows, vmin, vmax)
                s = (v.astype(np.float64) ** 2).sum(axis=1) \
                    - 2.0 * (v @ q).astype(np.float64)
                I_pred[m][qi, :kk] = ids[np.argsort(s)[:kk]]
                sh = s[half]
                ih = ids[half]
                kk2 = min(args.k, len(ih))
                I_half[m][qi, :kk2] = ih[np.argsort(sh)[:kk2]]
        dt = time.time() - t_np
        recs = {m: recall_at_k(I_pred[m], I_gt, args.k) for m in schemes}
        recs_h = {m: recall_at_k(I_half[m], I_gt, args.k) for m in schemes}
        print(f"nprobe={npb:>4} | " +
              " | ".join(f"{m}={recs[m]:.5f}" for m in schemes) +
              " | " + " | ".join(f"{m}@C/2={recs_h[m]:.5f}" for m in schemes) +
              f" | search={t_search:.1f}s total={dt:.1f}s", flush=True)
        for m in schemes:
            for name, r in ((m, recs[m]), (m + "_halfC", recs_h[m])):
                results.append({
                    "dataset": args.dataset, "nb": nb, "method": name,
                    "mode": "signed", "S": 0, "nlist": args.nlist, "nprobe": npb,
                    "rerank_C": C, f"recall@{args.k}": r,
                    "search_time_sec": t_search, "total_time_sec": dt,
                })
        pd.DataFrame(results).to_csv(out_path, index=False)
        gc.collect()

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
