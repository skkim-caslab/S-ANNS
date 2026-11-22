#!/usr/bin/env bash
# T2I-100M (200-d, inner product): every recall run behind Fig. 4 / Fig. 6 panel (d).
# The 100M crop of Yandex Text-to-Image-1B is fetched by download_t2i_100m.sh.
# FP32, PoT, BF16 and SQ8 come from one all-formats rerank of identical top-2000
# candidates (02_5); Mitchell reranks the same candidates (02_3).
# Outputs land in results/text2image100m/ (this directory); copy them to
# data/recall/text2image100m/ (nprobe 4..1024) and data/recall/high_recall/ (nprobe 1024, S=2..5).
# Requirements: 80 GB base, ~40 GB RAM for the FP16 index. Runtime: ~1 day on 28 cores.
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
CACHE="${CACHE:-../../data/cache}"
DATA="${DATA:-../../data/datasets/text2image100m}"
DS=text2image100m; BASE=base.100M.fbin; NL=32768; MAXQ="${MAXQ:-10000}"
COMMON=(--dataset $DS --data-dir "$DATA" --base-file $BASE --query-file query.public.100K.fbin
        --gt-file groundtruth.public.100K.ibin --nlist $NL --max-query $MAXQ --threads 28
        --chunk-size 1000000 --cache-dir "$CACHE")

# (1) inner-product coarse quantizer + assignment cache
$PY 02_2_text1B_quantization_shared_coarse.py --dataset $DS --data-dir "$DATA" --base-file $BASE \
  --query-file query.public.100K.fbin --gt-file groundtruth.public.100K.ibin --metric ip \
  --nlist $NL --threads 28 --chunk-size 1000000 --cache-dir "$CACHE" --assign-only || exit $?
# (2) all formats (FP32 / BF16 / SQ8 / PoT S=2..4), (3) Mitchell
$PY 02_5_text1B_allformats_recall.py "${COMMON[@]}" --nprobe-list 4 8 16 32 64 128 256 512 1024 \
  --S-list 2 3 4 --bmin -16 --bmax -2 || exit $?
$PY 02_3_text100m_mitchell_recall.py "${COMMON[@]}" --nprobe-list 4 8 16 32 64 128 256 512 1024 || exit $?
# (4) accuracy-only S=2..5 point at nprobe 1024 (Section IV-B text)
$PY 02_5_text1B_allformats_recall.py "${COMMON[@]}" --nprobe-list 1024 --S-list 2 3 4 5 \
  --bmin -16 --bmax -2 || exit $?
