#!/usr/bin/env bash
# GIST1M (960-d, L2): every recall run behind Fig. 4 / Fig. 6 panel (a).
# Every method ranks all probed vectors (no rerank cut), nlist = 16384.
# Outputs land in results/gist1M/ (this directory); copy them to data/ to plot:
#   quant_brute_force_*_nlist16384_*.csv  -> data/anns/GIST1M_*.csv
#   mitchell_*.csv, scalarq_*.csv         -> data/recall/gist1M/
# Runtime: about 2-4 h on a 16-core CPU; memory ~16 GB.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python3}"
DATA="${DATA:-../../data/datasets/gist1M}"     # gist_{base,query,learn}.fvecs, gist_groundtruth.ivecs
CACHE="${CACHE:-../../data/cache}"
NL=16384
NPROBE="1 2 4 8 12 16 24 32 48 64 96 128 144 160 176 192 208 224 240 256 320 384 448 512"

# (1) FP32 baseline + PoT S=1..4 (signed and unsigned codes), 15-position window [-14, 0]
$PY 02_2_gist1M_quantization_recall_bruteforce.py --data-dir "$DATA" --cache-dir "$CACHE" \
  --nlist $NL --nprobe-list $NPROBE --S-list 1 2 3 4 --modes signed positive \
  --bmin -14 --bmax 0

# (2) Mitchell logarithmic multiplier on the same candidates
$PY 02_3_gist1M_mitchell_recall.py --data-dir "$DATA" --cache-dir "$CACHE" \
  --nlist $NL --nprobe-list 32 64 128 256 512 --S 3 --bmin -14 --bmax 0

# (3) scalar-quantized storage (BF16 / FP16 / SQ8) against the PoT (S, window) grid
$PY 02_4_gist1M_scalar_quant_recall.py --data-dir "$DATA" --cache-dir "$CACHE" \
  --nlist $NL --nprobe-list 32 64 128 256 512 --S-list 2 3 4 --windows 15 7
$PY 02_4_gist1M_scalar_quant_recall.py --data-dir "$DATA" --cache-dir "$CACHE" \
  --nlist $NL --nprobe-list 32 64 128 256 512 --S-list 3 4 --windows 15
