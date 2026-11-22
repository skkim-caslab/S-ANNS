#!/usr/bin/env bash
# DEEP1B (96-d) and MSTuring1B (100-d), L2: every recall run behind Fig. 4 / Fig. 6.
# Usage: bash run_1b.sh deep1b | msturing1b
#
# The shared HNSW coarse quantizer and the 1B-vector list assignment are built
# once (--assign-only) and cached in $CACHE; every format then ranks the same
# probed lists (02_2) or the same top-2000 candidates of an FP16 index (02_3/4/5).
# Outputs land in results/<dataset>/ (this directory); copy them to data/ to plot:
#   quant_sharedcoarse_*_nlist<NL>_*.csv            -> data/anns/{DEEP1B,MSTuring1B}_*.csv
#   mitchell_*, scalarq_*, allformats_* (nlist NL)  -> data/recall/<dataset>/
#   allformats_*_nprobe1024-1024.csv (step 5)       -> data/recall/high_recall/
# Requirements per corpus: ~400 GB for the base vectors, ~200-250 GB RAM for the
# FP16 index, ~50 GB cache. Runtime: roughly 1-2 days per corpus on 32 cores.
set -uo pipefail
cd "$(dirname "$0")"
DS="${1:?usage: bash run_1b.sh deep1b|msturing1b}"
PY="${PY:-python3}"
CACHE="${CACHE:-../../data/cache}"
case "$DS" in
  deep1b)     NL=131072; SCRIPT=02_2_deep1B_quantization_shared_coarse.py
              NP02="8 16 32 64 128 256 512 1024"; NPRR="8 16 32 64 128 256 512 1024"
              NPAF="8 16 32 64 128 256 512 1024" ;;
  msturing1b) NL=262144; SCRIPT=02_2_msturing1B_quantization_shared_coarse.py
              NP02="4 8 16 32 64 128 256 512"; NPRR="2 4 8 16 32 64 128 256 512"
              NPAF="32 128 512 1024 2048 4096" ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
DATA="${DATA:-../../data/datasets/$DS}"

# (1) coarse quantizer + assignment, then FP32 + PoT S=2..4, one config per process
#     (--single exits 42 once every config is done; the OS reclaims RAM between configs)
$PY $SCRIPT --data-dir "$DATA" --cache-dir "$CACHE" --assign-only --nlist $NL --max-query 1000 || exit $?
while true; do
  $PY $SCRIPT --data-dir "$DATA" --cache-dir "$CACHE" --single --nlist $NL \
    --nprobe-list $NP02 --S-list 2 3 4 --modes signed --bmin -16 --bmax -2 --max-query 1000
  code=$?; [ $code -eq 42 ] && break; [ $code -ne 0 ] && exit $code
done

# (2) Mitchell, (3) BF16/SQ8, (4) all formats -- identical top-2000 candidates
$PY 02_3_1B_mitchell_recall.py   --dataset $DS --data-dir "$DATA" --cache-dir "$CACHE" --nlist $NL --nprobe-list $NPRR || exit $?
$PY 02_4_1B_scalar_quant_recall.py --dataset $DS --data-dir "$DATA" --cache-dir "$CACHE" --nlist $NL --nprobe-list $NPRR || exit $?
$PY 02_5_1B_allformats_recall.py --dataset $DS --data-dir "$DATA" --cache-dir "$CACHE" --nlist $NL --nprobe-list $NPAF \
  --S-list 3 4 || exit $?

# (5) DEEP1B only: accuracy-only S=2..5 point at nprobe 1024 (Section IV-B text)
if [ "$DS" = deep1b ]; then
  $PY 02_5_1B_allformats_recall.py --dataset $DS --data-dir "$DATA" --cache-dir "$CACHE" --nlist $NL \
    --nprobe-list 1024 --S-list 2 3 4 5 || exit $?
fi
