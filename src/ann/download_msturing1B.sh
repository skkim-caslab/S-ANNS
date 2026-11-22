#!/usr/bin/env bash
#
# Download MSTuring-1B (Microsoft Turing-ANNS, NeurIPS'21 big-ann-benchmarks).
# Usage: bash download_msturing1B.sh [target-dir]   (default: ../../data/datasets/msturing1b)
#
# Source: harsha-simhadri/big-ann-benchmarks  (Azure static web, https, resumable)
# Format: .fbin  -> header [uint32 N][uint32 d], then N*d float32, d=100.  NOT gzipped.
# Metric: L2 (Euclidean).
#
# Sizes (raw, no decompression step):
#   base1b.fbin        400,000,000,008 B  (~373 GiB)  = 8 + 1e9 x 100 x 4
#   query100K.fbin          40,000,008 B  (~38 MB)    = 8 + 1e5 x 100 x 4
#   query_gt100.bin         80,000,008 B  (~76 MB)    = ids(int32) + dists(float32), k=100
#
# Safe to re-run: wget -c resumes partial files.

set -euo pipefail

DEST="${1:-../../data/datasets/msturing1b}"
BASE_URL="https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS"

mkdir -p "$DEST"
cd "$DEST"

# wget -c : continue/resume   --tries=0 : retry forever on transient errors
WGET="wget -c --tries=0 --retry-connrefused --waitretry=10 --read-timeout=60"

echo "[1/3] query set (~38 MB)"
$WGET "$BASE_URL/query100K.fbin"

echo "[2/3] groundtruth (~76 MB)"
$WGET "$BASE_URL/query_gt100.bin"

echo "[3/3] base vectors (~373 GiB) -- this is the long one"
$WGET "$BASE_URL/base1b.fbin"

echo
echo "Done. Files in $DEST :"
ls -la "$DEST"

# Sanity check: expected base size = 8 + 1e9*100*4 = 400000000008 bytes
EXPECT=400000000008
ACTUAL=$(stat -c%s "$DEST/base1b.fbin" 2>/dev/null || echo 0)
if [ "$ACTUAL" = "$EXPECT" ]; then
  echo "OK: base1b.fbin size matches ($ACTUAL bytes)"
else
  echo "WARNING: base1b.fbin is $ACTUAL bytes, expected $EXPECT. Re-run to resume."
fi
