#!/usr/bin/env bash
#
# Download Deep1B (big-ann-benchmarks fbin distribution).
# Usage: bash download_deep1B.sh [target-dir]   (default: ../../data/datasets/deep1b)
#
# Source: harsha-simhadri/big-ann-benchmarks  (Yandex Object Storage, https, resumable)
# Format: .fbin  -> header [uint32 N][uint32 d], then N*d float32, d=96.  NOT gzipped.
#
# Sizes (raw, no decompression step):
#   base.1B.fbin            ~ 384 GB  (1e9 x 96 x 4 + 8)
#   query.public.10K.fbin   ~ 3.8 MB  (1e4 x 96 x 4 + 8)
#   groundtruth .bin        ~ 8 MB
#
# Safe to re-run: wget -c resumes partial files.

set -euo pipefail

DEST="${1:-../../data/datasets/deep1b}"
BASE_URL="https://storage.yandexcloud.net/yandex-research/ann-datasets"

mkdir -p "$DEST"
cd "$DEST"

# wget -c : continue/resume   --tries=0 : retry forever on transient errors
WGET="wget -c --tries=0 --retry-connrefused --waitretry=10 --read-timeout=60"

echo "[1/3] query set (~3.8 MB)"
$WGET "$BASE_URL/DEEP/query.public.10K.fbin"

echo "[2/3] groundtruth (~8 MB)"
$WGET -O groundtruth.public.10K.ibin "$BASE_URL/deep_new_groundtruth.public.10K.bin"

echo "[3/3] base vectors (~384 GB) -- this is the long one"
$WGET "$BASE_URL/DEEP/base.1B.fbin"

echo
echo "Done. Files in $DEST :"
ls -la "$DEST"

# Sanity check: expected base size = 8 + 1e9*96*4 = 384000000008 bytes
EXPECT=384000000008
ACTUAL=$(stat -c%s "$DEST/base.1B.fbin" 2>/dev/null || echo 0)
if [ "$ACTUAL" = "$EXPECT" ]; then
  echo "OK: base.1B.fbin size matches ($ACTUAL bytes)"
else
  echo "WARNING: base.1B.fbin is $ACTUAL bytes, expected $EXPECT. Re-run to resume."
fi
