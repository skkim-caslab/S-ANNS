#!/usr/bin/env bash
# Download the official big-ann T2I-100M crop (~80GB instead of 800GB):
#   base = first 100M vectors of base.1B.fbin, fetched with HTTP range requests
#          (resumable -- rerun this script to continue), header patched to n=100M
#   gt   = organizer-computed GT_100M/text2image-100M (same 100K public queries)
# Usage: bash download_t2i_100m.sh [target-dir]   (default: ../../data/datasets/text2image100m)
set -eu
DIR="${1:-../../data/datasets/text2image100m}"
mkdir -p "$DIR"; cd "$DIR"

TARGET=80000000008   # 8-byte header + 100e6 * 200 dims * 4 bytes
URL=https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
while :; do
  sz=$(stat -c%s base.100M.fbin 2>/dev/null || echo 0)
  if [ "$sz" -ge "$TARGET" ]; then break; fi
  echo "base.100M.fbin: resuming at byte $sz / $TARGET  $(date)"
  curl -fL -r "$sz-$((TARGET-1))" "$URL" >> base.100M.fbin || { echo "curl failed, retrying in 10s"; sleep 10; }
done
echo "base.100M.fbin complete ($TARGET bytes)"

python3 - <<'EOF'
import struct
with open("base.100M.fbin", "r+b") as f:
    n, d = struct.unpack("<II", f.read(8))
    assert d == 200, f"unexpected dim {d}"
    if n != 100_000_000:
        f.seek(0); f.write(struct.pack("<I", 100_000_000))
        print(f"header patched: n {n} -> 100000000 (d={d})")
    else:
        print(f"header already n=100000000 (d={d})")
EOF

wget -c https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
wget -c -O groundtruth.public.100K.ibin \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/text2image-100M

echo "--- done; expect base 80,000,000,008 / query 80,000,008 / gt 80,000,008 bytes ---"
ls -la
