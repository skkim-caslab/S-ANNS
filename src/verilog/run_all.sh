#!/bin/bash
# Synthesizes the eleven PE datapaths of Table I with Yosys (ASAP7) and runs
# OpenSTA at 350 MHz. Each design's area, timing and power report is written
# to data/verilog/<report>.out, which Table I and Fig. 5 read.
# Requires `yosys` and `sta` (OpenSTA) on PATH; binaries placed in this
# directory are picked up as well.
set -e
cd "$(dirname "$0")"
export PATH="$(pwd):$PATH"

dirs=(FP32MAC_Seq FP32MAC_BoothSerial FP32MAC_Array FP32MAC_Wallace
      FP32MAC_BoothWallace FP32MAC_Mitchell
      IsoBW_BF16_16 IsoBW_INT8A_32
      IsoBW_Shift_S2 IsoBW_Shift_S3 IsoBW_Shift_S4)

for d in "${dirs[@]}"; do
    echo "===== Running in $d ====="
    (
        cd "$d"
        yosys -s synth.ys      # writes the area report (tee -o) and synthesized.v
        sta -exit sta.tcl      # appends timing and power to the same report
    )
done
