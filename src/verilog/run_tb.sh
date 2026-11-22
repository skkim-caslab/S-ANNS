#!/bin/bash
# Functional checks of the RTL with Icarus Verilog (iverilog/vvp on PATH).
#   tb_shift   : 8-lane S-ANNS shift PE vs. a double-precision model of the
#                decoded PoT terms, 300 bursts x 8 lanes = 2,400 random products,
#                for S = 2, 3, 4 (|error| < 1e-9 required)
#   tb_cores   : 24x24 array / Wallace / Booth-Wallace significand multipliers (bit-exact)
#   tb_mac_par : parallel FP32 MAC PE with each multiplier core (CORE = 0..3)
#   tb_fp32mac_seq{sa,booth} : sequential shift-add and radix-4 Booth serial MACs
#   tb_isobw   : iso-bandwidth PEs (BF16, INT8 and the 17-lane S=3 shift PE)
set -e
cd "$(dirname "$0")"
B=$(mktemp -d)
C=FP32_common
run() { local top="$1"; shift; iverilog -g2012 -s "$top" -o "$B/sim" "$@" && vvp -n "$B/sim" | grep -E "PASS|FAIL"; }

for S in 2 3 4; do
    run tb -P tb.S=$S -P tb.SIGNED=1 $C/fp32_common.v $C/fp_shift_unit.v $C/tb_shift.v
done
run tb_cores $C/fp32_common.v $C/tb_cores.v
for CORE in 0 1 2 3; do
    run tb -P tb.CORE=$CORE $C/fp32_common.v $C/fp32mac_par.v $C/tb_mac_par.v
done
run tb $C/fp32_common.v $C/fp32mac_seqsa.v $C/tb_fp32mac_seqsa.v
run tb $C/fp32_common.v $C/fp32mac_seqbooth.v $C/tb_fp32mac_seqbooth.v
for top in tb_core tb_int8s tb_bf16 tb_int8a tb_shift17; do
    run $top $C/fp32_common.v IsoBW_common/isobw_pes.v IsoBW_common/tb_isobw.v
done
rm -rf "$B"
