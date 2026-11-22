# PE Datapaths (RTL)

RTL of the processing-element datapaths compared in Table I and Fig. 5, the
Yosys/OpenSTA scripts that synthesize them, and the functional testbenches.
Every PE retires one 32-byte burst of stored vector data per cycle with FP32
queries and shares the same 64-bit fixed-point accumulation back end, so only
the per-lane multiplication differs.

| Directory | Top module (parameters) | Table I row | Report (`data/verilog/`) |
|-----------|-------------------------|-------------|--------------------------|
| `FP32MAC_Seq/` | `fp32mac_seqsa` | Sequential shift-add (x25) | `fp32mac_seq.out` |
| `FP32MAC_BoothSerial/` | `fp32mac_seqbooth` | Booth serial, radix-4 (x14) | `fp32mac_bser.out` |
| `FP32MAC_Array/` | `fp32mac_par` (`CORE=0`) | Carry-save array | `fp32mac_array.out` |
| `FP32MAC_Wallace/` | `fp32mac_par` (`CORE=1`) | Wallace tree | `fp32mac_wallace.out` |
| `FP32MAC_BoothWallace/` | `fp32mac_par` (`CORE=2`) | Booth + Wallace (FP32 MAC) | `fp32mac_boothw.out` |
| `FP32MAC_Mitchell/` | `fp32mac_par` (`CORE=3`) | Mitchell log. (approx.) | `fp32mac_mitchell.out` |
| `IsoBW_BF16_16/` | `pe_bf16` (`LANES=16`) | BF16 MAC | `isobw_bf16_16.out` |
| `IsoBW_INT8A_32/` | `pe_int8a` (`LANES=32`) | INT8 dequant. MAC | `isobw_int8a_32.out` |
| `IsoBW_Shift_S2/` | `pe_shift` (`LANES=25, S=2`) | S-ANNS S=2 | `isobw_shift_s2.out` |
| `IsoBW_Shift_S3/` | `pe_shift` (`LANES=17, S=3`) | S-ANNS S=3 | `isobw_shift_s3.out` |
| `IsoBW_Shift_S4/` | `pe_shift` (`LANES=12, S=4`) | S-ANNS S=4 | `isobw_shift_s4.out` |

Shared sources:

- `FP32_common/fp32_common.v` – FP32 decode, carry-save helpers and the 24x24
  significand multipliers (array, Wallace, Booth-Wallace).
- `FP32_common/fp32mac_par.v`, `fp32mac_seqsa.v`, `fp32mac_seqbooth.v` – the
  8-lane FP32 MAC PEs (parallel cores and the two sequential multipliers).
- `FP32_common/fp_shift_unit.v` – the 8-lane S-ANNS shift PE used by the
  functional test of Section IV-A (`tb_shift.v`).
- `IsoBW_common/isobw_pes.v` – the iso-bandwidth PEs (`pe_shift`, `pe_bf16`,
  `pe_int8a`; `pe_fp32` and `pe_int8s` are not in Table I) with the
  shared reduction tree.
- `constraints.sdc` – 350 MHz clock (2.857 ns), I/O delays and load.
- `libs/asap7.lib` – merged ASAP7 TT library (see `libs/README.md`).

## Running

Install [Yosys](https://github.com/YosysHQ/yosys) and
[OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) so that `yosys` and
`sta` are on `PATH` (the included reports were produced with Yosys 0.53 and
OpenSTA; other versions may give slightly different area and power), then:

```bash
bash run_all.sh     # overwrites data/verilog/*.out, ~10-20 min
bash run_tb.sh      # functional tests, needs Icarus Verilog (iverilog, vvp)
```

Each report holds the Yosys `stat` area report followed by the OpenSTA
`report_checks` and `report_power` output (power in mW). `synthesized.v`
netlists are written next to each `synth.ys` and are not tracked.
