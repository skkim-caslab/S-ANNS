"""Fig. 5: PE (a) area and (b) power per streamed dimension, normalized to the
FP32 Booth-Wallace MAC (clipped bars annotated with their multiple).

Input : data/verilog/{fp32mac_*,isobw_shift_s{2,3,4}}.out
Output: fig5_pe_cost.png (this directory)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import paper_eval as pe  # noqa: E402

# (label, report, replicas to sustain one 32-byte burst per cycle, dims per burst)
DESIGNS = [("Seq", "fp32mac_seq", 25, 8), ("BSer", "fp32mac_bser", 14, 8),
           ("B+W", "fp32mac_boothw", 1, 8), ("Mit", "fp32mac_mitchell", 1, 8),
           ("S=2", "isobw_shift_s2", 1, 25), ("S=3", "isobw_shift_s3", 1, 17),
           ("S=4", "isobw_shift_s4", 1, 12)]


def main():
    pe.pe_cost_fig(out=os.path.join(HERE, "fig5_pe_cost.png"))
    am = pe.area("fp32mac_boothw") / 8
    pm = pe.power("fp32mac_boothw") / 8
    print("\nPer streamed dimension, normalized to the FP32 Booth-Wallace MAC:")
    print(f"  {'PE':5s} {'area':>7s} {'power':>7s}")
    for lbl, u, f, d in DESIGNS:
        print(f"  {lbl:5s} {f * pe.area(u) / d / am:7.3f} {f * pe.power(u) / d / pm:7.3f}")


if __name__ == "__main__":
    main()
