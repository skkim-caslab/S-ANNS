"""Table I: PE datapaths at one 32-byte burst per cycle with FP32 queries.

Every datapath retires one 32-byte burst of stored vector data per cycle, so
the storage format (bits per dimension, b/dim) sets the dimensions per burst,
floor(256 / b/dim). Sequential multipliers are replicated to sustain the burst
rate (25 shift-add units, 14 radix-4 Booth serial units). Area and power are
read from the Yosys/OpenSTA reports (ASAP7, 350 MHz); OpenSTA reports power
in mW for the ASAP7 library units.

Input : data/verilog/*.out (see src/verilog)
Output: printed table and table1_pe_datapaths.csv (this directory)
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VERI = os.path.join(HERE, "..", "..", "..", "data", "verilog")

# (row label, report, replicas, bits per dimension)
ROWS = [
    ("Sequential shift-add (x25)", "fp32mac_seq", 25, 32),
    ("Booth serial, radix-4 (x14)", "fp32mac_bser", 14, 32),
    ("Carry-save array", "fp32mac_array", 1, 32),
    ("Wallace tree", "fp32mac_wallace", 1, 32),
    ("Booth + Wallace (FP32 MAC)", "fp32mac_boothw", 1, 32),
    ("Mitchell log. (approx.)", "fp32mac_mitchell", 1, 32),
    ("BF16 MAC", "isobw_bf16_16", 1, 16),
    ("INT8 dequant. MAC", "isobw_int8a_32", 1, 8),
    ("S-ANNS S=2", "isobw_shift_s2", 1, 10),
    ("S-ANNS S=3", "isobw_shift_s3", 1, 15),
    ("S-ANNS S=4", "isobw_shift_s4", 1, 20),
]


def report(name):
    with open(os.path.join(VERI, name + ".out")) as f:
        txt = f.read()
    area = float(re.search(r"Chip area for (?:top )?module\s+'[^']+':\s+([\d.]+)", txt).group(1))
    power = float(re.findall(r"^Total\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+([\d.eE+\-]+)",
                             txt, re.M)[-1])
    return area, power


def main():
    rows = []
    for label, rep, reps, bits in ROWS:
        a, p = report(rep)
        dims = 256 // bits
        area, pwr = reps * a, reps * p
        rows.append(dict(pe=label, b_per_dim=bits, dims_per_burst=dims, area_um2=area,
                         power_mW=pwr, um2_per_dim=area / dims, uW_per_dim=pwr * 1e3 / dims))
    hdr = f"{'PE datapath':30s} {'b/dim':>5s} {'dims':>4s} {'Area(um2)':>10s} {'Power(mW)':>9s} {'um2/dim':>8s} {'uW/dim':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['pe']:30s} {r['b_per_dim']:5d} {r['dims_per_burst']:4d} {r['area_um2']:10,.0f} "
              f"{r['power_mW']:9.3f} {r['um2_per_dim']:8.1f} {r['uW_per_dim']:7.1f}")
    mac = next(r for r in rows if r["pe"].startswith("Booth + Wallace"))
    def sig3(x):                      # the paper quotes three significant digits
        return float(f"{x:.3g}")

    print("\nS-ANNS vs. FP32 MAC per streamed dimension (exact / from the rounded table entries):")
    for r in rows:
        if r["pe"].startswith("S-ANNS"):
            ea = (1 - r["um2_per_dim"] / mac["um2_per_dim"]) * 100
            ep = (1 - r["uW_per_dim"] / mac["uW_per_dim"]) * 100
            ra = (1 - sig3(r["um2_per_dim"]) / sig3(mac["um2_per_dim"])) * 100
            rp = (1 - sig3(r["uW_per_dim"]) / sig3(mac["uW_per_dim"])) * 100
            print(f"  {r['pe']}: area -{ea:.2f}% / -{ra:.1f}%, power -{ep:.2f}% / -{rp:.1f}%")
    out = os.path.join(HERE, "table1_pe_datapaths.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved '{out}'.")


if __name__ == "__main__":
    main()
