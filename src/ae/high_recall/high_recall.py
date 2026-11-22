"""Section IV-B (System Capacity and Speed): the high-recall operating points
quoted in the text.

  * accuracy-only S=5 runs at nprobe = 1024 (DEEP1B, T2I-100M), compared with
    the exact scan, BF16 and S=4 on the same top-2000 candidates;
  * the T2I-100M BF16 deficit against the exact scan across nprobe.

Input : data/recall/high_recall/*_nprobe1024.csv  (S=2..5 at nprobe 1024)
        data/recall/text2image100m/allformats_*.csv (Fig. 4 sweep)
Output: printed table
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, "..", "..", "..", "data", "recall")
METHODS = ["exact", "bf16", "sq8", "pot_S2", "pot_S3", "pot_S4", "pot_S5"]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    print("Recall@10 (%) at nprobe = 1024, identical top-2000 candidates")
    print(f"  {'corpus':16s}" + "".join(f"{m:>9s}" for m in METHODS))
    for path in sorted(glob.glob(os.path.join(REC, "high_recall", "*.csv"))):
        rs = {r["method"]: float(r["recall@10"]) * 100 for r in rows(path)}
        name = rs and rows(path)[0]["dataset"]
        print(f"  {name:16s}" + "".join(f"{rs[m]:9.2f}" if m in rs else f"{'-':>9s}"
                                      for m in METHODS))
        print(f"  {'':16s} S=5 gap to exact: {rs['exact'] - rs['pot_S5']:.2f} pt, "
              f"BF16 gap to exact: {rs['exact'] - rs['bf16']:.2f} pt")

    path = glob.glob(os.path.join(REC, "text2image100m", "allformats_*.csv"))[0]
    ex, bf = {}, {}
    for r in rows(path):
        if r["method"] == "exact":
            ex[int(r["nprobe"])] = float(r["recall@10"]) * 100
        elif r["method"] == "bf16":
            bf[int(r["nprobe"])] = float(r["recall@10"]) * 100
    print("\nT2I-100M BF16 deficit against the exact scan (pt, from recall rounded to 0.01%) by nprobe:")
    print("  " + "  ".join(f"{n}:{round(ex[n], 2) - round(bf[n], 2):.2f}" for n in sorted(ex) if n in bf))


if __name__ == "__main__":
    main()
