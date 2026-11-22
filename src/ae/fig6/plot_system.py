"""Fig. 6: System-level (a) memory footprint and (b, c) estimated scan-stage
search throughput at recall@10 targets of 0.9 and 0.95, normalized to FP32.

Input : data/anns/{GIST1M,MSTuring1B,DEEP1B}_*.csv
        data/recall/{gist1M,msturing1b,deep1b,text2image100m}/*.csv
Output: fig6_system.png (this directory)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import paper_eval as pe  # noqa: E402


def main():
    pe.system_fig(out=os.path.join(HERE, "fig6_system.png"))

    # the plotted values (same model as pe.system_fig)
    names = [n for n in pe.N_ if pe.has_data(n)]
    data = {n: pe.load(n) or pe.load_rerank(n) for n in names}
    for n in names:
        nl = int(pe.NLIST_[n])
        for lbl, meth in (("BF16", "bf16"), ("SQ8", "sq8")):
            pts = pe.load_method(n, meth)
            if pts:
                data[n][lbl] = [(nl, np_, rc / 100.0) for np_, rc in pts]
    fmts = [("S=2", ("signed", "2")), ("S=3", ("signed", "3")), ("S=4", ("signed", "4")),
            ("BF16", "BF16"), ("SQ8", "SQ8")]

    def gb(name, key):
        return pe.N_[name] * pe.D_[name] * pe._fmt_bits(key) / 8 / 1e9 + pe.index_gb(name)

    print("\n(a) Footprint incl. IVF index, normalized to FP32  [GB]")
    for n in names:
        cells = "  ".join(f"{l}={gb(n, k) / gb(n, 'FP32'):.2f}" for l, k in fmts[:3])
        print(f"  {n:11s} FP32={gb(n, 'FP32'):7.1f} GB  {cells}  "
              f"S=3: {gb(n, ('signed', '3')):.1f} GB")
    for tgt in (0.90, 0.95):
        print(f"\n({'b' if tgt == 0.90 else 'c'}) Throughput @ recall@10 = {tgt}, normalized to FP32"
              " (n/a = target not reached)")
        for n in names:
            be = pe._scan_bytes(data[n], "FP32", n, tgt)
            if not be:
                print(f"  {n:11s} FP32 does not reach the target (dropped from the panel)")
                continue
            cells = []
            for l, k in fmts:
                bs = pe._scan_bytes(data[n], k, n, tgt)
                cells.append(f"{l}={be / bs:.2f}" if bs else f"{l}=n/a")
            print(f"  {n:11s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
