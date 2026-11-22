"""Fig. 4: Recall@10 versus nprobe for PoT (S=2/3/4), scalar-quantized storage
(BF16, SQ8) and the Mitchell scan, all ranking identical probed candidates
against the FP32 baseline (GIST1M, MSTuring1B, DEEP1B, T2I-100M).

Input : data/anns/{GIST1M,MSTuring1B,DEEP1B}_*.csv
        data/recall/{gist1M,msturing1b,deep1b,text2image100m}/*.csv
Output: fig4_recall.png (this directory)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import paper_eval as pe  # noqa: E402


def main():
    pe.recall_fig(out=os.path.join(HERE, "fig4_recall.png"))
    # recall at the largest swept nprobe of every curve (saturated recall, %)
    print("\nRecall@10 (%) at the largest swept nprobe:")
    for name in pe.RECALL_PANELS:
        d = pe.load(name)
        nl = pe.NLIST_[name]
        rows = {}
        if "FP32" in d:
            rows["FP32"] = sorted((p, r * 100) for (l, p, r) in d["FP32"] if str(l) == nl)
            for s in ("2", "3", "4"):
                rows[f"S={s}"] = sorted((p, r * 100) for (l, p, r) in d.get(("signed", s), [])
                                        if str(l) == nl)
        else:
            rows["FP32"] = pe.load_method(name, "exact")
            for s in ("2", "3", "4"):
                rows[f"S={s}"] = pe.load_method(name, f"pot_S{s}")
        for meth, lbl in (("bf16", "BF16"), ("sq8", "SQ8"), ("mitchell", "Mitchell")):
            rows[lbl] = pe.load_method(name, meth)
        cells = "  ".join(f"{k}={v[-1][1]:.2f}@{v[-1][0]}" for k, v in rows.items() if v)
        print(f"  {name:11s} {cells}")


if __name__ == "__main__":
    main()
