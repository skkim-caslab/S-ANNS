"""
Fig. 2 -- motivation schematic: what each bank-level compute option costs in MEMORY
MODULES for the same dataset (DEEP1B, 64 GB DIMMs). Sizes cover the fp32 or
PoT vectors plus the IVF index (centroid table + 64-bit ids).

  (a) DPU (UPMEM-class): the DPU displaces ~50% of the module's capacity
      (UPMEM PIM-DIMMs ship half the density of a standard DIMM).
  (b) fp32 MAC: 0.66% die overhead, still storing 4-byte dimensions.
  (c) S-ANNS shift: 0.44% overhead AND the PoT code (S=3) compresses the
      vectors from 384 GB to 180 GB.

Drawn as module glyphs (DIMM capacity vs compute logic share); the compute
strip is drawn to scale for the DPU (50%) and at minimum visible height for
MAC/shift (their true shares, 0.66%/0.44%, are annotated).

Output: fig2_module_schematic.png (this directory); the module counts are printed.
"""
import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 15,
})

# DEEP1B under the IVF index the evaluation actually uses
N, D, NLIST, CAP = 1_000_000_000, 96, 131072, 64.0
IDX = (NLIST * D * 4 + N * 8) / 1e9            # centroids + 64-bit ids, ~8 GB
FP = N * D * 4 / 1e9 + IDX                     # fp32 vectors + index (GB)
FP_POT = N * D * (15 / 8) / 1e9 + IDX          # PoT S=3 vectors + index (GB)

N_DPU = math.ceil(FP / (CAP * 0.5))
N_MAC = math.ceil(FP / (CAP * (1 - 0.00662)))
N_SH = math.ceil(FP_POT / (CAP * (1 - 0.00436)))

PANELS = [
    ("(a) DPU (UPMEM-class)", N_DPU, "DPU", 0.50, "50%"),
    ("(b) fp32 MAC",          N_MAC, "MAC", 0.10, "0.66%"),
    ("(c) S-ANNS shift",      N_SH,  "Shift", 0.055, "0.44%"),
]
C_BG, C_MEM, C_LOGIC = "#e8f1e6", "white", "#d9d9d9"


def draw_panel(ax, title, count, lname, lfrac, ltext):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.15, 0.55), 9.7, 3.15,
                                boxstyle="round,pad=0.02", fc=C_BG, ec="#1f3b5c", lw=1.4))
    mw, mh, gap, x0, y0 = 3.9, 2.35, 0.5, 0.85, 0.95
    for i in range(2):
        x = x0 + i * (mw + gap)
        lh = mh * lfrac
        if lname == "DPU":
            # DPU displaces the lower half of the module
            ax.add_patch(Rectangle((x, y0 + lh), mw, mh - lh, fc=C_MEM, ec="black", lw=1.2))
            ax.add_patch(Rectangle((x, y0), mw, lh, fc=C_LOGIC, ec="black", lw=1.2))
            ax.text(x + mw/2, y0 + lh + (mh - lh)/2, "Bank", ha="center", va="center", fontsize=13)
            ax.text(x + mw/2, y0 + lh/2, lname, ha="center", va="center", fontsize=13)
        else:
            # PE unit sandwiched between the upper and lower banks
            bh = (mh - lh) / 2
            ax.add_patch(Rectangle((x, y0 + bh + lh), mw, bh, fc=C_MEM, ec="black", lw=1.2))
            ax.add_patch(Rectangle((x, y0 + bh), mw, lh, fc=C_LOGIC, ec="black", lw=1.2))
            ax.add_patch(Rectangle((x, y0), mw, bh, fc=C_MEM, ec="black", lw=1.2))
            ax.text(x + mw/2, y0 + bh + lh + bh/2, "Bank", ha="center", va="center", fontsize=13)
            ax.text(x + mw/2, y0 + bh + lh/2, lname, ha="center", va="center", fontsize=9.5)
            ax.text(x + mw/2, y0 + bh/2, "Bank", ha="center", va="center", fontsize=13)
    ax.text(9.75, 4.15, rf"$\times${count}", ha="right", va="center",
            fontsize=19, fontweight="bold")
    ax.text(0.45, 4.15, f"logic = {ltext} of module", ha="left", va="center",
            fontsize=11.5, color="#333333")
    ax.text(5.0, 0.12, title, ha="center", va="center", fontsize=15)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7))
    for ax, (title, count, lname, lfrac, ltext) in zip(axes, PANELS):
        draw_panel(ax, title, count, lname, lfrac, ltext)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.06)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_module_schematic.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved '{out}'.  modules: DPU x{N_DPU}, MAC x{N_MAC}, shift x{N_SH}")


if __name__ == "__main__":
    main()
