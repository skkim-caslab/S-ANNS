"""
Fig. 1: online-scan latency with and without the IVF-PQ look-up table, on a
DPU-based PIM (UPMEM, WRAM-resident table) and on a bank-level MAC PIM
(AiM-style, DRAM-resident table), for each fp32 dataset.

All numbers come from data/motivation/figure_values.csv; nothing is hardcoded
here. The CSV carries its own ratio annotation, which is reused verbatim so the
label can never drift from the measurement.

Input : data/motivation/figure_values.csv
Output: fig1_lut_motiv.png (this directory)
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "..", "data", "motivation", "figure_values.csv")
OUT = os.path.join(HERE, "fig1_lut_motiv.png")
C_NOLUT, C_LUT = "#7ba7cc", "#f2a35e"


def load():
    """csv -> {architecture: {dataset: (direct_ms, lut_ms, ratio_label)}}, order preserved"""
    archs, dsets, tbl = [], [], {}
    for r in csv.DictReader(open(CSV)):
        a, d = r["architecture"], r["dataset"]
        if a not in archs:
            archs.append(a)
        if d not in dsets:
            dsets.append(d)
        label = r["annotation"].split()[0].replace("x", r"$\times$")
        tbl.setdefault(a, {})[d] = (float(r["direct_scan_ms"]),
                                    float(r["ivfpq_lut_ms"]), label)
    return archs, dsets, tbl


def main():
    archs, dsets, tbl = load()
    n = len(dsets)
    x = np.arange(n * len(archs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.6, 1.75))
    labelled = False
    for ai, a in enumerate(archs):
        for di, d in enumerate(dsets):
            slot = ai * n + di
            nolut, lut, ratio = tbl[a][d]
            ax.bar(slot - w/2, nolut, w, color=C_NOLUT, edgecolor="black", lw=0.7,
                   zorder=3, label=None if labelled else "Direct scan (no LUT)")
            ax.bar(slot + w/2, lut, w, color=C_LUT, edgecolor="black", lw=0.7,
                   zorder=3, label=None if labelled else "IVF-PQ LUT scan")
            labelled = True
            # sit the ratio over whichever bar is taller, not between the pair
            tall_x = slot - w/2 if nolut >= lut else slot + w/2
            ax.text(tall_x, max(nolut, lut) * 1.35, ratio, ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.axvline(n - 0.5, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylim(4e-3, 60)
    ax.set_xlim(-0.8, n * len(archs) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(dsets * len(archs))
    ax.set_ylabel("Latency (ms)", fontsize=14)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7, zorder=0)
    for ai, a in enumerate(archs):
        ax.text(ai * n + (n - 1) / 2, -0.30, a, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=15)

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.80), ncol=2,
               frameon=False, columnspacing=2.0)
    plt.subplots_adjust(bottom=0.30, top=0.86)
    plt.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved '{OUT}'.")
    for a in archs:
        for d in dsets:
            nolut, lut, ratio = tbl[a][d]
            print(f"  {a:6} {d:11} {nolut:7.4f} / {lut:.4f} ms  {ratio}")


if __name__ == "__main__":
    main()
