"""
Shared plotting code for the evaluation figures of the S-ANNS paper.
Each figure has a thin driver in its own folder (src/ae/fig4, fig5, fig6).

Style: shared horizontal legend above the axes, panel names below the x-axis,
large fonts on a canvas about 2x the print width. The recall baseline is
labeled FP32 (an encoding), the PE-cost reference line MAC (a datapath);
S shades run 4-3-2 light to dark so S=4 stays clear of the black baseline
curve it nearly overlaps.

  recall_fig()  -> Fig. 4: Recall@10 vs nprobe per float32 dataset.
  pe_cost_fig() -> Fig. 5: (a) area, (b) power per streamed dimension,
                   normalized to the FP32 MAC.
  system_fig()  -> Fig. 6: (a) normalized footprint, (b, c) normalized
                   throughput at recall@10 = 0.9 / 0.95 (FP32 = 1).

Inputs (paths are resolved relative to the repository root):
  data/anns/<dataset>_*.csv       FP32 + PoT IVF recall sweeps (GIST1M, MSTuring1B, DEEP1B)
  data/recall/<dataset>/*.csv     same-candidates rerank runs (Mitchell, BF16/SQ8,
                                  all formats; T2I-100M FP32/PoT curves live here)
  data/verilog/*.out              Yosys/OpenSTA reports of the PE datapaths
"""
import os, csv, glob, re, math
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
})

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
ANN = os.path.join(ROOT, "data", "anns")
VERI = os.path.join(ROOT, "data", "verilog")
REC = os.path.join(ROOT, "data", "recall")
N_ = {"GIST1M": 1_000_000, "MSTuring1B": 1_000_000_000, "DEEP1B": 1_000_000_000,
      "T2I-100M": 100_000_000}
D_ = {"GIST1M": 960, "MSTuring1B": 100, "DEEP1B": 96, "T2I-100M": 200}
NLIST_ = {"GIST1M": "16384", "MSTuring1B": "262144", "DEEP1B": "131072",
          "T2I-100M": "32768"}
RECALL_PANELS = ["GIST1M", "MSTuring1B", "DEEP1B", "T2I-100M"]
SHORT_ = {"GIST1M": "GIST", "MSTuring1B": "MSTuring", "DEEP1B": "DEEP",
          "T2I-100M": "T2I"}


def has_data(name):
    """A dataset joins the figures once its recall CSVs exist locally."""
    return bool(load(name) or load_method(name, "exact"))
TBITS, EBITS, RTGT = 5, 32, 0.90
IDBYTES = 8            # FAISS stores 64-bit vector ids in the inverted lists


def index_gb(name):    # IVF index = fp32 centroid table + one id per vector
    return (int(NLIST_[name]) * D_[name] * 4 + N_[name] * IDBYTES) / 1e9


SHADE = {"2": "#08306b", "3": "#2b7bba", "4": "#92c5de"}
C_BASE = "#1a1a1a"
C_MITCH = "#c0392b"
# Same-candidates recall runs (src/ann/02_3_* Mitchell, 02_4_* scalar quant,
# 02_5_* all formats); a curve is drawn only when a CSV matching the panel's
# nlist exists.
MITCH_DIR = {"GIST1M": os.path.join(REC, "gist1M"),
             "MSTuring1B": os.path.join(REC, "msturing1b"),
             "DEEP1B": os.path.join(REC, "deep1b"),
             "T2I-100M": os.path.join(REC, "text2image100m")}


def load_method(name, method, patterns=("mitchell_*", "scalarq*", "allformats*")):
    """[(nprobe, recall%)] for one method at the panel's nlist, from the
    same-candidates experiment CSVs (Mitchell / scalar-quant / all-formats)."""
    d = MITCH_DIR.get(name)
    if not d:
        return []
    pts = {}
    for pat in patterns:
        for f in glob.glob(os.path.join(d, "%s_nlist%s_*.csv" % (pat, NLIST_[name]))) +                  glob.glob(os.path.join(d, "%s_nlist%s.csv" % (pat, NLIST_[name]))):
            for r in csv.DictReader(open(f)):
                if r["method"] == method and r["nlist"] == NLIST_[name]:
                    pts[int(r["nprobe"])] = float(r["recall@10"]) * 100
    return sorted(pts.items())


def load_mitchell(name):
    return load_method(name, "mitchell")


def _read(u): return open(os.path.join(VERI, u + ".out")).read()
def area(u): return float(re.search(r"Chip area for (?:top )?module\s+'[^']+':\s+([\d.]+)", _read(u)).group(1))
def power(u): return float(re.findall(r"^Total\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+[\d.eE+\-]+\s+([\d.eE+\-]+)", _read(u), re.M)[-1])


def load(name):
    d = {}
    for f in sorted(glob.glob(os.path.join(ANN, f"{name}_*.csv"))):
        for r in csv.DictReader(open(f)):
            if not r["recall@10"]:
                continue
            k = "FP32" if r["method"] == "original" else (r["mode"], r["S"])
            d.setdefault(k, []).append((int(r["nlist"]) if r["nlist"] else 0,
                                        int(r["nprobe"]), float(r["recall@10"])))
    return d


# ---------------- recall figure (2-column grid, single column width) ----------------
def recall_fig(out="eval_recall.png"):
    panels = [p for p in RECALL_PANELS if has_data(p)]
    ncol = 2
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.4, 2.05 * nrow))
    axf = np.atleast_1d(axes).ravel()
    for extra_ax in axf[len(panels):]:      # hide unused grid slots (odd panel count)
        extra_ax.axis("off")
    ymins = {"GIST1M": 55}
    for ax, name in zip(axf, panels):
        d = load(name)
        nl = NLIST_[name]
        curves = []                     # (xs, ys, plot kwargs) for main + inset
        if "FP32" in d:
            ex = sorted([(np_, rc * 100) for (l, np_, rc) in d["FP32"] if str(l) == nl])
        else:   # T2I-100M: every curve lives in the all-formats rerank CSV
            ex = load_method(name, "exact")
        curves.append(([p[0] for p in ex], [p[1] for p in ex],
                       dict(ls=(0, (4, 2)), lw=1.8, color=C_BASE,
                            label="FP32", zorder=5)))
        for s in ("2", "3", "4"):
            if "FP32" in d:
                pts = sorted([(np_, rc * 100) for (l, np_, rc) in d.get(("signed", s), [])
                              if str(l) == nl])
            else:
                pts = load_method(name, f"pot_S{s}")
            if pts:
                curves.append(([p[0] for p in pts], [p[1] for p in pts],
                               dict(ls="-", lw=1.4, color=SHADE[s],
                                    label=f"$S$={s}", zorder=3)))
        # the Mitchell sweep may cover nprobe values the FP32 run does not;
        # clip it to the shared range so every curve spans the same x
        extra = [("bf16", "#3e8f4a", 3), ("sq8", "#c9971d", 3),
                 ("mitchell", C_MITCH, 4)]
        for meth, col, zo in extra:
            mpts = [p for p in load_method(name, meth)
                    if ex[0][0] <= p[0] <= ex[-1][0]]
            if mpts:
                lbl = {"bf16": "BF16", "sq8": "SQ8", "mitchell": "Mitchell"}[meth]
                curves.append(([p[0] for p in mpts], [p[1] for p in mpts],
                               dict(ls="-", lw=1.4, color=col, label=lbl, zorder=zo)))
        for xs, ys, kw in curves:
            ax.plot(xs, ys, **kw)
        ax.set_xscale("log", base=2)
        if ymins.get(name) is not None:
            ax.set_ylim(ymins[name], 101)
            vis = [p[0] for p in ex if p[1] >= ymins[name]]
            if vis:
                ax.set_xlim(min(vis) * 0.8, max(p[0] for p in ex) * 1.25)
        ax.grid(ls=":", lw=0.5, alpha=0.7)
        ax.tick_params(labelsize=11)

        # zoom inset in the empty upper-left corner: the crowded top cluster
        # (FP32 / S=3 / S=4 / bf16 / SQ8) over the last two nprobe octaves
        xhi = ex[-1][0]
        xlo = xhi / 2
        cluster = {"FP32", "$S$=3", "$S$=4", "BF16", "SQ8"}
        ys_in = [y for xs, ys, kw in curves if kw.get("label") in cluster
                 for x, y in zip(xs, ys) if xlo <= x <= xhi]
        axins = ax.inset_axes([0.10, 0.56, 0.36, 0.40])
        for xs, ys, kw in curves:
            kw2 = dict(kw); kw2.pop("label", None)
            kw2["lw"] = kw["lw"] * 0.55        # thinner in the zoom so close curves separate
            axins.plot(xs, ys, **kw2)
        axins.set_xscale("log", base=2)
        axins.set_xlim(xlo * 0.95, xhi * 1.05)
        pad = (max(ys_in) - min(ys_in)) * 0.18 + 0.05
        axins.set_ylim(min(ys_in) - pad, min(max(ys_in) + pad, 100.0))
        axins.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=3, integer=True, prune="upper"))
        axins.tick_params(labelsize=6.5, pad=1.2, labelbottom=False, bottom=False)
        axins.grid(ls=":", lw=0.4, alpha=0.6)
        ax.indicate_inset_zoom(axins, edgecolor="0.45", lw=0.7)
    for r in range(nrow):               # y-label on each left-column panel
        axf[r * ncol].set_ylabel("Recall@10 (%)", fontsize=13)
    for ax, name, tag in zip(axf, panels,
                             [f"({c})" for c in "abcdefg"]):
        e = int(math.log2(int(NLIST_[name])))          # nlist is always a power of two
        ax.text(0.5, -0.17, f"{tag} {name} (nlist=$2^{{{e}}}$)",
                transform=ax.transAxes, ha="center", va="top", fontsize=12)
    h, l = [], []
    for a in axf[:len(panels)]:        # union of labels across all panels
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    order = ["FP32", "$S$=2", "$S$=3", "$S$=4", "BF16", "SQ8", "Mitchell"]
    pairs = sorted(zip(h, l), key=lambda p: order.index(p[1]) if p[1] in order else 99)
    h, l = [p[0] for p in pairs], [p[1] for p in pairs]
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=len(l),
               frameon=False, columnspacing=0.9, handletextpad=0.35, fontsize=11.5)
    plt.subplots_adjust(bottom=0.13, top=0.92, wspace=0.22, hspace=0.42)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved '{out}'.")


# ---------------- PE cost figure: (a) area, (b) power ----------------
def pe_cost_fig(out="eval_pe_cost.png"):
    # Per-dimension cost at one 32-byte burst per cycle, FP32 queries throughout.
    # FP32 designs carry 8 dims/burst (serial units replicated 25/14x to hold the
    # burst rate); the S-ANN units are the burst-rate lane counts (25/17/12 dims).
    # Reference = FP32 Booth+Wallace MAC per dim.
    base = [("Seq", "fp32mac_seq", 25, 8), ("BSer", "fp32mac_bser", 14, 8),
            ("B+W", "fp32mac_boothw", 1, 8), ("Mit", "fp32mac_mitchell", 1, 8)]
    cfgs = [("S2s", "isobw_shift_s2", 25), ("S3s", "isobw_shift_s3", 17),
            ("S4s", "isobw_shift_s4", 12)]
    Am, Pm = area("fp32mac_boothw") / 8, power("fp32mac_boothw") / 8
    vals = {"(a) Area": [f * area(u) / d / Am for _, u, f, d in base] + [area(u) / l / Am for _, u, l in cfgs],
            "(b) Power": [f * power(u) / d / Pm for _, u, f, d in base] + [power(u) / l / Pm for _, u, l in cfgs]}
    colors = {"(a) Area": "#7ba7cc", "(b) Power": "#f2a35e"}
    C_BASEBAR = "#b8b8b8"

    def _fmt(v):
        return f"{v:.1f}" if v >= 3 else f"{v:.2f}"

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 1.57))
    nb = len(base)
    n_all = nb + len(cfgs)
    x = np.arange(n_all)
    YMAX = 1.2
    for ax, (ttl, v) in zip(axes, vals.items()):
        barcols = [C_BASEBAR] * nb + [colors[ttl]] * len(cfgs)
        for xi, vi, ci in zip(x, v, barcols):
            h = min(vi, YMAX)          # off-scale bars fill the axis
            ax.bar(xi, h, 0.6, color=ci, edgecolor="black", lw=0.7, zorder=3)
            if vi > YMAX:              # annotate only the clipped bars
                ax.text(xi, h - 0.04, f"{vi:.1f}$\\times$", ha="center", va="top",
                        fontsize=9, fontweight="bold", rotation=90, zorder=5)
        ax.axhline(1.0, color="black", lw=1.1, ls="--", zorder=2)
        ax.axvline(nb - 0.5, color="gray", lw=0.8, ls=":", alpha=0.8)
        ax.set_ylim(0, YMAX)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks(x)
        ax.set_xticklabels(["Seq", "BSer", "B+W", "Mit"] + [c[0][1] for c in cfgs])
        for lbl in ax.get_xticklabels()[:nb]:
            lbl.set_fontsize(9.5)
        for lbl in ax.get_xticklabels()[nb:]:
            lbl.set_fontsize(12)
        ax.set_xlim(-0.7, n_all - 0.3)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.7, zorder=0)
        ax.tick_params(labelsize=12)
        # group names, then the panel name
        ax.text(1.5, -0.42, "Multiplier", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12)
        ax.text(5.0, -0.42, "S-ANNS $S$",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=12)
        ax.text(0.5, -0.78, ttl, transform=ax.transAxes, ha="center", va="top", fontsize=14)
    axes[0].set_ylabel("Norm. to\nFP32 MAC", fontsize=13)
    plt.subplots_adjust(bottom=0.44, top=0.96, wspace=0.22)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved '{out}'.")


# ---------------- system figure (1x2) ----------------
def _bits(mode, s):
    """bits per PoT term: signed carries a sign bit, unsigned does not"""
    return (TBITS if mode == "signed" else TBITS - 1) * int(s)


def _np_at_target(d, key, name, tgt=RTGT):
    """nprobe where recall crosses tgt at the panel's nlist (log2-interpolated,
    so the result is not quantized to the swept grid)."""
    nl = int(NLIST_[name])
    pts = sorted((np_, rc) for (l, np_, rc) in d.get(key, []) if l == nl)
    if not pts:
        return None
    if pts[0][1] >= tgt:
        return pts[0][0]
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        if r0 < tgt <= r1:
            f = (tgt - r0) / (r1 - r0)
            return 2 ** (math.log2(x0) + f * (math.log2(x1) - math.log2(x0)))
    return None


def _fmt_bits(key):
    if key == "FP32":
        return EBITS
    if key in ("BF16", "SQ8"):
        return {"BF16": 16, "SQ8": 8}[key]
    return _bits(*key)


def _scan_bytes(d, key, name, tgt=RTGT):
    """Bytes the bank streams per query at the recall target. Index traversal
    runs on the host, so only the probed lists count (C = nprobe*N/nlist);
    nlist is fixed per dataset so every encoding scans the same lists."""
    np_ = _np_at_target(d, key, name, tgt)
    if np_ is None:
        return None
    bpv = D_[name] * _fmt_bits(key) / 8.0
    return np_ * (N_[name] / int(NLIST_[name])) * bpv


def load_rerank(name):
    """All-formats rerank CSV -> the load() dict shape (recall as a fraction).
    Used for T2I-100M, whose FP32/PoT curves live in the rerank CSV only."""
    nl = int(NLIST_[name])
    d = {}
    for key, meth in [("FP32", "exact"), (("signed", "2"), "pot_S2"),
                      (("signed", "3"), "pot_S3"), (("signed", "4"), "pot_S4")]:
        pts = load_method(name, meth)
        if pts:
            d[key] = [(nl, np_, rc / 100.0) for np_, rc in pts]
    return d


C_BF16, C_SQ8 = "#3e8f4a", "#c9971d"          # match the recall-figure palette


def system_fig(out="eval_system.png"):
    names = [n for n in N_ if has_data(n)]
    data = {n: load(n) or load_rerank(n) for n in names}
    for n in names:                     # BF16/SQ8 recall curves for iso-target throughput
        nl = int(NLIST_[n])
        for lbl, meth in (("BF16", "bf16"), ("SQ8", "sq8")):
            pts = load_method(n, meth)
            if pts:
                data[n][lbl] = [(nl, np_, rc / 100.0) for np_, rc in pts]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.45))

    def draw(ax, fmts, value, w, panel_names):
        xs = np.arange(len(panel_names))
        for i, (lbl, key, col) in enumerate(fmts):
            for j, name in enumerate(panel_names):
                slot = xs[j] + (i - (len(fmts) - 1) / 2) * w
                v = value(name, key)
                if v:
                    ax.bar(slot, v, w, color=col, edgecolor="black", lw=0.6, zorder=3)
                else:
                    ax.text(slot, 0.06, "n/a", ha="center", va="bottom",
                            fontsize=8, rotation=90, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT_[n] for n in panel_names], fontsize=10.5)
        ax.set_xlim(-0.6, len(panel_names) - 0.4)

    def footprint(name, key):
        fe = N_[name] * D_[name] * EBITS / 8 / 1e9 + index_gb(name)
        fs = N_[name] * D_[name] * _fmt_bits(key) / 8 / 1e9 + index_gb(name)
        return fs / fe

    def throughput(tgt):
        def f(name, key):
            be = _scan_bytes(data[name], "FP32", name, tgt)
            bs = _scan_bytes(data[name], key, name, tgt)
            return be / bs if (be and bs) else 0
        return f

    POT = [(f"$S$={s}", ("signed", s), SHADE[s]) for s in ("2", "3", "4")]
    ALL = POT + [("BF16", "BF16", C_BF16), ("SQ8", "SQ8", C_SQ8)]

    ax = axes[0]
    draw(ax, POT, footprint, 0.26, names)
    ax.axhline(1.0, color="black", lw=1.1, ls="--")
    ax.set_ylim(0, 1.14); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Norm. to FP32", fontsize=12)
    ax.text(0.5, -0.52, "(a) Memory Footprint", transform=ax.transAxes, ha="center", fontsize=13)

    for ax, tgt, tag in ((axes[1], 0.90, "(b) Throughput @ Recall 0.9"),
                         (axes[2], 0.95, "(c) Throughput @ Recall 0.95")):
        # drop datasets whose FP32 baseline cannot reach the target (the
        # target sits at the corpus's recall ceiling, e.g. MSTuring at 0.95)
        reach = [n for n in names if _np_at_target(data[n], "FP32", n, tgt)]
        draw(ax, ALL, throughput(tgt), 0.17, reach)
        ax.axhline(1.0, color="black", lw=1.1, ls="--")
        ax.set_ylim(0, 3.5); ax.set_yticks([0, 1, 2, 3])
        ax.text(0.5, -0.52, tag, transform=ax.transAxes, ha="center", fontsize=13)

    for lbl, key, col in ALL:
        axes[0].bar(np.nan, 0, color=col, edgecolor="black", lw=0.6, label=lbl)

    for ax in axes:
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.7, zorder=0)
        ax.tick_params(labelsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.835), ncol=6, frameon=False,
               columnspacing=1.2, handletextpad=0.45, fontsize=11.5)
    plt.subplots_adjust(bottom=0.38, top=0.82, wspace=0.24)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved '{out}'.")


def main():
    recall_fig()
    pe_cost_fig()
    system_fig()


if __name__ == "__main__":
    main()
