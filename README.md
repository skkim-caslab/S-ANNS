# S-ANNS: Energy-Efficient Approximate Nearest Neighbor Search via Shift-Based PIM Architecture

[![Paper](https://img.shields.io/badge/IEEE%20CAL-2026-blue)](https://ieeexplore.ieee.org/document/11738663/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FLCA.2026.3743912-blue)](https://doi.org/10.1109/LCA.2026.3743912)

This repository contains the PoT encoder, recall experiments, RTL and figure scripts of **S-ANNS** (Shift-ANNS, IEEE CAL 2026).

> **S-ANNS: Energy-Efficient Approximate Nearest Neighbor Search via Shift-Based PIM Architecture**<br>
> Seulki Kim<sup>1,\*</sup>, Yewon Jang<sup>2,\*</sup>, Minho Kim<sup>1</sup>, Daehoon Kim<sup>2</sup><br>
> <sup>1</sup>DGIST, <sup>2</sup>Yonsei University &nbsp; (<sup>\*</sup>Seulki Kim and Yewon Jang contributed equally to this work.)<br>
> *IEEE Computer Architecture Letters*, vol. 25, no. 2, pp. 436–439, 2026<br>
> [[IEEE Xplore]](https://ieeexplore.ieee.org/document/11738663/) [[DOI]](https://doi.org/10.1109/LCA.2026.3743912) [[Project Page]](https://caslab-yonsei.github.io/publications/cal26-skim/)

## Abstract

Approximate Nearest Neighbor Search (ANNS) underpins Retrieval-Augmented
Generation, recommendation, and vector databases but is memory-bandwidth bound
at scale, motivating Processing-in-Memory (PIM). Existing PIM architectures,
however, trade programmability for efficiency. DPU-based designs support
flexible compression, but their general-purpose cores consume DRAM capacity
critical to billion-scale corpora, while MAC-based bank-level designs stream
only scalar-quantized vectors and cannot support LUT-based compression schemes
commonly used for float32 corpora. We present S-ANNS (Shift-ANNS), a bank-level
PIM architecture that uses Power-of-Two (PoT) quantization to encode float32
vectors as compact signed-shift representations in DRAM. This representation
replaces floating-point multiplication with shift-and-add operations during
distance computation while reducing storage overhead. With S signed terms per
dimension, the shift PE reduces per-dimension area and power relative to an
FP32 MAC by 49.4% and 58.9% at S=2 and by 21.6% and 29.2% at S=4. At S=4, the
maximum recall loss is 0.7 percentage points relative to FP32. Across GIST1M,
MSTuring1B, DEEP1B, and T2I-100M at recall@10 of 0.9, S-ANNS achieves 1.4–1.9×
higher search throughput at 0.3–0.7× the footprint.

## Repository Structure

```text
S-ANNS/
├── src/
│   ├── pot/           # Stand-alone PoT encoder/decoder and a small example
│   ├── ann/           # FAISS IVF recall experiments, per-corpus runners, download helpers
│   ├── verilog/       # RTL of the PE datapaths + Yosys/OpenSTA flow and testbenches
│   └── ae/            # Figure and table scripts (fig1, fig2, fig4, fig5, fig6, table1, high_recall)
├── data/
│   ├── anns/          # FP32 + PoT recall sweeps (GIST1M, MSTuring1B, DEEP1B)
│   ├── recall/        # Same-candidates rerank runs, one folder per corpus
│   ├── verilog/       # Yosys/OpenSTA reports of the Table I PEs
│   └── motivation/    # LUT-scan latency model values (Fig. 1)
├── requirements.txt
└── CITATION.cff
```

Details: [`src/verilog/README.md`](src/verilog/README.md).

## Getting Started

### Requirements

- Python 3.9+ and the packages in [`requirements.txt`](requirements.txt): NumPy and Matplotlib for the figure scripts; pandas and `faiss-cpu` for `src/ann/`
- Synthesis only: Yosys and OpenSTA on `PATH` (reports were produced with Yosys 0.53); Icarus Verilog for the testbenches; the ASAP7 library is included in `src/verilog/libs/`
- Optional: the Times New Roman font (otherwise Matplotlib falls back to its default serif font)

### Setup

```bash
pip install -r requirements.txt
export SANNS_HOME=$(pwd)
```

## Reproducing the Paper Figures and Tables

The figures and tables are redrawn from the result data in `data/`; each script writes its output
next to itself (Table I and Sec. IV-B also print their tables). Fig. 3 is a design diagram and has
no script.

| Figure / Table | Script | Data |
|---|---|---|
| Fig. 1 | `src/ae/fig1/plot_lut_motiv.py` | `data/motivation/` |
| Fig. 2 | `src/ae/fig2/plot_module_schematic.py` | computed in the script |
| Fig. 4 | `src/ae/fig4/plot_recall.py` | `data/anns/`, `data/recall/` |
| Table I | `src/ae/table1/get_table1.py` | `data/verilog/` |
| Fig. 5 | `src/ae/fig5/plot_pe_cost.py` | `data/verilog/` |
| Fig. 6 | `src/ae/fig6/plot_system.py` | `data/anns/`, `data/recall/` |
| Sec. IV-B (high recall) | `src/ae/high_recall/high_recall.py` | `data/recall/` |

```bash
cd $SANNS_HOME/src/ae
for s in fig1/plot_lut_motiv.py fig2/plot_module_schematic.py fig4/plot_recall.py table1/get_table1.py \
         fig5/plot_pe_cost.py fig6/plot_system.py high_recall/high_recall.py; do python3 $s; done
```

Fig. 6 throughput covers the in-bank scan stage only: with the IVF index traversed on the host, a
query streams `nprobe · N / nlist` vectors (`nlist` fixed per corpus), and the `nprobe` that reaches the recall target is
interpolated on a log2 scale from the Fig. 4 sweeps. The footprint includes the IVF index (FP32
centroids and 64-bit vector ids).

## Regenerating the Data (optional)

### Recall Experiments

The CSVs in `data/anns/` and `data/recall/` come from the scripts in `src/ann/`. One runner per
corpus repeats the exact runs and writes to `src/ann/results/<corpus>/`; copy the files over those
in `data/` (the runner headers list the mapping) to redraw the figures. Corpora go to
`data/datasets/<corpus>/` by default; index caches go to `data/cache/` (up to ~50 GB per
billion-scale corpus).

| Corpus | Source | Download |
|---|---|---|
| GIST1M (960-d, L2) | [TEXMEX](http://corpus-texmex.irisa.fr/) (`ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz`) | extract to `data/datasets/gist1M/` |
| DEEP1B (96-d, L2) | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) (Yandex Deep1B) | `src/ann/download_deep1B.sh` |
| MSTuring1B (100-d, L2) | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) (Microsoft Turing-ANNS) | `src/ann/download_msturing1B.sh` |
| T2I-100M (200-d, inner product) | [big-ann-benchmarks](https://big-ann-benchmarks.com/neurips21.html) (Yandex Text-to-Image, first 100M vectors + official 100M ground truth) | `src/ann/download_t2i_100m.sh` |

| Runner (in `src/ann/`) | Disk | Peak RAM | Runtime |
|---|---|---|---|
| `run_gist1m.sh` | ~6 GB | ~16 GB | 2–4 h |
| `run_1b.sh deep1b` | 384 GB | ~250 GB | 1–2 days |
| `run_1b.sh msturing1b` | 400 GB | ~250 GB | 1–2 days |
| `run_t2i100m.sh` | 80 GB | ~60 GB | ~1 day |

```bash
cd $SANNS_HOME/src/ann
bash run_gist1m.sh              # optional overrides: PY=<python> DATA=<dataset dir> CACHE=<cache dir>
bash run_1b.sh deep1b
bash run_1b.sh msturing1b
bash run_t2i100m.sh
```

Runtime: rough estimates for a 28–32-core server, not re-measured for this release.

On the billion-scale corpora and T2I-100M, every method (including FP32) ranks the same 2,000
nearest candidates from an FP16-stored IVF index (`02_3` / `02_4` / `02_5`); the `*_halfC` column
repeats the ranking on the top 1,000 candidates as a truncation check. PoT codes use a 15-position
exponent window (`--bmin/--bmax`: [-14, 0] for GIST1M, [-16, -2] for the other corpora) and 5 bits
per signed term.

### PoT Encoder

`src/pot/pot.py` is a compact, stand-alone PoT encoder/decoder (greedy nearest power of two,
15-position window, 5-bit signed codes) that the RTL testbench mirrors; the experiment scripts carry
their own encoders, which pick the error-minimizing S-term code from the full candidate set.

```bash
cd $SANNS_HOME/src/pot && python3 example.py
```

### Hardware Synthesis

Synthesizes the six FP32 MAC baselines, the BF16 and INT8 MACs and the S-ANNS PEs (S = 2, 3, 4)
with Yosys (ASAP7, typical corner, 0.7 V, 25 °C) and analyzes them with OpenSTA at 350 MHz;
overwrites `data/verilog/*.out`. `run_tb.sh` runs the functional checks (Icarus Verilog),
including the 2,400-product shift-PE test. See [`src/verilog/README.md`](src/verilog/README.md).

```bash
cd $SANNS_HOME/src/verilog
bash run_all.sh && bash run_tb.sh
```

Runtime: about 10–20 minutes.

## Citation

```bibtex
@article{kim2026sanns,
  author  = {Kim, Seulki and Jang, Yewon and Kim, Minho and Kim, Daehoon},
  title   = {{S-ANNS}: Energy-Efficient Approximate Nearest Neighbor Search via Shift-Based {PIM} Architecture},
  journal = {IEEE Computer Architecture Letters},
  volume  = {25},
  number  = {2},
  pages   = {436--439},
  year    = {2026},
  doi     = {10.1109/LCA.2026.3743912}
}
```

## Acknowledgements

The recall experiments use [FAISS](https://github.com/facebookresearch/faiss) and the
[big-ann-benchmarks](https://github.com/harsha-simhadri/big-ann-benchmarks) corpora.
Hardware evaluation uses [Yosys](https://github.com/YosysHQ/yosys),
[OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) and the
[ASAP7](https://github.com/The-OpenROAD-Project/asap7) PDK (BSD 3-Clause, see [`src/verilog/libs/README.md`](src/verilog/libs/README.md)).
