# ASAP7 standard-cell library

`asap7.lib` is a single Liberty file that merges the typical-corner (TT, 0.7 V,
25 °C) SRAM-flavor libraries of the ASAP7 7 nm predictive PDK
(`asap7sc7p5t_{AO,INVBUF,OA,SEQ,SIMPLE}_SRAM_TT_nldm_*.lib`, cell names
`*_ASAP7_75t_SRAM`). It is used unmodified by every `synth.ys` / `sta.tcl` in
this directory tree.

ASAP7 is developed by Arizona State University and ARM Research and distributed
under the BSD 3-Clause license:
<https://github.com/The-OpenROAD-Project/asap7> (standard cells:
`asap7sc7p5t_28`). See that repository for the full license text.
