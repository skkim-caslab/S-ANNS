read_liberty ../libs/asap7.lib
read_verilog synthesized.v
link_design fp32mac_seqsa
read_sdc ../constraints.sdc
report_checks -path_delay max >> ../../../data/verilog/fp32mac_seq.out
report_power >> ../../../data/verilog/fp32mac_seq.out
