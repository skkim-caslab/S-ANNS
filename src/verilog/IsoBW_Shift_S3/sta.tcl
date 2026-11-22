read_liberty ../libs/asap7.lib
read_verilog synthesized.v
link_design pe_shift
read_sdc ../constraints.sdc
report_checks -path_delay max >> ../../../data/verilog/isobw_shift_s3.out
report_power >> ../../../data/verilog/isobw_shift_s3.out
