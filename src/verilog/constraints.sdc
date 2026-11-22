create_clock -name clk -period 2.857 [get_ports clk]

set_clock_uncertainty 0.1 [get_clocks clk]

set_input_delay -clock clk 1.0 [all_inputs]

set_output_delay -clock clk 0.5 [all_outputs]

set_load 5.0 [all_outputs]

set_max_fanout 16 [current_design]
set_max_transition 0.2 [current_design]
