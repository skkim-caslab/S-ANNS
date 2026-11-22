// Parallel FP32 MAC PE, 8 lanes, one 32-byte burst per cycle.
// Per lane: sign XOR, 8-bit exponent add, 24x24 significand multiply (CORE
// selects the implementation), truncate-normalize to 24 bits, then barrel-shift
// alignment into a shared signed 64-bit fixed-point window and an adder tree.
// The accumulator holds units of 2^-61 relative to the real dot product.
// CORE: 0 = carry-save array, 1 = Wallace tree, 2 = Booth+Wallace,
//       3 = Mitchell logarithmic approximation (no multiplier at all).
module fp32mac_par #(
    parameter CORE = 1
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         en,
    input  wire [255:0] q_vec_in,
    input  wire [255:0] v_vec_in,
    output reg  [71:0]  acc_out
);
    wire signed [63:0] mult_res [0:7];

    genvar i, k;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lanes
            wire        sa, sb, za, zb;
            wire [7:0]  ea, eb;
            wire [23:0] siga, sigb;
            fp32_decode da (.x(q_vec_in[i*32 +: 32]), .sgn(sa), .e(ea), .sig(siga), .is_z(za));
            fp32_decode db (.x(v_vec_in[i*32 +: 32]), .sgn(sb), .e(eb), .sig(sigb), .is_z(zb));

            wire [23:0] p24;
            wire        hi;
            if (CORE == 0) begin : g_array
                wire [47:0] p48;
                mult24_array u_core (.a(siga), .b(sigb), .p(p48));
                assign hi  = p48[47];
                assign p24 = hi ? p48[47:24] : p48[46:23];
            end else if (CORE == 1) begin : g_wallace
                wire [47:0] p48;
                mult24_wallace u_core (.a(siga), .b(sigb), .p(p48));
                assign hi  = p48[47];
                assign p24 = hi ? p48[47:24] : p48[46:23];
            end else if (CORE == 2) begin : g_boothw
                wire [47:0] p48;
                mult24_boothw u_core (.a(siga), .b(sigb), .p(p48));
                assign hi  = p48[47];
                assign p24 = hi ? p48[47:24] : p48[46:23];
            end else begin : g_mitchell
                // log-domain: characteristics = exponents, fractions add
                wire [23:0] fsum = {1'b0, siga[22:0]} + {1'b0, sigb[22:0]};
                assign hi  = fsum[23];
                assign p24 = hi ? fsum : {1'b1, fsum[22:0]};
            end

            // value = p24 * 2^E2, E2 = ea+eb-277 (+1 when product >= 2)
            wire signed [10:0] e2  = {2'd0, ea} + {2'd0, eb} - (hi ? 11'd276 : 11'd277);
            wire signed [10:0] sh  = e2 + 11'd61;   // fixed window LSB = 2^-61
            wire               ok  = !za && !zb && (sh >= 0) && (sh <= 11'sd39);
            wire [63:0] mag = ok ? ({40'd0, p24} << sh[5:0]) : 64'd0;
            assign mult_res[i] = (sa ^ sb) ? (-$signed(mag)) : $signed(mag);
        end
    endgenerate

    wire signed [64:0] lvl1 [0:3];
    wire signed [65:0] lvl2 [0:1];
    wire signed [66:0] tree_sum;
    generate
        for (k = 0; k < 4; k = k + 1) begin : add_l1
            assign lvl1[k] = mult_res[2*k] + mult_res[2*k+1];
        end
        for (k = 0; k < 2; k = k + 1) begin : add_l2
            assign lvl2[k] = lvl1[2*k] + lvl1[2*k+1];
        end
    endgenerate
    assign tree_sum = lvl2[0] + lvl2[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)      acc_out <= 72'd0;
        else if (en)     acc_out <= acc_out + { {5{tree_sum[66]}}, tree_sum };
    end
endmodule
