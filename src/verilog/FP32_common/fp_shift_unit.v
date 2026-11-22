// S-ANN FP32-native shift PE, 8 lanes, one 32-byte burst per cycle.
// The database dimension is stored as S PoT codes: {sign, kidx[3:0]} when
// SIGNED (5 bits/term) or {kidx[3:0]} (4 bits/term); kidx==0 is the null term.
// Q_i x (+-2^k) needs no multiplier in FP32: the term's fixed-point image is the
// 24-bit query significand barrel-shifted by E_Q + kidx (one small adder), which
// folds the float-to-fixed alignment into the same shift that implements the
// multiply. The window/backend matches the MAC PEs: accumulator units are
// 2^(KMIN-48) relative to the real dot product (KMIN = window bottom exponent).
module fp_shift_unit #(
    parameter S      = 3,
    parameter SIGNED = 1
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       en,
    input  wire [255:0]               q_vec_in,
    input  wire [8*S*(SIGNED+4)-1:0]  v_enc_vec_in,
    output reg  [71:0]                acc_out
);
    localparam TB = SIGNED + 4;   // bits per term

    wire signed [65:0] lane_sum [0:7];

    genvar i, t, k;
    generate
        for (i = 0; i < 8; i = i + 1) begin : lanes
            wire        sq, zq;
            wire [7:0]  eq;
            wire [23:0] sigq;
            fp32_decode dq (.x(q_vec_in[i*32 +: 32]), .sgn(sq), .e(eq), .sig(sigq), .is_z(zq));

            // The S terms of one dimension share the query significand and
            // differ only in the 4-bit code, so they are combined at narrow
            // width first (shift by c-1, at most 14 positions); one alignment
            // shifter per dimension then moves the sum into the fixed window.
            wire signed [38:0] term [0:S-1];
            for (t = 0; t < S; t = t + 1) begin : terms
                wire [TB-1:0] f = v_enc_vec_in[(i*S+t)*TB +: TB];
                wire [3:0]    c = f[3:0];                       // 0 = null term
                wire          sv = SIGNED ? f[4] : 1'b0;        // 1 = negative term
                wire [37:0] mag = (c == 4'd0) ? 38'd0
                                : ({14'd0, sigq} << (c - 4'd1));
                assign term[t] = sv ? (-$signed({1'b0, mag})) : $signed({1'b0, mag});
            end

            wire signed [40:0] inner;
            if (S == 2) begin : s2
                assign inner = term[0] + term[1];
            end else if (S == 3) begin : s3
                assign inner = term[0] + term[1] + term[2];
            end else begin : s4
                assign inner = (term[0] + term[1]) + (term[2] + term[3]);
            end

            // alignment: value = inner * 2^(eq-150+KMIN); window LSB 2^(KMIN-48)
            wire signed [8:0] osh = {1'b0, eq} - 9'd102;
            wire ok = !zq && (osh >= 0) && (osh <= 9'sd25);
            wire signed [65:0] shifted = $signed({{25{inner[40]}}, inner}) << osh[4:0];
            wire signed [65:0] lane_v = ok ? (sq ? -shifted : shifted) : 66'sd0;
            assign lane_sum[i] = lane_v;
        end
    endgenerate

    wire signed [66:0] lvl1 [0:3];
    wire signed [67:0] lvl2 [0:1];
    wire signed [68:0] tree_sum;
    generate
        for (k = 0; k < 4; k = k + 1) begin : add_l1
            assign lvl1[k] = lane_sum[2*k] + lane_sum[2*k+1];
        end
        for (k = 0; k < 2; k = k + 1) begin : add_l2
            assign lvl2[k] = lvl1[2*k] + lvl1[2*k+1];
        end
    endgenerate
    assign tree_sum = lvl2[0] + lvl2[1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)      acc_out <= 72'd0;
        else if (en)     acc_out <= acc_out + { {3{tree_sum[68]}}, tree_sum };
    end
endmodule
