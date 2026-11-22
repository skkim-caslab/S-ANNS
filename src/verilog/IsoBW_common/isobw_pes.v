// Iso-bandwidth PE suite: every design consumes one 32-byte burst of STORED
// vector data per cycle, so the lane count follows the storage format:
//   FP32 (32 b/dim)      ->  8 lanes
//   bf16 (16 b/dim)      -> 16 lanes
//   int8 ( 8 b/dim)      -> 32 lanes
//   S-ANN 5S b/dim       -> floor(256/5S) lanes (25 / 17 / 12 for S=2/3/4)
// The alignment window, reduction tree and accumulator are shared by every PE
// (red_tree below), so differences are confined to the per-lane arithmetic.
//
// Query side follows the recall experiments: bf16 and asymmetric int8 keep an
// FP32 query (the per-dimension affine scale/offset of SQ8 folds into the query
// once per query, so the per-candidate operation is fp32 x uint8); symmetric
// int8 quantizes the query too and accumulates in pure integer.

// ---------------- shared balanced reduction tree (<=32 inputs) ----------------
module red_tree #(
    parameter N = 8,
    parameter W = 64
) (
    input  wire [N*W-1:0]       in_bus,
    output wire signed [W+4:0]  sum
);
    wire signed [W+4:0] l0 [0:31];
    wire signed [W+4:0] l1 [0:15];
    wire signed [W+4:0] l2 [0:7];
    wire signed [W+4:0] l3 [0:3];
    wire signed [W+4:0] l4 [0:1];
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : pad
            if (i < N) assign l0[i] = $signed(in_bus[i*W +: W]);
            else       assign l0[i] = 0;
        end
        for (i = 0; i < 16; i = i + 1) begin : t1
            assign l1[i] = l0[2*i] + l0[2*i+1];
        end
        for (i = 0; i < 8; i = i + 1) begin : t2
            assign l2[i] = l1[2*i] + l1[2*i+1];
        end
        for (i = 0; i < 4; i = i + 1) begin : t3
            assign l3[i] = l2[2*i] + l2[2*i+1];
        end
        for (i = 0; i < 2; i = i + 1) begin : t4
            assign l4[i] = l3[2*i] + l3[2*i+1];
        end
    endgenerate
    assign sum = l4[0] + l4[1];
endmodule


// ---------------- radix-4 Booth 24x8 (unsigned), 5 partial products ----------
module mult24x8_bw (
    input  wire [23:0] a,
    input  wire [7:0]  b,
    output wire [31:0] p
);
    localparam W = 34;
    wire [W-1:0] ax = {10'd0, a};
    wire [W-1:0] an = -ax;
    wire [9:0]   b2 = {2'b00, b};
    wire [5*W-1:0] pp_bus;
    genvar j;
    generate
        for (j = 0; j < 5; j = j + 1) begin : booth
            wire [2:0] d;
            if (j == 0) assign d = {b2[1], b2[0], 1'b0};
            else        assign d = {b2[2*j+1], b2[2*j], b2[2*j-1]};
            reg [W-1:0] ppv;
            always @* begin
                case (d)
                    3'b000, 3'b111: ppv = {W{1'b0}};
                    3'b001, 3'b010: ppv = ax;
                    3'b011:         ppv = ax << 1;
                    3'b100:         ppv = an << 1;
                    default:        ppv = an;
                endcase
            end
            assign pp_bus[j*W +: W] = ppv << (2*j);
        end
    endgenerate
    wire [4*W-1:0] s1;
    wire [3*W-1:0] s2;
    wire [2*W-1:0] s3;
    csa_stage_w #(5, W) st1 (pp_bus, s1);
    csa_stage_w #(4, W) st2 (s1, s2);
    csa_stage_w #(3, W) st3 (s2, s3);
    wire [W-1:0] full = s3[W-1:0] + s3[2*W-1:W];
    assign p = full[31:0];
endmodule


// ---------------- FP32 x FP32 (reference point, 8 lanes) --------------------
module pe_fp32 #(
    parameter LANES = 8
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 en,
    input  wire [LANES*32-1:0]  q_vec_in,
    input  wire [LANES*32-1:0]  v_vec_in,
    output reg  [71:0]          acc_out
);
    wire [LANES*64-1:0] term_bus;
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lanes
            wire sa, sb, za, zb;
            wire [7:0] ea, eb;
            wire [23:0] siga, sigb;
            fp32_decode da (.x(q_vec_in[i*32 +: 32]), .sgn(sa), .e(ea), .sig(siga), .is_z(za));
            fp32_decode db (.x(v_vec_in[i*32 +: 32]), .sgn(sb), .e(eb), .sig(sigb), .is_z(zb));
            wire [47:0] p48;
            mult24_boothw u_core (.a(siga), .b(sigb), .p(p48));
            wire hi = p48[47];
            wire [23:0] p24 = hi ? p48[47:24] : p48[46:23];
            wire signed [10:0] sh = {2'd0, ea} + {2'd0, eb} - (hi ? 11'd276 : 11'd277) + 11'd61;
            wire ok = !za && !zb && (sh >= 0) && (sh <= 11'sd39);
            wire [63:0] mag = ok ? ({40'd0, p24} << sh[5:0]) : 64'd0;
            assign term_bus[i*64 +: 64] = (sa ^ sb) ? (-$signed(mag)) : $signed(mag);
        end
    endgenerate
    wire signed [68:0] tree_sum;
    red_tree #(LANES, 64) u_red (term_bus, tree_sum);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  acc_out <= 72'd0;
        else if (en) acc_out <= acc_out + { {3{tree_sum[68]}}, tree_sum };
    end
endmodule


// ---------------- FP32 query x bf16 stored (16 lanes) ----------------------
module pe_bf16 #(
    parameter LANES = 16
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 en,
    input  wire [LANES*32-1:0]  q_vec_in,
    input  wire [LANES*16-1:0]  v_vec_in,
    output reg  [71:0]          acc_out
);
    wire [LANES*64-1:0] term_bus;
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lanes
            wire sa, za;
            wire [7:0] ea;
            wire [23:0] siga;
            fp32_decode da (.x(q_vec_in[i*32 +: 32]), .sgn(sa), .e(ea), .sig(siga), .is_z(za));
            wire [15:0] vb = v_vec_in[i*16 +: 16];
            wire        sb = vb[15];
            wire [7:0]  eb = vb[14:7];
            wire [7:0]  sigb = {1'b1, vb[6:0]};
            wire        zb = (eb == 8'd0);
            wire [31:0] p32;
            mult24x8_bw u_core (.a(siga), .b(sigb), .p(p32));
            // p32 in [2^30, 2^32): take the top 24 bits, same exponent math as fp32
            wire hi = p32[31];
            wire [23:0] p24 = hi ? p32[31:8] : p32[30:7];
            wire signed [10:0] sh = {2'd0, ea} + {2'd0, eb} - (hi ? 11'd276 : 11'd277) + 11'd61;
            wire ok = !za && !zb && (sh >= 0) && (sh <= 11'sd39);
            wire [63:0] mag = ok ? ({40'd0, p24} << sh[5:0]) : 64'd0;
            assign term_bus[i*64 +: 64] = (sa ^ sb) ? (-$signed(mag)) : $signed(mag);
        end
    endgenerate
    wire signed [68:0] tree_sum;
    red_tree #(LANES, 64) u_red (term_bus, tree_sum);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  acc_out <= 72'd0;
        else if (en) acc_out <= acc_out + { {3{tree_sum[68]}}, tree_sum };
    end
endmodule


// ---------------- FP32 query x uint8 code, asymmetric SQ8 (32 lanes) -------
// The per-dimension scale folds into the query once per query, so the
// per-candidate work is significand(24) x code(8); the product is aligned by
// the query exponent alone. Window LSB is 2^(-55) here (32-bit magnitude,
// 32 binades of query exponent range) but the width, tree and accumulator
// match every other PE.
module pe_int8a #(
    parameter LANES = 32
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 en,
    input  wire [LANES*32-1:0]  q_vec_in,
    input  wire [LANES*8-1:0]   v_vec_in,
    output reg  [71:0]          acc_out
);
    wire [LANES*64-1:0] term_bus;
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lanes
            wire sa, za;
            wire [7:0] ea;
            wire [23:0] siga;
            fp32_decode da (.x(q_vec_in[i*32 +: 32]), .sgn(sa), .e(ea), .sig(siga), .is_z(za));
            wire [7:0] code = v_vec_in[i*8 +: 8];
            wire [31:0] p32;
            mult24x8_bw u_core (.a(siga), .b(code), .p(p32));
            wire signed [10:0] sh = {3'd0, ea} - 11'd95;      // eq=127 -> 32
            wire ok = !za && (code != 8'd0) && (sh >= 0) && (sh <= 11'sd32);
            wire [63:0] mag = ok ? ({32'd0, p32} << sh[5:0]) : 64'd0;
            assign term_bus[i*64 +: 64] = sa ? (-$signed(mag)) : $signed(mag);
        end
    endgenerate
    wire signed [68:0] tree_sum;
    red_tree #(LANES, 64) u_red (term_bus, tree_sum);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  acc_out <= 72'd0;
        else if (en) acc_out <= acc_out + { {3{tree_sum[68]}}, tree_sum };
    end
endmodule


// ---------------- int8 x int8 symmetric (32 lanes) -------------------------
// Both operands quantized, one shared scale, so no alignment is needed and the
// accumulation is pure integer -- the cheapest possible "small int8 multiplier".
module pe_int8s #(
    parameter LANES = 32
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 en,
    input  wire [LANES*8-1:0]   q_vec_in,
    input  wire [LANES*8-1:0]   v_vec_in,
    output reg  [31:0]          acc_out
);
    wire [LANES*16-1:0] term_bus;
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lanes
            wire signed [7:0] qa = q_vec_in[i*8 +: 8];
            wire signed [7:0] vb = v_vec_in[i*8 +: 8];
            assign term_bus[i*16 +: 16] = qa * vb;
        end
    endgenerate
    wire signed [20:0] tree_sum;
    red_tree #(LANES, 16) u_red (term_bus, tree_sum);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  acc_out <= 32'd0;
        else if (en) acc_out <= acc_out + { {11{tree_sum[20]}}, tree_sum };
    end
endmodule


// ---------------- S-ANN shift PE, lane count parameterized ----------------
module pe_shift #(
    parameter LANES  = 17,
    parameter S      = 3,
    parameter SIGNED = 1
) (
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            en,
    input  wire [LANES*32-1:0]             q_vec_in,
    input  wire [LANES*S*(SIGNED+4)-1:0]   v_enc_vec_in,
    output reg  [71:0]                     acc_out
);
    localparam TB = SIGNED + 4;
    wire [LANES*66-1:0] lane_bus;
    genvar i, t;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lanes
            wire sq, zq;
            wire [7:0] eq;
            wire [23:0] sigq;
            fp32_decode dq (.x(q_vec_in[i*32 +: 32]), .sgn(sq), .e(eq), .sig(sigq), .is_z(zq));
            wire signed [38:0] term [0:S-1];
            for (t = 0; t < S; t = t + 1) begin : terms
                wire [TB-1:0] f = v_enc_vec_in[(i*S+t)*TB +: TB];
                wire [3:0]    c = f[3:0];
                wire          sv = SIGNED ? f[4] : 1'b0;
                wire [37:0] mag = (c == 4'd0) ? 38'd0 : ({14'd0, sigq} << (c - 4'd1));
                assign term[t] = sv ? (-$signed({1'b0, mag})) : $signed({1'b0, mag});
            end
            wire signed [40:0] inner;
            if (S == 2)      assign inner = term[0] + term[1];
            else if (S == 3) assign inner = term[0] + term[1] + term[2];
            else             assign inner = (term[0] + term[1]) + (term[2] + term[3]);
            wire signed [8:0] osh = {1'b0, eq} - 9'd102;
            wire ok = !zq && (osh >= 0) && (osh <= 9'sd25);
            wire signed [65:0] shifted = $signed({{25{inner[40]}}, inner}) << osh[4:0];
            assign lane_bus[i*66 +: 66] = ok ? (sq ? -shifted : shifted) : 66'sd0;
        end
    endgenerate
    wire signed [70:0] tree_sum;
    red_tree #(LANES, 66) u_red (lane_bus, tree_sum);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)  acc_out <= 72'd0;
        else if (en) acc_out <= acc_out + { tree_sum[70], tree_sum };
    end
endmodule
