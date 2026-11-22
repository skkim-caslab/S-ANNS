// Common building blocks for the FP32 PE suite: FP32 field decode (FTZ),
// width-parametric carry-save stages, 24x24 mantissa-multiplier cores, and
// Wallace reduction trees. All PEs share the same alignment/accumulation
// backend (24-bit significand terms barrel-shifted into a 64-bit fixed window).

module fp32_decode (
    input  wire [31:0] x,
    output wire        sgn,
    output wire [7:0]  e,
    output wire [23:0] sig,
    output wire        is_z
);
    assign sgn  = x[31];
    assign e    = x[30:23];
    assign is_z = (x[30:23] == 8'd0);   // zeros and denormals flush to zero
    assign sig  = {1'b1, x[22:0]};      // hidden bit restored
endmodule

module csa_w #(parameter W = 48) (
    input  wire [W-1:0] a,
    input  wire [W-1:0] b,
    input  wire [W-1:0] c,
    output wire [W-1:0] sum,
    output wire [W-1:0] carry
);
    assign sum   = a ^ b ^ c;
    assign carry = ((a & b) | (a & c) | (b & c)) << 1;
endmodule

module csa_stage_w #(
    parameter N_IN = 24,
    parameter W    = 48
) (
    input  wire [N_IN*W-1:0]                  in_bus,
    output wire [(2*(N_IN/3)+N_IN%3)*W-1:0]   out_bus
);
    localparam N_GRP = N_IN / 3;
    localparam N_REM = N_IN % 3;
    genvar g;
    generate
        for (g = 0; g < N_GRP; g = g + 1) begin : grp
            csa_w #(W) u_csa (
                .a    (in_bus[(3*g+0)*W +: W]),
                .b    (in_bus[(3*g+1)*W +: W]),
                .c    (in_bus[(3*g+2)*W +: W]),
                .sum  (out_bus[(2*g+0)*W +: W]),
                .carry(out_bus[(2*g+1)*W +: W])
            );
        end
        for (g = 0; g < N_REM; g = g + 1) begin : rem
            assign out_bus[(2*N_GRP+g)*W +: W] = in_bus[(3*N_GRP+g)*W +: W];
        end
    endgenerate
endmodule

// Unsigned 24x24 partial products, one 48-bit row per multiplier bit.
module ppgen24 (
    input  wire [23:0]      a,
    input  wire [23:0]      b,
    output wire [24*48-1:0] pp_bus
);
    wire [47:0] ax = {24'd0, a};
    genvar i;
    generate
        for (i = 0; i < 24; i = i + 1) begin : rows
            assign pp_bus[i*48 +: 48] = b[i] ? (ax << i) : 48'd0;
        end
    endgenerate
endmodule

// Array core: linear carry-save chain (one CSA row per bit) + final CPA.
module mult24_array (
    input  wire [23:0] a,
    input  wire [23:0] b,
    output wire [47:0] p
);
    wire [24*48-1:0] pp_bus;
    ppgen24 u_pp (.a(a), .b(b), .pp_bus(pp_bus));
    wire [47:0] cs [0:22];
    wire [47:0] cc [0:22];
    assign cs[0] = pp_bus[0*48 +: 48];
    assign cc[0] = pp_bus[1*48 +: 48];
    genvar k;
    generate
        for (k = 1; k <= 22; k = k + 1) begin : chain
            csa_w #(48) u_row (
                .a(cs[k-1]), .b(cc[k-1]), .c(pp_bus[(k+1)*48 +: 48]),
                .sum(cs[k]), .carry(cc[k])
            );
        end
    endgenerate
    assign p = cs[22] + cc[22];
endmodule

// Wallace core: 3:2 tree 24->16->11->8->6->4->3->2 + final CPA.
module mult24_wallace (
    input  wire [23:0] a,
    input  wire [23:0] b,
    output wire [47:0] p
);
    wire [24*48-1:0] pp_bus;
    ppgen24 u_pp (.a(a), .b(b), .pp_bus(pp_bus));
    wire [16*48-1:0] s1;
    wire [11*48-1:0] s2;
    wire [8*48-1:0]  s3;
    wire [6*48-1:0]  s4;
    wire [4*48-1:0]  s5;
    wire [3*48-1:0]  s6;
    wire [2*48-1:0]  s7;
    csa_stage_w #(24, 48) st1 (pp_bus, s1);
    csa_stage_w #(16, 48) st2 (s1, s2);
    csa_stage_w #(11, 48) st3 (s2, s3);
    csa_stage_w #(8, 48)  st4 (s3, s4);
    csa_stage_w #(6, 48)  st5 (s4, s5);
    csa_stage_w #(4, 48)  st6 (s5, s6);
    csa_stage_w #(3, 48)  st7 (s6, s7);
    assign p = s7[47:0] + s7[95:48];
endmodule

// Booth+Wallace core: radix-4 Booth over {2'b00,b} (13 signed digits), 13 rows
// summed mod 2^48 by a 3:2 tree 13->9->6->4->3->2 + final CPA.
module mult24_boothw (
    input  wire [23:0] a,
    input  wire [23:0] b,
    output wire [47:0] p
);
    wire [47:0] ax = {24'd0, a};
    wire [47:0] an = -ax;
    wire [25:0] b2 = {2'b00, b};
    wire [13*48-1:0] pp_bus;
    genvar j;
    generate
        for (j = 0; j < 13; j = j + 1) begin : booth
            wire [2:0] d;
            if (j == 0) assign d = {b2[1], b2[0], 1'b0};
            else        assign d = {b2[2*j+1], b2[2*j], b2[2*j-1]};
            reg [47:0] ppv;
            always @* begin
                case (d)
                    3'b000, 3'b111: ppv = 48'd0;
                    3'b001, 3'b010: ppv = ax;
                    3'b011:         ppv = ax << 1;
                    3'b100:         ppv = an << 1;
                    default:        ppv = an;       // 101, 110
                endcase
            end
            assign pp_bus[j*48 +: 48] = ppv << (2*j);
        end
    endgenerate
    wire [9*48-1:0] s1;
    wire [6*48-1:0] s2;
    wire [4*48-1:0] s3;
    wire [3*48-1:0] s4;
    wire [2*48-1:0] s5;
    csa_stage_w #(13, 48) st1 (pp_bus, s1);
    csa_stage_w #(9, 48)  st2 (s1, s2);
    csa_stage_w #(6, 48)  st3 (s2, s3);
    csa_stage_w #(4, 48)  st4 (s3, s4);
    csa_stage_w #(3, 48)  st5 (s4, s5);
    assign p = s5[47:0] + s5[95:48];
endmodule
