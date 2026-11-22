// Sequential shift-add FP32 MAC PE: sign XOR and exponent add are single-cycle,
// but the 24x24 significand multiply reuses ONE 25-bit adder per lane over 24
// iterations (classic right-shift form). Steady-state period: 24 iteration
// cycles + 1 accumulate cycle = 25 cycles per burst, so equal streaming
// bandwidth requires 25 copies. Backend/window identical to fp32mac_par.
module fp32mac_seqsa (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         en,
    input  wire [255:0] q_vec_in,
    input  wire [255:0] v_vec_in,
    output reg  [71:0]  acc_out,
    output reg          busy
);
    localparam IDLE = 2'd0, RUN = 2'd1, ACC = 2'd2;
    reg [1:0] state;
    reg [4:0] cnt;
    reg [23:0] m   [0:7];   // multiplicand significand
    reg [23:0] ah  [0:7];   // product high
    reg [23:0] ql  [0:7];   // multiplier, then product low
    reg [7:0]  ea_r [0:7];
    reg [7:0]  eb_r [0:7];
    reg        sgn_r [0:7];
    reg        z_r   [0:7];

    // decode current burst inputs (used only in latch cycles)
    wire        sa_w [0:7]; wire        sb_w [0:7];
    wire [7:0]  ea_w [0:7]; wire [7:0]  eb_w [0:7];
    wire [23:0] siga_w [0:7]; wire [23:0] sigb_w [0:7];
    wire        za_w [0:7]; wire        zb_w [0:7];
    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : dec
            fp32_decode da (.x(q_vec_in[g*32 +: 32]), .sgn(sa_w[g]), .e(ea_w[g]), .sig(siga_w[g]), .is_z(za_w[g]));
            fp32_decode db (.x(v_vec_in[g*32 +: 32]), .sgn(sb_w[g]), .e(eb_w[g]), .sig(sigb_w[g]), .is_z(zb_w[g]));
        end
    endgenerate

    // shared normalize/align backend over the registered products
    wire signed [63:0] mult_res [0:7];
    generate
        for (g = 0; g < 8; g = g + 1) begin : post
            wire [47:0] p48 = {ah[g], ql[g]};
            wire        hi  = p48[47];
            wire [23:0] p24 = hi ? p48[47:24] : p48[46:23];
            wire signed [10:0] e2 = {2'd0, ea_r[g]} + {2'd0, eb_r[g]} - (hi ? 11'd276 : 11'd277);
            wire signed [10:0] sh = e2 + 11'd61;
            wire ok = !z_r[g] && (sh >= 0) && (sh <= 11'sd39);
            wire [63:0] mag = ok ? ({40'd0, p24} << sh[5:0]) : 64'd0;
            assign mult_res[g] = sgn_r[g] ? (-$signed(mag)) : $signed(mag);
        end
    endgenerate
    wire signed [64:0] lvl1 [0:3];
    wire signed [65:0] lvl2 [0:1];
    wire signed [66:0] tree_sum;
    generate
        for (g = 0; g < 4; g = g + 1) begin : add_l1
            assign lvl1[g] = mult_res[2*g] + mult_res[2*g+1];
        end
        for (g = 0; g < 2; g = g + 1) begin : add_l2
            assign lvl2[g] = lvl1[2*g] + lvl1[2*g+1];
        end
    endgenerate
    assign tree_sum = lvl2[0] + lvl2[1];

    integer i;
    reg [24:0] sum25;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; cnt <= 5'd0; acc_out <= 72'd0; busy <= 1'b0;
            for (i = 0; i < 8; i = i + 1) begin
                m[i] <= 24'd0; ah[i] <= 24'd0; ql[i] <= 24'd0;
                ea_r[i] <= 8'd0; eb_r[i] <= 8'd0; sgn_r[i] <= 1'b0; z_r[i] <= 1'b1;
            end
        end else begin
            case (state)
                IDLE: if (en) begin
                    for (i = 0; i < 8; i = i + 1) begin
                        m[i]  <= sigb_w[i]; ah[i] <= 24'd0; ql[i] <= siga_w[i];
                        ea_r[i] <= ea_w[i]; eb_r[i] <= eb_w[i];
                        sgn_r[i] <= sa_w[i] ^ sb_w[i];
                        z_r[i] <= za_w[i] | zb_w[i];
                    end
                    cnt <= 5'd0; busy <= 1'b1; state <= RUN;
                end
                RUN: begin
                    for (i = 0; i < 8; i = i + 1) begin
                        sum25 = {1'b0, ah[i]} + (ql[i][0] ? {1'b0, m[i]} : 25'd0);
                        ah[i] <= sum25[24:1];
                        ql[i] <= {sum25[0], ql[i][23:1]};
                    end
                    cnt <= cnt + 5'd1;
                    if (cnt == 5'd23) state <= ACC;
                end
                ACC: begin
                    // tree_sum reads the old registers, so the next burst can
                    // latch in this same cycle: period = 24 + 1 = 25 cycles.
                    acc_out <= acc_out + { {5{tree_sum[66]}}, tree_sum };
                    if (en) begin
                        for (i = 0; i < 8; i = i + 1) begin
                            m[i]  <= sigb_w[i]; ah[i] <= 24'd0; ql[i] <= siga_w[i];
                            ea_r[i] <= ea_w[i]; eb_r[i] <= eb_w[i];
                            sgn_r[i] <= sa_w[i] ^ sb_w[i];
                            z_r[i] <= za_w[i] | zb_w[i];
                        end
                        cnt <= 5'd0; state <= RUN;
                    end else begin
                        busy <= 1'b0; state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
