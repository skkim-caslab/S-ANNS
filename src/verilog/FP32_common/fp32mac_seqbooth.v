// Radix-4 Booth serial FP32 MAC PE: the 24-bit significand multiply runs one
// signed-digit add per cycle over 13 digits (classic right-shift form, one
// 28-bit adder per lane). Steady-state period: 13 + 1 = 14 cycles per burst,
// so equal streaming bandwidth requires 14 copies. Backend as fp32mac_par.
module fp32mac_seqbooth (
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
    reg [3:0] cnt;
    reg [23:0] m    [0:7];
    reg [53:0] r    [0:7];   // {28-bit signed acc, 26-bit multiplier tail}
    reg        prev [0:7];
    reg [7:0]  ea_r [0:7];
    reg [7:0]  eb_r [0:7];
    reg        sgn_r [0:7];
    reg        z_r   [0:7];

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

    // Booth addend per lane (combinational)
    reg signed [27:0] addend [0:7];
    integer j;
    always @* begin
        for (j = 0; j < 8; j = j + 1) begin
            case ({r[j][1], r[j][0], prev[j]})
                3'b000, 3'b111: addend[j] = 28'sd0;
                3'b001, 3'b010: addend[j] = $signed({4'd0, m[j]});
                3'b011:         addend[j] = $signed({3'd0, m[j], 1'b0});
                3'b100:         addend[j] = -$signed({3'd0, m[j], 1'b0});
                default:        addend[j] = -$signed({4'd0, m[j]});   // 101, 110
            endcase
        end
    end

    wire signed [63:0] mult_res [0:7];
    generate
        for (g = 0; g < 8; g = g + 1) begin : post
            wire [47:0] p48 = r[g][47:0];
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
    reg signed [27:0] t28;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; cnt <= 4'd0; acc_out <= 72'd0; busy <= 1'b0;
            for (i = 0; i < 8; i = i + 1) begin
                m[i] <= 24'd0; r[i] <= 54'd0; prev[i] <= 1'b0;
                ea_r[i] <= 8'd0; eb_r[i] <= 8'd0; sgn_r[i] <= 1'b0; z_r[i] <= 1'b1;
            end
        end else begin
            case (state)
                IDLE: if (en) begin
                    for (i = 0; i < 8; i = i + 1) begin
                        m[i] <= sigb_w[i];
                        r[i] <= {28'd0, 2'b00, siga_w[i]};
                        prev[i] <= 1'b0;
                        ea_r[i] <= ea_w[i]; eb_r[i] <= eb_w[i];
                        sgn_r[i] <= sa_w[i] ^ sb_w[i];
                        z_r[i] <= za_w[i] | zb_w[i];
                    end
                    cnt <= 4'd0; busy <= 1'b1; state <= RUN;
                end
                RUN: begin
                    for (i = 0; i < 8; i = i + 1) begin
                        t28 = $signed(r[i][53:26]) + addend[i];
                        r[i] <= { {2{t28[27]}}, t28, r[i][25:2] };
                        prev[i] <= r[i][1];
                    end
                    cnt <= cnt + 4'd1;
                    if (cnt == 4'd12) state <= ACC;
                end
                ACC: begin
                    // next burst latches here as well: period = 13 + 1 = 14.
                    acc_out <= acc_out + { {5{tree_sum[66]}}, tree_sum };
                    if (en) begin
                        for (i = 0; i < 8; i = i + 1) begin
                            m[i] <= sigb_w[i];
                            r[i] <= {28'd0, 2'b00, siga_w[i]};
                            prev[i] <= 1'b0;
                            ea_r[i] <= ea_w[i]; eb_r[i] <= eb_w[i];
                            sgn_r[i] <= sa_w[i] ^ sb_w[i];
                            z_r[i] <= za_w[i] | zb_w[i];
                        end
                        cnt <= 4'd0; state <= RUN;
                    end else begin
                        busy <= 1'b0; state <= IDLE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
