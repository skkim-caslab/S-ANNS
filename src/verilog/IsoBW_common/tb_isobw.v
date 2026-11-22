`timescale 1ns/1ps
// Verification for the iso-bandwidth PE suite.
//   tb_core   : mult24x8_bw must be bit-exact
//   tb_int8s  : symmetric int8 PE must be bit-exact against an integer golden
//   tb_bf16   : fp32 x bf16 PE against a double-precision golden
//   tb_int8a  : fp32 x uint8 PE against a double-precision golden
//   tb_shift17: 17-lane S=3 shift PE against a double-precision golden

module tb_core;
    reg [23:0] a;
    reg [7:0]  b;
    wire [31:0] p;
    integer n, err;
    mult24x8_bw u (.a(a), .b(b), .p(p));
    initial begin
        err = 0;
        for (n = 0; n < 300000; n = n + 1) begin
            a = $random; b = $random;
            if (n < 512) begin a = {1'b1, 23'($random)}; b = n[7:0]; end
            #1;
            if (p !== (a * b)) begin
                err = err + 1;
                $display("FAIL a=%h b=%h p=%h golden=%h", a, b, p, a * b);
                if (err > 4) $finish;
            end
        end
        if (err == 0) $display("PASS mult24x8_bw exact over %0d vectors", n);
        $finish;
    end
endmodule


module tb_int8s;
    localparam L = 32;
    reg clk = 0, rst_n = 0, en = 0;
    reg [L*8-1:0] q, v;
    wire [31:0] acc;
    reg signed [31:0] golden;
    reg signed [7:0] qa, va;
    integer burst, lane;
    pe_int8s #(L) u (.clk(clk), .rst_n(rst_n), .en(en),
                     .q_vec_in(q), .v_vec_in(v), .acc_out(acc));
    always #1 clk = ~clk;
    initial begin
        golden = 0;
        #5 rst_n = 1;
        @(negedge clk);
        en = 1;
        for (burst = 0; burst < 300; burst = burst + 1) begin
            for (lane = 0; lane < L; lane = lane + 1) begin
                qa = $random; va = $random;
                q[lane*8 +: 8] = qa;
                v[lane*8 +: 8] = va;
                golden = golden + qa * va;
            end
            @(negedge clk);
        end
        en = 0;
        @(negedge clk);
        if (acc === golden) $display("PASS pe_int8s exact acc=%h", acc);
        else $display("FAIL pe_int8s acc=%h golden=%h", acc, golden);
        $finish;
    end
endmodule


module tb_bf16;
    localparam L = 16;
    reg clk = 0, rst_n = 0, en = 0;
    reg [L*32-1:0] q;
    reg [L*16-1:0] v;
    wire [71:0] acc;
    real golden, absacc, accr, tol;
    integer burst, lane;
    reg [31:0] qa, vf;
    reg [15:0] vb;
    pe_bf16 #(L) u (.clk(clk), .rst_n(rst_n), .en(en),
                    .q_vec_in(q), .v_vec_in(v), .acc_out(acc));
    always #1 clk = ~clk;

    function real fp2r(input [31:0] x);
        integer e; real f;
        begin
            if (x[30:23] == 8'd0) fp2r = 0.0;
            else begin
                e = x[30:23] - 127;
                f = 1.0 + $itor({9'd0, x[22:0]}) / 8388608.0;
                fp2r = f * (2.0 ** e);
                if (x[31]) fp2r = -fp2r;
            end
        end
    endfunction
    function real v72r(input [71:0] u_);
        reg [71:0] mag; integer b;
        begin
            mag = u_[71] ? (~u_ + 72'd1) : u_;
            v72r = 0.0;
            for (b = 71; b >= 0; b = b - 1) v72r = v72r * 2.0 + (mag[b] ? 1.0 : 0.0);
            if (u_[71]) v72r = -v72r;
        end
    endfunction

    initial begin
        golden = 0.0; absacc = 0.0;
        #5 rst_n = 1;
        @(negedge clk);
        en = 1;
        for (burst = 0; burst < 200; burst = burst + 1) begin
            for (lane = 0; lane < L; lane = lane + 1) begin
                qa = $random; qa[30:23] = 8'd117 + {5'd0, qa[26:24]};
                vf = $random; vf[30:23] = 8'd117 + {5'd0, vf[26:24]};
                vb = vf[31:16];                       // bf16 = top 16 bits
                q[lane*32 +: 32] = qa;
                v[lane*16 +: 16] = vb;
                golden = golden + fp2r(qa) * fp2r({vb, 16'd0});
                absacc = absacc + ((fp2r(qa) * fp2r({vb, 16'd0})) < 0.0 ?
                                   -(fp2r(qa) * fp2r({vb, 16'd0})) :
                                    (fp2r(qa) * fp2r({vb, 16'd0})));
            end
            @(negedge clk);
        end
        en = 0;
        @(negedge clk);
        accr = v72r(acc) / (2.0 ** 61);
        tol = absacc / (2.0 ** 21) + 1e-9;
        if (((accr - golden) < tol) && ((golden - accr) < tol))
            $display("PASS pe_bf16 acc=%e golden=%e err=%e", accr, golden, accr - golden);
        else
            $display("FAIL pe_bf16 acc=%e golden=%e err=%e tol=%e", accr, golden, accr - golden, tol);
        $finish;
    end
endmodule


module tb_int8a;
    localparam L = 32;
    reg clk = 0, rst_n = 0, en = 0;
    reg [L*32-1:0] q;
    reg [L*8-1:0]  v;
    wire [71:0] acc;
    real golden, accr;
    integer burst, lane;
    reg [31:0] qa;
    reg [7:0] code;
    pe_int8a #(L) u (.clk(clk), .rst_n(rst_n), .en(en),
                     .q_vec_in(q), .v_vec_in(v), .acc_out(acc));
    always #1 clk = ~clk;

    function real fp2r(input [31:0] x);
        integer e; real f;
        begin
            if (x[30:23] == 8'd0) fp2r = 0.0;
            else begin
                e = x[30:23] - 127;
                f = 1.0 + $itor({9'd0, x[22:0]}) / 8388608.0;
                fp2r = f * (2.0 ** e);
                if (x[31]) fp2r = -fp2r;
            end
        end
    endfunction
    function real v72r(input [71:0] u_);
        reg [71:0] mag; integer b;
        begin
            mag = u_[71] ? (~u_ + 72'd1) : u_;
            v72r = 0.0;
            for (b = 71; b >= 0; b = b - 1) v72r = v72r * 2.0 + (mag[b] ? 1.0 : 0.0);
            if (u_[71]) v72r = -v72r;
        end
    endfunction

    initial begin
        golden = 0.0;
        #5 rst_n = 1;
        @(negedge clk);
        en = 1;
        for (burst = 0; burst < 200; burst = burst + 1) begin
            for (lane = 0; lane < L; lane = lane + 1) begin
                qa = $random; qa[30:23] = 8'd117 + {5'd0, qa[26:24]};
                code = $random;
                q[lane*32 +: 32] = qa;
                v[lane*8 +: 8] = code;
                golden = golden + fp2r(qa) * $itor({24'd0, code});
            end
            @(negedge clk);
        end
        en = 0;
        @(negedge clk);
        accr = v72r(acc) / (2.0 ** 55);
        if (((accr - golden) < 1e-3) && ((golden - accr) < 1e-3))
            $display("PASS pe_int8a acc=%e golden=%e err=%e", accr, golden, accr - golden);
        else
            $display("FAIL pe_int8a acc=%e golden=%e err=%e", accr, golden, accr - golden);
        $finish;
    end
endmodule


module tb_shift17;
    localparam L = 17, S = 3, SG = 1, TB_ = SG + 4;
    reg clk = 0, rst_n = 0, en = 0;
    reg [L*32-1:0] q;
    reg [L*S*TB_-1:0] v_enc;
    wire [71:0] acc;
    real golden, accr, vr, r, a, term, vhat;
    integer burst, lane, t, kf, kk;
    reg [31:0] qa;
    reg sgn;
    reg [3:0] c;
    pe_shift #(L, S, SG) u (.clk(clk), .rst_n(rst_n), .en(en),
                            .q_vec_in(q), .v_enc_vec_in(v_enc), .acc_out(acc));
    always #1 clk = ~clk;

    function real fp2r(input [31:0] x);
        integer e; real f;
        begin
            if (x[30:23] == 8'd0) fp2r = 0.0;
            else begin
                e = x[30:23] - 127;
                f = 1.0 + $itor({9'd0, x[22:0]}) / 8388608.0;
                fp2r = f * (2.0 ** e);
                if (x[31]) fp2r = -fp2r;
            end
        end
    endfunction
    function real v72r(input [71:0] u_);
        reg [71:0] mag; integer b;
        begin
            mag = u_[71] ? (~u_ + 72'd1) : u_;
            v72r = 0.0;
            for (b = 71; b >= 0; b = b - 1) v72r = v72r * 2.0 + (mag[b] ? 1.0 : 0.0);
            if (u_[71]) v72r = -v72r;
        end
    endfunction

    initial begin
        golden = 0.0;
        #5 rst_n = 1;
        @(negedge clk);
        en = 1;
        for (burst = 0; burst < 200; burst = burst + 1) begin
            for (lane = 0; lane < L; lane = lane + 1) begin
                qa = $random;
                if ((qa % 20) == 0) qa = 32'd0;
                else qa[30:23] = 8'd107 + {4'd0, qa[30:27]};
                q[lane*32 +: 32] = qa;
                vr = $itor($random % 1000000) / 1000000.0;
                r = vr; vhat = 0.0;
                for (t = 0; t < S; t = t + 1) begin
                    sgn = (r < 0.0);
                    a = sgn ? -r : r;
                    if (a <= (2.0 ** -14) / 2.0) c = 4'd0;
                    else begin
                        kf = -60;
                        for (kk = -20; kk <= 1; kk = kk + 1)
                            if ((2.0 ** kk) <= a) kf = kk;
                        if (a > 1.5 * (2.0 ** kf)) kf = kf + 1;
                        if (kf < -14) kf = -14;
                        if (kf > 0) kf = 0;
                        c = kf + 15;
                        term = (sgn ? -1.0 : 1.0) * (2.0 ** kf);
                        r = r - term;
                        vhat = vhat + term;
                    end
                    v_enc[(lane*S+t)*TB_ +: TB_] = {sgn, c};
                end
                golden = golden + fp2r(qa) * vhat;
            end
            @(negedge clk);
        end
        en = 0;
        @(negedge clk);
        accr = v72r(acc) / (2.0 ** 62);
        if (((accr - golden) < 1e-9) && ((golden - accr) < 1e-9))
            $display("PASS pe_shift17 acc=%e golden=%e err=%e", accr, golden, accr - golden);
        else
            $display("FAIL pe_shift17 acc=%e golden=%e err=%e", accr, golden, accr - golden);
        $finish;
    end
endmodule
