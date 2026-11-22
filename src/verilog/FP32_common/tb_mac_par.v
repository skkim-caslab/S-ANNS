`timescale 1ns/1ps
// Parallel FP32 MAC PE testbench: random FP32 operands in ~[-2,2), golden dot
// in double precision. Exact cores must land within truncation+flush bounds;
// Mitchell within its 11.1% per-product bound.
module tb;
    parameter CORE = 1;
    reg clk = 0, rst_n = 0, en = 0;
    reg [255:0] q, v;
    wire [71:0] acc;
    real golden, absacc, tol, accr;
    integer burst, lane, seed;
    reg [31:0] qa, va;

    fp32mac_par #(.CORE(CORE)) u (.clk(clk), .rst_n(rst_n), .en(en),
                                  .q_vec_in(q), .v_vec_in(v), .acc_out(acc));
    always #1 clk = ~clk;

    function [31:0] mkfp(input integer dummy);
        reg [31:0] r;
        begin
            r = $random;
            if ((r % 20) == 0) mkfp = 32'd0;                 // occasional zero
            else mkfp = {r[31], 8'd117 + {5'd0, r[26:24]} + {4'd0, r[30:27]},
                         r[22:0]};                            // exp 117..139-> clamp
        end
    endfunction

    function real fp2r(input [31:0] x);
        integer e;
        real f;
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

    function real v72r(input [71:0] u);
        reg [71:0] mag;
        integer b;
        begin
            mag = u[71] ? (~u + 72'd1) : u;
            v72r = 0.0;
            for (b = 71; b >= 0; b = b - 1)
                v72r = v72r * 2.0 + (mag[b] ? 1.0 : 0.0);
            if (u[71]) v72r = -v72r;
        end
    endfunction

    initial begin
        golden = 0.0; absacc = 0.0;
        #5 rst_n = 1;
        @(negedge clk);
        en = 1;
        for (burst = 0; burst < 300; burst = burst + 1) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                qa = mkfp(0); va = mkfp(0);
                // clamp exponents to <= 127 so |x| < 2 (window assumption)
                if (qa[30:23] > 8'd127) qa[30:23] = 8'd127;
                if (va[30:23] > 8'd127) va[30:23] = 8'd127;
                q[lane*32 +: 32] = qa;
                v[lane*32 +: 32] = va;
                golden = golden + fp2r(qa) * fp2r(va);
                absacc = absacc + ((fp2r(qa) * fp2r(va)) < 0.0 ?
                                   -(fp2r(qa) * fp2r(va)) : (fp2r(qa) * fp2r(va)));
            end
            @(negedge clk);
        end
        en = 0;
        @(negedge clk);
        accr = v72r(acc) / (2.0 ** 61);
        if (CORE == 3) tol = absacc * 0.12;
        else           tol = absacc / (2.0 ** 21) + 1e-9;
        if (((accr - golden) < tol) && ((golden - accr) < tol))
            $display("PASS CORE=%0d acc=%e golden=%e err=%e tol=%e",
                     CORE, accr, golden, accr - golden, tol);
        else
            $display("FAIL CORE=%0d acc=%e golden=%e err=%e tol=%e",
                     CORE, accr, golden, accr - golden, tol);
        $finish;
    end
endmodule
