`timescale 1ns/1ps
module tb;
    reg clk = 0, rst_n = 0, en = 0;
    reg [255:0] q, v;
    wire [71:0] acc;
    wire busy;
    real golden, absacc, tol, accr, snapg;
    integer burst, lane;
    integer acc_cnt = 0;
    time first_acc = 0, last_acc = 0;
    reg [31:0] qa, va;

    fp32mac_seqbooth u (.clk(clk), .rst_n(rst_n), .en(en),
            .q_vec_in(q), .v_vec_in(v), .acc_out(acc), .busy(busy));
    always #1 clk = ~clk;

    always @(posedge clk) if (rst_n && u.state == 2'd2) begin
        if (acc_cnt == 0) first_acc = $time;
        last_acc = $time;
        acc_cnt = acc_cnt + 1;
    end

    function [31:0] mkfp(input integer dummy);
        reg [31:0] r;
        begin
            r = $random;
            if ((r % 20) == 0) mkfp = 32'd0;
            else begin
                mkfp = {r[31], 8'd117, r[22:0]};
                mkfp[30:23] = 8'd117 + {5'd0, r[26:24]} + {4'd0, r[30:27]};
                if (mkfp[30:23] > 8'd127) mkfp[30:23] = 8'd127;
            end
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
        // phase A: one-shot bursts, functional check
        for (burst = 0; burst < 40; burst = burst + 1) begin
            @(negedge clk);
            for (lane = 0; lane < 8; lane = lane + 1) begin
                qa = mkfp(0); va = mkfp(0);
                q[lane*32 +: 32] = qa;
                v[lane*32 +: 32] = va;
                golden = golden + fp2r(qa) * fp2r(va);
                absacc = absacc + ((fp2r(qa)*fp2r(va)) < 0.0 ? -(fp2r(qa)*fp2r(va)) : (fp2r(qa)*fp2r(va)));
            end
            en = 1;
            @(negedge clk);
            en = 0;
            wait (busy == 1);
            wait (busy == 0);
        end
        @(negedge clk);
        accr = v72r(acc) / (2.0 ** 61);
        tol = absacc / (2.0 ** 21) + 1e-9;
        if (((accr - golden) < tol) && ((golden - accr) < tol))
            $display("PASS phase A acc=%e golden=%e err=%e", accr, golden, accr - golden);
        else
            $display("FAIL phase A acc=%e golden=%e err=%e tol=%e", accr, golden, accr - golden, tol);

        // phase B: back-to-back, expect one accumulate every 14 cycles
        snapg = v72r(acc); acc_cnt = 0;
        for (lane = 0; lane < 8; lane = lane + 1) begin
            q[lane*32 +: 32] = 32'h3f800000;   // 1.0
            v[lane*32 +: 32] = 32'h3f800000;
        end
        @(negedge clk);
        en = 1;
        repeat (1 + 5*14) @(negedge clk);
        en = 0;
        wait (busy == 0);
        @(negedge clk);
        accr = (v72r(acc) - snapg) / (2.0 ** 61);
        if ((accr > 47.9) && (accr < 48.1) && acc_cnt == 6 &&
            (last_acc - first_acc) == (acc_cnt - 1) * 14 * 2)
            $display("PASS phase B: 6 bursts, period %0d cycles",
                      (last_acc - first_acc) / ((acc_cnt - 1) * 2));
        else
            $display("FAIL phase B: delta=%e acc_cnt=%0d spacing=%0d",
                      accr, acc_cnt,
                      acc_cnt > 1 ? (last_acc - first_acc) / ((acc_cnt - 1) * 2) : 0);
        $finish;
    end
endmodule
