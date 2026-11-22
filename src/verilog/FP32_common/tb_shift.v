`timescale 1ns/1ps
// FP32-native S-ANN shift PE testbench. Queries are random FP32 in [-1,1];
// database values are PoT-encoded in the testbench (greedy nearest, 15-position
// window [kmin,kmax]=[-14,0], code 0 = null) mirroring the golden model
// (src/pot/pot.py). Golden = double-precision dot(q, v_hat); within the
// window everything is exactly representable, so the match must be tight.
module tb;
    parameter S = 3;
    parameter SIGNED = 1;
    localparam TB_ = SIGNED + 4;
    reg clk = 0, rst_n = 0, en = 0;
    reg [255:0] q;
    reg [8*S*TB_-1:0] v_enc;
    wire [71:0] acc;
    real golden, accr, vr, r, a, term, vhat;
    integer burst, lane, t, kf, kk;
    reg [31:0] qa;
    reg sgn;
    reg [3:0] c;

    fp_shift_unit #(.S(S), .SIGNED(SIGNED)) u (.clk(clk), .rst_n(rst_n), .en(en),
        .q_vec_in(q), .v_enc_vec_in(v_enc), .acc_out(acc));
    always #1 clk = ~clk;

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
        golden = 0.0;
        #5 rst_n = 1;
        @(negedge clk);
        en = 1;
        for (burst = 0; burst < 300; burst = burst + 1) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                // query: FP32 in about [-1,1], occasional zero
                qa = $random;
                if ((qa % 20) == 0) qa = 32'd0;
                else qa[30:23] = 8'd107 + {4'd0, qa[30:27]};   // exp 107..122
                q[lane*32 +: 32] = qa;

                // database value in [-1,1] (or [0,1] unsigned), PoT-encode
                vr = $itor($random % 1000000) / 1000000.0;
                if (!SIGNED && vr < 0.0) vr = -vr;
                r = vr; vhat = 0.0;
                for (t = 0; t < S; t = t + 1) begin
                    sgn = (r < 0.0);
                    a = sgn ? -r : r;
                    if ((!SIGNED && r < 0.0) || (a <= (2.0 ** -14) / 2.0)) begin
                        c = 4'd0;
                    end else begin
                        kf = -60;
                        for (kk = -20; kk <= 1; kk = kk + 1)
                            if ((2.0 ** kk) <= a) kf = kk;
                        if (a > 1.5 * (2.0 ** kf)) kf = kf + 1;
                        if (kf < -14) kf = -14;
                        if (kf > 0) kf = 0;
                        c = kf + 15;                     // k - kmin + 1
                        term = (sgn ? -1.0 : 1.0) * (2.0 ** kf);
                        r = r - term;
                        vhat = vhat + term;
                    end
                    if (SIGNED)
                        v_enc[(lane*S+t)*TB_ +: TB_] = {sgn, c};
                    else
                        v_enc[(lane*S+t)*TB_ +: TB_] = c;
                end
                golden = golden + fp2r(qa) * vhat;
            end
            @(negedge clk);
        end
        en = 0;
        @(negedge clk);
        accr = v72r(acc) / (2.0 ** 62);
        if (((accr - golden) < 1e-9) && ((golden - accr) < 1e-9))
            $display("PASS S=%0d SIGNED=%0d acc=%e golden=%e err=%e",
                     S, SIGNED, accr, golden, accr - golden);
        else
            $display("FAIL S=%0d SIGNED=%0d acc=%e golden=%e err=%e",
                     S, SIGNED, accr, golden, accr - golden);
        $finish;
    end
endmodule
