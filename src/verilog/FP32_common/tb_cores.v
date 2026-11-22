`timescale 1ns/1ps
module tb_cores;
    reg  [23:0] a, b;
    wire [47:0] pa, pw, pb;
    reg  [47:0] golden;
    integer n, errors = 0;
    mult24_array   ua (.a(a), .b(b), .p(pa));
    mult24_wallace uw (.a(a), .b(b), .p(pw));
    mult24_boothw  ub (.a(a), .b(b), .p(pb));
    initial begin
        for (n = 0; n < 200000; n = n + 1) begin
            a = {1'b1, $random} ; b = {1'b1, $random};  // significand-like (MSB set)
            if (n < 1000) begin a = $random; b = $random; end  // plus general values
            #1;
            golden = a * b;
            if (pa !== golden) begin errors = errors + 1; $display("array  FAIL a=%h b=%h p=%h g=%h", a, b, pa, golden); end
            if (pw !== golden) begin errors = errors + 1; $display("wallace FAIL a=%h b=%h p=%h g=%h", a, b, pw, golden); end
            if (pb !== golden) begin errors = errors + 1; $display("boothw FAIL a=%h b=%h p=%h g=%h", a, b, pb, golden); end
            if (errors > 5) $finish;
        end
        if (errors == 0) $display("PASS all cores exact over %0d vectors", n);
        $finish;
    end
endmodule
