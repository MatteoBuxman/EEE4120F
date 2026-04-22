// =============================================================================
// EEE4120F Practical 4 — StarCore-1 Processor
// File        : ALU_tb.v
// Description : Testbench for the ALU module (Task 1).
//               Applies all 8 operations with multiple input pairs and checks
//               both the result output and the zero flag.
//               Produces automated PASS/FAIL output and a waveform dump.
//
// Run:
//   iverilog -Wall -I ../src -o ../build/alu_sim ../src/ALU.v ALU_tb.v
//   cd ../test && ../build/alu_sim
//   gtkwave ../waves/alu_tb.vcd &
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module ALU_tb;

    // -------------------------------------------------------------------------
    // DUT port connections
    // Inputs to the DUT are declared as reg (so the testbench can drive them).
    // Outputs from the DUT are declared as wire (driven by the DUT).
    // -------------------------------------------------------------------------
    reg  [15:0] a;
    reg  [15:0] b;
    reg  [ 2:0] alu_control;
    wire [15:0] result;
    wire        zero;

    // -------------------------------------------------------------------------
    // DUT instantiation — named port connections
    // -------------------------------------------------------------------------
    ALU uut (
        .a           (a),
        .b           (b),
        .alu_control (alu_control),
        .result      (result),
        .zero        (zero)
    );

    // -------------------------------------------------------------------------
    // Waveform dump — always include this block
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("./waves/alu_tb.vcd");
        $dumpvars(0, ALU_tb);
    end

    // -------------------------------------------------------------------------
    // Failure counter
    // -------------------------------------------------------------------------
    integer fail_count;
    integer test_id;

    initial begin
        fail_count = 0;
        test_id    = 1;
    end

    // -------------------------------------------------------------------------
    // Reusable check task
    // Compares 'got' against 'expected' and prints PASS or FAIL.
    // Increments fail_count on mismatch.
    // -------------------------------------------------------------------------
    task check_result;
        input [15:0] got;
        input [15:0] expected;
        input [63:0] id;        // test number for display
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d]: result = %0d (0x%h), expected = %0d (0x%h)",
                         id, got, got, expected, expected);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [T%0d]: result = %0d (0x%h)", id, got, got);
            end
        end
    endtask

    task check_zero;
        input got;
        input expected;
        input [63:0] id;
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d] zero flag: got = %b, expected = %b", id, got, expected);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [T%0d] zero flag = %b", id, got);
            end
        end
    endtask

    // =========================================================================
    // STIMULUS AND CHECKING
    // =========================================================================
    initial begin
        $display("=== ALU Testbench ===");
        $display("--- ADD (alu_control = 3'b000) ---");

        alu_control = 3'b000;
        a = 16'd10;     b = 16'd5;      #10;
        check_result(result, 16'd15, test_id); test_id = test_id + 1;

        a = 16'hFFFF;   b = 16'd1;      #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;

        a = 16'd0;      b = 16'd0;      #10;
        check_result(result, 16'd0, test_id); test_id = test_id + 1;


        $display("--- SUB (alu_control = 3'b001) ---");

        alu_control = 3'b001;
        a = 16'd10;     b = 16'd5;      #10;
        check_result(result, 16'd5, test_id); test_id = test_id + 1;

        a = 16'd7;      b = 16'd7;      #10;
        check_result(result, 16'd0, test_id); test_id = test_id + 1;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;

        a = 16'd5;      b = 16'd10;     #10;
        check_result(result, 16'hFFFB, test_id); test_id = test_id + 1;


        $display("--- INV / NOT (alu_control = 3'b010) ---");

        alu_control = 3'b010;
        a = 16'h0000;   b = 16'd0;      #10;
        check_result(result, 16'hFFFF, test_id); test_id = test_id + 1;

        a = 16'hFFFF;   b = 16'd0;      #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;

        a = 16'hA5A5;   b = 16'd0;      #10;
        check_result(result, 16'h5A5A, test_id); test_id = test_id + 1;


        $display("--- SHL (alu_control = 3'b011) ---");

        alu_control = 3'b011;
        a = 16'h0001;   b = 16'd4;      #10;
        check_result(result, 16'h0010, test_id); test_id = test_id + 1;

        a = 16'h0003;   b = 16'd2;      #10;
        check_result(result, 16'h000C, test_id); test_id = test_id + 1;

        a = 16'hFFFF;   b = 16'd8;      #10;
        check_result(result, 16'hFF00, test_id); test_id = test_id + 1;


        $display("--- SHR (alu_control = 3'b100) ---");

        alu_control = 3'b100;
        a = 16'h0080;   b = 16'd4;      #10;
        check_result(result, 16'h0008, test_id); test_id = test_id + 1;

        a = 16'hFFFF;   b = 16'd8;      #10;
        check_result(result, 16'h00FF, test_id); test_id = test_id + 1;

        a = 16'h0001;   b = 16'd1;      #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;


        $display("--- AND (alu_control = 3'b101) ---");

        alu_control = 3'b101;
        a = 16'hFFFF;   b = 16'h0F0F;   #10;
        check_result(result, 16'h0F0F, test_id); test_id = test_id + 1;

        a = 16'hAAAA;   b = 16'h5555;   #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;

        a = 16'h0000;   b = 16'hBEEF;   #10;
        check_result(result, 16'h0000, test_id); test_id = test_id + 1;


        $display("--- OR (alu_control = 3'b110) ---");

        alu_control = 3'b110;
        a = 16'h0F0F;   b = 16'hF0F0;   #10;
        check_result(result, 16'hFFFF, test_id); test_id = test_id + 1;

        a = 16'hAAAA;   b = 16'h5555;   #10;
        check_result(result, 16'hFFFF, test_id); test_id = test_id + 1;

        a = 16'h0000;   b = 16'hBEEF;   #10;
        check_result(result, 16'hBEEF, test_id); test_id = test_id + 1;


        $display("--- SLT (alu_control = 3'b111) ---");

        alu_control = 3'b111;
        a = 16'd5;      b = 16'd10;     #10;
        check_result(result, 16'd1, test_id); test_id = test_id + 1;

        a = 16'd10;     b = 16'd10;     #10;
        check_result(result, 16'd0, test_id); test_id = test_id + 1;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;

        a = 16'd15;     b = 16'd3;      #10;
        check_result(result, 16'd0, test_id); test_id = test_id + 1;


        $display("--- Zero flag edge cases ---");

        // Zero asserted: SUB with equal operands
        alu_control = 3'b001;
        a = 16'h1234;   b = 16'h1234;   #10;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;

        // Zero de-asserted: ADD producing non-zero result
        alu_control = 3'b000;
        a = 16'd3;      b = 16'd4;      #10;
        check_zero(zero, 1'b0, test_id); test_id = test_id + 1;

        // Zero asserted: INV of 16'hFFFF
        alu_control = 3'b010;
        a = 16'hFFFF;   b = 16'd0;      #10;
        check_zero(zero, 1'b1, test_id); test_id = test_id + 1;


        // -----------------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------------
        $display("");
        if (fail_count == 0)
            $display("=== ALL %0d TESTS PASSED ===", test_id - 1);
        else
            $display("=== %0d / %0d TESTS FAILED ===", fail_count, test_id - 1);

        $finish;
    end

endmodule
