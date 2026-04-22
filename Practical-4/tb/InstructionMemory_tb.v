// =============================================================================
// EEE4120F Practical 4 — StarCore-1 Processor
// File        : InstructionMemory_tb.v
// Description : Testbench for the Instruction Memory module (Task 3).
//               Walks the PC through all valid addresses and verifies the
//               correct instruction word is output combinationally.
//
// Run:
//   iverilog -Wall -I ../src -o ../build/im_sim ../src/InstructionMemory.v InstructionMemory_tb.v
//   cd ../test && ../build/im_sim
//   gtkwave ../waves/im_tb.vcd &
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module InstructionMemory_tb;

    reg  [15:0] pc;
    wire [15:0] instruction;

    InstructionMemory uut (.pc(pc), .instruction(instruction));

    initial begin
        $dumpfile("./waves/im_tb.vcd");
        $dumpvars(0, InstructionMemory_tb);
    end

    integer fail_count;
    integer test_id;
    integer i;
    // Expected instruction words — these must match the contents of test.prog.
    reg [15:0] expected [0:14];

    initial begin
        fail_count = 0;
        test_id    = 1;

        $display("=== InstructionMemory Testbench ===");

        // Expected program (matches ./test/test.prog exactly)
        expected[0]  = 16'b0000010000000000; // LD  R0, R2+0
        expected[1]  = 16'b0000010001000001; // LD  R1, R2+1
        expected[2]  = 16'b0010000001010000; // ADD R2, R0, R1
        expected[3]  = 16'b0001001010000000; // ST  R2, R1+0
        expected[4]  = 16'b0011000001010000; // SUB R2, R0, R1
        expected[5]  = 16'b0111000001010000; // AND R2, R0, R1
        expected[6]  = 16'b1000000001010000; // OR  R2, R0, R1
        expected[7]  = 16'b1001000001010000; // SLT R2, R0, R1
        expected[8]  = 16'b0010000000000000; // ADD R0, R0, R0
        expected[9]  = 16'b1011000001000001; // BEQ R0, R1, +1
        expected[10] = 16'b1100000001000000; // BNE R0, R1, +0
        expected[11] = 16'b1101000000000000; // JMP 0x000
        expected[12] = 16'b0000000000000000;
        expected[13] = 16'b0000000000000000;
        expected[14] = 16'b0000000000000000;

        // Walk PC through every word-aligned address
        for (i = 0; i < 15; i = i + 1) begin
            pc = i * 2; #5;
            if (instruction !== expected[i]) begin
                $display("FAIL [T%0d]: PC=%0d got %b exp %b",
                         test_id, pc, instruction, expected[i]);
                fail_count = fail_count + 1;
            end else begin
                $display("PASS [T%0d]: PC=%0d instr=%b", test_id, pc, instruction);
            end
            test_id = test_id + 1;
        end

        // Combinational behaviour: change PC without any clock edge
        pc = 16'd0;  #1;
        if (instruction !== expected[0]) begin
            $display("FAIL [T%0d]: combinational read at PC=0", test_id);
            fail_count = fail_count + 1;
        end else begin
            $display("PASS [T%0d]: combinational output observed (no clock needed)", test_id);
        end
        test_id = test_id + 1;

        $display("");
        if (fail_count == 0)
            $display("=== ALL %0d TESTS PASSED ===", test_id - 1);
        else
            $display("=== %0d / %0d TESTS FAILED ===", fail_count, test_id - 1);
        $finish;
    end

endmodule
