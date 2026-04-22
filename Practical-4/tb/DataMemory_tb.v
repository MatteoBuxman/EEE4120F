// =============================================================================
// EEE4120F Practical 4 — StarCore-1 Processor
// File        : DataMemory_tb.v
// Description : Testbench for the Data Memory module (Task 4).
//               Verifies synchronous write, gated combinational read,
//               write followed by immediate read, and disabled-write safety.
//
// Run:
//   iverilog -Wall -I ../src -o ../build/dm_sim ../src/DataMemory.v DataMemory_tb.v
//   cd ../test && ../build/dm_sim
//   gtkwave ../waves/dm_tb.vcd &
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module DataMemory_tb;

    reg        clk;
    reg  [15:0] mem_access_addr;
    reg  [15:0] mem_write_data;
    reg        mem_write_en;
    reg        mem_read;
    wire [15:0] mem_read_data;

    DataMemory uut (
        .clk             (clk),
        .mem_access_addr (mem_access_addr),
        .mem_write_data  (mem_write_data),
        .mem_write_en    (mem_write_en),
        .mem_read        (mem_read),
        .mem_read_data   (mem_read_data)
    );

    initial clk = 1'b0;
    always  #5 clk = ~clk;

    initial begin
        $dumpfile("./waves/dm_tb.vcd");
        $dumpvars(0, DataMemory_tb);
    end

    integer fail_count;
    integer test_id;
    integer i;

    // Expected initial contents from test.data
    reg [15:0] init_data [0:7];

    task check16;
        input [15:0] got;
        input [15:0] expected;
        input [63:0] id;
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d]: got = 0x%h, expected = 0x%h", id, got, expected);
                fail_count = fail_count + 1;
            end else
                $display("PASS [T%0d]: value = 0x%h", id, got);
        end
    endtask

    initial begin
        fail_count      = 0;
        test_id         = 1;
        mem_write_en    = 1'b0;
        mem_read        = 1'b0;
        mem_access_addr = 16'd0;
        mem_write_data  = 16'd0;

        init_data[0] = 16'h0001;
        init_data[1] = 16'h0002;
        init_data[2] = 16'h0003;
        init_data[3] = 16'h0004;
        init_data[4] = 16'h0005;
        init_data[5] = 16'h0006;
        init_data[6] = 16'h0007;
        init_data[7] = 16'h0008;

        $display("=== DataMemory Testbench ===");

        // Wait a clock edge so $readmemb has taken effect and the sim is stable
        @(posedge clk); #1;

        // ------------------------------------------------------------------
        // TEST GROUP 1: Read back initial values loaded from test.data
        // ------------------------------------------------------------------
        $display("--- Group 1: Verify $readmemb initialisation ---");

        mem_read = 1'b1;
        for (i = 0; i < 8; i = i + 1) begin
            mem_access_addr = i[15:0]; #2;
            check16(mem_read_data, init_data[i], test_id);
            test_id = test_id + 1;
        end

        // ------------------------------------------------------------------
        // TEST GROUP 2: Write new values to all 8 locations, then read back
        // ------------------------------------------------------------------
        $display("--- Group 2: Write then read all 8 locations ---");

        for (i = 0; i < 8; i = i + 1) begin
            mem_read        = 1'b0;
            mem_write_en    = 1'b1;
            mem_access_addr = i[15:0];
            mem_write_data  = 16'hA000 + i[15:0];
            @(posedge clk); #1;
            mem_write_en    = 1'b0;

            mem_read        = 1'b1;
            mem_access_addr = i[15:0]; #2;
            check16(mem_read_data, 16'hA000 + i[15:0], test_id);
            test_id = test_id + 1;
        end

        // ------------------------------------------------------------------
        // TEST GROUP 3: mem_read = 0 must produce 16'd0 output
        // ------------------------------------------------------------------
        $display("--- Group 3: mem_read disabled -> output must be 0 ---");

        mem_read        = 1'b0;
        mem_access_addr = 16'd0; #2;
        check16(mem_read_data, 16'd0, test_id); test_id = test_id + 1;

        mem_access_addr = 16'd5; #2;
        check16(mem_read_data, 16'd0, test_id); test_id = test_id + 1;

        // ------------------------------------------------------------------
        // TEST GROUP 4: Write then immediately read on the next cycle
        // ------------------------------------------------------------------
        $display("--- Group 4: Write followed by immediate read ---");

        mem_write_en    = 1'b1;
        mem_access_addr = 16'd3;
        mem_write_data  = 16'hBEEF;
        @(posedge clk); #1;
        mem_write_en    = 1'b0;

        mem_read        = 1'b1;
        mem_access_addr = 16'd3; #2;
        check16(mem_read_data, 16'hBEEF, test_id); test_id = test_id + 1;

        // ------------------------------------------------------------------
        // TEST GROUP 5: Disabled write must not alter memory
        // ------------------------------------------------------------------
        $display("--- Group 5: mem_write_en=0 must not overwrite memory ---");

        mem_write_en    = 1'b0;
        mem_access_addr = 16'd3;
        mem_write_data  = 16'hDEAD;
        @(posedge clk); #1;

        mem_read        = 1'b1;
        mem_access_addr = 16'd3; #2;
        check16(mem_read_data, 16'hBEEF, test_id); test_id = test_id + 1;

        $display("");
        if (fail_count == 0)
            $display("=== ALL %0d TESTS PASSED ===", test_id - 1);
        else
            $display("=== %0d / %0d TESTS FAILED ===", fail_count, test_id - 1);
        $finish;
    end

endmodule
