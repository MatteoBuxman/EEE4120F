// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Matteo Buxman, BXMMAT001
//   - Emmanuel Basua, BSXEMM001

// File        : DataMemory.v
// Description : Data Memory (RAM).
//               8 words × 16 bits. Contents loaded at simulation start from
//               the binary file ./test/test.data using $readmemb.
//               Writes are synchronous (positive-edge clocked).
//               Reads are combinational and gated by the mem_read enable.
//
// Task 4 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module DataMemory (
    input        clk,

    // Shared address bus (used for both reads and writes)
    input  [15:0] mem_access_addr,  // Byte address; only lower bits used for indexing

    // Write port
    input  [15:0] mem_write_data,   // Data to store
    input        mem_write_en,      // Assert to write on the next posedge clk

    // Read port
    input        mem_read,          // Assert to enable the read output
    output [15:0] mem_read_data     // Read result; 16'd0 when mem_read is de-asserted
);

    reg [`COL-1:0] memory [`ROW_D-1:0];

    wire [2:0] ram_addr = mem_access_addr[2:0];

    integer log_fd;
    initial begin
        $readmemb("./test/test.data", memory);
    end

    initial begin
        log_fd = $fopen(`DMEM_LOG);
        $fmonitor(log_fd, "t=%0t  [0]=%h [1]=%h [2]=%h [3]=%h",
                  $time, memory[0], memory[1], memory[2], memory[3]);
        `SIM_TIME;
        $fclose(log_fd);
    end

    always @(posedge clk) begin
        if (mem_write_en)
            memory[ram_addr] <= mem_write_data;
    end

    assign mem_read_data = mem_read ? memory[ram_addr] : 16'd0;

endmodule
