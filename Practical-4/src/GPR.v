// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER: 18
//
// MEMBERS:
//   - Matteo Buxman, BXMMAT001
//   - Emmanuel Basua, BSXEMM001

// File        : GPR.v
// Description : General Purpose Register File.
//               8 registers, each 16 bits wide (R0–R7).
//               Two asynchronous (combinational) read ports.
//               One synchronous (clocked, positive-edge) write port.
//
// Task 2 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module GPR (
    input        clk,

    // --- Write port (synchronous) -------------------------------------------
    input        reg_write_en,          // Write enable; write occurs on posedge clk
    input  [2:0] reg_write_dest,        // Destination register address (0–7)
    input  [15:0] reg_write_data,       // Data to write

    // --- Read port 1 (asynchronous) ------------------------------------------
    input  [2:0] reg_read_addr_1,       // Source register 1 address (RS1)
    output [15:0] reg_read_data_1,      // Data from RS1 — available immediately

    // --- Read port 2 (asynchronous) ------------------------------------------
    input  [2:0] reg_read_addr_2,       // Source register 2 address (RS2 / WS for I-type)
    output [15:0] reg_read_data_2       // Data from RS2 — available immediately
);

    reg [15:0] reg_array [7:0];

    integer i;
    initial begin
        for (i = 0; i < 8; i = i + 1)
            reg_array[i] <= 16'd0;
    end

    always @(posedge clk) begin
        if (reg_write_en)
            reg_array[reg_write_dest] <= reg_write_data;
    end

    assign reg_read_data_1 = reg_array[reg_read_addr_1];
    assign reg_read_data_2 = reg_array[reg_read_addr_2];

endmodule
