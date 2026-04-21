// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Matteo Buxman, BXMMAT001
//   - Emmanuel Basua, BSXEMM001

// File        : InstructionMemory.v
// Description : Instruction Memory (ROM).
//               16 words × 16 bits. Contents loaded at simulation start from
//               the binary file ./test/test.prog using $readmemb.
//               This is a purely combinational module — the instruction output
//               updates immediately when the PC changes. No clock input.
//
// Task 3 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module InstructionMemory (
    input  [15:0] pc,           // Program Counter (byte address)
    output [15:0] instruction   // Fetched 16-bit instruction word
);

    reg [`COL-1:0] memory [`ROW_I-1:0];

    wire [3:0] rom_addr = pc[4:1];

    initial begin
        $readmemb("./test/test.prog", memory, 0, 14);
    end

    assign instruction = memory[rom_addr];

endmodule
