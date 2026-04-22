// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Emmanuel Basua, BSXEMM001
//   - Matteo Buxman, BXMMAT001

// File        : StarCore1.v
// Description : Top-level StarCore-1 processor module.
//               Connects the Datapath and ControlUnit together.
//               The only external input is the clock signal; all internal
//               signals flow between the two sub-modules via wires.
//
// Task 8 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module StarCore1 (
    input clk       // System clock — drives both the Datapath and GPR/DataMemory
);

    // =========================================================================
    // INTERNAL CONTROL WIRES
    // =========================================================================
    wire        jump;
    wire        beq;
    wire        bne;
    wire        mem_read;
    wire        mem_write;
    wire        alu_src;
    wire        reg_dst;
    wire        mem_to_reg;
    wire        reg_write;
    wire [1:0]  alu_op;
    wire [3:0]  opcode;
    // =========================================================================
    // DATAPATH INSTANTIATION
    // =========================================================================
    Datapath DU (
        .clk        (clk),
        .jump       (jump),
        .beq        (beq),
        .bne        (bne),
        .mem_read   (mem_read),
        .mem_write  (mem_write),
        .alu_src    (alu_src),
        .reg_dst    (reg_dst),
        .mem_to_reg (mem_to_reg),
        .reg_write  (reg_write),
        .alu_op     (alu_op),
        .opcode     (opcode)    // Output to the Control Unit
    );
    // =========================================================================
    // CONTROL UNIT INSTANTIATION
    // =========================================================================
    ControlUnit CU (
        .opcode     (opcode),   // Input from the Datapath
        .alu_op     (alu_op),
        .jump       (jump),
        .beq        (beq),
        .bne        (bne),
        .mem_read   (mem_read),
        .mem_write  (mem_write),
        .alu_src    (alu_src),
        .reg_dst    (reg_dst),
        .mem_to_reg (mem_to_reg),
        .reg_write  (reg_write)
    );
endmodule
