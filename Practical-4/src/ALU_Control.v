// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Emmanuel Basua, BSXEMM001
//   - Matteo Buxman, BXMMAT001

// File        : ALU_Control.v
// Description : ALU Control Unit.
//               Maps the 2-bit ALUOp signal (from the Main Control Unit) and
//               the 4-bit instruction opcode to the 3-bit ALUcnt signal that
//               drives the ALU's operation select input.
//               This is a purely combinational module.
//
// Task 5 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module ALU_Control (
    input  [1:0] ALUOp,         // From ControlUnit:
                                //   2'b10 = memory access (always ADD for address)
                                //   2'b01 = branch      (always SUB for comparison)
                                //   2'b00 = R-type      (decode from opcode)
    input  [3:0] Opcode,        // Instruction opcode field [15:12]
    output reg [2:0] ALU_Cnt    // To ALU alu_control input
);

    
// Concatenate ALUOp and Opcode into a 6-bit control word
    wire [5:0] control_in;
    assign control_in = {ALUOp, Opcode};

    always @(*) begin
        casex (control_in)
            // Result is always ADD for address offset calculation
            6'b10_xxxx : ALU_Cnt = 3'b000; 

            // Result is always SUB to compare the two registers
            6'b01_xxxx : ALU_Cnt = 3'b001;

            // R-Type Instructions: Mapping Opcode to ALU Function
            6'b00_0010 : ALU_Cnt = 3'b000; // ADD
            6'b00_0011 : ALU_Cnt = 3'b001; // SUB
            6'b00_0100 : ALU_Cnt = 3'b010; // INV 
            6'b00_0101 : ALU_Cnt = 3'b011; // SHL 
            6'b00_0110 : ALU_Cnt = 3'b100; // SHR
            6'b00_0111 : ALU_Cnt = 3'b101; // AND
            6'b00_1000 : ALU_Cnt = 3'b110; // OR
            6'b00_1001 : ALU_Cnt = 3'b111; // SLT

            // Default: Fail-safe to ADD for undefined opcodes
            default    : ALU_Cnt = 3'b000;
        endcase
    end
endmodule
