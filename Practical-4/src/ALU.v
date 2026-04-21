// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Matteo Buxman, BXMMAT001
//   - Emmanuel Basua, BSXEMM001

// File        : ALU.v
// Description : 16-bit Arithmetic and Logic Unit (ALU).
//               Implements all arithmetic and logic operations required by
//               the StarCore ISA. This is a purely combinational module —
//               it has no clock input and no internal state.
//
// Task 1 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module ALU (
    input  [15:0] a,            // Operand A  — connected to GPR read data 1
    input  [15:0] b,            // Operand B  — connected to ALUSrc mux output
    input  [ 2:0] alu_control,  // Operation select — driven by ALU_Control unit
    output reg [15:0] result,   // Computed result  — fed to DataMemory and write-back mux
    output         zero         // Zero flag: asserted (1) when result == 16'd0
);

    assign zero = (result == 16'd0);

    always @(*) begin
        case (alu_control)
            3'b000: result = a + b;
            3'b001: result = a - b;
            3'b010: result = ~a;
            3'b011: result = a << b[3:0];
            3'b100: result = a >> b[3:0];
            3'b101: result = a & b;
            3'b110: result = a | b;
            3'b111: result = (a < b) ? 16'd1 : 16'd0;
            default: result = a + b;
        endcase
    end

endmodule
