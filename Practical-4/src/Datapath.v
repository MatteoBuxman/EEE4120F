// =========================================================================
// Practical 4: StarCore-1 — Single-Cycle Processor in Verilog
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Emmanuel Basua, BSXEMM001
//   - Matteo Buxman, BXMMAT001

// File        : Datapath.v
// Description : StarCore-1 Datapath.
//               Integrates all sub-components (Tasks 1–6) and implements the
//               full data-flow of the processor. Control signals arrive from
//               an external ControlUnit module (instantiated in StarCore1.v).
//               The opcode of the current instruction is exposed as an output
//               so the ControlUnit can decode it.
//
//               Internal structure (in order of data flow):
//               1.  Program Counter (PC) register
//               2.  PC+2 adder
//               3.  Instruction Memory (ROM)
//               4.  Register-file write-address multiplexer (RegDst)
//               5.  General Purpose Register File (GPR)
//               6.  Immediate sign-extension
//               7.  ALUSrc multiplexer
//               8.  ALU Control Unit
//               9.  ALU
//               10. Branch address adder + branch/sequential mux
//               11. Jump address computation + jump mux
//               12. Data Memory (RAM)
//               13. Write-back multiplexer (MemToReg)
//
// Task 7 — Student Implementation Required
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module Datapath (
    input        clk,

    // --- Control signals from ControlUnit ------------------------------------
    input        jump,          // Select jump target PC
    input        beq,           // Enable branch-on-equal
    input        bne,           // Enable branch-on-not-equal
    input        mem_read,      // Enable data memory read
    input        mem_write,     // Enable data memory write (posedge clk)
    input        alu_src,       // 0 = RS2; 1 = sign-extended immediate
    input        reg_dst,       // 0 = instr[8:6] (I-type); 1 = instr[5:3] (R-type)
    input        mem_to_reg,    // 0 = ALU result; 1 = memory read data
    input        reg_write,     // Enable register file write (posedge clk)
    input  [1:0] alu_op,        // ALU operation class for ALU_Control

    // --- Output to ControlUnit -----------------------------------------------
    output [3:0] opcode         // Instruction opcode field [15:12]
);

    // =========================================================================
    // INTERNAL SIGNAL DECLARATIONS
    // All internal signals that interconnect sub-components go here.
    // =========================================================================

    // --- Program Counter ------------------------------------------------------
    reg  [15:0] pc_current;             // Current PC value (register)
    wire [15:0] pc_next;                // Next PC value (combinational)
    wire [15:0] pc2;                    // PC + 2 (sequential next address)

    // --- Instruction fetch ----------------------------------------------------
    wire [15:0] instr;                  // Fetched instruction word

    // --- Register file --------------------------------------------------------
    wire [2:0]  reg_write_dest;         // Write-back register address (after RegDst mux)
    wire [15:0] reg_write_data;         // Write-back data (after MemToReg mux)
    wire [2:0]  reg_read_addr_1;        // RS1 address (from instr[11:9])
    wire [2:0]  reg_read_addr_2;        // RS2 address (from instr[8:6])
    wire [15:0] reg_read_data_1;        // Data from RS1
    wire [15:0] reg_read_data_2;        // Data from RS2

    // --- Immediate extension --------------------------------------------------
    wire [15:0] ext_im;                 // Sign-extended 6-bit immediate

    // --- ALU ------------------------------------------------------------------
    wire [15:0] alu_operand_b;          // ALUSrc mux output (RS2 or immediate)
    wire [2:0]  alu_control;            // ALU function select from ALU_Control
    wire [15:0] alu_result;             // ALU computed result
    wire        zero_flag;              // ALU zero output

    // --- Branch / Jump PC computation ----------------------------------------
    wire [15:0] pc_branch;              // Branch target address
    wire        beq_taken;              // BEQ condition satisfied
    wire        bne_taken;              // BNE condition satisfied
    wire [15:0] pc_after_branch;        // PC selected after branch evaluation
    wire [12:0] jump_target;            // Jump target (12 bits + appended 0)
    wire [15:0] pc_jump;                // Full 16-bit jump target address

    // --- Data memory ----------------------------------------------------------
    wire [15:0] mem_read_data;          // Data read from memory


    // =========================================================================
    // 1. PROGRAM COUNTER
    // =========================================================================
    initial begin
        pc_current <= 16'd0;
    end

    always @(posedge clk) begin
        pc_current <= pc_next;
    end

    assign pc2 = pc_current + 16'd2; // Standard PC increment 
    // =========================================================================
    // 2. INSTRUCTION MEMORY
    // Instantiate InstructionMemory; connect pc_current and instr.
    // =========================================================================
    InstructionMemory im (
        .pc          (pc_current),
        .instruction (instr)
    );

    assign opcode = instr[15:12];
    // =========================================================================
    // 3. REGISTER FILE WRITE-ADDRESS MULTIPLEXER (RegDst)
    // =========================================================================
    assign reg_write_dest  = reg_dst ? instr[5:3] : instr[8:6];
    assign reg_read_addr_1 = instr[11:9]; // RS1
    assign reg_read_addr_2 = instr[8:6];  // RS2
    // =========================================================================
    // 4. GENERAL PURPOSE REGISTER FILE
    // =========================================================================
    GPR reg_file (
        .clk              (clk),
        .reg_write_en     (reg_write),
        .reg_write_dest   (reg_write_dest),
        .reg_write_data   (reg_write_data),
        .reg_read_addr_1  (reg_read_addr_1),
        .reg_read_data_1  (reg_read_data_1),
        .reg_read_addr_2  (reg_read_addr_2),
        .reg_read_data_2  (reg_read_data_2)
    )
    // =========================================================================
    // 5. IMMEDIATE SIGN-EXTENSION
    // =========================================================================
    assign ext_im = { {10{instr[5]}}, instr[5:0] };
    // =========================================================================
    // 6. ALUSrc MULTIPLEXER
    // =========================================================================
    assign alu_operand_b = alu_src ? ext_im : reg_read_data_2;  
    // =========================================================================
    // 7. ALU CONTROL UNIT
    // =========================================================================
    ALU_Control alu_ctrl (
        .ALUOp   (alu_op),
        .Opcode  (instr[15:12]),
        .ALU_Cnt (alu_control)
    );

    // =========================================================================
    // 8. ALU
    // =========================================================================
    ALU alu_unit (
        .a           (reg_read_data_1),
        .b           (alu_operand_b),
        .alu_control (alu_control),
        .result      (alu_result),
        .zero        (zero_flag)
    );

    // =========================================================================
    // 9. BRANCH ADDRESS COMPUTATION AND PC-NEXT MUX CHAIN
    // =========================================================================

    // Branch Target = (PC+2) + (Offset << 1)
    assign pc_branch       = pc2 + {ext_im[14:0], 1'b0};
    
    // Logic to decide if we branch
    assign beq_taken       = beq & zero_flag;
    assign bne_taken       = bne & ~zero_flag;
    
    // First Mux: Choose between PC+2 and Branch Target
    assign pc_after_branch = (beq_taken | bne_taken) ? pc_branch : pc2;
    
    // Jump Logic: {Upper 3 bits of PC+2, 12-bit offset, 0}
    assign jump_target     = {instr[11:0], 1'b0};
    assign pc_jump         = {pc2[15:13], jump_target};
    
    // Final Mux: Choose between Branch Result and Jump Target
    assign pc_next         = jump ? pc_jump : pc_after_branch;

    // =========================================================================
    // 10. DATA MEMORY
    // =========================================================================
    DataMemory dm (
        .clk             (clk),
        .mem_access_addr (alu_result),
        .mem_write_data  (reg_read_data_2),
        .mem_write_en    (mem_write),
        .mem_read        (mem_read),
        .mem_read_data   (mem_read_data)
    );
    // =========================================================================
    // 11. WRITE-BACK MULTIPLEXER (MemToReg)
    // =========================================================================
    assign reg_write_data = mem_to_reg ? mem_read_data : alu_result;
endmodule
