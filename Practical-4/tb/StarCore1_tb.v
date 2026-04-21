// =============================================================================
// EEE4120F Practical 4 — StarCore-1 Processor
// File        : StarCore1_tb.v
// Description : Integration testbench for the full StarCore-1 processor (Task 8).
//               Runs the program stored in test.prog and verifies processor
//               behaviour over multiple clock cycles using hierarchical signal
//               references.
//
//               This testbench does NOT drive the processor's datapath signals
//               directly — it only drives the clock and observes internal state
//               via hierarchical references.
//
// *** IMPORTANT — Expected compile behaviour with the skeleton ***
// When you first compile this testbench against the skeleton source files,
// iverilog will report "Unable to bind wire/reg/memory" errors for every
// hierarchical reference below (uut.DU.pc_current, uut.DU.instr, etc.).
// This is EXPECTED. Those signals do not yet exist because the Datapath
// module body is empty. The errors will disappear once you implement the
// internal signal declarations and sub-module instantiations in Datapath.v
// and StarCore1.v as required by Tasks 7 and 8.
//
// Hierarchical signal reference examples (valid after implementation):
//   uut.DU.pc_current              — Program Counter (reg in Datapath)
//   uut.DU.instr                   — Currently fetched instruction word (wire)
//   uut.DU.alu_result              — ALU output (wire)
//   uut.DU.zero_flag               — ALU zero flag (wire)
//   uut.DU.reg_file.reg_array[N]   — Register RN value (inside GPR instance)
//   uut.DU.dm.memory[N]            — Data memory word N (inside DataMemory instance)
//   uut.CU.reg_write               — ControlUnit reg_write output
//   uut.CU.alu_op                  — ControlUnit alu_op output
//
// The instance names used here (DU for Datapath, CU for ControlUnit, reg_file
// for GPR, dm for DataMemory) MUST match the names you use when instantiating
// those modules in StarCore1.v and Datapath.v respectively.
//
// Run:
//   iverilog -Wall -I ../src -o ../build/star_sim \
//       ../src/Parameter.v ../src/ALU.v ../src/GPR.v \
//       ../src/InstructionMemory.v ../src/DataMemory.v \
//       ../src/ALU_Control.v ../src/ControlUnit.v \
//       ../src/Datapath.v ../src/StarCore1.v StarCore1_tb.v
//   cd ../test && ../build/star_sim
//   gtkwave ../waves/star.vcd &
// =============================================================================

`timescale 1ns / 1ps
`include "../src/Parameter.v"

module StarCore1_tb;

    // -------------------------------------------------------------------------
    // Clock
    // -------------------------------------------------------------------------
    reg clk;
    initial clk = 1'b0;
    always  #5 clk = ~clk;     // 10 ns period — 100 MHz

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    StarCore1 uut (.clk(clk));

    // -------------------------------------------------------------------------
    // Waveform dump — captures ALL signals in the design hierarchy
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("../waves/star.vcd");
        $dumpvars(0, StarCore1_tb);
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
    // Check tasks — compare 16-bit values observed via hierarchical reference
    // -------------------------------------------------------------------------
    task check16;
        input [15:0] got;
        input [15:0] expected;
        input [63:0] id;
        begin
            if (got !== expected) begin
                $display("FAIL [T%0d]: got = 0x%h (%0d), expected = 0x%h (%0d)",
                         id, got, got, expected, expected);
                fail_count = fail_count + 1;
            end else
                $display("PASS [T%0d]: value = 0x%h (%0d)", id, got, got);
        end
    endtask

    // -------------------------------------------------------------------------
    // Cycle-by-cycle execution trace
    // This always block fires on every rising clock edge and prints the current
    // processor state. It is your primary debugging tool.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        $display("%0t ns | PC=0x%h | instr=%b | R0=%3d R1=%3d R2=%3d R3=%3d | alu=%0d z=%b",
            $time,
            uut.DU.pc_current,
            uut.DU.instr,
            uut.DU.reg_file.reg_array[0],
            uut.DU.reg_file.reg_array[1],
            uut.DU.reg_file.reg_array[2],
            uut.DU.reg_file.reg_array[3],
            uut.DU.alu_result,
            uut.DU.zero_flag
        );
    end
    // =========================================================================
    // MAIN STIMULUS BLOCK
    // =========================================================================
    initial begin
        $display("=== StarCore-1 Integration Testbench ===");
        $display("=== Program loaded from ./test/test.prog ===");
        $display("=== Data memory loaded from ./test/test.data ===");
        $display("");

        // -----------------------------------------------------------------------
        // Wait for the simulation to run long enough for your program to
        // complete at least one full pass. Adjust SIM_TIME in Parameter.v
        // if your program needs more cycles.
        // -----------------------------------------------------------------------
        `SIM_TIME;

        // -----------------------------------------------------------------------
        // POST-SIMULATION VERIFICATION
        // -----------------------------------------------------------------------
        // -----------------------------------------------------------------------
        // STEP 1: Verify register values after execution.
         // ----------------------------------------------------------------------
        $display("Checking R0 after LD (expect Mem[0] = 0x0001):");
        check16(uut.DU.reg_file.reg_array[0], 16'h0001, test_id);
        test_id = test_id + 1;

        $display("Checking R1 after LD (expect Mem[1] = 0x0002):");
        check16(uut.DU.reg_file.reg_array[1], 16'h0002, test_id);
        test_id = test_id + 1;

        $display("Checking R2 after ADD R0+R1 (expect 0x0001+0x0002 = 0x0003):");
        check16(uut.DU.reg_file.reg_array[2], 16'h0003, test_id);
        test_id = test_id + 1;
        // -----------------------------------------------------------------------
        // STEP 2: Verify data memory after ST instruction.
        // -----------------------------------------------------------------------
        $display("Checking DataMem[2] after ST R2 -> Mem[R1+0]:");
        check16(uut.DU.dm.memory[2], 16'h0003, test_id);
        test_id = test_id + 1;
        // -----------------------------------------------------------------------
        // STEP 3: Verify additional R-type instruction results.
        // -----------------------------------------------------------------------
        $display("Checking SUB R2 (expect 0xFFFF):");
        check16(uut.DU.reg_file.reg_array[2], 16'hFFFF, test_id); 
        test_id = test_id + 1;
        // -----------------------------------------------------------------------
        // STEP 4: Advanced Verification based on test.prog sequence
        // -----------------------------------------------------------------------
        $display("");
        $display("--- STEP 4: Logic, SLT, and Branch Verification ---");

        // 4a. Verify the result of the last R-type to write to R2 (SLT)
        // SLT R2, R0, R1 -> (1 < 2) is true, so R2 should be 1.
        $display("Checking final R2 value (Result of SLT R0 < R1):");
        check16(uut.DU.reg_file.reg_array[2], 16'h0001, test_id);
        test_id = test_id + 1;

        // 4b. Verify the result of the final ADD (R0 = R0 + R0)
        // R0 was 1, after index 8 it should be 2.
        $display("Checking final R0 value (Result of ADD R0, R0, R0):");
        check16(uut.DU.reg_file.reg_array[0], 16'h0002, test_id);
        test_id = test_id + 1;

        // 4c. Verify Branch Logic (BEQ R0, R1, target)
        // At index 9, R0=2 and R1=2. The branch MUST be taken.
        // If taken, PC skips index 10 (the Jump) and lands on index 11.
        // PC usually increments by 1 normally, so target +1 means PC = 9 + 1 + 1 = 11.
        $display("Checking PC to verify BEQ (2==2) took the branch to index 11:");
        // Note: Check if your PC is byte-addressed (x2) or word-addressed.
        check16(uut.DU.pc_current, 16'h000b, test_id); 
        test_id = test_id + 1;

        // 4d. Verify Memory Persistence
        // Check that the ST instruction at index 3 actually stayed in memory.
        $display("Checking Mem[2] persistence (Result of ST R2):");
        check16(uut.DU.dm.memory[2], 16'h0003, test_id);
        test_id = test_id + 1;
        // -----------------------------------------------------------------------
        // Print register and memory state (safe to uncomment after Task 7)
        // -----------------------------------------------------------------------
        $display("");
        $display("--- Final Register File State ---");
        $display("R0=0x%h  R1=0x%h  R2=0x%h  R3=0x%h",
            uut.DU.reg_file.reg_array[0], uut.DU.reg_file.reg_array[1],
            uut.DU.reg_file.reg_array[2], uut.DU.reg_file.reg_array[3]);
        $display("R4=0x%h  R5=0x%h  R6=0x%h  R7=0x%h",
            uut.DU.reg_file.reg_array[4], uut.DU.reg_file.reg_array[5],
            uut.DU.reg_file.reg_array[6], uut.DU.reg_file.reg_array[7]);
        
        $display("");
        $display("--- Final Data Memory State ---");
        $display("Mem[0]=0x%h  Mem[1]=0x%h  Mem[2]=0x%h  Mem[3]=0x%h",
            uut.DU.dm.memory[0], uut.DU.dm.memory[1],
            uut.DU.dm.memory[2], uut.DU.dm.memory[3]);
        $display("Mem[4]=0x%h  Mem[5]=0x%h  Mem[6]=0x%h  Mem[7]=0x%h",
            uut.DU.dm.memory[4], uut.DU.dm.memory[5],
            uut.DU.dm.memory[6], uut.DU.dm.memory[7]);

        // -----------------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------------
        $display("");
        if (fail_count == 0)
            $display("=== ALL %0d INTEGRATION TESTS PASSED ===", test_id - 1);
        else
            $display("=== %0d / %0d INTEGRATION TESTS FAILED ===", fail_count, test_id - 1);

        $finish;
    end

endmodule
