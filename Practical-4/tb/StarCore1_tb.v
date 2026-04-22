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
        $dumpfile("./waves/star.vcd");
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
    // SAFETY TIMEOUT — ends the simulation if the checks below hang.
    // =========================================================================
    initial begin
        `SIM_TIME;
        $display("*** SIM_TIME elapsed before all checks completed ***");
        $finish;
    end

    // =========================================================================
    // MAIN STIMULUS BLOCK
    // Each assertion is gated on pc_current reaching a specific value, so the
    // check fires at the exact cycle the target instruction's write has just
    // committed — regardless of the program looping via JMP 0.
    //
    // Program timing (byte-addressed PC, one instruction per clock):
    //   pc=0  : LD  R0, Mem[0]       R0  <- 0x0001   (committed at pc->2)
    //   pc=2  : LD  R1, Mem[1]       R1  <- 0x0002   (committed at pc->4)
    //   pc=4  : ADD R2, R0, R1       R2  <- 0x0003   (committed at pc->6)
    //   pc=6  : ST  R2, Mem[R1+0]    Mem[2] <- 0x0003 (committed at pc->8)
    //   pc=8  : SUB R2, R0, R1       R2  <- 0xFFFF   (committed at pc->10)
    //   pc=10 : AND R2, R0, R1       R2  <- 0x0000
    //   pc=12 : OR  R2, R0, R1       R2  <- 0x0003
    //   pc=14 : SLT R2, R0, R1       R2  <- 0x0001   (committed at pc->16)
    //   pc=16 : ADD R0, R0, R0       R0  <- 0x0002   (committed at pc->18)
    //   pc=18 : BEQ R0, R1, +1       R0==R1 -> pc_next = 22 (0x16)
    //   pc=22 : JMP 0                pc_next = 0
    // =========================================================================
    initial begin
        $display("=== StarCore-1 Integration Testbench ===");
        $display("=== Program loaded from ./test/test.prog ===");
        $display("=== Data memory loaded from ./test/test.data ===");
        $display("");

        // T1 — R0 after LD
        while (uut.DU.pc_current !== 16'd2)  @(posedge clk);
        $display("Checking R0 after LD (expect Mem[0] = 0x0001):");
        check16(uut.DU.reg_file.reg_array[0], 16'h0001, test_id); test_id = test_id + 1;

        // T2 — R1 after LD
        while (uut.DU.pc_current !== 16'd4)  @(posedge clk);
        $display("Checking R1 after LD (expect Mem[1] = 0x0002):");
        check16(uut.DU.reg_file.reg_array[1], 16'h0002, test_id); test_id = test_id + 1;

        // T3 — R2 after ADD R0+R1
        while (uut.DU.pc_current !== 16'd6)  @(posedge clk);
        $display("Checking R2 after ADD R0+R1 (expect 0x0003):");
        check16(uut.DU.reg_file.reg_array[2], 16'h0003, test_id); test_id = test_id + 1;

        // T4 — Mem[2] after ST
        while (uut.DU.pc_current !== 16'd8)  @(posedge clk);
        $display("Checking DataMem[2] after ST R2 -> Mem[R1+0] (expect 0x0003):");
        check16(uut.DU.dm.memory[2], 16'h0003, test_id); test_id = test_id + 1;

        // T5 — R2 after SUB
        while (uut.DU.pc_current !== 16'd10) @(posedge clk);
        $display("Checking R2 after SUB R0-R1 (expect 0xFFFF):");
        check16(uut.DU.reg_file.reg_array[2], 16'hFFFF, test_id); test_id = test_id + 1;

        // T6 — R2 after SLT (1 < 2 -> 1)
        while (uut.DU.pc_current !== 16'd16) @(posedge clk);
        $display("Checking R2 after SLT R0<R1 (expect 0x0001):");
        check16(uut.DU.reg_file.reg_array[2], 16'h0001, test_id); test_id = test_id + 1;

        // T7 — R0 after ADD R0,R0,R0
        while (uut.DU.pc_current !== 16'd18) @(posedge clk);
        $display("Checking R0 after ADD R0+R0 (expect 0x0002):");
        check16(uut.DU.reg_file.reg_array[0], 16'h0002, test_id); test_id = test_id + 1;

        // T8 — BEQ (R0==R1) must branch to pc = 0x0016 (byte-addressed)
        @(posedge clk);
        $display("Checking PC after BEQ (R0==R1 must branch to 0x0016):");
        check16(uut.DU.pc_current, 16'h0016, test_id); test_id = test_id + 1;

        // T9 — Mem[2] persisted through the full instruction sequence
        $display("Checking Mem[2] persistence end-of-pass (expect 0x0003):");
        check16(uut.DU.dm.memory[2], 16'h0003, test_id); test_id = test_id + 1;
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
