import matplotlib.pyplot as plt
import numpy as np

# Benchmark data: (dataset, threads/procs, tinit, tcomp, ttotal)
omp_data = {
    4:  {1: (0.000080400, 0.000003400, 0.000083800),
         2: (0.000028000, 0.000075000, 0.000103000),
         4: (0.000031200, 0.000140600, 0.000171800),
         8: (0.000027600, 0.000174000, 0.000201600)},
    5:  {1: (0.000821200, 0.000014400, 0.000835600),
         2: (0.000032400, 0.000084200, 0.000116600),
         4: (0.000032000, 0.000134200, 0.000166200),
         8: (0.000036000, 0.000235000, 0.000271000)},
    6:  {1: (0.000830600, 0.000007800, 0.000838400),
         2: (0.000033400, 0.000080400, 0.000113800),
         4: (0.000033000, 0.000135200, 0.000168200),
         8: (0.000040800, 0.000214600, 0.000255400)},
    7:  {1: (0.000379000, 0.000017200, 0.000396200),
         2: (0.000035600, 0.000086000, 0.000121600),
         4: (0.000035600, 0.000144000, 0.000179600),
         8: (0.000037200, 0.000220600, 0.000257800)},
    8:  {1: (0.000816800, 0.000078000, 0.000894800),
         2: (0.000043800, 0.000148400, 0.000192200),
         4: (0.000042800, 0.000181200, 0.000224000),
         8: (0.000045000, 0.000316200, 0.000361200)},
    9:  {1: (0.000822400, 0.000209200, 0.001031600),
         2: (0.000029600, 0.000189200, 0.000218800),
         4: (0.000031000, 0.000189200, 0.000220200),
         8: (0.000035000, 0.000279400, 0.000314400)},
    10: {1: (0.000937000, 0.003660800, 0.004597800),
         2: (0.000087400, 0.002582600, 0.002670000),
         4: (0.000092800, 0.001750600, 0.001843400),
         8: (0.000096600, 0.001450000, 0.001546600)},
}

mpi_data = {
    4:  {1: (0.269081000, 0.000006400, 0.269087400),
         2: (0.298242400, 0.000045800, 0.298288200),
         4: (0.321309000, 0.000054400, 0.321363400),
         8: (0.361682000, 0.000056400, 0.361738400)},
    5:  {1: (0.257643400, 0.000007600, 0.257651000),
         2: (0.304144000, 0.000037000, 0.304181000),
         4: (0.326890400, 0.000041000, 0.326931400),
         8: (0.377325800, 0.000213800, 0.377539600)},
    6:  {1: (0.258925200, 0.000015600, 0.258940800),
         2: (0.304658800, 0.000047000, 0.304705800),
         4: (0.315381600, 0.000040600, 0.315422200),
         8: (0.387020000, 0.000053400, 0.387073400)},
    7:  {1: (0.260584000, 0.000036000, 0.260620000),
         2: (0.302063200, 0.000093400, 0.302156600),
         4: (0.314337600, 0.000063800, 0.314401400),
         8: (0.368903600, 0.000053400, 0.368957000)},
    8:  {1: (0.262610600, 0.000228200, 0.262838800),
         2: (0.303617000, 0.000191000, 0.303808000),
         4: (0.317909800, 0.000109000, 0.318018800),
         8: (0.397400200, 0.000080600, 0.397480800)},
    9:  {1: (0.269072200, 0.000711200, 0.269783400),
         2: (0.296779200, 0.000451000, 0.297230200),
         4: (0.324299800, 0.000329000, 0.324628800),
         8: (0.387724800, 0.000220800, 0.387945600)},
    10: {1: (0.262440000, 0.004850200, 0.267290200),
         2: (0.304052400, 0.004387400, 0.308439800),
         4: (0.323637200, 0.001767800, 0.325405000),
         8: (0.377174000, 0.000857200, 0.378031200)},
}

procs = [1, 2, 4, 8]
outdir = "/home/buxman/Documents/EEE4120F/Practical-3/plots"
import os
os.makedirs(outdir, exist_ok=True)

# ---- Plot 1: Computation time comparison for energy10 ----
fig, ax = plt.subplots(figsize=(6, 4))
omp_comp = [omp_data[10][p][1] * 1000 for p in procs]  # ms
mpi_comp = [mpi_data[10][p][1] * 1000 for p in procs]
x = np.arange(len(procs))
w = 0.35
ax.bar(x - w/2, omp_comp, w, label='OpenMP', color='#2196F3')
ax.bar(x + w/2, mpi_comp, w, label='MPI', color='#FF9800')
ax.set_xlabel('Threads / Processes')
ax.set_ylabel('Computation Time (ms)')
ax.set_title('Computation Time: OpenMP vs MPI (N=10)')
ax.set_xticks(x)
ax.set_xticklabels(procs)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{outdir}/comp_time_comparison.pdf')
plt.savefig(f'{outdir}/comp_time_comparison.png', dpi=150)
plt.close()

# ---- Plot 2: Computation speedup for energy10 ----
fig, ax = plt.subplots(figsize=(6, 4))
omp_base = omp_data[10][1][1]
mpi_base = mpi_data[10][1][1]
omp_speedup = [omp_base / omp_data[10][p][1] for p in procs]
mpi_speedup = [mpi_base / mpi_data[10][p][1] for p in procs]
ax.plot(procs, omp_speedup, 'o-', label='OpenMP', color='#2196F3', linewidth=2)
ax.plot(procs, mpi_speedup, 's-', label='MPI', color='#FF9800', linewidth=2)
ax.plot(procs, procs, '--', label='Ideal', color='gray', alpha=0.5)
ax.set_xlabel('Threads / Processes')
ax.set_ylabel('Speedup ($T_1 / T_p$)')
ax.set_title('Computation Speedup (N=10)')
ax.set_xticks(procs)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{outdir}/comp_speedup.pdf')
plt.savefig(f'{outdir}/comp_speedup.png', dpi=150)
plt.close()

# ---- Plot 3: Total time comparison — stacked init+comp, side by side ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey=False)

omp_init_vals = [omp_data[10][p][0] * 1000 for p in procs]
omp_comp_vals = [omp_data[10][p][1] * 1000 for p in procs]
mpi_init_vals = [mpi_data[10][p][0] * 1000 for p in procs]
mpi_comp_vals = [mpi_data[10][p][1] * 1000 for p in procs]

x_single = np.arange(len(procs))
w_single = 0.5

ax1.bar(x_single, omp_init_vals, w_single, label='$T_{init}$', color='#90CAF9')
ax1.bar(x_single, omp_comp_vals, w_single, bottom=omp_init_vals, label='$T_{comp}$', color='#1565C0')
ax1.set_xlabel('Threads')
ax1.set_ylabel('Time (ms)')
ax1.set_title('OpenMP (N=10)')
ax1.set_xticks(x_single)
ax1.set_xticklabels(procs)
ax1.legend(fontsize=8)
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x_single, mpi_init_vals, w_single, label='$T_{init}$', color='#FFCC80')
ax2.bar(x_single, mpi_comp_vals, w_single, bottom=mpi_init_vals, label='$T_{comp}$', color='#E65100')
ax2.set_xlabel('Processes')
ax2.set_ylabel('Time (ms)')
ax2.set_title('MPI (N=10)')
ax2.set_xticks(x_single)
ax2.set_xticklabels(procs)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

fig.suptitle('Total Time Breakdown: OpenMP vs MPI', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{outdir}/total_time_comparison.pdf', bbox_inches='tight')
plt.savefig(f'{outdir}/total_time_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ---- Plot 4: Init time breakdown ----
fig, ax = plt.subplots(figsize=(6, 4))
omp_init = [omp_data[10][p][0] * 1000 for p in procs]
mpi_init = [mpi_data[10][p][0] * 1000 for p in procs]
ax.bar(x - w/2, omp_init, w, label='OpenMP', color='#2196F3')
ax.bar(x + w/2, mpi_init, w, label='MPI', color='#FF9800')
ax.set_xlabel('Threads / Processes')
ax.set_ylabel('Initialisation Time (ms)')
ax.set_title('Initialisation Time: OpenMP vs MPI (N=10)')
ax.set_xticks(x)
ax.set_xticklabels(procs)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{outdir}/init_time_comparison.pdf')
plt.savefig(f'{outdir}/init_time_comparison.png', dpi=150)
plt.close()

# ---- Plot 5: Computation time scaling across datasets ----
fig, ax = plt.subplots(figsize=(6, 4))
datasets = [4, 5, 6, 7, 8, 9, 10]
omp1 = [omp_data[d][1][1] * 1e6 for d in datasets]  # microseconds
mpi1 = [mpi_data[d][1][1] * 1e6 for d in datasets]
ax.plot(datasets, omp1, 'o-', label='OpenMP (1 thread)', color='#2196F3', linewidth=2)
ax.plot(datasets, mpi1, 's-', label='MPI (1 process)', color='#FF9800', linewidth=2)
ax.set_xlabel('Number of Cities (N)')
ax.set_ylabel('Computation Time ($\\mu$s)')
ax.set_title('Serial Computation Time vs Problem Size')
ax.set_xticks(datasets)
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(f'{outdir}/scaling_by_dataset.pdf')
plt.savefig(f'{outdir}/scaling_by_dataset.png', dpi=150)
plt.close()

# ---- Plot 6: Efficiency for energy10 ----
fig, ax = plt.subplots(figsize=(6, 4))
omp_eff = [s/p for s, p in zip(omp_speedup, procs)]
mpi_eff = [s/p for s, p in zip(mpi_speedup, procs)]
ax.plot(procs, omp_eff, 'o-', label='OpenMP', color='#2196F3', linewidth=2)
ax.plot(procs, mpi_eff, 's-', label='MPI', color='#FF9800', linewidth=2)
ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.5, label='Ideal')
ax.set_xlabel('Threads / Processes')
ax.set_ylabel('Efficiency ($S_p / p$)')
ax.set_title('Parallel Efficiency (N=10)')
ax.set_xticks(procs)
ax.set_ylim(0, 1.5)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{outdir}/efficiency.pdf')
plt.savefig(f'{outdir}/efficiency.png', dpi=150)
plt.close()

print("All plots saved to", outdir)

# Print tables for LaTeX
print("\n=== OpenMP energy10 table ===")
print("Threads & Tinit (ms) & Tcomp (ms) & Ttotal (ms) & Comp Speedup & Efficiency")
for p in procs:
    tinit, tcomp, ttotal = omp_data[10][p]
    speedup = omp_base / tcomp
    eff = speedup / p
    print(f"{p} & {tinit*1000:.4f} & {tcomp*1000:.4f} & {ttotal*1000:.4f} & {speedup:.2f} & {eff:.2f}")

print("\n=== MPI energy10 table ===")
print("Procs & Tinit (ms) & Tcomp (ms) & Ttotal (ms) & Comp Speedup & Efficiency")
for p in procs:
    tinit, tcomp, ttotal = mpi_data[10][p]
    speedup = mpi_base / tcomp
    eff = speedup / p
    print(f"{p} & {tinit*1000:.4f} & {tcomp*1000:.4f} & {ttotal*1000:.4f} & {speedup:.2f} & {eff:.2f}")

# Print all datasets table
print("\n=== All datasets, 1 thread/proc ===")
for d in datasets:
    omp_t = omp_data[d][1][1] * 1e6
    mpi_t = mpi_data[d][1][1] * 1e6
    print(f"N={d}: OMP={omp_t:.1f}us, MPI={mpi_t:.1f}us")
