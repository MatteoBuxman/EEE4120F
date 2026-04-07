// =========================================================================
// Practical 3: Minimum Energy Consumption Freight Route Optimization
// =========================================================================
//
// GROUP NUMBER:
//
// MEMBERS:
//   - Matteo Buxman, BXMMAT001
//   - Emmanuel Basua, BSXEMM001

// ========================================================================
//  PART 2: Minimum Energy Consumption Freight Route Optimization using OpenMPI
// =========================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include "mpi.h"

#define MAX_N 10

// ============================================================================
// Global variables
// ============================================================================

int n; // If this is -1, it signals an error/exit
int adj[MAX_N][MAX_N];

// Per-process best solution
int best_cost;
int best_path[MAX_N];

// ============================================================================
// Timer: returns time in seconds
// ============================================================================

double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// ============================================================================
// Usage function
// ============================================================================

void Usage(char *program) {
  printf("Usage: mpirun -np <num> %s [options]\n", program);
  printf("-i <file>\tInput file name\n");
  printf("-o <file>\tOutput file name\n");
  printf("-h \t\tDisplay this help\n");
}

// ============================================================================
// Branch and Bound: recursive DFS with pruning
// ============================================================================

void branch_and_bound(int depth, int current_cost, int path[], int visited)
{
    if (depth == n) {
        // Complete path found - check if it's the best
        if (current_cost < best_cost) {
            best_cost = current_cost;
            memcpy(best_path, path, n * sizeof(int));
        }
        return;
    }

    int last_city = path[depth - 1];

    for (int i = 0; i < n; i++) {
        if (!(visited & (1 << i))) {
            int new_cost = current_cost + adj[last_city][i];
            // Prune: only explore if partial cost is already less than best known
            if (new_cost < best_cost) {
                path[depth] = i;
                branch_and_bound(depth + 1, new_cost, path, visited | (1 << i));
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    int rank, nprocs;
    int opt;
    int i, j;
    char *input_file = NULL;
    char *output_file = NULL;
    FILE *infile = NULL;
    FILE *outfile = NULL;
    int success_flag = 1; // 1 = good, 0 = error/help encountered

    double t_init_start, t_init_end, t_comp_start, t_comp_end;

    // Start timing initialisation
    t_init_start = gettime();

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);


    if (rank == 0) {
        n = -1;

        while ((opt = getopt(argc, argv, "i:o:h")) != -1)
        {
            switch (opt)
            {
                case 'i':
                    input_file = optarg;
                    break;

                case 'o':
                    output_file = optarg;
                    break;

                case 'h':
                    Usage(argv[0]);
                    success_flag = 0;
                    break;

                default:
                    Usage(argv[0]);
                    success_flag = 0;
            }
        }



        if (success_flag) {
            infile = fopen(input_file, "r");
            if (infile == NULL) {
                fprintf(stderr, "Error: Cannot open input file '%s'\n", input_file);
                perror("");
                success_flag = 0;
            } else {

                fscanf(infile, "%d", &n);

                for (i = 1; i < n; i++)
                {
                    for (j = 0; j < i; j++)
                    {
                        fscanf(infile, "%d", &adj[i][j]);
                        adj[j][i] = adj[i][j];
                    }
                }
                fclose(infile);
            }
        }
        if (success_flag) {
            outfile = fopen(output_file, "w");
            if (outfile == NULL) {
                fprintf(stderr, "Error: Cannot open output file '%s'\n", output_file);
                perror("");
                success_flag = 0;
            }
        }

    }

    // Broadcast n to all processes (n == -1 signals error/exit)
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (n == -1) {
        MPI_Finalize();
        return 0;
    }

    // Broadcast adjacency matrix to all processes
    MPI_Bcast(&adj[0][0], MAX_N * MAX_N, MPI_INT, 0, MPI_COMM_WORLD);

    // End initialisation timing
    t_init_end = gettime();

    // ================================================================
    // Computation: Parallel Branch and Bound
    // ================================================================
    // Strategy: distribute first-level branches (from city 0) among processes.
    // City 0 is the starting hub. The (n-1) possible first moves go to
    // cities 1, 2, ..., n-1. Each process handles a subset of these branches
    // in round-robin fashion.
    // ================================================================

    t_comp_start = gettime();

    best_cost = INT_MAX;
    int path[MAX_N];
    path[0] = 0; // Always start at city 0 (City 1 in 1-indexed)

    // Each process takes branches where (branch_index % nprocs == rank)
    // branch_index 0 => first city = 1, branch_index 1 => first city = 2, etc.
    for (int b = rank; b < n - 1; b += nprocs) {
        int first_city = b + 1; // Cities 1..n-1
        path[1] = first_city;
        int initial_cost = adj[0][first_city];
        int visited = (1 << 0) | (1 << first_city);

        if (initial_cost < best_cost) {
            branch_and_bound(2, initial_cost, path, visited);
        }
    }

    // ================================================================
    // Gather results: find the global minimum across all processes
    // ================================================================

    // Find the global minimum cost and the rank which holds it in one reduction
    struct { int cost; int rank; } local_val, global_val;
    local_val.cost = best_cost;
    local_val.rank = rank;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

    int global_best_cost = global_val.cost;
    int global_winner = global_val.rank;

    // The winner sends its best path to rank 0
    int final_path[MAX_N];
    if (global_winner == 0) {
        memcpy(final_path, best_path, n * sizeof(int));
    } else {
        if (rank == global_winner) {
            MPI_Send(best_path, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            MPI_Recv(final_path, n, MPI_INT, global_winner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    t_comp_end = gettime();

    // ================================================================
    // Output results (rank 0 only)
    // ================================================================

    if (rank == 0) {
        double t_init = t_init_end - t_init_start;
        double t_comp = t_comp_end - t_comp_start;
        double t_total = t_init + t_comp;

        printf("Cities: %d\n", n);
        printf("Processes: %d\n", nprocs);
        printf("Optimal Route: ");
        for (i = 0; i < n; i++) {
            printf("%d", final_path[i] + 1); // Convert to 1-indexed
            if (i < n - 1) printf(" -> ");
        }
        printf("\n");
        printf("Minimum Energy: %d\n", global_best_cost);
        printf("Initialisation Time: %.6f seconds\n", t_init);
        printf("Computation Time:    %.6f seconds\n", t_comp);
        printf("Total Time:          %.6f seconds\n", t_total);

        // Write to output file
        if (outfile != NULL) {
            fprintf(outfile, "Cities: %d\n", n);
            fprintf(outfile, "Processes: %d\n", nprocs);
            fprintf(outfile, "Optimal Route: ");
            for (i = 0; i < n; i++) {
                fprintf(outfile, "%d", final_path[i] + 1);
                if (i < n - 1) fprintf(outfile, " -> ");
            }
            fprintf(outfile, "\n");
            fprintf(outfile, "Minimum Energy: %d\n", global_best_cost);
            fprintf(outfile, "Initialisation Time: %.6f seconds\n", t_init);
            fprintf(outfile, "Computation Time:    %.6f seconds\n", t_comp);
            fprintf(outfile, "Total Time:          %.6f seconds\n", t_total);
            fclose(outfile);
        }
    }

    MPI_Finalize();
    return 0;
}
