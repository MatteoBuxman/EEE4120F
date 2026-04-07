// =========================================================================
// Practical 3: Minimum Energy Consumption Freight Route Optimization
// =========================================================================
//
// GROUP NUMBER: 18
//
// MEMBERS:
//   - Emmanuel Basua, BSXEMM001
//   - Matteo Buxman, BXMMAT001
// ========================================================================
//  PART 1: Minimum Energy Consumption Freight Route Optimization using OpenMP
// =========================================================================
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

#define MAX_N 10

// ============================================================================
// Global variables
// ============================================================================
int procs = 1;
int n;
int adj[MAX_N][MAX_N];

// ============================================================================
// Shared best solution (protected by lock)
// ============================================================================
int          best_cost;
int          best_path[MAX_N];
omp_lock_t   best_lock;

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
  printf("Usage: %s [options]\n", program);
  printf("-p <num>\tNumber of processors/threads to use\n");
  printf("-i <file>\tInput file name\n");
  printf("-o <file>\tOutput file name\n");
  printf("-h \t\tDisplay this help\n");
}

// ============================================================================
// Branch-and-Bound recursive solver
//
//   path    : cities visited so far (0-indexed)
//   visited : bitmask of visited cities
//   depth   : number of cities placed in path so far
//   cost    : accumulated energy cost so far
// ============================================================================
void bnb(int *path, int visited, int depth, int cost)
{
    /* Pruning: already as expensive as (or worse than) best known */
    if (cost >= best_cost) return;

    /* All cities visited -> complete route found */
    if (depth == n) {
        omp_set_lock(&best_lock);
        if (cost < best_cost) {
            best_cost = cost;
            memcpy(best_path, path, n * sizeof(int));
        }
        omp_unset_lock(&best_lock);
        return;
    }

    /* Try every unvisited city as the next stop */
    for (int next = 0; next < n; next++) {
        if (visited & (1 << next)) continue;

        int new_cost = cost + adj[path[depth - 1]][next];

        /* Prune before recursing */
        if (new_cost >= best_cost) continue;

        path[depth] = next;
        bnb(path, visited | (1 << next), depth + 1, new_cost);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv)
{
    double t_init_start = gettime();

    int opt;
    int i, j;
    char *input_file  = NULL;
    char *output_file = NULL;
    FILE *infile      = NULL;
    FILE *outfile     = NULL;
    int success_flag  = 1;

    while ((opt = getopt(argc, argv, "p:i:o:h")) != -1)
    {
        switch (opt)
        {
            case 'p':
                procs = atoi(optarg);
                break;
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
            if (fscanf(infile, "%d", &n) != 1) {
                fprintf(stderr, "Error: Failed to read N from input file\n");
                    return 1;
            }
            for (i = 1; i < n; i++) {
                for (j = 0; j < i; j++) {
                    if (fscanf(infile, "%d", &adj[i][j]) != 1) {
                        fprintf(stderr, "Error: Failed to read matrix value\n");
                        return 1;
                    }
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

    if (!success_flag) return 1;

    printf("Running with %d processes/threads on a graph with %d nodes\n", procs, n);

    /* ------------------------------------------------------------------ */
    /* Initialise shared best and lock                                     */
    /* ------------------------------------------------------------------ */
    best_cost = INT_MAX;
    memset(best_path, -1, sizeof(best_path));
    omp_init_lock(&best_lock);

    double t_init_end = gettime();
    double Tinit      = t_init_end - t_init_start;

    /* ------------------------------------------------------------------ */
    /* Parallel Branch-and-Bound                                           */
    /*                                                                     */
    /* City 0 is always the start (Central Distribution Hub).             */
    /* Parallelise over the (n-1) choices for the second city.            */
    /* Dynamic scheduling balances load since sub-tree sizes vary widely. */
    /* ------------------------------------------------------------------ */
    omp_set_num_threads(procs);

    double t_comp_start = gettime();

    #pragma omp parallel for schedule(dynamic, 1) \
        shared(best_cost, best_path, adj, n)
    for (int b = 0; b < n - 1; b++) {
        int second = b + 1;

        int local_path[MAX_N];
        local_path[0] = 0;
        local_path[1] = second;

        int visited = (1 << 0) | (1 << second);
        int cost    = adj[0][second];

        bnb(local_path, visited, 2, cost);
    }

    double t_comp_end = gettime();
    double Tcomp      = t_comp_end - t_comp_start;

    omp_destroy_lock(&best_lock);

    /* ------------------------------------------------------------------ */
    /* Write results to output file and stdout                             */
    /* ------------------------------------------------------------------ */
    fprintf(outfile, "Optimal route (1-indexed): ");
    printf("Optimal route (1-indexed): ");
    for (i = 0; i < n; i++) {
        fprintf(outfile, "%d", best_path[i] + 1);
        printf("%d", best_path[i] + 1);
        if (i < n - 1) {
            fprintf(outfile, " -> ");
            printf(" -> ");
        }
    }
    fprintf(outfile, "\nMinimum energy cost: %d kWh\n", best_cost);
    printf("\nMinimum energy cost: %d kWh\n", best_cost);

    fprintf(outfile, "Threads used : %d\n", procs);
    fprintf(outfile, "Tinit        : %.6f s\n", Tinit);
    fprintf(outfile, "Tcomp        : %.6f s\n", Tcomp);
    fprintf(outfile, "Ttotal       : %.6f s\n", Tinit + Tcomp);

    printf("Threads used : %d\n", procs);
    printf("Tinit        : %.6f s\n", Tinit);
    printf("Tcomp        : %.6f s\n", Tcomp);
    printf("Ttotal       : %.6f s\n", Tinit + Tcomp);

    fclose(outfile);
    return 0;
}