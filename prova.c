#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#define MATRIX_SIZE 84
#define MATRIX_COLUMNS 6
#define MATRIX_LINES 14
//CHANGE PROCESS SIZE TO 6
#define PROCESS_SIZE 6
// #define PROCESS_SIZE 3
#define THREAD_SIZE 5


void solveMPIHalfMatrixSENDREC(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm){

    int u_dim = (Nx % 2 == 0) ? Nx / 2 + 1 : Nx / 2 + 2;

    int local_Nx = u_dim / size;
    int gather_Nx = local_Nx;
    int remainder = u_dim % size;
    if (rank < remainder) local_Nx += 1;

    if (rank == 0 || rank == size - 1) {
        local_Nx += 1;
    } else {
        local_Nx += 2;
    }

    double *u = NULL;
    if (rank == 0) {
        u = aligned_alloc(32, u_dim * Nt * sizeof(double));
        if (!u) {
            printf("Errore nell'allocazione della memoria.\n");
            MPI_Finalize();
            return;
        }
        u[(u_dim - 2)] = central_heat;
    }

    double *local_current_row = aligned_alloc(32, local_Nx * sizeof(double));
    double *next_row = aligned_alloc(32, local_Nx * sizeof(double));
    memset(local_current_row, 0, local_Nx * sizeof(double));
    memset(next_row, 0, local_Nx * sizeof(double));

    //elimina
    if (rank == size - 1){
        local_current_row[local_Nx - 2] = central_heat;
    }
    //elimina sopra

    for (int t = 0; t < Nt - 1; t++) {
        MPI_Request requests[4];
        int req_count = 0;

        printf("qui\n");
        // Exchange boundary data asynchronously
        if (rank > 0) {
            MPI_Isend(&local_current_row[1], 1, MPI_DOUBLE, rank - 1, 1, comm, &requests[req_count++]);
            MPI_Irecv(&local_current_row[0], 1, MPI_DOUBLE, rank - 1, 2, comm, &requests[req_count++]);
        }
        if (rank < size - 1) {
            MPI_Isend(&local_current_row[local_Nx - 2], 1, MPI_DOUBLE, rank + 1, 2, comm, &requests[req_count++]);
            MPI_Irecv(&local_current_row[local_Nx - 1], 1, MPI_DOUBLE, rank + 1, 1, comm, &requests[req_count++]);
        }
        printf("dopo qui\n");
        // Wait for all non-blocking communication to finish
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Compute the next row
        for (int i = 1; i < local_Nx - 1; i++) {
            next_row[i] = s * local_current_row[i - 1] +
                          (1 - 2 * s) * local_current_row[i] +
                          s * local_current_row[i + 1];
        }

        if (rank == 0) {
            next_row[0] = local_current_row[0];  // Left boundary condition
        }
        if (rank == size - 1) {
            next_row[local_Nx - 1] = next_row[local_Nx - 3];  // Symmetry at right boundary
        }
        printf("prima di gather\n");
        MPI_Gather(&local_current_row[1], gather_Nx, MPI_DOUBLE, &u[(t) * u_dim + 1], gather_Nx, MPI_DOUBLE, 0, comm);
        if (rank == 0){
            memcpy(&u[(t + 1) * u_dim], local_current_row, gather_Nx * sizeof(double));
        }
        printf("dopo di gather\n t= %d rank %d\n", t, rank);


        // Prepare for next iteration
        memcpy(local_current_row, next_row, local_Nx * sizeof(double));

    }

    if (rank == 0) {

        free(u);
    }

    free(local_current_row);
    free(next_row);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    double elapsed_time, start_time, end_time;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start_time = MPI_Wtime();  
    solveMPIHalfMatrixSENDREC(131072.0, 1048576.0,100.0, 0.31, rank, size, 3, MPI_COMM_WORLD);  // Passa il nuovo comunicatore
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;

    if (rank == 0){
        printf("  time %.9f  ", elapsed_time);
    }
    

    MPI_Finalize();

    return 0;


}



