#include "all_implementation.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

void solveMPIHalfMatrixBROADCAST(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time){

    int u_dim = (Nx % 2 == 0) ? Nx / 2 + 1 : Nx / 2 + 2;

    int local_Nx = u_dim / size;
    int remainder = u_dim % size;
    if (rank < remainder) local_Nx += 1;

    double *u = NULL;
    if (rank == 0) {
        u = aligned_alloc(32, u_dim*Nt*sizeof(double));
        if (!u) {
            printf("Error in memory allocation.\n");
            MPI_Finalize();
            return;
        }
        u[(u_dim - 2)] = central_heat;
    }

    double *current_row = aligned_alloc(32, u_dim * sizeof(double));
    memset(current_row, 0, u_dim * sizeof(double));
    current_row[u_dim - 2] = central_heat;

    double *next_row = aligned_alloc(32, local_Nx * sizeof(double));
    memset(next_row, 0, local_Nx * sizeof(double));

    int *recvcounts = NULL;
    int *displs = NULL;
    int global_idx;

    if (rank == 0) {
        recvcounts = aligned_alloc(32, size * sizeof(int));
        displs = aligned_alloc(32, size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            recvcounts[i] = u_dim / size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    double process_comm_time = 0.0;

    for (int j = 0; j < Nt - 1; j++) {

        double comm_start = MPI_Wtime();
        MPI_Bcast(current_row, u_dim, MPI_DOUBLE, 0, comm);
        double comm_end = MPI_Wtime();
        process_comm_time = (comm_end - comm_start);

        for (int i = 0; i < local_Nx; i++) {
            global_idx = rank * (u_dim / size) + (rank < remainder ? rank : remainder) + i;
            if (global_idx != 0 ) {
                next_row[i] = s * current_row[global_idx - 1] +
                              (1 - 2 * s) * current_row[global_idx] +
                              s * current_row[global_idx + 1];
            }
            if (global_idx == u_dim-1){
                next_row[i] = next_row[i - 2];
            }
        }

        comm_start = MPI_Wtime();
        MPI_Gatherv(next_row, local_Nx, MPI_DOUBLE, current_row, recvcounts, displs, MPI_DOUBLE, 0, comm);
        comm_end = MPI_Wtime();
        process_comm_time = (comm_end - comm_start);

        if (rank == 0) {
            // Updating the matrix u with the new computed line
            for (int i = 1; i < u_dim; i++) {
                u[i + (j + 1)*u_dim] = current_row[i];
            }
        }
    }

    double max_time = 0.0;
    MPI_Reduce(&process_comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {

        if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 9);

        *comm_time = max_time;

        free(recvcounts);
        free(displs);
        free(u);
    }

    free(current_row);
    free(next_row);
}


void solveMPIHalfMatrixBMATRIX(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time){

    int u_dim = (Nx % 2 == 0) ? Nx / 2 + 1 : Nx / 2 + 2;

    int local_Nx = u_dim / size;
    int remainder = u_dim % size;
    if (rank < remainder) local_Nx += 1;

    double *u = NULL;
    if (rank == 0) {
        u = aligned_alloc(32, u_dim*Nt*sizeof(double));
        if (!u) {
            printf("Error in memory allocation.\n");
            MPI_Finalize();
            return;
        }
        u[(u_dim - 2)] = central_heat;
    }

    double *current_row = aligned_alloc(32, u_dim * sizeof(double));
    memset(current_row, 0, u_dim * sizeof(double));
    current_row[u_dim - 2] = central_heat;

    double *next_row = aligned_alloc(32, local_Nx * sizeof(double));
    memset(next_row, 0, local_Nx * sizeof(double));

    int *recvcounts = NULL;
    int *displs = NULL;
    int global_idx;

    if (rank == 0) {
        recvcounts = aligned_alloc(32, size * sizeof(int));
        displs = aligned_alloc(32, size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            recvcounts[i] = u_dim / size + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += recvcounts[i];
        }
    }

    double process_comm_time = 0.0;

    for (int j = 0; j < Nt - 1; j++) {

        if (rank == 0) {
            memcpy(current_row, &u[j * u_dim], u_dim * sizeof(double));
        }

        double comm_start = MPI_Wtime();
        MPI_Bcast(current_row, u_dim, MPI_DOUBLE, 0, comm);
        double comm_end = MPI_Wtime();
        process_comm_time = (comm_end - comm_start);

        for (int i = 0; i < local_Nx; i++) {
            global_idx = rank * (u_dim / size) + (rank < remainder ? rank : remainder) + i;
            if (global_idx != 0 ) {
                next_row[i] = s * current_row[global_idx - 1] +
                              (1 - 2 * s) * current_row[global_idx] +
                              s * current_row[global_idx + 1];
            }
            if (global_idx == u_dim-1){
                next_row[i] = next_row[i - 2];
            }
        }

        comm_start = MPI_Wtime();
        MPI_Gatherv(next_row, local_Nx, MPI_DOUBLE, &u[(j + 1) * u_dim], recvcounts, displs, MPI_DOUBLE, 0, comm);
        comm_end = MPI_Wtime();
        process_comm_time += (comm_end - comm_start);
    }

    double max_time = 0.0;
    MPI_Reduce(&process_comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {

        if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 10);
        
        *comm_time = max_time;

        free(recvcounts);
        free(displs);
        free(u);
    }

    free(current_row);
    free(next_row);
}

void solveMPIHalfMatrixSMATRIX(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time){

    int u_dim = (Nx % 2 == 0) ? Nx / 2 + 1 : Nx / 2 + 2;

    int local_Nx = u_dim / size;
    int remainder = u_dim % size;
    if (rank < remainder) local_Nx += 1;

    int gather_Nx = local_Nx;

    if (rank == 0 || rank == size - 1){
        local_Nx += 1;
    }else{
        local_Nx += 2;
    }

    double *u = NULL;
    if (rank == 0) {

        u = aligned_alloc(32, u_dim*Nt*sizeof(double));
        if (!u) {
            printf("Error in memory allocation.\n");
            MPI_Finalize();
            return;
        }
        u[(u_dim - 2)] = central_heat;
    }

    double *current_row = (rank == 0) ? (double *)aligned_alloc(32, u_dim * sizeof(double)) : NULL;
    double *local_current_row = aligned_alloc(32, (local_Nx) * sizeof(double));
    double *next_row = aligned_alloc(32, local_Nx * sizeof(double));
    memset(local_current_row, 0, local_Nx * sizeof(double));
    memset(next_row, 0, local_Nx * sizeof(double));

    int *recvcounts = NULL;
    int *displs = NULL;
    int *sendvcounts = NULL;
    int *displs_sendvcounts = NULL;

    recvcounts = aligned_alloc(32, size * sizeof(int));
    displs = aligned_alloc(32, size * sizeof(int));
    sendvcounts = aligned_alloc(32, size * sizeof(int));
    displs_sendvcounts = aligned_alloc(32, size * sizeof(int));

    int offset_rec = 0;
    int offset_send = 0;

    for (int i = 0; i < size; i++) {
        if (i == 0 || i == size - 1) {
            recvcounts[i] = u_dim / size + (i < remainder ? 1 : 0) + 1;
        }else{
            recvcounts[i] = u_dim / size + (i < remainder ? 1 : 0) + 2;
        }

        displs[i] = offset_rec;
        offset_rec += recvcounts[i] - 2;
        sendvcounts[i] = u_dim / size + (i < remainder ? 1 : 0);
        displs_sendvcounts[i] = offset_send;
        offset_send += sendvcounts[i];

    }

    if (rank == 0) {
        current_row[u_dim - 2] = central_heat; 
    }

    double process_comm_time = 0.0;

    for (int j = 0; j < Nt - 1; j++) {

        if (rank == 0) {
            memcpy(current_row, &u[j * u_dim], u_dim * sizeof(double));
        }

        double comm_start = MPI_Wtime();
        MPI_Scatterv(current_row, recvcounts, displs, MPI_DOUBLE, local_current_row, local_Nx, MPI_DOUBLE, 0, comm);
        double comm_end = MPI_Wtime();
        process_comm_time = (comm_end - comm_start);

        for (int i = 0; i < local_Nx - 1; i++) {
            int global_idx = rank * (u_dim / size) + (rank < remainder ? rank : remainder) + i;
            if (i == 0){
                if (global_idx == 0) {
                    next_row[i] = local_current_row[i];
                }
            }else{
                next_row[i] = s * local_current_row[i - 1] +
                              (1 - 2 * s) * local_current_row[i] +
                              s * local_current_row[i + 1];
                if (global_idx == u_dim-2){
                    next_row[i + 1] = next_row[i - 1];
                }
            }
        }

        if (rank == 0){

            comm_start = MPI_Wtime();
            MPI_Gatherv(next_row, gather_Nx, MPI_DOUBLE, &u[(j+1)*u_dim], sendvcounts, displs_sendvcounts, MPI_DOUBLE, 0, comm);
            comm_end = MPI_Wtime();
            process_comm_time += (comm_end - comm_start);
            
            u[(j+1)*u_dim + u_dim - 1] = u[(j+1)*u_dim + u_dim - 3];

        }else{
            comm_start = MPI_Wtime();
            MPI_Gatherv(next_row + 1, gather_Nx, MPI_DOUBLE, &u[(j+1)*u_dim], sendvcounts, displs_sendvcounts, MPI_DOUBLE, 0, comm);
            comm_end = MPI_Wtime();
            process_comm_time += (comm_end - comm_start);
        }
    }

    double max_time = 0.0;
    MPI_Reduce(&process_comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {

        if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 11);
        
        *comm_time = max_time;

        free(recvcounts);
        free(displs);
        free(u);
        free(current_row);
    }

    free(local_current_row);
    free(next_row);
}

void solveMPIHalfMatrixSENDREC(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time){

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
            printf("Error in memory allocation.\n");
            MPI_Finalize();
            return;
        }
        u[(u_dim - 2)] = central_heat;
    }

    double *local_current_row = aligned_alloc(32, local_Nx * sizeof(double));
    double *next_row = aligned_alloc(32, local_Nx * sizeof(double));
    memset(local_current_row, 0, local_Nx * sizeof(double));
    memset(next_row, 0, local_Nx * sizeof(double));

    if (rank == size - 1){
        local_current_row[local_Nx - 2] = central_heat;
    }

    double process_comm_time = 0.0;

    for (int t = 0; t < Nt - 1; t++) {

         double comm_start = MPI_Wtime();

        // Exchange boundary data asynchronously
        if (rank > 0) {
            MPI_Send(&local_current_row[1], 1, MPI_DOUBLE, rank - 1, 1, comm);
            MPI_Recv(&local_current_row[0], 1, MPI_DOUBLE, rank - 1, 2, comm, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&local_current_row[local_Nx - 2], 1, MPI_DOUBLE, rank + 1, 2, comm);
            MPI_Recv(&local_current_row[local_Nx - 1], 1, MPI_DOUBLE, rank + 1, 1, comm, MPI_STATUS_IGNORE);
        }

        double comm_end = MPI_Wtime();
        process_comm_time = (comm_end - comm_start);

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

        comm_start = MPI_Wtime();
        MPI_Gather(&local_current_row[1], gather_Nx, MPI_DOUBLE, &u[(t) * u_dim + 1], gather_Nx, MPI_DOUBLE, 0, comm);
        comm_end = MPI_Wtime();
        process_comm_time += (comm_end - comm_start);

        if (rank == 0){
            memcpy(&u[(t + 1) * u_dim], local_current_row, gather_Nx * sizeof(double));
        }

        // Prepare for next iteration
        memcpy(local_current_row, next_row, local_Nx * sizeof(double));

    }


    double max_time = 0.0;
    MPI_Reduce(&process_comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {

        if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 12);

        *comm_time = max_time;

        free(u);
    }

    free(local_current_row);
    free(next_row);
}

void solveMPIHalfMatrixISENDIRECV(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time){

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
            printf("Error in memory allocation.\n");
            MPI_Finalize();
            return;
        }
        u[(u_dim - 2)] = central_heat;
    }

    double *local_current_row = aligned_alloc(32, local_Nx * sizeof(double));
    double *next_row = aligned_alloc(32, local_Nx * sizeof(double));
    memset(local_current_row, 0, local_Nx * sizeof(double));
    memset(next_row, 0, local_Nx * sizeof(double));

    if (rank == size - 1){
        local_current_row[local_Nx - 2] = central_heat;
    }

    double process_comm_time = 0.0;

    for (int t = 0; t < Nt - 1; t++) {
        MPI_Request requests[4];
        int req_count = 0;

        double comm_start = MPI_Wtime();

        // Exchange boundary data asynchronously
        if (rank > 0) {
            MPI_Isend(&local_current_row[1], 1, MPI_DOUBLE, rank - 1, 1, comm, &requests[req_count++]);
            MPI_Irecv(&local_current_row[0], 1, MPI_DOUBLE, rank - 1, 2, comm, &requests[req_count++]);
        }
        if (rank < size - 1) {
            MPI_Isend(&local_current_row[local_Nx - 2], 1, MPI_DOUBLE, rank + 1, 2, comm, &requests[req_count++]);
            MPI_Irecv(&local_current_row[local_Nx - 1], 1, MPI_DOUBLE, rank + 1, 1, comm, &requests[req_count++]);
        }

        // Wait for all non-blocking communication to finish
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        double comm_end = MPI_Wtime();
        process_comm_time = (comm_end - comm_start);

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

        comm_start = MPI_Wtime();
        MPI_Gather(&local_current_row[1], gather_Nx, MPI_DOUBLE, &u[(t) * u_dim + 1], gather_Nx, MPI_DOUBLE, 0, comm);
        comm_end = MPI_Wtime();
        process_comm_time += (comm_end - comm_start);
        
        if (rank == 0){
            memcpy(&u[(t + 1) * u_dim], local_current_row, gather_Nx * sizeof(double));
        }

        // Prepare for next iteration
        memcpy(local_current_row, next_row, local_Nx * sizeof(double));

    }

    double max_time = 0.0;
    MPI_Reduce(&process_comm_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {

        if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 13);
        
        *comm_time = max_time;
        free(u);
    }

    free(local_current_row);
    free(next_row);
}
