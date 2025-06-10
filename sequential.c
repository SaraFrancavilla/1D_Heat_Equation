#include "all_implementation.h"
#include <string.h>


void solveSeqFullMatrix(int Nx, int Nt, float central_heat, double s, int index) {

    // Allocating memory for a mono-dimensional array
    double *u = (double *)calloc(Nx * Nt, sizeof(double));
    if (u == NULL) {
        printf("Errore nell'allocazione della memoria.\n");
        return;
    }

    u[Nx / 2] = central_heat;  // Heat source in the middle of the first row

    // Temporal iteration using the explicit formula
    for (int j = 0; j < Nt - 1; j++) {
        for (int i = 1; i < Nx - 1; i++) {
            u[i + (j + 1)*Nx] = s * u[i - 1 + (j*Nx)] + (1 - 2 * s) * u[i + (j*Nx)] + s * u[i + 1 + (j*Nx)];
        }
    }

    if (index == 1) compareResultsFile(Nx, &u[(Nt - 1)*Nx], 0);

    // Deallocating memory
    free(u);
}

void solveSeqHalfMatrix(int Nx, int Nt, float central_heat, double s, int index) {

    int u_dim;

    if (Nx % 2 == 0){
        u_dim = Nx / 2 + 1; 
    }else{
        u_dim = Nx / 2 + 2; 
    }

    double *u = (double *)calloc(u_dim * Nt, sizeof(double));
    if (u == NULL) {
        printf("Errore nell'allocazione della memoria.\n");
        return;
    }

    u[u_dim - 2] = central_heat;

    for (int j = 0; j < Nt - 1; j++) {
        for (int i = 1; i < u_dim ; i++) {
            if (i == u_dim - 1){
                u[i + (j + 1)*u_dim] = u[(i - 2) + (j + 1)*u_dim];
            }else{
                u[i + (j + 1)*u_dim] = s * u[(i - 1) + j*u_dim] + (1 - 2 * s) * u[i + j*u_dim] + s * u[(i + 1) + j*u_dim];
            }
        }
    }

    if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 1);

    free(u);
}

void solveSeqSwapVec(int Nx, int Nt, float central_heat, double s, int index){

    int u_dim;
    if (Nx % 2 == 0) {
        u_dim = Nx / 2 + 1;
    } else {
        u_dim = Nx / 2 + 2;

    double *u = (double *)calloc(u_dim * Nt, sizeof(double));
    if (u == NULL) {
        printf("Errore nell'allocazione della memoria.\n");
        return;
    }

    double *swap_v = (double *)calloc(u_dim, sizeof(double));

    u[(u_dim - 2)] = central_heat; 

    for (int j = 0; j < Nt - 1; j++) {
        
        // Computing the new line (swap_v)
        for (int i = 1; i < u_dim; i++) { 
            if (i == u_dim - 1){
                swap_v[i] = swap_v[i-2];
            }else{
            swap_v[i] = s * u[(i - 1) + j*u_dim] + (1 - 2 * s) * u[i + j*u_dim] + s * u[(i + 1) + j*u_dim];
            }
        }
        
        memcpy(&u[(j+1)*u_dim], swap_v, sizeof(double) * u_dim); // Copying swap_v in u
    }

    if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 2);

    free(u);
    free(swap_v);
}

