#include "all_implementation.h"
void solveOMPHalfMatrixUNROLLL(int Nx, int Nt, float central_heat, double s, int index){

    int u_dim;
    double a, b, c;

    if (Nx % 2 == 0){
        u_dim = Nx / 2 + 1; 
    }else{
        u_dim = Nx / 2 + 2; 
    }

    // Allocating memory for a mono-dimensional array
    double *u = aligned_alloc(32, u_dim*Nt*sizeof(double));
    if (u == NULL) {
        printf("Errore nell'allocazione della memoria.\n");
        return;
    }
    memset(u, 0, u_dim*Nt*sizeof(double));

    u[u_dim - 2] = central_heat;

    for (int j = 0; j < Nt - 1; j++) {
        #pragma unroll full
        #pragma omp for simd aligned(u:32) schedule(static) safelen(8)
        for (int i = 1; i < u_dim ; i++) {
        
            a = s * u[(i - 1) + j*u_dim];
            b = (1 - 2 * s) * u[i + j*u_dim];
            c = s * u[(i + 1) + j*u_dim];
            u[i + (j + 1)*u_dim] = a + b+ c;
            
        }
        #pragma omp single
        u[u_dim - 1 + (j + 1)*u_dim] = u[u_dim - 3 + (j + 1)*u_dim];
        #pragma omp barrier
    }

    if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 3);

    free(u);
}

void solveOMPHalfMatrixCOLLAPSE(int Nx, int Nt, float central_heat, double s, int index){

    int u_dim;

    double a, b, c;

    if (Nx % 2 == 0){
        u_dim = Nx / 2 + 1; 
    }else{
        u_dim = Nx / 2 + 2; 
    }

    double *u = aligned_alloc(32, u_dim*Nt*sizeof(double));
    if (u == NULL) {
        printf("Errore nell'allocazione della memoria.\n");
        return;
    }
    memset(u, 0, u_dim*Nt*sizeof(double));

    u[(u_dim - 2)] = central_heat;

    int block_size = u_dim / omp_get_num_threads();
    for (int j = 0; j < Nt - 1; j++) {
        #pragma omp for schedule (static, block_size)
        for (int i = 1; i < u_dim ; i++) {
        
            a = s * u[(i - 1) + j*u_dim];
            b = (1 - 2 * s) * u[i + j*u_dim];
            c = s * u[(i + 1) + j*u_dim];
            u[i + (j + 1)*u_dim] = a + b+ c;
            
        }
        u[u_dim - 1 + (j + 1)*u_dim] = u[u_dim - 3 + (j + 1)*u_dim];
    }


    if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 4);

    free(u);
}


void solveOMPHalfMatrixALIGNED(int Nx, int Nt, float central_heat, double s, int index){

    int u_dim;
    double a, b, c;

    if (Nx % 2 == 0){
        u_dim = Nx / 2 + 1; 
    }else{
        u_dim = Nx / 2 + 2; 

    double *u = aligned_alloc(32, u_dim*Nt*sizeof(double));
    if (u == NULL) {
        printf("Errore nell'allocazione della memoria.\n");
        return;
    }
    memset(u, 0, u_dim*Nt*sizeof(double));

    u[(u_dim - 2)] = central_heat;
    
    for (int j = 0; j < Nt - 1; j++) {
        #pragma omp simd aligned(u:32)
        for (int i = 1; i < u_dim ; i++) {
            
            a = s * u[(i - 1) + j*u_dim];
            b = (1 - 2 * s) * u[i + j*u_dim];
            c = s * u[(i + 1) + j*u_dim];
            
            u[i + (j + 1)*u_dim] = a + b + c;
            
        }
        u[u_dim - 1 + (j + 1)*u_dim] = u[u_dim - 3 + (j + 1)*u_dim];
        
    }

    if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 7);

    free(u);
}


void solveOMPSwapVecALIGNED(int Nx, int Nt, float central_heat, double s, int index){

    int u_dim;
    double a, b, c;

    if (Nx % 2 == 0){
        u_dim = Nx / 2 + 1; 
    }else{
        u_dim = Nx / 2 + 2; 
    }

    double *u = aligned_alloc(32, u_dim*Nt*sizeof(double));

    memset(u, 0, u_dim*Nt*sizeof(double));

    double *swap_v = aligned_alloc(32, u_dim*sizeof(double));
    memset(swap_v, 0, u_dim*sizeof(double));

    u[(u_dim - 2)] = central_heat;

    #pragma omp simd aligned(u, swap_v :32)
    for (int j = 0; j < Nt - 1; j++) {
        for (int i = 1; i < u_dim ; i++) {

            a = s * u[(i - 1) + j*u_dim];
            b = (1 - 2 * s) * u[i + j*u_dim];
            c = s * u[(i + 1) + j*u_dim];
            
            swap_v[i] = a + b + c;
            
        }
        swap_v[u_dim - 1] = swap_v[(u_dim - 3)];
        
        for (int i = 1; i < u_dim ; i++) {
            u[i + (j + 1)*u_dim] = swap_v[i];
        }
    }

    if (index == 1) compareResultsFile(u_dim, &u[(Nt - 1)*u_dim], 8);

    free(u);
    free(swap_v);
}
