#ifndef ALL_IMPLEMENTATION_H
#define ALL_IMPLEMENTATION_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#define MATRIX_SIZE 72 //84  72
#define MATRIX_COLUMNS 6
#define MATRIX_LINES 12 //14  12
//CHANGE PROCESS SIZE TO 6
#define PROCESS_SIZE 6
// #define PROCESS_SIZE 3
#define THREAD_SIZE 5

//In order: Nx, Nt, central_heat, s, proc_num, thread_num  //dx, dt
extern const float matrix_configs[MATRIX_SIZE];
extern const int num_threads[THREAD_SIZE];
extern const int num_processes[PROCESS_SIZE];


void solveSeqFullMatrix(int Nx, int Nt, float central_heat, double s, int index);  //id : 0

void solveSeqHalfMatrix(int Nx, int Nt, float central_heat, double s, int index); //id : 1

void solveSeqSwapVec(int Nx, int Nt, float central_heat, double s, int index); //id : 2


//OMP

void solveOMPHalfMatrixUNROLLL(int Nx, int Nt, float central_heat, double s, int index); //id : 3

void solveOMPHalfMatrixCOLLAPSE(int Nx, int Nt, float central_heat, double s, int index); //id : 4

// void solveOMPHalfMatrixBARRIER(int Nx, int Nt, float central_heat, double s, int index); //id : 5

void solveOMPHalfMatrixALIGNED(int Nx, int Nt, float central_heat, double s, int index); //id : 7

void solveOMPSwapVecALIGNED(int Nx, int Nt, float central_heat, double s, int index); //id : 8

//MPI

void solveMPIHalfMatrixBROADCAST(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time); //id : 9

void solveMPIHalfMatrixBMATRIX(int Nx, int Nt, float central_heat, double s, int rank, int size, int index,  MPI_Comm comm, double *comm_time); //id : 10

void solveMPIHalfMatrixSMATRIX(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time); //id : 11

void solveMPIHalfMatrixSENDREC(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time); //id : 12

void solveMPIHalfMatrixISENDIRECV(int Nx, int Nt, float central_heat, double s, int rank, int size, int index, MPI_Comm comm, double *comm_time); //id : 13

//grouping functions
void compareResults(int Nx, int Nt, float central_heat, double s, int rank, int size, int index);
void executionTimes(int rank, int index);

void searchingOptimalConfiguration(int rank, int size, int index);

//write to file useful functions
void compareResultsFile(int dim, double *u, int id);
void create_clean_file(const char *filename);

#endif